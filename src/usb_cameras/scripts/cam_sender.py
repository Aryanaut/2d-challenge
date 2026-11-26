#!/usr/bin/env python3
"""Send two USB webcams to the robot over the network (Windows laptop now, Raspberry Pi later),
reading the QR codes HERE at full resolution.

Needs:  pip install opencv-python numpy pyzbar      (pyzbar optional, reads small codes better)

1. Find which camera number is which (saves cam_<n>.jpg next to this script):
       python cam_sender.py --list
2. Stream to the robot (robot runs: ros2 run usb_cameras net_receiver):
       python cam_sender.py --robot 192.168.1.71 --left 0 --right 2 --preview

What happens:
  * each camera is captured at 1920x1080 (usually the webcam's widest view)
  * every camera has its own QR thread that reads the FULL 1920x1080 picture --scan-hz times
    a second -> codes about 3x further away than from a 640-wide picture
  * the robot gets a small preview (--send-width, default 640 wide) for viewing/VLM, plus the
    decoded QR results (text, corners in full-res pixels, image size, field of view).
    The robot publishes them on /left/qr_codes and /right/qr_codes; qr_scanner turns them into
    direction, distance, "report once" and the log, exactly as for the front camera.
  * every NEW code is printed here once, with what kind of content it is (links are only
    described, never opened)
Ctrl+C (or q in a preview window) to stop.

On a Raspberry Pi use fewer scans:  --fps 10 --scan-hz 2
--no-decode turns the QR reading off (then it is just a camera link, as before).
"""

import argparse
import collections
import csv
import ipaddress
import json
import os
import re
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import cv2
import numpy as np

try:
    from pyzbar import pyzbar
    ZBAR_QR = [pyzbar.ZBarSymbol.QRCODE]
except Exception:          # not installed, or the zbar DLL is missing on Windows
    pyzbar = None
    ZBAR_QR = None

MAGIC = b"CAM1"                      # must match usb_cameras/net_protocol.py on the robot
_LENS = struct.Struct("!II")
IS_WINDOWS = os.name == "nt"
MAX_CODES = 20


def pack_frame(camera, seq, jpeg, width, height, hfov_deg, t_capture):
    header = json.dumps({"type": "frame", "camera": camera, "seq": seq, "width": width, "height": height,
                         "hfov_deg": hfov_deg, "t_capture": t_capture}).encode("utf-8")
    return MAGIC + _LENS.pack(len(header), len(jpeg)) + header + jpeg


def pack_qr(camera, seq, width, height, hfov_deg, t_capture, codes, scan_ms):
    """QR result of one scan. width/height = the FULL resolution the corners refer to."""
    header = json.dumps({"type": "qr", "camera": camera, "seq": seq, "width": width, "height": height,
                         "hfov_deg": hfov_deg, "t_capture": t_capture, "scan_ms": round(scan_ms, 1),
                         "codes": codes[:MAX_CODES]}).encode("utf-8")
    return MAGIC + _LENS.pack(len(header), 0) + header


# ---------------------------------------------------------------- what is in the code?
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
              "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc", "s.id", "qrco.de", "lnkd.in"}
_DOMAIN_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?::\d{1,5})?(?:[/?#].*)?$", re.I)


def classify(text: str) -> dict:
    """Describe a QR payload. Never contacts the network."""
    t = text.strip()
    low = t.lower()
    info = {"kind": "text", "is_link": False, "summary": t if len(t) <= 60 else t[:57] + "...", "warnings": []}

    if low.startswith(("http://", "https://")) or low.startswith("www.") or (" " not in t and _DOMAIN_RE.match(t)):
        url = t if low.startswith(("http://", "https://")) else "http://" + t
        p = urlparse(url)
        host = (p.hostname or "").lower()
        info.update(kind="website link", is_link=True, url=t, domain=host,
                    secure=p.scheme == "https", summary=f"LINK -> {host}{p.path if p.path not in ('', '/') else ''}")
        if p.scheme != "https":
            info["warnings"].append("not https (unencrypted)" if low.startswith("http://") else "no https:// given")
        try:
            ipaddress.ip_address(host)
            info["warnings"].append("points to a raw IP address, not a named website")
        except ValueError:
            pass
        if host in SHORTENERS:
            info["warnings"].append("URL shortener - the real destination is hidden")
        if host.startswith("xn--") or ".xn--" in host:
            info["warnings"].append("internationalised domain (can imitate other sites)")
        if p.username or "@" in (p.netloc or ""):
            info["warnings"].append("contains '@' in the address (can disguise the real site)")
        return info
    prefixes = [("wifi:", "Wi-Fi login"), ("mailto:", "email address"), ("matmsg:", "email message"),
                ("tel:", "phone number"), ("smsto:", "SMS message"), ("sms:", "SMS message"),
                ("geo:", "map location"), ("begin:vcard", "contact card (vCard)"), ("mecard:", "contact card"),
                ("begin:vevent", "calendar event"), ("bitcoin:", "crypto payment"), ("upi:", "payment request")]
    for prefix, kind in prefixes:
        if low.startswith(prefix):
            info.update(kind=kind)
            if prefix == "wifi:":
                m = re.search(r"S:([^;]*)", t)
                info["summary"] = f"Wi-Fi network '{m.group(1) if m else '?'}'"
            elif prefix in ("bitcoin:", "upi:"):
                info["warnings"].append("payment request")
            return info
    if len(t) == 1 and t.isalnum():
        info["kind"] = "single character (arena clue)"
    elif t.startswith(("{", "[")):
        info["kind"] = "JSON / structured data"
    return info


# ---------------------------------------------------------------- reading QR codes
def _quad(pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    return cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32) if len(pts) != 4 else pts


def _side(q):
    return float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]))


class QrReader:
    """Same approach as the robot's qr_scanner: zbar on the full frame, then find
    small/unread codes and zoom into them; anything still unread is reported as 'seen'."""

    def __init__(self):
        self.decoder = cv2.QRCodeDetector()
        try:
            params = cv2.QRCodeDetectorAruco_Params()
            params.minModuleSizeInPyramid = 1.0
            self.finder = cv2.QRCodeDetectorAruco(params)
        except AttributeError:
            self.finder = cv2.QRCodeDetector()
        self.clahe = cv2.createCLAHE(2.0, (8, 8))

    def _zbar(self, gray, scale=1.0, off=(0, 0)):
        if pyzbar is None:
            return []
        out = []
        for d in pyzbar.decode(gray, symbols=ZBAR_QR):
            try:
                text = d.data.decode("utf-8")
            except UnicodeDecodeError:
                text = d.data.decode("latin-1")
            out.append((text, _quad([(p.x, p.y) for p in d.polygon]) / scale + np.float32(off)))
        return out

    def _cv(self, gray, scale=1.0, off=(0, 0)):
        try:
            ok, texts, pts, _ = self.decoder.detectAndDecodeMulti(gray)
        except cv2.error:
            return []
        if not ok or pts is None:
            return []
        return [(t, _quad(q) / scale + np.float32(off)) for t, q in zip(texts, pts) if t]

    def read(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        hits = self._zbar(gray) or self._cv(gray)
        try:
            ok, quads = self.finder.detectMulti(gray)
        except cv2.error:
            ok, quads = False, None
        seen = []
        for q in ([] if not ok or quads is None else [_quad(x) for x in quads]):
            c = q.mean(axis=0)
            if any(np.linalg.norm(h[1].mean(axis=0) - c) < 0.6 * _side(h[1]) + 10 for h in hits):
                continue
            s = _side(q)
            half = 0.9 * s + 8
            x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
            x1, y1 = int(min(W, c[0] + half)), int(min(H, c[1] + half))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            scale = float(np.clip(240 / max(s, 1), 1, 8))
            up = cv2.resize(gray[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            got = []
            for img in (up, self.clahe.apply(up)):
                got = self._zbar(img, scale, (x0, y0)) or self._cv(img, scale, (x0, y0))
                if got:
                    break
            if got:
                hits += got
            else:
                seen.append(q)
        return hits, seen


# ---------------------------------------------------------------- cameras
def open_camera(source, width, height, fps, exposure=None):
    src = int(source) if str(source).isdigit() else source
    if IS_WINDOWS and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)       # DirectShow: opens fast, honours MJPG
    elif isinstance(src, str) and src.startswith("/dev/"):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # needed for 1080p and two cams on one bus
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if exposure is not None:
        # Webcams lengthen the exposure in dim light and then CANNOT keep the frame rate. A fixed short
        # exposure keeps fps up AND reduces motion blur (helps QR reading while walking).
        # Windows (DirectShow): value is log2(seconds): -5 = 1/32 s, -6 = 1/64 s, -7 = 1/128 s.
        # Linux/Pi (V4L2): value is in 100 us units: 100 = 10 ms, 50 = 5 ms.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if IS_WINDOWS else 1)   # manual exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap


class Camera(threading.Thread):
    def __init__(self, name, source, width, height, fps, exposure=None):
        super().__init__(daemon=True)
        self.name, self.source, self.exposure = name, source, exposure
        self.width, self.height, self.fps = width, height, fps
        self.lock = threading.Lock()
        self.frame, self.stamp, self.frames = None, None, 0
        self.running = True

    def latest(self):
        with self.lock:
            return self.frame, self.stamp

    def run(self):
        cap, fails = None, 0
        while self.running:
            if cap is None:
                cap = open_camera(self.source, self.width, self.height, self.fps, self.exposure)
                if cap is None:
                    print(f"[{self.name}] cannot open camera {self.source}, retrying...")
                    time.sleep(2)
                    continue
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[{self.name}] opened camera {self.source} at {w}x{h}, "
                      f"camera reports {cap.get(cv2.CAP_PROP_FPS):.0f} fps max")
                if (w, h) != (self.width, self.height) and str(self.source).isdigit():
                    print(f"[{self.name}] NOTE: asked for {self.width}x{self.height}, camera gave {w}x{h}")
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:
                    print(f"[{self.name}] camera {self.source} stopped giving frames, reopening")
                    cap.release()
                    cap, fails = None, 0
                time.sleep(0.02)
                continue
            fails = 0
            with self.lock:
                self.frame, self.stamp = frame, time.time()
                self.frames += 1
        if cap is not None:
            cap.release()


# ---------------------------------------------------------------- QR reading at full resolution
class Decoder(threading.Thread):
    """Reads QR codes from the newest FULL-RES frame of one camera, scan_hz times a second.
    Each result goes into `outbox` for the robot; new codes are printed once."""

    def __init__(self, cam, args, outbox, log_lock):
        super().__init__(daemon=True)
        self.cam, self.args, self.outbox, self.log_lock = cam, args, outbox, log_lock
        self.reader = QrReader()             # one per thread: OpenCV detectors are not thread-safe
        self.lock = threading.Lock()
        self.result = None                   # (stamp, hits, seen, (w, h)) for the preview
        self.scans, self.scan_ms, self.seq = 0, 0.0, 0
        self.last_print = {}                 # text -> time it was last announced
        self.running = True

    def latest(self):
        with self.lock:
            return self.result

    def run(self):
        last_stamp = None
        period = 1.0 / max(0.1, self.args.scan_hz)
        while self.running:
            frame, stamp = self.cam.latest()
            if frame is None or stamp == last_stamp:
                time.sleep(0.01)
                continue
            last_stamp = stamp
            t0 = time.time()
            hits, seen = self.reader.read(frame)
            ms = (time.time() - t0) * 1000
            h, w = frame.shape[:2]
            codes = [{"text": t, "decoded": True, "corners": np.round(q, 1).tolist(), "method": "edge"}
                     for t, q in hits]
            codes += [{"text": "", "decoded": False, "corners": np.round(q, 1).tolist(), "method": "edge-seen"}
                      for q in seen]
            msg = pack_qr(self.cam.name, self.seq, w, h, self.args.hfov, stamp, codes, ms)
            self.seq += 1
            if len(self.outbox) == self.outbox.maxlen:
                self.outbox.popleft()            # robot not connected: keep only the newest results
            self.outbox.append(msg)
            with self.lock:
                self.result = (stamp, hits, seen, (w, h))
                self.scans += 1
                self.scan_ms = 0.8 * self.scan_ms + 0.2 * ms if self.scan_ms else ms
            for text, quad in hits:
                self._announce(text, quad, w)
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _announce(self, text, quad, width):
        now = time.time()
        if now - self.last_print.get(text, -1e9) < 10.0:       # same code again within 10 s: quiet
            self.last_print[text] = now
            return
        self.last_print[text] = now
        info = classify(text)
        fx = (width / 2) / np.tan(np.radians(self.args.hfov) / 2)
        side = _side(quad)
        dist = fx * self.args.qr_size / max(side, 1)
        lines = [f"\n[{self.cam.name}] QR {text!r}  ({info['kind']}, {side:.0f} px, ~{dist:.2f} m)"]
        if info["is_link"]:
            lines.append(f"    website: {info['domain']}  https: {'yes' if info['secure'] else 'NO'}")
        lines += [f"    WARNING: {w}" for w in info["warnings"]]
        print("\n".join(lines))
        if self.args.log:
            with self.log_lock:
                try:
                    new_file = not os.path.exists(self.args.log)
                    with open(self.args.log, "a", newline="", encoding="utf-8") as f:
                        wr = csv.writer(f)
                        if new_file:
                            wr.writerow(["time", "camera", "text", "type", "is_link", "domain", "warnings",
                                         "size_px", "distance_m"])
                        wr.writerow([time.strftime("%H:%M:%S"), self.cam.name, text, info["kind"], info["is_link"],
                                     info.get("domain", ""), "; ".join(info["warnings"]), round(side),
                                     round(dist, 2)])
                except OSError:
                    pass


# ---------------------------------------------------------------- helpers
def list_cameras(max_index=6):
    here = os.path.dirname(os.path.abspath(__file__))
    print("Probing camera numbers 0..%d (the laptop's built-in camera is usually 0)" % (max_index - 1))
    for i in range(max_index):
        cap = open_camera(i, 1920, 1080, 15)
        if cap is None:
            print(f"  {i}: -")
            continue
        frame = None
        t0 = time.time()
        while time.time() - t0 < 2.0:
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            print(f"  {i}: opens but gives no picture")
            continue
        path = os.path.join(here, f"cam_{i}.jpg")
        cv2.imwrite(path, frame)
        print(f"  {i}: OK {frame.shape[1]}x{frame.shape[0]}  -> {path}")
    print("Open the cam_<n>.jpg files (or cover one camera and run again) to see which is left/right.")


def shrink(frame, width):
    h, w = frame.shape[:2]
    if width <= 0 or w <= width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- network loop
def stream(args, cams, decoders, outbox):
    seq = {c.name: 0 for c in cams}
    sent_stamp = {c.name: None for c in cams}
    stats_t, stats = time.time(), {c.name: [0, 0] for c in cams}     # frames, bytes
    cap_count = {c.name: c.frames for c in cams}
    scan_count = {d.cam.name: d.scans for d in decoders}
    period = 1.0 / args.fps
    sock = None
    pool = ThreadPoolExecutor(max_workers=max(1, len(cams)))   # shrink + JPEG-encode cameras in parallel

    def encode(frame):
        small = shrink(frame, args.send_width)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        return (buf.tobytes(), small.shape[1], small.shape[0]) if ok else None

    while True:
        if sock is None:
            try:
                sock = socket.create_connection((args.robot, args.port), timeout=3)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(5)
                print(f"connected to robot {args.robot}:{args.port}")
                outbox.clear()                   # don't flood the robot with stale results
            except OSError as e:
                print(f"cannot reach robot {args.robot}:{args.port} ({e}) - is net_receiver running? retrying...")
                sock = None
                if not preview_wait(cams, decoders, args, 2000):
                    return
                continue
        t0 = time.time()
        try:
            while outbox:                        # QR results first: they are tiny and matter most
                sock.sendall(outbox.popleft())
            jobs = []
            for c in cams:
                frame, stamp = c.latest()
                if frame is None or stamp == sent_stamp[c.name]:
                    continue
                jobs.append((c, stamp, pool.submit(encode, frame)))
            for c, stamp, job in jobs:
                res = job.result()
                if res is None:
                    continue
                jpeg, w, h = res
                sock.sendall(pack_frame(c.name, seq[c.name], jpeg, w, h, args.hfov, stamp))
                seq[c.name] += 1
                sent_stamp[c.name] = stamp
                stats[c.name][0] += 1
                stats[c.name][1] += len(jpeg)
        except OSError as e:
            print(f"connection lost ({e}), reconnecting...")
            try:
                sock.close()
            except OSError:
                pass
            sock = None
            continue
        if time.time() - stats_t >= 5:
            dt = time.time() - stats_t
            parts = []
            for c in cams:
                s = (f"{c.name}: camera {(c.frames - cap_count[c.name]) / dt:.1f} fps, "
                     f"sent {stats[c.name][0] / dt:.1f} fps, {stats[c.name][1] / dt / 1024:.0f} kB/s")
                d = next((d for d in decoders if d.cam is c), None)
                if d is not None:
                    s += f", QR scans {(d.scans - scan_count[c.name]) / dt:.1f}/s ({d.scan_ms:.0f} ms each)"
                    scan_count[c.name] = d.scans
                parts.append(s)
            print("  |  ".join(parts))
            stats_t, stats = time.time(), {c.name: [0, 0] for c in cams}
            cap_count = {c.name: c.frames for c in cams}
        wait_ms = max(1, int((period - (time.time() - t0)) * 1000))
        if not preview_wait(cams, decoders, args, wait_ms):
            return


def preview_wait(cams, decoders, args, wait_ms):
    """Show preview windows (if asked) and wait. Returns False when the user presses q."""
    if not args.preview:
        time.sleep(wait_ms / 1000.0)
        return True
    for c in cams:
        frame, _ = c.latest()
        if frame is None:
            continue
        view = shrink(frame, 960)
        k = view.shape[1] / frame.shape[1]
        d = next((d for d in decoders if d.cam is c), None)
        res = d.latest() if d else None
        if res is not None:
            _, hits, seen, _ = res
            for text, quad in hits:
                q = (quad * k).astype(np.int32)
                cv2.polylines(view, [q], True, (0, 200, 0), 2)
                x, y = q.min(axis=0)
                cv2.putText(view, f"{classify(text)['summary'][:30]} ({_side(quad):.0f}px)",
                            (int(x), int(max(18, y - 6))), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            for quad in seen:
                q = (quad * k).astype(np.int32)
                cv2.polylines(view, [q], True, (0, 140, 255), 2)
        h, w = frame.shape[:2]
        cv2.putText(view, f"{c.name}  capture {w}x{h}  sending {args.send_width or w} wide",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(f"{c.name} camera", view)
    return (cv2.waitKey(wait_ms) & 0xFF) != ord("q")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="find camera numbers and save a photo from each")
    ap.add_argument("--robot", default="192.168.1.71", help="robot IP address")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--left", default="1", help="left camera number (or device path / video file)")
    ap.add_argument("--right", default="2", help="right camera number; 'none' to send one camera")
    ap.add_argument("--width", type=int, default=1920, help="capture width (full resolution for QR reading)")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--send-width", type=int, default=640, help="width of the images sent to the robot (0 = full)")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality of the sent images")
    ap.add_argument("--hfov", type=float, default=70.0,
                    help="webcam horizontal field of view (deg) IN THE CAPTURE MODE (1080p is often wider)")
    ap.add_argument("--no-decode", action="store_true", help="don't read QR codes here (robot does it)")
    ap.add_argument("--scan-hz", type=float, default=4.0, help="QR scans per second per camera")
    ap.add_argument("--qr-size", type=float, default=0.09, help="printed QR black square width (m), for distances")
    ap.add_argument("--log", default="qr_sender_log.csv", help="CSV of new codes ('' = no log)")
    ap.add_argument("--preview", action="store_true", help="show the camera images on this computer")
    ap.add_argument("--exposure", type=float, default=None,
                    help="fixed exposure: keeps fps up and blur down (Windows: -6 = 1/64 s; Linux: 100 = 10 ms)")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return
    cams = [Camera(n, s, args.width, args.height, args.fps, args.exposure)
            for n, s in (("left", args.left), ("right", args.right)) if str(s).lower() != "none"]
    if not cams:
        sys.exit("no cameras selected")
    outbox = collections.deque(maxlen=50)
    log_lock = threading.Lock()
    decoders = [] if args.no_decode else [Decoder(c, args, outbox, log_lock) for c in cams]
    if decoders:
        print(f"QR reading here at full resolution, {args.scan_hz:g} scans/s per camera, "
              f"{'zbar + OpenCV' if pyzbar else 'OpenCV only (pip install pyzbar reads small codes better)'}")
    for t in cams + decoders:
        t.start()
    try:
        stream(args, cams, decoders, outbox)
    except KeyboardInterrupt:
        pass
    finally:
        for t in cams + decoders:
            t.running = False
        if args.preview:
            cv2.destroyAllWindows()
        print("stopped")


if __name__ == "__main__":
    main()
