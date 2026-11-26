"""Try the QR reader without ROS.

  ros2 run qr_scanner qr_test photos/*.jpg                 # decode saved images
  ros2 run qr_scanner qr_test --camera /dev/video4         # live (q to quit if a display exists)
  ros2 run qr_scanner qr_test img.jpg --key-map I:storeroom J:kitchen --qr-size 0.09 --fx 910

Prints every code with its pixel size, the method that read it, and (if --fx is given)
the distance estimated from the printed QR size.
"""

import argparse
import glob
import os
import sys

import cv2

from .localize import depth_from_size
from .payload import decode_payload, parse_key_map
from .qr_decode import QrReader, draw


def report(name, hits, args, key_map):
    if not hits:
        print(f"{name}: no QR")
    for h in hits:
        dist = depth_from_size(h.side_px, args.fx, args.qr_size) if args.fx else None
        dist_s = f" ~{dist:.2f} m" if dist else ""
        if h.decoded:
            dec = decode_payload(h.text, args.decode_mode, args.caesar_shift, key_map)
            print(f"{name}: '{h.text}' -> '{dec}'  [{h.side_px:.0f}px, {h.method}]{dist_s}")
        else:
            print(f"{name}: QR spotted but unreadable [{h.side_px:.0f}px]{dist_s} - move closer / hold still")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--camera")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--decode-mode", default="raw")
    ap.add_argument("--caesar-shift", type=int, default=0)
    ap.add_argument("--key-map", nargs="*", default=[])
    ap.add_argument("--qr-size", type=float, default=0.09, help="printed black square width (m)")
    ap.add_argument("--fx", type=float, default=0.0, help="camera focal length in pixels (from camera_info)")
    ap.add_argument("--save", help="folder to save annotated images")
    args = ap.parse_args(argv)
    key_map = parse_key_map(args.key_map)
    reader = QrReader()

    if args.camera:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            sys.exit(f"cannot open {args.camera}")
        gui = bool(os.environ.get("DISPLAY"))
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            hits = reader.read(frame)
            if hits:
                report("live", hits, args, key_map)
            if gui:
                cv2.imshow("qr_test", draw(frame, hits))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        return

    paths = [p for pat in args.images for p in sorted(glob.glob(pat))]
    if not paths:
        ap.error("give image files or --camera")
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        hits = reader.read(img)
        report(os.path.basename(path), hits, args, key_map)
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            cv2.imwrite(os.path.join(args.save, os.path.basename(path)), draw(img, hits))


if __name__ == "__main__":
    main()
