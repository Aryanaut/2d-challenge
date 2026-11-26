"""Camera capture helpers (OpenCV only, no ROS) - shared by the node and the probe tool."""

import glob
import os
import re
import threading
import time

import cv2
import numpy as np


def open_camera(device, width=640, height=480, fps=15, mjpg=True):
    """Open a /dev/video* path, a /dev/v4l/by-path|by-id link, an index, or a video file.
    MJPG matters with two webcams on one USB bus: raw YUYV at 640x480 x2 often exceeds the
    bandwidth and the second camera then delivers no frames."""
    dev = device
    if isinstance(dev, str) and dev.isdigit():
        dev = int(dev)
    if isinstance(dev, str) and dev.startswith("/dev/"):
        cap = cv2.VideoCapture(os.path.realpath(dev), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        return None
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always hand out the newest frame
    except Exception:
        pass
    return cap


def camera_info_k(width, height, hfov_deg):
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return [fx, 0.0, width / 2.0, 0.0, fx, height / 2.0, 0.0, 0.0, 1.0]


def list_video_devices():
    """All /dev/video* devices plus their stable by-path / by-id names."""
    devs = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0))
    links = {}
    for d in ("/dev/v4l/by-path", "/dev/v4l/by-id"):
        for link in sorted(glob.glob(os.path.join(d, "*"))):
            links.setdefault(os.path.realpath(link), []).append(link)
    return [(d, links.get(d, [])) for d in devs]


class CameraThread:
    """Reads one camera in the background, reconnecting if it is unplugged.
    latest() returns (frame, capture_time) or (None, None)."""

    def __init__(self, name, device, width=640, height=480, fps=15, mjpg=True,
                 log=print, reconnect_s=2.0):
        self.name, self.device = name, device
        self.width, self.height, self.fps, self.mjpg = width, height, fps, mjpg
        self.log, self.reconnect_s = log, reconnect_s
        self.lock = threading.Lock()
        self.frame, self.stamp = None, None
        self.frames = 0
        self.size = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.name}")
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def latest(self):
        with self.lock:
            return self.frame, self.stamp

    def _run(self):
        cap, fails, warned = None, 0, False
        while self.running:
            if cap is None:
                cap = open_camera(self.device, self.width, self.height, self.fps, self.mjpg)
                if cap is None:
                    if not warned:
                        self.log(f"{self.name}: cannot open {self.device} - retrying every {self.reconnect_s:.0f} s")
                        warned = True
                    time.sleep(self.reconnect_s)
                    continue
                warned, fails = False, 0
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.log(f"{self.name}: opened {self.device} at {w}x{h}")
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 20:                 # unplugged or stuck: reopen
                    self.log(f"{self.name}: lost {self.device}, reconnecting")
                    cap.release()
                    cap = None
                time.sleep(0.02)
                continue
            fails = 0
            with self.lock:
                self.frame, self.stamp = frame, time.time()
                self.frames += 1
                self.size = (frame.shape[1], frame.shape[0])
        if cap is not None:
            cap.release()
