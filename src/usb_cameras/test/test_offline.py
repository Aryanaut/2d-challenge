"""Offline tests (no ROS, no cameras): python3 -m pytest test/ -q
A short video file stands in for a webcam - OpenCV opens both the same way."""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from usb_cameras.capture import CameraThread, camera_info_k, open_camera  # noqa: E402


def _fake_camera(path, n=60):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
    for i in range(n):
        img = np.full((240, 320, 3), (i * 4) % 255, np.uint8)
        w.write(img)
    w.release()
    return str(path)


def test_camera_info_k():
    k = camera_info_k(640, 480, 90.0)
    assert abs(k[0] - 320) < 1e-6 and k[2] == 320 and k[5] == 240 and k[8] == 1.0


def test_open_missing_device():
    assert open_camera("/dev/video_does_not_exist") is None


def test_thread_reads_and_reconnects(tmp_path):
    src = _fake_camera(tmp_path / "cam.avi")
    logs = []
    cam = CameraThread("left", src, 320, 240, mjpg=False, log=logs.append, reconnect_s=0.1)
    cam.start()
    deadline = time.time() + 5
    while cam.frames < 5 and time.time() < deadline:
        time.sleep(0.05)
    frame, stamp = cam.latest()
    assert frame is not None and frame.shape == (240, 320, 3) and stamp is not None
    time.sleep(1.0)                       # the file ends -> treated as unplugged -> reopened
    cam.stop()
    assert any("opened" in l for l in logs)
    assert any("lost" in l for l in logs) or cam.frames >= 60


def test_missing_device_keeps_retrying_quietly():
    logs = []
    cam = CameraThread("right", "/dev/video_nope", log=logs.append, reconnect_s=0.1)
    cam.start()
    time.sleep(0.5)
    cam.stop()
    assert cam.latest() == (None, None)
    assert len(logs) == 1 and "cannot open" in logs[0]      # warned once, not spammed


# ---------- network protocol ----------
def test_protocol_roundtrip_and_sender_compat():
    import socket
    from usb_cameras.net_protocol import pack_frame, read_frame
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import cam_sender
    a, b = socket.socketpair()
    jpeg = bytes(range(256)) * 100
    a.sendall(cam_sender.pack_frame("left", 7, jpeg, 640, 480, 70.0, 123.5))   # laptop side
    header, data = read_frame(b)                                                # robot side
    assert header["camera"] == "left" and header["seq"] == 7 and header["width"] == 640 and data == jpeg
    assert pack_frame("left", 7, jpeg, 640, 480, 70.0, 123.5) == cam_sender.pack_frame("left", 7, jpeg, 640, 480, 70.0, 123.5)
    a.close(); b.close()


def test_protocol_rejects_garbage():
    import socket
    import struct
    import pytest
    from usb_cameras.net_protocol import ProtocolError, read_frame
    a, b = socket.socketpair()
    a.sendall(b"GET / HTTP/1.1\r\n\r\n")                     # someone points a browser at the port
    with pytest.raises(ProtocolError):
        read_frame(b)
    a2, b2 = socket.socketpair()
    a2.sendall(b"CAM1" + struct.pack("!II", 10, 999_999_999))  # absurd size: refused, nothing allocated
    with pytest.raises(ProtocolError):
        read_frame(b2)
    for s in (a, b, a2, b2):
        s.close()


def test_qr_message_roundtrip_and_sender_compat():
    import socket
    from usb_cameras.net_protocol import pack_qr, read_frame
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import cam_sender
    codes = [{"text": "I", "decoded": True, "corners": [[10, 10], [60, 10], [60, 60], [10, 60]], "method": "edge"}]
    msg = cam_sender.pack_qr("right", 3, 1920, 1080, 78.0, 5.0, codes, 80.0)
    assert msg == pack_qr("right", 3, 1920, 1080, 78.0, 5.0, codes, 80.0)
    a, b = socket.socketpair()
    a.sendall(msg + cam_sender.pack_frame("right", 4, b"jpg", 640, 360, 78.0, 5.1))
    header, data = read_frame(b)
    assert header["type"] == "qr" and data == b"" and header["codes"][0]["text"] == "I"
    assert header["width"] == 1920                                      # corners refer to full res
    header, data = read_frame(b)
    assert header["type"] == "frame" and data == b"jpg"
    a.close(); b.close()


def test_sender_reads_small_codes_at_full_resolution():
    """The point of reading on the laptop/Pi: a code that is lost in the 640-wide preview is read at 1080p."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import cam_sender
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.integers(80, 180, (1080, 1920), dtype=np.uint8), (7, 7), 0)
    qr = cv2.QRCodeEncoder.create().encode("J")
    side = int(60 * qr.shape[0] / (qr.shape[0] - 8))                   # 60 px black square
    img[500:500 + side, 1300:1300 + side] = cv2.resize(qr, (side, side), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_GRAY2BGR)
    reader = cam_sender.QrReader()
    hits, _ = reader.read(img)
    assert [t for t, _ in hits] == ["J"]
    small_hits, _ = reader.read(cam_sender.shrink(img, 640))
    assert small_hits == []
    assert cam_sender.shrink(img, 640).shape[:2] == (360, 640)
