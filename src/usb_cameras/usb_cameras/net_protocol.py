"""Wire format for sending camera frames over TCP (pure Python, shared by sender and receiver).

Each message:
    b"CAM1" | uint32 header_len | uint32 jpeg_len | header (UTF-8 JSON) | JPEG bytes
Two kinds, told apart by header["type"] (missing = "frame", so old senders still work):
  frame: {"type": "frame", "camera": "left", "seq": 12, "width": 640, "height": 360,
          "hfov_deg": 70.0, "t_capture": <sender clock, seconds>}            + JPEG
  qr:    {"type": "qr", "camera": "left", "seq": 5, "width": 1920, "height": 1080,  (FULL res)
          "hfov_deg": 70.0, "t_capture": ..., "scan_ms": 80.0,
          "codes": [{"text": "I", "decoded": true, "corners": [[x, y] x4], "method": "edge"}, ...]}
          + no JPEG (jpeg_len 0). Corners are pixels of the full-resolution picture.
All integers are big-endian. The receiver checks the magic and size limits, so a wrong
program connecting to the port is rejected instead of crashing anything.
"""

import json
import socket
import struct

MAGIC = b"CAM1"
_LENS = struct.Struct("!II")
MAX_HEADER = 64 * 1024
MAX_JPEG = 8 * 1024 * 1024
DEFAULT_PORT = 5600


class ProtocolError(Exception):
    pass


def pack_frame(camera: str, seq: int, jpeg: bytes, width: int, height: int,
               hfov_deg: float, t_capture: float) -> bytes:
    header = json.dumps({"type": "frame", "camera": camera, "seq": seq, "width": width, "height": height,
                         "hfov_deg": hfov_deg, "t_capture": t_capture}).encode("utf-8")
    return MAGIC + _LENS.pack(len(header), len(jpeg)) + header + jpeg


def pack_qr(camera: str, seq: int, width: int, height: int, hfov_deg: float, t_capture: float,
            codes: list, scan_ms: float = 0.0) -> bytes:
    header = json.dumps({"type": "qr", "camera": camera, "seq": seq, "width": width, "height": height,
                         "hfov_deg": hfov_deg, "t_capture": t_capture, "scan_ms": scan_ms,
                         "codes": codes}).encode("utf-8")
    return MAGIC + _LENS.pack(len(header), 0) + header


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def read_frame(sock: socket.socket):
    """Block until one whole frame arrives. Returns (header dict, jpeg bytes)."""
    start = recv_exact(sock, 4 + _LENS.size)
    if start[:4] != MAGIC:
        raise ProtocolError("not a camera stream (bad magic)")
    hlen, jlen = _LENS.unpack(start[4:])
    if hlen > MAX_HEADER or jlen > MAX_JPEG:
        raise ProtocolError(f"frame too large (header {hlen}, jpeg {jlen})")
    header = json.loads(recv_exact(sock, hlen).decode("utf-8"))
    return header, recv_exact(sock, jlen)
