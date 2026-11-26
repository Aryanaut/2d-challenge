#!/usr/bin/env python3
"""Read the Lite3 0x0901 robot state straight off UDP, without ROS and without the explorer.

Uses the same parser as robot_state_bridge (src/lite3_explorer/lite3_explorer/robot_state.py, loaded by file path, so
nothing needs building or sourcing). Prints one line per state change, a status line every --every seconds (packet
rate, payload layout, rpy, battery) and anything on the port that is NOT a 0x0901 report (wrong port / format).

    python3 scripts/robot_state_probe.py                    # listen on *:43897, accept reports from 192.168.1.120
    python3 scripts/robot_state_probe.py --source ''        # accept reports from any sender
    python3 scripts/robot_state_probe.py --port 43897 --every 1 --timeout 5
    python3 scripts/robot_state_probe.py --selftest         # no robot: send fake packets to itself and check the parse

Stop explorer.sh's robot_state_bridge first: both bind the port with SO_REUSEPORT, and Linux then splits unicast
packets between them, so each sees only some of the reports.

Exit code: 0 ok, 1 no 0x0901 packet within --timeout, 2 self-test failed.
"""
import argparse
import importlib.util
import math
import os
import socket
import struct
import sys
import threading
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_RS_PATH = os.path.join(_HERE, '..', 'src', 'lite3_explorer', 'lite3_explorer', 'robot_state.py')


def load_robot_state(path=_RS_PATH):
    spec = importlib.util.spec_from_file_location('robot_state', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = load_robot_state()


def full_fields(packet):
    """All RobotStateUpload fields as a dict (+ 'layout'), or None. parse() keeps only the mode; this is for checking
    that the layout is right: with the wrong one rpy / battery come out as garbage."""
    body = packet[rs._HEAD.size:]
    for name, layout in (('aligned', rs._ALIGNED), ('packed', rs._PACKED)):
        if len(body) == layout.size:
            v = layout.unpack_from(body)
            return dict(layout=name, basic=v[0], gait=v[1], policy=v[2], rpy=v[3:6], rpy_vel=v[6:9], xyz_acc=v[9:12],
                        pos_world=v[12:15], vel_world=v[15:18], vel_body=v[18:21], touch_down=v[21],
                        charging=v[22], error=v[23], motion=v[24], battery=v[25], task=v[26], ultrasound=v[31:33])
    return None


def fake_packet(basic, gait, policy, motion, battery=87.0, rpy=(0.0, 0.0, 0.0), packed=False):
    """A 0x0901 packet as the motion host would send it (for --selftest)."""
    fmt = rs._PACKED if packed else rs._ALIGNED
    doubles = list(rpy) + [0.0] * 15
    body = fmt.pack(basic, gait, policy, *doubles, 0, False, 0, motion, battery, 0, False, False, False, False, 0.0, 0.0)
    return struct.pack('<3I', rs.STATE_CODE, len(body), 1) + body


def open_socket(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((ip, port))
    except OSError as e:
        sys.exit(f'cannot listen on {ip or "*"}:{port}: {e}. --ip must be an address of THIS computer '
                 "(ip -4 addr), or '' for all interfaces; is another program already on the port?")
    sock.settimeout(0.2)
    return sock


def fmt_status(f, rate):
    r, p, y = (math.degrees(a) for a in f['rpy'])
    return (f'{rate:5.1f} Hz  {f["layout"]:7s}  {rs.describe((f["basic"], f["gait"], f["policy"], f["motion"]))}  '
            f'rpy {r:+6.1f} {p:+6.1f} {y:+7.1f} deg  battery {f["battery"]:.0f}%'
            + (f'  error 0x{f["error"]:X}' if f['error'] else '') + ('  CHARGING' if f['charging'] else ''))


def listen(ip, port, every, timeout, count, stop=None, source=None):
    """Print the robot state from UDP. Returns the list of distinct modes seen, in order (None if nothing arrived)."""
    sock = open_socket(ip, port)
    print(f'listening for 0x0901 on udp {ip or "*"}:{port}' + (f' from {source}' if source else '')
          + '  (Ctrl-C to stop)', flush=True)
    start = last_print = time.monotonic()
    n, n_window, other, modes, last, last_mode = 0, 0, Counter(), [], None, None
    try:
        while not (stop and stop.is_set()):
            now = time.monotonic()
            if n == 0 and timeout and now - start > timeout:
                print(f'no 0x0901 packet in {timeout:.0f} s'
                      + (f'; other packets seen: {dict(other)}' if other else '; nothing at all arrived on this port'))
                return None
            if count and n >= count:
                break
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                data = None
            now = time.monotonic()
            if data is not None and source and addr[0] != source:
                key = (f'from {addr[0]}', len(data))
                if key not in other:
                    print(f'  ignored: {len(data)} bytes from {addr[0]} (not {source})', flush=True)
                other[key] += 1
                data = None
            if data is not None:
                s = rs.parse(data)
                if s is None:
                    code = struct.unpack_from('<I', data)[0] if len(data) >= 4 else None
                    key = (f'0x{code:04X}' if code is not None else 'short', len(data))
                    if key not in other:
                        print(f'  not a robot state: code {key[0]} {len(data)} bytes from {addr[0]}', flush=True)
                    other[key] += 1
                    continue
                n += 1
                n_window += 1
                last = data
                if s.mode != last_mode:
                    print(f'[{now - start:7.2f}s] state {rs.describe(s)}  mode={s.mode}  battery {s.battery:.0f}%'
                          + (f'  error 0x{s.error:X}' if s.error else '') + f'  from {addr[0]}', flush=True)
                    last_mode = s.mode
                    modes.append(s.mode)
            if every and now - last_print >= every:
                if last is not None:
                    print('  ' + fmt_status(full_fields(last), n_window / (now - last_print)), flush=True)
                elif not timeout:
                    print('  ...no 0x0901 packets yet', flush=True)
                last_print, n_window = now, 0
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    print(f'{n} state packets, {len(modes)} distinct states' + (f', other packets {dict(other)}' if other else ''))
    return modes if n else None


def selftest(port):
    """Send a scripted sequence (aligned and packed, plus a junk packet) to ourselves and check what is parsed."""
    seq = [(6, 0, 0, 0), (6, 13, 0, 1), (16, 0, 0, 1), (6, 0, 0, 0)]
    stop = threading.Event()
    result = {}

    def rx():
        result['modes'] = listen('127.0.0.1', port, every=0, timeout=3.0, count=len(seq) * 10, stop=stop)

    t = threading.Thread(target=rx)
    t.start()
    time.sleep(0.3)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(struct.pack('<3I', 0x0902, 4, 1) + b'\0' * 4, ('127.0.0.1', port))     # must be ignored
    for i, mode in enumerate(seq):
        for _ in range(10):
            tx.sendto(fake_packet(*mode, packed=bool(i % 2), rpy=(0.1, -0.05, 1.0)), ('127.0.0.1', port))
            time.sleep(0.01)
    time.sleep(0.5)
    stop.set()
    t.join()
    tx.close()
    ok = result.get('modes') == seq
    f = full_fields(fake_packet(6, 13, 0, 1, battery=55.0, rpy=(0.1, -0.05, 1.0), packed=True))
    ok = ok and f['battery'] == 55.0 and abs(f['rpy'][2] - 1.0) < 1e-9 and f['layout'] == 'packed'
    print('SELFTEST', 'PASS' if ok else f'FAIL: expected {seq}, got {result.get("modes")}')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default='', help='local address to listen on (default: all interfaces)')
    ap.add_argument('--source', default=os.environ.get('STATE_SOURCE', '192.168.1.120'),
                    help="only accept reports sent from this address (default 192.168.1.120, the motion host; "
                         "'' = any sender)")
    ap.add_argument('--port', type=int, default=int(os.environ.get('STATE_PORT', 43897)))
    ap.add_argument('--every', type=float, default=2.0, help='status line period, s (0 = only state changes)')
    ap.add_argument('--timeout', type=float, default=10.0, help='give up if no 0x0901 packet by then (0 = wait forever)')
    ap.add_argument('--count', type=int, default=0, help='stop after this many state packets (0 = run until Ctrl-C)')
    ap.add_argument('--selftest', action='store_true', help='no robot: loop fake packets through 127.0.0.1')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.port + 1) else 2)
    sys.exit(0 if listen(a.ip, a.port, a.every, a.timeout, a.count, source=a.source) is not None else 1)


if __name__ == '__main__':
    main()
