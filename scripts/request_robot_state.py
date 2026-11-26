#!/usr/bin/env python3
"""Ask the Lite3 motion host for its robot state (command code 0x0901, type 1) and print the replies.

Standalone: plain Python 3, no ROS, imports nothing from this repo. Copy it to any computer on the robot's network.

Sends the 12-byte command head  uint32 code=0x0901, uint32 parameters_size=0, uint32 type=1  (little endian) to the
motion host, then prints every UDP packet that comes back to the socket it sent from. 0x0901 replies are decoded as
RobotStateUpload (natural C alignment 208 bytes, or packed 201 bytes); anything else is shown as code / size / hex.

    python3 request_robot_state.py                               # 192.168.1.120:43893, from a free local port, 2 Hz, 10 s
    python3 request_robot_state.py --once                        # one request, wait --duration for the reply
    python3 request_robot_state.py --local-port 43897            # send from / listen on a fixed port
    python3 request_robot_state.py --selftest                    # no robot: fake motion host on 127.0.0.1

Replies are expected at the address and port the request came from. If the host instead reports to a fixed port,
see who already holds it (sudo ss -uanp | grep <port>): that program is the one receiving the state.

Exit code: 0 a 0x0901 reply arrived, 1 none did, 2 self-test failed.
"""
import argparse
import math
import socket
import struct
import sys
import threading
import time

STATE_CODE = 0x0901
HEAD = struct.Struct('<3I')                            # code, parameters_size, type
ALIGNED = struct.Struct('<3i4x18dI?3xIidi4?2d')        # RobotStateUpload, natural alignment: 208 bytes
PACKED = struct.Struct('<3i18dI?Iidi4?2d')             # same, packed: 201 bytes

NAMES = {
    (1, 0, 0): 'sitting', (4, 0, 0): 'prepare', (5, 0, 0): 'sit-to-stand',
    (6, 0, 0): 'standing, flat slow', (6, 0, 1): 'stepping, flat slow',
    (6, 2, 0): 'standing, RUG general', (6, 2, 1): 'stepping, RUG general',
    (6, 4, 0): 'standing, flat medium', (6, 4, 1): 'stepping, flat medium',
    (6, 5, 0): 'standing, flat fast', (6, 5, 1): 'stepping, flat fast',
    (6, 6, 0): 'standing, RUG grip', (6, 6, 1): 'stepping, RUG grip',
    (6, 13, 0): 'standing, RUG H-step', (6, 13, 1): 'stepping, RUG H-step',
    (7, 0, 0): 'sitting', (8, 0, 0): 'lose-control protection', (9, 0, 0): 'posture adjustment',
    (16, 0, 14): 'AI basic gait', (16, 16, 1): 'AI jump gait', (16, 18, 1): 'AI stand gait', (16, 20, 1): 'AI fast gait',
}


def request(code=STATE_CODE, ctype=1):
    return HEAD.pack(code, 0, ctype)


def decode(packet):
    """dict of RobotStateUpload fields for a 0x0901 packet, else None."""
    if len(packet) < HEAD.size:
        return None
    code, size, ctype = HEAD.unpack_from(packet)
    if code != STATE_CODE:
        return None
    body = packet[HEAD.size:]
    for name, lay in (('aligned', ALIGNED), ('packed', PACKED)):
        if len(body) == lay.size:
            v = lay.unpack_from(body)
            return dict(layout=name, size=size, type=ctype, basic=v[0], gait=v[1], policy=v[2], rpy=v[3:6],
                        pos_world=v[12:15], vel_body=v[18:21], charging=v[22], error=v[23], motion=v[24],
                        battery=v[25], task=v[26])
    return dict(layout=f'unknown ({len(body)}-byte body, parameters_size {size})', type=ctype)


def describe(f):
    if 'basic' not in f:
        return f'0x0901 with {f["layout"]}: layout not recognised'
    mid = f['policy'] if f['basic'] == 16 else f['gait']
    key = (f['basic'], mid, f['motion'])
    r, p, y = (math.degrees(a) for a in f['rpy'])
    return (f'({key[0]},{key[1]},{key[2]}) {NAMES.get(key, "?")}  [basic {f["basic"]} gait {f["gait"]} policy '
            f'{f["policy"]} motion {f["motion"]}]  rpy {r:+.1f} {p:+.1f} {y:+.1f} deg  battery {f["battery"]:.0f}%  '
            f'{f["layout"]}' + (f'  error 0x{f["error"]:X}' if f['error'] else '')
            + ('  CHARGING' if f['charging'] else ''))


def run(host, port, local_ip, local_port, rate, duration, once, code, ctype, quiet_repeats=True):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((local_ip, local_port))
    except OSError as e:
        sys.exit(f'cannot bind {local_ip or "*"}:{local_port}: {e}. Another program holds it '
                 f'(sudo ss -uanp | grep :{local_port}); use --local-port 0')
    sock.settimeout(0.05)
    me = sock.getsockname()
    pkt = request(code, ctype)
    print(f'sending code 0x{code:04X} type {ctype} ({pkt.hex(" ")}) to {host}:{port} from {me[0] or "*"}:{me[1]}'
          + (' once' if once else f' at {rate:g} Hz') + f', listening {duration:g} s', flush=True)
    start = time.monotonic()
    next_send, sent, got, other, last_key = start, 0, 0, 0, None
    try:
        while time.monotonic() - start < duration:
            now = time.monotonic()
            if now >= next_send and (not once or sent == 0):
                sock.sendto(pkt, (host, port))
                sent += 1
                next_send = now + (1.0 / rate if rate > 0 else 1e9)
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except ConnectionRefusedError:            # ICMP port unreachable from the host
                print(f'  {host}:{port} refused (ICMP port unreachable): nothing listens on that port', flush=True)
                continue
            t = time.monotonic() - start
            f = decode(data)
            if f is None:
                other += 1
                code_in = struct.unpack_from('<I', data)[0] if len(data) >= 4 else None
                print(f'[{t:6.2f}s] {len(data)} bytes from {addr[0]}:{addr[1]}  code '
                      + (f'0x{code_in:08X}' if code_in is not None else '-') + f'  {data[:32].hex(" ")}', flush=True)
                continue
            got += 1
            line = describe(f)
            key = line.split('  rpy')[0]
            if not quiet_repeats or key != last_key or got == 1:
                print(f'[{t:6.2f}s] from {addr[0]}:{addr[1]}  {line}', flush=True)
            last_key = key
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    print(f'{sent} requests sent, {got} state replies, {other} other packets')
    if not got:
        print('no 0x0901 reply. Check: the command port (--port), that this computer may talk to the motion host, '
              'and whether the replies go to a fixed address (sudo tcpdump -ni any udp and host ' + host + ')')
    return got


def selftest():
    """A fake motion host on 127.0.0.1 that answers each 0x0901 type-1 request with a state packet."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(('127.0.0.1', 0))
    srv.settimeout(0.1)
    port, stop, seen = srv.getsockname()[1], threading.Event(), []

    def host():
        n = 0
        while not stop.is_set():
            try:
                data, addr = srv.recvfrom(64)
            except socket.timeout:
                continue
            seen.append(data)
            if HEAD.unpack_from(data) == (STATE_CODE, 0, 1):
                lay = PACKED if n % 2 else ALIGNED
                body = lay.pack(6 if n < 2 else 16, 13 if n < 2 else 0, 0, 0.1, -0.05, 1.0, *[0.0] * 15,
                                0, False, 0, 1, 77.0, 0, False, False, False, False, 0.0, 0.0)
                srv.sendto(HEAD.pack(STATE_CODE, len(body), 1) + body, addr)
                n += 1

    th = threading.Thread(target=host)
    th.start()
    got = run('127.0.0.1', port, '127.0.0.1', 0, rate=10, duration=0.5, once=False, code=STATE_CODE, ctype=1)
    stop.set()
    th.join()
    srv.close()
    ok = got >= 3 and seen and seen[0] == bytes.fromhex('01090000 00000000 01000000')
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='192.168.1.120', help='motion host address')
    ap.add_argument('--port', type=int, default=43893, help='motion host command port')
    ap.add_argument('--local-ip', default='', help='local address to send from / listen on (default all)')
    ap.add_argument('--local-port', type=int, default=0, help='local port (default 0 = any free port)')
    ap.add_argument('--rate', type=float, default=2.0, help='requests per second')
    ap.add_argument('--duration', type=float, default=10.0, help='seconds to run')
    ap.add_argument('--once', action='store_true', help='send a single request')
    ap.add_argument('--code', type=lambda s: int(s, 0), default=STATE_CODE, help='command code (default 0x0901)')
    ap.add_argument('--type', dest='ctype', type=int, default=1, help='command type field (default 1)')
    ap.add_argument('--all', action='store_true', help='print every state reply, not only changes')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 2)
    got = run(a.host, a.port, a.local_ip, a.local_port, a.rate, a.duration, a.once, a.code, a.ctype,
              quiet_repeats=not a.all)
    sys.exit(0 if got else 1)


if __name__ == '__main__':
    main()
