#!/usr/bin/env python3
"""Relay the Lite3 0x0901 robot state reports from the robot computer to the explorer computer.

The motion host (192.168.1.120) streams its reports only to 192.168.1.50:43897, where jetson2motion owns the port.
This script runs on 192.168.1.50, copies the 0x0901 packets off the wire with a raw packet socket (it listens in, it
does not take the port, so jetson2motion keeps getting everything) and sends them unchanged to the explorer computer,
where robot_state_bridge (explorer.sh) listens on 43897 as usual.

Standalone: plain Python 3 standard library, Linux, needs root (or CAP_NET_RAW).

    sudo python3 relay_robot_state.py                         # -> 192.168.1.71:43897, at most 10 Hz
    sudo python3 relay_robot_state.py --to 192.168.1.71:43897 --to 192.168.1.80:43897
    sudo python3 relay_robot_state.py --max-rate 0            # every packet (~200 Hz)
    sudo python3 relay_robot_state.py --codes 0x0901,0x10901  # also relay the IMU report
    python3 relay_robot_state.py --selftest                   # no root, no robot: checks the packet filter

Prints a status line every --every seconds: packets seen / relayed and the last robot state, e.g.
    relayed 50/1000 (10.0 Hz)  state (6,13,1) basic 6 gait 13 policy 0 motion 1  battery 87%
"""
import argparse
import socket
import struct
import sys
import time

ETH_P_IP = 0x0800
STATE_CODE = 0x0901


def udp_payload(frame, src_ip, dst_port):
    """UDP payload of an Ethernet/IPv4 frame from src_ip (bytes, 4) to dst_port, else None."""
    if len(frame) < 14 + 20 + 8 or frame[12:14] != b'\x08\x00':
        return None
    ip = 14
    ihl = (frame[ip] & 0x0F) * 4
    if frame[ip] >> 4 != 4 or frame[ip + 9] != 17 or frame[ip + 12:ip + 16] != src_ip:
        return None
    if struct.unpack_from('!H', frame, ip + 6)[0] & 0x3FFF:       # a fragment: these reports are never fragmented
        return None
    udp = ip + ihl
    if struct.unpack_from('!H', frame, udp + 2)[0] != dst_port:
        return None
    ulen = struct.unpack_from('!H', frame, udp + 4)[0]
    return bytes(frame[udp + 8:udp + ulen])


def state_text(p):
    """'(6,13,1) basic 6 gait 13 policy 0 motion 1  battery 87%' for a 0x0901 packet (208-byte aligned body)."""
    if len(p) != 12 + 208:
        return f'0x0901 with a {len(p) - 12}-byte body (expected 208)'
    basic, gait, policy = struct.unpack_from('<3i', p, 12)
    motion, = struct.unpack_from('<i', p, 12 + 172)
    battery, = struct.unpack_from('<d', p, 12 + 176)
    mid = policy if basic == 16 else gait
    return (f'({basic},{mid},{motion}) basic {basic} gait {gait} policy {policy} motion {motion}  '
            f'battery {battery:.0f}%')


def parse_dest(s):
    host, _, port = s.rpartition(':')
    return (host, int(port)) if host else (s, 43897)


def relay(iface, src, port, dests, codes, max_rate, every):
    try:
        rx = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_IP))
    except PermissionError:
        sys.exit('a raw packet socket needs root: run with sudo')
    rx.bind((iface, 0))
    rx.settimeout(0.5)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    src_b, min_dt = socket.inet_aton(src), (1.0 / max_rate if max_rate > 0 else 0.0)
    print(f'relaying codes {", ".join(f"0x{c:04X}" for c in sorted(codes))} from {src} -> *:{port} on {iface} to '
          + ', '.join(f'{h}:{p}' for h, p in dests)
          + (f' at most {max_rate:g} Hz per code' if min_dt else ' (every packet)'), flush=True)
    seen, sent, last_sent, last_state, failing = 0, 0, {}, None, set()
    t_print = time.monotonic()
    n_window = 0
    try:
        while True:
            try:
                frame, addr = rx.recvfrom(65535)
            except socket.timeout:
                frame, addr = None, None
            now = time.monotonic()
            if frame is not None and addr[2] != socket.PACKET_OUTGOING:
                p = udp_payload(frame, src_b, port)
                if p is not None and len(p) >= 12:
                    code = struct.unpack_from('<I', p)[0]
                    if code in codes:
                        seen += 1
                        if code == STATE_CODE:
                            last_state = p
                        if now - last_sent.get(code, -1e9) >= min_dt:
                            for d in dests:
                                try:
                                    tx.sendto(p, d)
                                    if d in failing:
                                        failing.discard(d)
                                        print(f'  sending to {d[0]}:{d[1]} works again', flush=True)
                                except OSError as e:
                                    if d not in failing:     # log once, not 10x/s while the explorer PC is off
                                        failing.add(d)
                                        print(f'  send to {d[0]}:{d[1]} failed: {e} (retrying quietly)', flush=True)
                            last_sent[code] = now
                            sent += 1
                            n_window += 1
            if every and now - t_print >= every:
                print(f'relayed {sent}/{seen} ({n_window / (now - t_print):.1f} Hz)  '
                      + (f'state {state_text(last_state)}' if last_state else
                         f'no 0x0901 from {src} seen on {iface} yet'), flush=True)
                t_print, n_window = now, 0
    except KeyboardInterrupt:
        print(f'relayed {sent} of {seen} packets')
    finally:
        rx.close()
        tx.close()


def selftest():
    """Build Ethernet/IPv4/UDP frames by hand and check what the filter keeps."""
    body = struct.pack('<3i4x18dI?3xIidi4?2d', 6, 13, 0, *[0.0] * 18, 0, False, 0, 1, 87.0, 0,
                       False, False, False, False, 0.0, 0.0)
    state = struct.pack('<3I', STATE_CODE, len(body), 1) + body

    def frame(payload, src='192.168.1.120', dport=43897, proto=17, frag=0):
        udp = struct.pack('!4H', 43780, dport, 8 + len(payload), 0) + payload
        ip = struct.pack('!BBHHHBBH4s4s', 0x45, 0, 20 + len(udp), 0, frag, 64, proto, 0,
                         socket.inet_aton(src), socket.inet_aton('192.168.1.50'))
        return b'\x00' * 12 + b'\x08\x00' + ip + udp + b'\x00' * 4          # trailing Ethernet padding

    src = socket.inet_aton('192.168.1.120')
    checks = [
        udp_payload(frame(state), src, 43897) == state,
        udp_payload(frame(state, src='192.168.1.121'), src, 43897) is None,
        udp_payload(frame(state, dport=43898), src, 43897) is None,
        udp_payload(frame(state, proto=6), src, 43897) is None,
        udp_payload(frame(state, frag=0x2000), src, 43897) is None,
        state_text(state) == '(6,13,1) basic 6 gait 13 policy 0 motion 1  battery 87%',
        parse_dest('192.168.1.71') == ('192.168.1.71', 43897),
        parse_dest('10.0.0.2:5000') == ('10.0.0.2', 5000),
    ]
    print('SELFTEST', 'PASS' if all(checks) else f'FAIL {checks}')
    return all(checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iface', default='eth0', help='interface the reports arrive on (default eth0)')
    ap.add_argument('--src', default='192.168.1.120', help='motion host address')
    ap.add_argument('--port', type=int, default=43897, help='port the motion host reports to')
    ap.add_argument('--to', action='append', type=parse_dest,
                    help='destination host[:port], repeatable (default 192.168.1.71:43897)')
    ap.add_argument('--codes', default='0x0901', help='comma-separated report codes to relay (default 0x0901)')
    ap.add_argument('--max-rate', type=float, default=10.0, help='max relayed packets/s per code (default 10, 0 = all)')
    ap.add_argument('--every', type=float, default=5.0, help='status line period, s (0 = quiet)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 2)
    codes = {int(c, 0) for c in a.codes.split(',') if c.strip()}
    relay(a.iface, a.src, a.port, a.to or [('192.168.1.71', 43897)], codes, a.max_rate, a.every)


if __name__ == '__main__':
    main()
