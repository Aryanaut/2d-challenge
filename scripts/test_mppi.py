#!/usr/bin/env python3
"""Live test of the Nav2 MPPI (Omni) controller on the running robot.

Start the stack first:   LAUNCH_ARGS="controller:=mppi" ./scripts/explorer.sh
Pause the mission so it does not fight the test goal (e.g. stop the mission_manager node, or run the bare Nav2 launch).

Sends one NavigateToPose goal offset from the robot's current pose, in the robot frame (default: 0.6 m to the left,
same heading), and checks:
  1. controller_server runs motion_model Omni
  2. the goal is reached (status SUCCEEDED, final error <= --tol)
  3. the robot moved sideways: max |vy| on /cmd_vel >= --min-vy
  4. it did not turn to do so: heading change <= --max-turn-deg
Prints a report and exits 0 on pass, 1 on fail, 2 when the setup is wrong (no Nav2, no pose).

  ./test_mppi.py                    # 0.6 m left
  ./test_mppi.py --dy -0.6          # right
  ./test_mppi.py --dx 0.6           # straight ahead, no sideways part
  ./test_mppi.py --heading 90 --dy 0.6   # face map +y, then slide 0.6 m to its left (map -x), heading held
  ./test_mppi.py --dy 0.6 --yaw-deg 45   # sidestep and end turned 45 deg left
  ./test_mppi.py --dy 0.6 --yaw-map 90   # sidestep and end facing map +y
  ./test_mppi.py --dy 0.6 --yaw-deg 45 --turn-first   # turn 45 deg in place first, then sidestep with the heading held
  ./test_mppi.py --dx 0.5 --dy 0.5  # diagonal
  ./test_mppi.py --selftest         # check the pass/fail logic offline, no robot
"""
import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Result:
    status_ok: bool = False
    final_err_m: float = float('inf')
    max_vx: float = 0.0
    max_vy: float = 0.0
    max_wz: float = 0.0
    turn_deg: float = 0.0
    yaw_requested: bool = False        # a goal heading other than the start heading was asked for
    final_yaw_err_deg: float = 0.0     # |final heading - goal heading|
    turn_first: bool = False           # the heading was changed by a Spin before the linear motion
    duration_s: float = 0.0
    notes: List[str] = field(default_factory=list)


def evaluate(r: Result, tol: float, min_vy: float, max_turn_deg: float, yaw_tol_deg: float = 20.0) -> Tuple[bool, List[str]]:
    """Pass/fail from the measured run; returns (passed, one line per check)."""
    checks = [
        (r.status_ok, 'goal accepted and SUCCEEDED'),
        (r.final_err_m <= tol, f'final error {r.final_err_m:.2f} m <= {tol:.2f} m'),
        (r.max_vy >= min_vy, f'max |vy| {r.max_vy:.2f} m/s >= {min_vy:.2f} (moved sideways)'),
    ]
    if r.yaw_requested:
        checks.append((r.final_yaw_err_deg <= yaw_tol_deg, f'final heading {r.final_yaw_err_deg:.0f} deg from the goal heading <= {yaw_tol_deg:.0f}'))
    if r.turn_first or not r.yaw_requested:      # the heading must not move during the linear motion
        checks.append((r.turn_deg <= max_turn_deg, f'heading change during the move {r.turn_deg:.0f} deg <= {max_turn_deg:.0f} (heading held)'))
    return all(ok for ok, _ in checks), [('PASS  ' if ok else 'FAIL  ') + text for ok, text in checks]


def selftest() -> int:
    def mk(**kw):
        base = dict(status_ok=True, final_err_m=0.08, max_vx=0.1, max_vy=0.2, max_wz=0.3, turn_deg=5.0, duration_s=5.0)
        base.update(kw)
        return Result(**base)
    ok = lambda r: evaluate(r, 0.40, 0.05, 10)[0]
    assert ok(mk())
    assert not ok(mk(max_vy=0.0))             # never used vy (diff-drive behaviour)
    assert not ok(mk(turn_deg=90.0))          # turned instead of sidestepping
    assert ok(mk(turn_deg=90.0, yaw_requested=True, final_yaw_err_deg=5.0))       # asked for a turn: only the end heading counts
    assert not ok(mk(yaw_requested=True, final_yaw_err_deg=60.0))                 # ended facing the wrong way
    tf = dict(yaw_requested=True, turn_first=True, final_yaw_err_deg=5.0)
    assert ok(mk(turn_deg=4.0, **tf))                                             # turned first, then moved straight
    assert not ok(mk(turn_deg=40.0, **tf))                                        # turned first but drifted during the move
    assert not ok(mk(status_ok=False))        # goal failed
    assert not ok(mk(final_err_m=0.9))        # stopped short
    print('selftest ok')
    return 0


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def motion_model() -> str:
    out = subprocess.run(['ros2', 'param', 'get', '/controller_server', 'FollowPath.motion_model'],
                         capture_output=True, text=True, timeout=10)
    return (out.stdout + out.stderr).strip()


def run(args) -> int:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from action_msgs.msg import GoalStatus
    import tf2_ros

    rclpy.init()
    node = Node('test_mppi')
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)
    cmds: List[Tuple[float, float, float]] = []
    node.create_subscription(Twist, args.cmd_topic, lambda m: cmds.append((m.linear.x, m.linear.y, m.angular.z)), 20)

    def pose():
        try:
            t = buf.lookup_transform('map', 'base_link', rclpy.time.Time()).transform
            return t.translation.x, t.translation.y, yaw_of(t.rotation)
        except Exception:
            return None

    def spin_until(pred, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not pred():
            rclpy.spin_once(node, timeout_sec=0.05)
        return pred()

    mm = motion_model()
    print('controller_server FollowPath.motion_model ->', mm)
    if 'Omni' not in mm:
        print('SETUP  controller is not MPPI Omni: start with controller:=mppi'); return 2
    if not spin_until(lambda: pose() is not None, 10.0):
        print('SETUP  no map -> base_link transform'); return 2
    client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
    if not client.wait_for_server(timeout_sec=10.0):
        print('SETUP  navigate_to_pose action server not available'); return 2

    x0, y0, yaw0 = pose()
    if args.heading is not None:                  # --heading: the heading to hold, map frame
        args.yaw_map, args.turn_first = args.heading, True
    gyaw = math.radians(args.yaw_map) if args.yaw_map is not None else yaw0 + math.radians(args.yaw_deg)
    fy = gyaw if args.heading is not None else yaw0           # dx / dy are forward / left of the held heading
    gx = x0 + args.dx * math.cos(fy) - args.dy * math.sin(fy)
    gy = y0 + args.dx * math.sin(fy) + args.dy * math.cos(fy)
    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x, goal.pose.pose.position.y = gx, gy
    goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(gyaw / 2), math.cos(gyaw / 2)
    print(f'start ({x0:.2f}, {y0:.2f}) yaw {math.degrees(yaw0):.0f} deg -> goal ({gx:.2f}, {gy:.2f}), '
          f'offset dx={args.dx:+.2f} dy={args.dy:+.2f} m in the robot frame, goal heading {math.degrees(wrap(gyaw)):.0f} deg ({"start heading" if abs(wrap(gyaw - yaw0)) < 1e-6 else "turn " + format(math.degrees(wrap(gyaw - yaw0)), "+.0f") + " deg"})')

    r = Result()
    if args.turn_first and abs(wrap(gyaw - yaw0)) > math.radians(1.0):
        from nav2_msgs.action import Spin
        spin = ActionClient(node, Spin, 'spin')
        if not spin.wait_for_server(timeout_sec=10.0):
            print('SETUP  spin action server not available'); return 2
        sg = Spin.Goal()
        sg.target_yaw = float(wrap(gyaw - yaw0))
        sg.time_allowance.sec = int(args.timeout)
        print(f'turning {math.degrees(sg.target_yaw):+.0f} deg in place (Spin) before the linear motion')
        ssend = spin.send_goal_async(sg)
        if not spin_until(ssend.done, 10.0) or not ssend.result().accepted:
            print('FAIL  spin rejected'); return 1
        sres = ssend.result().get_result_async()
        spin_until(sres.done, args.timeout + 5.0)
        if not sres.done() or sres.result().status != GoalStatus.STATUS_SUCCEEDED:
            print('FAIL  the turn before the move did not finish'); return 1
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
        r.turn_first = True
        p = pose()
        print(f'heading after the turn {math.degrees(p[2]):.0f} deg (goal heading {math.degrees(wrap(gyaw)):.0f})')
        yaw_move = p[2]
    else:
        yaw_move = yaw0
    cmds.clear()
    t0 = time.monotonic()
    send = client.send_goal_async(goal)
    if not spin_until(send.done, 10.0) or not send.result().accepted:
        print('FAIL  goal rejected'); return 1
    res = send.result().get_result_async()
    max_turn = 0.0
    while not res.done() and time.monotonic() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        p = pose()
        if p:
            max_turn = max(max_turn, abs(math.degrees(wrap(p[2] - yaw_move))))
    if not res.done():
        send.result().cancel_goal_async()
        r.notes.append(f'timed out after {args.timeout:.0f} s, goal cancelled')
    else:
        r.status_ok = res.result().status == GoalStatus.STATUS_SUCCEEDED
        if not r.status_ok:
            r.notes.append(f'action status {res.result().status}')
    r.duration_s = time.monotonic() - t0
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)
    p = pose()
    r.final_err_m = math.hypot(p[0] - gx, p[1] - gy) if p else float('inf')
    r.turn_deg = max_turn
    r.yaw_requested = abs(wrap(gyaw - yaw0)) > math.radians(1.0)
    r.final_yaw_err_deg = abs(math.degrees(wrap(p[2] - gyaw))) if p else 180.0
    r.max_vx = max((abs(c[0]) for c in cmds), default=0.0)
    r.max_vy = max((abs(c[1]) for c in cmds), default=0.0)
    r.max_wz = max((abs(c[2]) for c in cmds), default=0.0)

    ok, lines = evaluate(r, args.tol, args.min_vy, args.max_turn_deg, args.yaw_tol_deg)
    print(f'\n{len(cmds)} cmd_vel samples on {args.cmd_topic}, {r.duration_s:.1f} s; '
          f'max |vx| {r.max_vx:.2f}  |vy| {r.max_vy:.2f}  |wz| {r.max_wz:.2f}')
    print('\n'.join(lines + [f'NOTE  {n}' for n in r.notes]))
    print('\nRESULT:', 'PASS' if ok else 'FAIL')
    rclpy.shutdown()
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dx', type=float, default=None, help='goal offset forward, robot frame (m); 0 when not given')
    ap.add_argument('--dy', type=float, default=None, help='goal offset to the left, robot frame (m); 0 when not given, 0.6 if neither is given')
    ap.add_argument('--heading', type=float, default=None,
                    help='the heading to maintain, degrees in the map frame (0 = map +x, 90 = map +y). The robot turns to it '
                         'in place first, then moves --dx / --dy FROM THAT HEADING (dx forward, dy left of it) holding it. '
                         'Without it the current heading is held and dx / dy follow the robot.')
    ap.add_argument('--yaw-deg', type=float, default=0.0,
                    help='goal heading as a turn from the start heading (deg, left positive); 0 = keep the heading')
    ap.add_argument('--yaw-map', type=float, default=None, help='goal heading in the map frame (deg); overrides --yaw-deg')
    ap.add_argument('--turn-first', action='store_true',
                    help='turn to the goal heading in place (Nav2 Spin) BEFORE the linear motion; the move itself is then '
                         'checked for heading drift. The goal position is still offset from the start pose and heading.')
    ap.add_argument('--yaw-tol-deg', type=float, default=20.0,
                    help='with a goal heading: max final heading error (deg); yaw_goal_tolerance is 0.6 rad = 34 deg')
    ap.add_argument('--tol', type=float, default=0.40, help='max final distance to the goal (m); = xy_goal_tolerance in nav2_params_mppi.yaml')
    ap.add_argument('--min-vy', type=float, default=0.05, help='min peak |vy| to count as lateral motion (m/s)')
    ap.add_argument('--max-turn-deg', type=float, default=10.0, help='max heading change allowed (deg)')
    ap.add_argument('--timeout', type=float, default=30.0)
    ap.add_argument('--cmd-topic', default='/cmd_vel')
    ap.add_argument('--selftest', action='store_true', help='check the pass/fail logic offline')
    args = ap.parse_args()
    if args.dx is None and args.dy is None:
        args.dy = 0.6                             # no offset given: the default sideways test
    args.dx = 0.0 if args.dx is None else args.dx
    args.dy = 0.0 if args.dy is None else args.dy
    return selftest() if args.selftest else run(args)


if __name__ == '__main__':
    sys.exit(main())
