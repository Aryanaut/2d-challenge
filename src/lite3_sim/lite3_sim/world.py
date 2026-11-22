"""The simulated world for lite3_sim (no ROS): maze geometry, robot, lidar, SLAM map, Nav2 stand-ins, curtain camera.

Everything is computed in the MAP frame. Like slam_toolbox on the real dog, the map frame starts at the robot's
start pose (map_origin: start), so the mission manager has to find the walls and snap its tile grid to them
exactly as on the robot. map_origin: maze puts the map frame on the maze's south-west corner instead (tile maths
then line up with the maze file, handy for reading tile labels).

What stands in for what:
  slam_toolbox          OccupancyMapper: log-odds map from the simulated scans, perfect pose
  Nav2 global costmap   static layer (the SLAM map) + /virtual_walls + inflation (global footprint, 0.25 m)
  Nav2 local costmap    4 m rolling window of the SLAM map + inflation (body footprint, 0.10 m)
  NavFn planner         GridPlanner (inscribed = wall, unknown allowed, robot cell cleared, goal tolerance)
  controller_server     NavTask: pure pursuit like RPP (vx, wz; rotate to heading), or omni (vx, vy, wz) like MPPI
  bt_navigator          NavTask replans every replan_period_s; on no path / no progress / collision ahead it runs
                        the default tree's recoveries (clear, spin, wait, back up) and retries, then aborts
  waypoint_follower     WaypointsTask: one NavTask per pose, missed_waypoints, stop_on_failure false
  behavior_server spin  SpinTask: relative yaw, aborts on collision or after the time allowance
  curtain_detector      curtain camera: curtain points in the depth camera's view -> CurtainMap
Not simulated: terrain, stairs, gaits, the AI. The robot collides with wall panels and slit panels, not curtains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lite3_sim.costmap import (INSCRIBED, GridInfo, GridPlanner, OccupancyMapper, inflate, path_length)
from lite3_sim.maze import CURTAIN, Maze

try:
    from lite3_explorer.curtain_core import CurtainMap
except ImportError:                                     # pragma: no cover
    CurtainMap = None

RUNNING, SUCCEEDED, FAILED, CANCELLED = 'running', 'succeeded', 'failed', 'cancelled'

DEFAULTS = {
    'map_origin': 'start',            # start (like SLAM) | maze
    'speed_scale': 1.0,               # >1 runs the robot faster than real time (the mission's own timers are wall time)
    'robot': {'length_m': 0.61, 'width_m': 0.37, 'max_lin_acc': 1.0, 'max_ang_acc': 2.5},
    'lidar': {'range_min': 0.35, 'range_max': 4.0, 'angle_increment': 0.0087, 'rate_hz': 10.0,
              'noise_m': 0.005, 'x_offset_m': 0.0},
    'slam': {'resolution': 0.05, 'margin_m': 2.0, 'update_hz': 2.0},
    'global_costmap': {'update_hz': 1.0, 'footprint_length_m': 0.68, 'footprint_width_m': 0.42,
                       'padding_m': 0.03, 'inflation_radius_m': 0.25, 'cost_scaling_factor': 1.0},
    'local_costmap': {'update_hz': 2.0, 'size_m': 4.0, 'inflation_radius_m': 0.10, 'cost_scaling_factor': 1.0},
    'planner': {'tolerance_m': 0.4, 'allow_unknown': True, 'unknown_cost': 4.0},
    'controller': {'type': 'rpp',     # rpp (vx, wz) | omni (vx, vy, wz: MPPI's Omni model)
                   'desired_linear_vel': 0.35, 'max_angular_vel': 0.6, 'lookahead_m': 0.6,
                   'rotate_to_heading_min_angle': 0.6, 'rotate_vel': 0.6, 'approach_dist_m': 0.6,
                   'min_approach_vel': 0.08, 'xy_goal_tolerance': 0.4,
                   'regulated_min_radius': 0.9, 'regulated_min_speed': 0.12, 'yaw_goal_tolerance': 0.6,
                   'replan_period_s': 1.0, 'max_plan_failures': 3,
                   'collision_detection': True, 'collision_horizon_s': 0.3, 'failure_tolerance_s': 3.0,
                   # bt_navigator default tree (navigate_to_pose_w_replanning_and_recovery): on a planner / controller
                   # failure run the next recovery (round robin) and retry, up to number_of_retries times
                   'bt_recoveries': True, 'number_of_retries': 6, 'recovery_spin_rad': 1.57, 'recovery_wait_s': 5.0,
                   'backup_dist_m': 0.30, 'backup_speed': 0.05,
                   'progress_window_s': 30.0, 'progress_min_m': 0.3},
    'spin': {'max_vel': 1.0, 'min_vel': 0.3, 'acc': 2.5},
    'cmd_vel_timeout_s': 0.5,
    'curtain_camera': {'enabled': True, 'rate_hz': 2.0, 'hfov_deg': 81.4, 'range_min': 0.30, 'range_max': 3.5,
                       'x_offset_m': 0.30, 'min_columns': 0.30, 'point_step_m': 0.05, 'noise_m': 0.015,
                       'cell_m': 0.05, 'confirm_hits': 3},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def compose(a, b):
    """2D pose composition a (+) b; poses are (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def invert(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


# ============================================================================================ geometry
class Geometry:
    """Maze segments in the map frame: what the lidar sees, what the robot bumps into, where the curtains are."""

    def __init__(self, maze: Maze, map_T_maze):
        def tf(x, y):
            return compose(map_T_maze, (x, y, 0.0))[:2]
        segs, kinds = [], []
        for s in maze.segments:
            (x0, y0), (x1, y1) = tf(s.x0, s.y0), tf(s.x1, s.y1)
            segs.append((x0, y0, x1, y1))
            kinds.append(s.kind)
        self.segs = np.array(segs, float).reshape(-1, 4)
        self.kinds = np.array(kinds, object)
        self.is_curtain = self.kinds == CURTAIN
        self.half_thickness = 0.5 * maze.wall_thickness
        # collision points along every wall / slit panel (curtains are walked through)
        pts = []
        for (x0, y0, x1, y1), curtain in zip(self.segs, self.is_curtain):
            if curtain:
                continue
            n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.02)))
            t = np.linspace(0.0, 1.0, n + 1)
            pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1))
        self.solid_pts = np.concatenate(pts) if pts else np.zeros((0, 2))
        cpts, cseg = [], []
        for k in np.flatnonzero(self.is_curtain):
            x0, y0, x1, y1 = self.segs[k]
            n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.025)))
            t = (np.arange(n) + 0.5) / n
            cpts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1))
            cseg += [k] * n
        self.curtain_pts = np.concatenate(cpts) if cpts else np.zeros((0, 2))
        self.curtain_seg = np.array(cseg, int)
        lo = self.segs[:, [0, 1]].min(axis=0) if len(self.segs) else np.zeros(2)
        hi = self.segs[:, [2, 3]].max(axis=0) if len(self.segs) else np.zeros(2)
        lo = np.minimum(lo, self.segs[:, [2, 3]].min(axis=0)) if len(self.segs) else lo
        hi = np.maximum(hi, self.segs[:, [0, 1]].max(axis=0)) if len(self.segs) else hi
        self.bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    def raycast(self, ox, oy, angles, max_range, skip=None):
        """(ranges, segment index) of the first segment hit by each ray; inf / -1 when none within max_range."""
        angles = np.asarray(angles, float).reshape(-1)
        if len(self.segs) == 0:
            return np.full(len(angles), np.inf), np.full(len(angles), -1)
        dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]
        px, py = self.segs[None, :, 0], self.segs[None, :, 1]
        ex, ey = self.segs[None, :, 2] - px, self.segs[None, :, 3] - py
        wx, wy = px - ox, py - oy
        denom = dx * ey - dy * ex
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (wx * ey - wy * ex) / denom
            u = (wx * dy - wy * dx) / denom
        ok = (np.abs(denom) > 1e-12) & (t >= 0.0) & (u >= -1e-9) & (u <= 1.0 + 1e-9) & (t <= max_range)
        if skip is not None:
            ok &= ~skip[None, :]
        t = np.where(ok, t, np.inf)
        idx = np.argmin(t, axis=1)
        rng = t[np.arange(len(angles)), idx]
        return rng, np.where(np.isfinite(rng), idx, -1)

    def collides(self, x, y, yaw, length, width) -> bool:
        if len(self.solid_pts) == 0:
            return False
        hl, hw = 0.5 * length + self.half_thickness, 0.5 * width + self.half_thickness
        d = self.solid_pts - (x, y)
        near = np.abs(d).max(axis=1) <= math.hypot(hl, hw)
        if not near.any():
            return False
        d = d[near]
        c, s = math.cos(yaw), math.sin(yaw)
        lx = c * d[:, 0] + s * d[:, 1]
        ly = -s * d[:, 0] + c * d[:, 1]
        return bool(((np.abs(lx) < hl) & (np.abs(ly) < hw)).any())


# ============================================================================================ Nav2 stand-ins
@dataclass
class Task:
    """A motion request the world executes (one at a time; a new one preempts the old)."""
    kind: str
    state: str = RUNNING
    reason: str = ''
    started: float = 0.0
    seq: int = 0
    feedback: dict = field(default_factory=dict)

    @property
    def done(self):
        return self.state != RUNNING

    def finish(self, state, reason=''):
        if self.state == RUNNING:
            self.state, self.reason = state, reason

    def command(self, world, dt):            # -> (vx, vy, wz)
        return 0.0, 0.0, 0.0


class SpinTask(Task):
    def __init__(self, target_yaw, allowance_s, **kw):
        super().__init__('spin', **kw)
        self.target, self.allowance = float(target_yaw), float(allowance_s)
        self.turned, self.last_yaw = 0.0, None

    def command(self, world, dt):
        p = world.pose
        if self.last_yaw is not None:
            self.turned += wrap(p[2] - self.last_yaw)
        self.last_yaw = p[2]
        remaining = self.target - self.turned
        self.feedback = {'angular_distance_traveled': abs(self.turned)}
        if abs(remaining) < 0.02:
            self.finish(SUCCEEDED)
            return 0.0, 0.0, 0.0
        if world.time - self.started > self.allowance:
            self.finish(FAILED, 'time allowance exceeded')
            return 0.0, 0.0, 0.0
        sc = world.cfg['spin']
        v = max(sc['min_vel'], min(sc['max_vel'], math.sqrt(2.0 * sc['acc'] * abs(remaining))))
        wz = math.copysign(v, remaining)
        nxt = p[2] + wz * dt * 4.0                    # Nav2 simulates a little ahead
        if world.geometry.collides(p[0], p[1], nxt, world.length, world.width):
            self.finish(FAILED, 'collision ahead')
            return 0.0, 0.0, 0.0
        return 0.0, 0.0, wz


class NavTask(Task):
    """NavigateToPose: plan, follow, replan, check progress, rotate to the goal heading."""

    def __init__(self, goal, **kw):
        super().__init__('navigate', **kw)
        self.goal = (float(goal[0]), float(goal[1]), float(goal[2]))
        self.path: Optional[List[Tuple[float, float]]] = None
        self.idx = 0
        self.last_plan = -1e9
        self.plan_failures = 0
        self.progress_ref = None              # (time, x, y)
        self.at_goal = False
        self.failing_since = None             # RPP collision detection: controller failing since this time
        self.retries = 0                      # bt_navigator RecoveryNode retries used
        self.recovery = None                  # running recovery: dict(kind, ...)
        self.recoveries_run = []

    def _replan(self, world):
        self.last_plan = world.time
        path = world.plan(world.pose[:2], self.goal[:2])
        if path is None:
            self.plan_failures += 1
            if self.plan_failures >= int(world.cfg['controller']['max_plan_failures']):
                self._fail(world, 'no path to the goal')
            return
        self.plan_failures = 0
        self.path, self.idx = path, 0

    # -- bt_navigator recoveries
    RECOVERY_CYCLE = ('clear', 'spin', 'wait', 'backup')

    def _fail(self, world, reason):
        c = world.cfg['controller']
        if not c.get('bt_recoveries', True) or self.retries >= int(c['number_of_retries']):
            self.finish(FAILED, reason + (f' (after {self.retries} recoveries: {", ".join(self.recoveries_run)})'
                                          if self.retries else ''))
            return
        kind = self.RECOVERY_CYCLE[self.retries % len(self.RECOVERY_CYCLE)]
        self.retries += 1
        self.recoveries_run.append(kind)
        world.log(f'navigate: {reason}; recovery {self.retries}/{c["number_of_retries"]}: {kind}')
        x, y, yaw = world.pose
        self.recovery = {'kind': kind, 'start': world.time, 'x': x, 'y': y, 'turned': 0.0, 'last_yaw': yaw}
        self.path, self.plan_failures, self.failing_since, self.progress_ref = None, 0, None, None

    def _recover(self, world, dt):
        c, r = world.cfg['controller'], self.recovery
        x, y, yaw = world.pose
        done, cmd = False, (0.0, 0.0, 0.0)
        if r['kind'] == 'clear':
            world.clear_costmaps()
            done = True
        elif r['kind'] == 'wait':
            done = world.time - r['start'] >= float(c['recovery_wait_s'])
        elif r['kind'] == 'spin':
            r['turned'] += abs(wrap(yaw - r['last_yaw']))
            r['last_yaw'] = yaw
            wz = float(world.cfg['spin']['max_vel'])
            if r['turned'] >= float(c['recovery_spin_rad']) or world.time - r['start'] > 10.0 or \
                    world.geometry.collides(x, y, yaw + wz * 0.4, world.length, world.width):
                done = True
            else:
                cmd = (0.0, 0.0, wz)
        elif r['kind'] == 'backup':
            v = -abs(float(c['backup_speed']))
            back = (x + math.cos(yaw) * v * 1.0, y + math.sin(yaw) * v * 1.0)
            if math.hypot(x - r['x'], y - r['y']) >= float(c['backup_dist_m']) or world.time - r['start'] > 15.0 or \
                    world.geometry.collides(back[0], back[1], yaw, world.length, world.width):
                done = True
            else:
                cmd = (v, 0.0, 0.0)
        if done:
            self.recovery = None
            return 0.0, 0.0, 0.0
        return cmd

    def command(self, world, dt):
        if self.recovery is not None:
            return self._recover(world, dt)
        cmd = self._command(world, dt)
        c = world.cfg['controller']
        if self.done or not c.get('collision_detection', True) or not any(cmd):
            self.failing_since = None
            return cmd
        # RPP use_collision_detection: the footprint projected along the command must stay clear for
        # collision_horizon_s; otherwise the controller fails, and after failure_tolerance_s Nav2 aborts the goal
        x, y, yaw = world.pose
        vx, vy, wz = cmd
        horizon = float(c['collision_horizon_s'])
        for k in range(1, 6):
            t = horizon * k / 5.0
            a = yaw + wz * t
            px = x + (math.cos(yaw) * vx - math.sin(yaw) * vy) * t
            py = y + (math.sin(yaw) * vx + math.cos(yaw) * vy) * t
            if abs(vx) + abs(vy) < 1e-6:
                px, py = x, y
            if world.geometry.collides(px, py, a, world.length, world.width):
                if self.failing_since is None:
                    self.failing_since = world.time
                    world.log('controller: collision ahead, stopping')
                elif world.time - self.failing_since > float(c['failure_tolerance_s']):
                    self._fail(world, 'collision ahead (controller failed)')
                return 0.0, 0.0, 0.0
        self.failing_since = None
        return cmd

    def _command(self, world, dt):
        c = world.cfg['controller']
        x, y, yaw = world.pose
        gx, gy, gyaw = self.goal
        dist_goal = math.hypot(gx - x, gy - y)
        if self.progress_ref is None:
            self.progress_ref = (world.time, x, y)
        # goal checker (stateful: once inside xy tolerance only the heading is fixed)
        if self.at_goal or dist_goal <= c['xy_goal_tolerance']:
            self.at_goal = True
            err = wrap(gyaw - yaw)
            if abs(err) <= c['yaw_goal_tolerance']:
                self.finish(SUCCEEDED)
                return 0.0, 0.0, 0.0
            return 0.0, 0.0, math.copysign(max(0.3, min(c['rotate_vel'], 1.5 * abs(err))), err)
        # progress checker
        t0, x0, y0 = self.progress_ref
        if math.hypot(x - x0, y - y0) >= c['progress_min_m']:
            self.progress_ref = (world.time, x, y)
        elif world.time - t0 > c['progress_window_s']:
            self._fail(world, f'no progress ({c["progress_min_m"]} m in {c["progress_window_s"]} s)')
            return 0.0, 0.0, 0.0
        if self.path is None or world.time - self.last_plan >= c['replan_period_s']:
            self._replan(world)
            if self.done or self.path is None:
                return 0.0, 0.0, 0.0
        self.feedback = {'distance_remaining': path_length(self.path[self.idx:]), 'current': (x, y, yaw)}
        pts = np.asarray(self.path)
        # nearest path point ahead of the last one, then the lookahead point
        win = pts[self.idx:min(len(pts), self.idx + 60)]
        self.idx += int(np.argmin(np.hypot(win[:, 0] - x, win[:, 1] - y)))
        d = np.hypot(pts[self.idx:, 0] - x, pts[self.idx:, 1] - y)
        far = np.flatnonzero(d >= c['lookahead_m'])
        lx, ly = pts[self.idx + far[0]] if len(far) else pts[-1]
        v = c['desired_linear_vel']
        if dist_goal < c['approach_dist_m']:
            v = max(c['min_approach_vel'], v * dist_goal / c['approach_dist_m'])
        bearing = math.atan2(ly - y, lx - x)
        err = wrap(bearing - yaw)
        if c['type'] == 'omni':
            ld = max(1e-6, math.hypot(lx - x, ly - y))
            vx, vy = v * math.cos(err), v * math.sin(err)
            seg_yaw = bearing if ld > 0.05 else gyaw
            wz = max(-c['max_angular_vel'], min(c['max_angular_vel'], 1.5 * wrap(seg_yaw - yaw)))
            return vx, vy, wz
        if abs(err) > c['rotate_to_heading_min_angle']:
            return 0.0, 0.0, math.copysign(c['rotate_vel'], err)
        ld = max(1e-6, math.hypot(lx - x, ly - y))
        curvature = 2.0 * math.sin(err) / ld
        # RPP use_regulated_linear_velocity_scaling: slow down on tight curves (not while approaching the goal)
        if abs(curvature) > 1e-6 and 1.0 / abs(curvature) < c['regulated_min_radius'] and \
                dist_goal >= c['approach_dist_m']:
            v = max(c['regulated_min_speed'], v * (1.0 / abs(curvature)) / c['regulated_min_radius'])
        wz = max(-c['max_angular_vel'], min(c['max_angular_vel'], v * curvature))
        return v, 0.0, wz


class WaypointsTask(Task):
    """FollowWaypoints: drive through the poses one by one; a failed pose is 'missed' and skipped."""

    def __init__(self, goals, **kw):
        super().__init__('waypoints', **kw)
        self.goals = [tuple(map(float, g)) for g in goals]
        self.i = 0
        self.missed: List[int] = []
        self.sub: Optional[NavTask] = None

    def command(self, world, dt):
        while self.i < len(self.goals):
            if self.sub is None:
                self.sub = NavTask(self.goals[self.i], started=world.time)
            cmd = self.sub.command(world, dt)
            if not self.sub.done:
                self.feedback = {'current_waypoint': self.i}
                return cmd
            if self.sub.state != SUCCEEDED:
                self.missed.append(self.i)
                world.log(f'waypoint {self.i} missed: {self.sub.reason}')
            self.i += 1
            self.sub = None
        self.finish(SUCCEEDED, f'missed {self.missed}' if self.missed else '')
        return 0.0, 0.0, 0.0

    @property
    def path(self):
        return self.sub.path if self.sub is not None else None


# ============================================================================================ the world
class World:
    def __init__(self, maze: Maze, cfg: Optional[dict] = None, logger=None):
        self.maze = maze
        self.cfg = cfg = deep_merge(DEFAULTS, cfg or {})
        self._log = logger
        self.length, self.width = float(cfg['robot']['length_m']), float(cfg['robot']['width_m'])
        start_maze = maze.start
        if cfg['map_origin'] == 'start':
            self.map_T_maze = invert(start_maze)            # the map frame IS the start pose
        elif cfg['map_origin'] == 'maze':
            self.map_T_maze = (0.0, 0.0, 0.0)
        else:
            raise ValueError(f"map_origin must be 'start' or 'maze', not {cfg['map_origin']!r}")
        self.geometry = Geometry(maze, self.map_T_maze)
        self.pose = list(compose(self.map_T_maze, start_maze))
        self.vel = [0.0, 0.0, 0.0]                           # body vx, vy, wz
        if self.geometry.collides(*self.pose, self.length, self.width):
            self.log('WARNING: the robot starts in collision with a wall (check start.offset_m).')
        self.time = 0.0
        self.bumped = False
        self.task: Optional[Task] = None
        self._seq = 0
        self.cmd_vel = None                                  # (vx, vy, wz, received_time)
        self._warned_cmd = False

        # SLAM map covering the maze + margin
        sc = cfg['slam']
        res, m = float(sc['resolution']), float(sc['margin_m'])
        x0, y0, x1, y1 = self.geometry.bounds
        x0, y0 = math.floor((x0 - m) / res) * res, math.floor((y0 - m) / res) * res
        self.info = GridInfo(int(math.ceil((x1 + m - x0) / res)), int(math.ceil((y1 + m - y0) / res)), res, x0, y0)
        self.mapper = OccupancyMapper(self.info)
        self.map_grid = self.mapper.grid()
        self.map_version = 0
        self.virtual_pts = np.zeros((0, 2))
        self.global_costmap = None
        self.global_version = 0
        self.local_costmap = None                           # (grid, GridInfo)
        self.planner = GridPlanner(unknown_cost=float(cfg['planner']['unknown_cost']),
                                   tolerance_m=float(cfg['planner']['tolerance_m']),
                                   allow_unknown=bool(cfg['planner']['allow_unknown']))
        self.last_scan = None                                # (angles_robot, ranges, stamp)
        self._next = {'scan': 0.0, 'map': 0.0, 'global': 0.0, 'local': 0.0, 'curtain': 0.0}
        cc = cfg['curtain_camera']
        self.curtains = CurtainMap(float(cc['cell_m']), int(cc['confirm_hits'])) if CurtainMap else None
        self.curtain_version = 0
        self.curtain_seen_now = np.zeros((0, 2))
        self.rng = np.random.default_rng(0)
        self.events: List[str] = []

    def log(self, msg):
        self.events.append(msg)
        if self._log:
            self._log(msg)

    # -------------------------------------------------------------------- frames
    def maze_to_map(self, x, y, yaw=0.0):
        return compose(self.map_T_maze, (x, y, yaw))

    @property
    def maze_pose_in_map(self):
        return self.map_T_maze

    # -------------------------------------------------------------------- commands
    def start_task(self, task: Task) -> Task:
        if self.task is not None and not self.task.done:
            self.task.finish(FAILED, f'preempted by {task.kind}')
        self._warned_cmd = False
        self._seq += 1
        task.seq, task.started = self._seq, self.time
        self.task = task
        return task

    def cancel(self, task: Optional[Task]):
        if task is not None and task.state == RUNNING:
            task.finish(CANCELLED, 'cancelled')

    def spin(self, yaw, allowance_s):
        return self.start_task(SpinTask(yaw, allowance_s))

    def navigate(self, goal):
        return self.start_task(NavTask(goal))

    def follow_waypoints(self, goals):
        return self.start_task(WaypointsTask(goals))

    def set_cmd_vel(self, vx, vy, wz):
        self.cmd_vel = (float(vx), float(vy), float(wz), self.time)

    def set_virtual_walls(self, pts):
        self.virtual_pts = np.asarray(pts, float).reshape(-1, 2)

    def clear_costmaps(self):
        """Nav2 clear_entirely_*: the obstacle layers forget the virtual walls until the next /virtual_walls."""
        self.virtual_pts = np.zeros((0, 2))
        self._next['global'] = self._next['local'] = self.time

    def plan(self, start_xy, goal_xy):
        if self.global_costmap is None:
            self.update_global_costmap()
        self.planner.set_costmap(self.global_costmap, self.info, self.global_version)
        return self.planner.plan(start_xy, goal_xy)

    # -------------------------------------------------------------------- simulation step
    def step(self, dt: float):
        dt = dt * float(self.cfg['speed_scale'])
        self.time += dt
        cmd = (0.0, 0.0, 0.0)
        task = self.task
        if task is not None and task.done:
            self.task = task = None
        if task is not None:
            cmd = task.command(self, dt)
        if self.cmd_vel is not None:
            vx, vy, wz, stamp = self.cmd_vel
            if self.time - stamp <= float(self.cfg['cmd_vel_timeout_s']) * float(self.cfg['speed_scale']):
                moving = any(abs(v) > 1e-6 for v in (vx, vy, wz))
                if task is not None and moving and not self._warned_cmd:
                    self._warned_cmd = True
                    self.log(f'/cmd_vel received while Nav2 runs a {task.kind} goal: /cmd_vel wins')
                if task is None or moving:          # a zero Twist does not stop a running Nav2 goal
                    cmd = (vx, vy, wz)
            else:
                self.cmd_vel = None
        self._integrate(cmd, dt)
        self._sensors()

    def _integrate(self, cmd, dt):
        rc = self.cfg['robot']
        lin, ang = rc['max_lin_acc'] * dt, rc['max_ang_acc'] * dt
        self.vel = [v + max(-lin, min(lin, c - v)) for v, c in zip(self.vel[:2], cmd[:2])] + \
                   [self.vel[2] + max(-ang, min(ang, cmd[2] - self.vel[2]))]
        vx, vy, wz = self.vel
        x, y, yaw = self.pose
        c, s = math.cos(yaw + 0.5 * wz * dt), math.sin(yaw + 0.5 * wz * dt)
        nx, ny, nyaw = x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, wrap(yaw + wz * dt)
        g = self.geometry
        self.bumped = False
        for cand in ((nx, ny, nyaw), (x, y, nyaw), (nx, ny, yaw)):
            if not g.collides(*cand, self.length, self.width):
                if cand != (nx, ny, nyaw):
                    self.bumped = True
                self.pose = list(cand)
                return
        self.bumped = True
        self.vel = [0.0, 0.0, 0.0]

    def _due(self, key, hz):
        if hz <= 0 or self.time < self._next[key]:
            return False
        self._next[key] = max(self._next[key] + 1.0 / hz, self.time)
        return True

    def _sensors(self):
        cfg = self.cfg
        if self._due('scan', cfg['lidar']['rate_hz']):
            self.scan()
        if self._due('map', cfg['slam']['update_hz']):
            self.map_grid = self.mapper.grid()
            self.map_version += 1
        if self._due('global', cfg['global_costmap']['update_hz']):
            self.update_global_costmap()
        if self._due('local', cfg['local_costmap']['update_hz']):
            self.update_local_costmap()
        if cfg['curtain_camera']['enabled'] and self._due('curtain', cfg['curtain_camera']['rate_hz']):
            self.sense_curtains()

    # -------------------------------------------------------------------- sensors
    def scan(self):
        lc = self.cfg['lidar']
        inc = float(lc['angle_increment'])
        rel = np.arange(-math.pi, math.pi, inc)
        x, y, yaw = self.pose
        ox, oy = x + math.cos(yaw) * lc['x_offset_m'], y + math.sin(yaw) * lc['x_offset_m']
        rng, idx = self.geometry.raycast(ox, oy, rel + yaw, float(lc['range_max']))
        if lc['noise_m'] > 0:
            rng = rng + self.rng.normal(0.0, lc['noise_m'], len(rng))
        blind = rng < lc['range_min']                        # too close: no reading at all (clears nothing)
        rng[blind] = np.inf
        hit = np.isfinite(rng) & (rng <= lc['range_max'])
        use = ~blind
        self.mapper.integrate(ox, oy, (rel + yaw)[use], np.where(hit, rng, np.inf)[use], hit[use],
                              float(lc['range_max']))
        self.last_scan = (rel, np.where(hit, rng, np.inf), self.time)
        return self.last_scan

    def update_global_costmap(self):
        gc = self.cfg['global_costmap']
        grid = self.mapper.grid()
        lethal = grid == 100
        if len(self.virtual_pts):
            info = self.info
            c = np.floor((self.virtual_pts[:, 0] - info.origin_x) / info.resolution).astype(int)
            r = np.floor((self.virtual_pts[:, 1] - info.origin_y) / info.resolution).astype(int)
            ok = (r >= 0) & (r < info.height) & (c >= 0) & (c < info.width)
            lethal[r[ok], c[ok]] = True
        inscribed = min(gc['footprint_length_m'], gc['footprint_width_m']) / 2.0 + gc['padding_m']
        self.global_costmap = inflate(lethal, grid == -1, self.info.resolution, inscribed,
                                      float(gc['inflation_radius_m']), float(gc['cost_scaling_factor']))
        self.global_version += 1

    def update_local_costmap(self):
        lc = self.cfg['local_costmap']
        res = self.info.resolution
        n = int(round(float(lc['size_m']) / res))
        x, y, _ = self.pose
        ox = math.floor((x - 0.5 * n * res) / res) * res
        oy = math.floor((y - 0.5 * n * res) / res) * res
        info = GridInfo(n, n, res, ox, oy)
        r0, c0 = self.info.world_to_cell(ox + 0.5 * res, oy + 0.5 * res)
        grid = self.mapper.grid()
        win = np.full((n, n), -1, np.int16)
        rs, cs = max(0, r0), max(0, c0)
        re, ce = min(self.info.height, r0 + n), min(self.info.width, c0 + n)
        if re > rs and ce > cs:
            win[rs - r0:re - r0, cs - c0:ce - c0] = grid[rs:re, cs:ce]
        inscribed = min(self.length, self.width) / 2.0
        cm = inflate(win == 100, np.zeros_like(win, bool), res, inscribed, float(lc['inflation_radius_m']),
                     float(lc['cost_scaling_factor']))
        self.local_costmap = (cm, info)

    def sense_curtains(self):
        """Depth camera: curtain points in its field of view, in line of sight, over at least min_columns of the
        image width (like the real detector) -> CurtainMap."""
        cc = self.cfg['curtain_camera']
        self.curtain_seen_now = np.zeros((0, 2))
        g = self.geometry
        if self.curtains is None or len(g.curtain_pts) == 0:
            return False
        x, y, yaw = self.pose
        cx, cy = x + math.cos(yaw) * cc['x_offset_m'], y + math.sin(yaw) * cc['x_offset_m']
        d = g.curtain_pts - (cx, cy)
        dist = np.hypot(d[:, 0], d[:, 1])
        bearing = np.array([wrap(a) for a in np.arctan2(d[:, 1], d[:, 0]) - yaw])
        half = math.radians(cc['hfov_deg']) / 2.0
        view = (dist >= cc['range_min']) & (dist <= cc['range_max']) & (np.abs(bearing) <= half)
        if not view.any():
            return False
        idx = np.flatnonzero(view)
        # line of sight: nothing but the curtain's own segment closer than the point
        skip = np.zeros(len(g.segs), bool)
        vis = []
        for k in idx:
            skip[:] = False
            skip[g.curtain_seg[k]] = True
            rng, _ = g.raycast(cx, cy, [math.atan2(d[k, 1], d[k, 0])], dist[k] - 0.01, skip=skip)
            vis.append(not np.isfinite(rng[0]))
        idx = idx[np.array(vis, bool)]
        if len(idx) == 0:
            return False
        span = bearing[idx].max() - bearing[idx].min()
        if span < cc['min_columns'] * 2.0 * half:
            return False
        step = max(1, int(round(cc['point_step_m'] / 0.025)))
        pts = g.curtain_pts[idx[::step]] + self.rng.normal(0.0, cc['noise_m'], (len(idx[::step]), 2))
        self.curtain_seen_now = pts
        if self.curtains.add(pts, (cx, cy)) or len(pts):
            self.curtain_version += 1
        return True

    def curtain_points(self) -> np.ndarray:
        return self.curtains.confirmed() if self.curtains is not None else np.zeros((0, 4))

    # -------------------------------------------------------------------- lookups
    def cost_at(self, grid_info_pair, x, y):
        grid, info = grid_info_pair
        r, c = info.world_to_cell(x, y)
        return int(grid[r, c]) if info.contains(r, c) else None

    def robot_in_inscribed(self) -> bool:
        if self.global_costmap is None:
            return False
        v = self.cost_at((self.global_costmap, self.info), self.pose[0], self.pose[1])
        return v is not None and v >= INSCRIBED

    def tile_of_maze(self) -> Tuple[int, int]:
        """The maze tile (i, j) the robot is on (ground truth)."""
        mx, my, _ = compose(invert(self.map_T_maze), tuple(self.pose))
        s = self.maze.tile_size
        return int(math.floor(mx / s)), int(math.floor(my / s))
