"""Run lite3_explorer's mission logic against the simulated world WITHOUT ROS, faster than real time.

The same mission modes, exploration profiles, tile tracker and end conditions as the mission_manager node, with
the world's Nav2 stand-ins called directly. Use it to check a maze file quickly (seconds instead of minutes) or in
tests; use the ROS simulation (sim.launch.py) to watch and debug a run in RViz.

    ros2 run lite3_sim headless --maze ~/my_maze.yaml
    ros2 run lite3_sim headless --maze ~/my_maze.yaml --set modes.exploring.curtains=false --set tiles.max_failures=1
    python3 -m lite3_sim.headless --maze src/lite3_sim/mazes/curtain_loop.yaml --mission-params my_mission.yaml

What the mission_manager node does and this does not: AI tile reports, terrain classification, the re-evaluation of
tiles after the virtual walls change (_reevaluate_tiles), RViz output. The virtual walls ARE fed to the costmap.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional

import numpy as np
import yaml

from lite3_explorer import map_grid as G
from lite3_explorer import turn_check as TC
from lite3_explorer.mission.config import DEFAULTS as MISSION_DEFAULTS, merge_flat
from lite3_explorer.mission.core import ModeMachine
from lite3_explorer.mission.traversal import FAILED, RUNNING, SUCCEEDED, Traversal, TraversalResult
from lite3_explorer.tile_tracker import TileTracker
from lite3_explorer.tiles import tile_label as T_label

from lite3_sim import world as W
from lite3_sim.maze import load_maze

# Mission settings the simulation always uses (on top of the mission YAML): no AI, no terrain classifier, no
# animation player, curtain and stairs modes on.
SIM_MISSION = {
    'use_ai': False,
    'terrain_classify_service': '',
    'modes.exploring.classify_target': False,
    'modes.exploring.quickscan_on_edge': False,
    'modes.exploring.curtains': True,
    'modes.exploring.stairs': True,
    'modes.idle.start_wait_s': 5.0,
}


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def mission_flat_from_yaml(path):
    with open(os.path.expanduser(path), encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}
    node = data.get('mission_manager', data)
    return flatten(node.get('ros__parameters', node))


# ------------------------------------------------------------------------------------------ Nav2 stand-ins
class _Scanner:
    def __init__(self, ctx):
        self.ctx, self.task = ctx, None

    ready = True

    def start(self, deg, allowance):
        self.task = self.ctx.world.spin(math.radians(deg), allowance)

    def poll(self):
        t = self.task
        if t is None or t.state == W.RUNNING:
            return None
        return SUCCEEDED if t.state == W.SUCCEEDED else FAILED

    def cancel(self):
        self.ctx.world.cancel(self.task)
        self.task = None

    def has_room(self):
        lc = self.ctx.world.local_costmap
        if lc is None:
            return True
        grid, info = lc
        x, y, _ = self.ctx.world.pose
        ok, _ = TC.can_turn_in_place(grid, (info.origin_x, info.origin_y), info.resolution, x, y,
                                     float(self.ctx.cfg['robot_length_m']), float(self.ctx.cfg['robot_width_m']),
                                     float(self.ctx.cfg['turn_room_margin_m']))
        return ok


class _Paths:
    ready = True

    def __init__(self, ctx):
        self.ctx, self.results = ctx, None

    def start(self, goals):
        w = self.ctx.world
        self.results = [w.plan(w.pose[:2], g) for g in goals]
        return True

    def poll(self):
        return self.results

    def cancel(self):
        self.results = None


class SimTraversal(Traversal):
    """nav2_pose / nav2_route profiles on the simulated Nav2."""

    def __init__(self, ctx, name, cfg):
        super().__init__(ctx, name, cfg)
        self.uses_route = cfg.get('type') == 'nav2_route'
        self.task = None

    def available(self):
        return True

    def start(self, goal_xy, yaw, route=None):
        w = self.ctx.world
        if self.uses_route:
            poses = list(route or [])[:-1] + [(goal_xy[0], goal_xy[1], yaw)]
            self.task = w.follow_waypoints(poses)
        else:
            self.task = w.navigate((goal_xy[0], goal_xy[1], yaw))
        return True

    def poll(self):
        t = self.task
        if t is None:
            return TraversalResult(FAILED, 'not started')
        if t.state == W.RUNNING:
            return TraversalResult(RUNNING)
        if t.state == W.SUCCEEDED:
            if isinstance(t, W.WaypointsTask) and (len(t.goals) - 1) in t.missed:
                return TraversalResult(FAILED, 'the final waypoint was missed')
            return TraversalResult(SUCCEEDED)
        return TraversalResult(FAILED, t.reason or t.state)

    def cancel(self):
        self.ctx.world.cancel(self.task)


class _NoAiCall:
    done, ok, result = True, False, {}

    def failure(self):
        return 'AI disabled (simulation)'


# ------------------------------------------------------------------------------------------ context
class HeadlessCtx:
    """The mission_manager's context API on top of a World."""

    def __init__(self, world: W.World, cfg: dict, verbose: bool = False):
        self.world, self.cfg, self.verbose = world, cfg, verbose
        tcfg = cfg['tiles']
        self.tiles = TileTracker(float(tcfg['size_m']), float(tcfg['reach_radius_m']), int(tcfg['max_failures']),
                                 min_known=float(tcfg.get('min_known', 0.5)),
                                 scene_min_known=float(tcfg.get('scene_min_known', 0.85)),
                                 scene_max_free=float(tcfg.get('scene_max_free', 0.80)))
        self.tiles.skip_scene = bool(cfg['modes'].get('exploring', {}).get('scenes', False))   # like the node
        self.tiles.stair_min_points = int(tcfg.get('stair_min_points', 4))
        self.ai_commands = []                          # (time, on, cmd_code) the stairs mode sent
        self._stair_version = -1
        self.scenes_described = []                     # (label, pose) the scene mode asked the (absent) AI about
        self.scanner, self.paths = _Scanner(self), _Paths(self)
        self.opening_done, self.initial_heading = False, None
        self.events, self.logs = [], []
        self.complete = None
        self._traversals = {}
        self._cmd = None
        self._map_version = -1

    # time / logging
    def now(self):
        return self.world.time

    def log(self, level, msg):
        self.logs.append((round(self.world.time, 1), level, msg))
        if self.verbose:
            print(f'[{self.world.time:7.1f}] {level:5s} {msg}')

    def event(self, kind, **data):
        self.events.append((round(self.world.time, 1), kind, data))

    def on_mode_change(self, mode, phase):
        pass

    # scene mode: no AI in the simulation -> a call that has already failed (as mission_manager with use_ai false)
    def animation_markers(self):
        return []

    def describe_scene(self, label, pose, view=None):
        self.scenes_described.append((label, list(pose) if pose else None))
        return _NoAiCall()

    def record_scene(self, label, pose, call):
        pass

    def on_complete(self, reason):
        self.complete = reason

    # robot state
    def pose(self):
        return [round(v, 3) for v in self.world.pose]

    def local_map_ready(self, pose, radius_m=1.5, min_known=0.5):
        if self.tiles.map is None:
            return False
        grid, info = self.tiles.map
        r, c = info.world_to_cell(pose[0], pose[1])
        rad = int(radius_m / info.resolution)
        win = grid[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1]
        return win.size > 0 and (win >= 0).mean() >= min_known

    def ready_for_start(self):
        return self.tiles.map is not None, ([] if self.tiles.map is not None else ['/map'])

    def gait_hold(self, kind):
        pass

    def classify_tile(self, label):
        pass

    def terrain_status(self, label):
        return 'seen'

    def animation_play(self, name):
        return False

    def animation_state(self):
        return 'done'

    def animation_cancel(self):
        pass

    # Nav2
    def traversal(self, name):
        if name not in self._traversals:
            profiles = self.cfg['traversals']
            if name not in profiles:
                raise KeyError(f'no traversal profile {name!r}')
            self._traversals[name] = SimTraversal(self, name, profiles[name])
        return self._traversals[name]

    def costmap_snapshot(self):
        w = self.world
        if w.global_costmap is None:
            return None
        i = w.info
        return w.global_costmap, G.GridInfo(i.width, i.height, i.resolution, i.origin_x, i.origin_y)

    def costmap_cost_at_robot(self):
        snap = self.costmap_snapshot()
        if snap is None:
            return None
        return G.value_at(snap[0], snap[1], *self.world.pose[:2])

    def clear_costmaps(self):
        self.world.clear_costmaps()

    def set_target(self, label, goal_xy, route):
        pass

    def publish_goal(self, *a):
        pass

    # curtain / mppi modes
    def curtain_points(self):
        return self.world.curtain_points()

    def drive(self, vx, wz, vy=0.0):
        self.world.set_cmd_vel(vx, vy, wz)

    def stop(self):
        self.world.set_cmd_vel(0.0, 0.0, 0.0)

    # stairs mode: /simple_cmd straight to the simulated robot
    def ai_mode(self, on, cmd_code):
        self.ai_commands.append((round(self.world.time, 1), bool(on), int(cmd_code)))
        self.world.simple_cmd(cmd_code)
        return True

    # what the node's timers do
    def sync_map(self):
        w = self.world
        if w.map_version != self._map_version:
            self._map_version = w.map_version
            i = w.info
            self.tiles.set_map(w.map_grid.copy(), G.GridInfo(i.width, i.height, i.resolution, i.origin_x, i.origin_y))

    def sync_stairs(self):
        """The node's /stairs/points subscription: new confirmed stair cells -> staircase tiles."""
        w = self.world
        if w.stair_version != self._stair_version:
            self._stair_version = w.stair_version
            if self.tiles.set_stair_points(w.stair_points()):
                self.tiles.update()

    def virtual_walls(self):
        if not (self.tiles.ready and self.opening_done):
            return
        pts = []
        for (x0, y0), (x1, y1) in self.tiles.virtual_walls():
            n = max(1, int(math.dist((x0, y0), (x1, y1)) / 0.05))
            pts += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n + 1)]
        self.world.set_virtual_walls(pts)


# ------------------------------------------------------------------------------------------ runner
class HeadlessRun:
    def __init__(self, maze_path: str, mission_flat: Optional[dict] = None, sim_cfg: Optional[dict] = None,
                 verbose: bool = False, overrides: Optional[dict] = None):
        self.maze = load_maze(maze_path)
        self.world = W.World(self.maze, sim_cfg)
        flat = dict(mission_flat or {})
        flat.update(SIM_MISSION)                             # the simulation's settings win over the mission YAML,
        flat['tiles.size_m'] = self.maze.tile_size          # so do the maze's tiles ...
        flat.update(overrides or {})                         # ... and explicit overrides over everything
        self.cfg = merge_flat(MISSION_DEFAULTS, flat)
        self.ctx = HeadlessCtx(self.world, self.cfg, verbose)
        self.machine = ModeMachine(self.ctx, self.cfg)
        self.truth_visited = set()

    def run(self, max_time_s: float = 1800.0, dt: float = 0.05):
        w, ctx, m = self.world, self.ctx, self.machine
        tick = float(self.cfg['tick_period_s'])
        next_tick = next_walls = next_track = 0.0
        m.start()
        while w.time < max_time_s and not ctx.complete:
            w.step(dt)
            ctx.sync_map()
            ctx.sync_stairs()
            if w.time >= next_tick:
                next_tick += tick
                m.tick()
            if w.time >= next_track:
                next_track += 0.5
                if ctx.tiles.ready and ctx.opening_done:
                    ctx.tiles.track(ctx.pose())
                    ctx.tiles.update()
            if w.time >= next_walls:
                next_walls += 1.0
                ctx.virtual_walls()
            self._truth_track()
        return self.result()

    def _truth_track(self):
        mx, my, _ = W.compose(W.invert(self.world.map_T_maze), tuple(self.world.pose))
        s = self.maze.tile_size
        i, j = int(mx // s), int(my // s)
        if math.hypot(mx - (i + 0.5) * s, my - (j + 0.5) * s) <= float(self.cfg['tiles']['reach_radius_m']) * 1.5:
            self.truth_visited.add((i, j))

    def result(self) -> dict:
        reach = self.maze.reachable_tiles()
        return {
            'maze': self.maze.name,
            'complete': self.ctx.complete,
            'sim_time_s': round(self.world.time, 1),
            'modes': [t[2] for t in self.machine.transitions],
            'tile_counts': self.ctx.tiles.counts() if self.ctx.tiles.tiles else {},
            'unreachable': dict(self.ctx.tiles.unreachable),
            'truth_tiles_reachable': len(reach),
            'truth_tiles_visited': len(self.truth_visited & reach),
            'truth_tiles_missed': sorted(reach - self.truth_visited),
            'curtain_events': [e for e in self.ctx.events if e[1] == 'curtain'],
            'scene_events': [e for e in self.ctx.events if e[1] in ('scene', 'scene_done')],
            'stairs_events': [e for e in self.ctx.events if e[1] in ('stairs', 'stairs_crossed', 'stairs_failed')],
            'ai_commands': [(t, on, f'0x{code:08X}') for t, on, code in self.ctx.ai_commands],
            'staircase_tiles': sorted(T_label(t) for t in self.ctx.tiles.stairs),
            'truth_stairs': sorted(self.maze.stairs),
            'truth_scenes': sorted(self.maze.scenes),
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--maze', required=True)
    ap.add_argument('--mission-params', default='', help="mission YAML (e.g. lite3_explorer's config/mission.yaml)")
    ap.add_argument('--sim-params', default='', help='lite3_sim YAML (config/sim.yaml)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='mission parameter override, e.g. modes.exploring.curtains=false (YAML value)')
    ap.add_argument('--max-time', type=float, default=1800.0, help='simulated seconds')
    ap.add_argument('-v', '--verbose', action='store_true', help='print the mission log')
    a = ap.parse_args(argv)
    flat = mission_flat_from_yaml(a.mission_params) if a.mission_params else {}
    overrides = {}
    for kv in a.set:
        k, _, v = kv.partition('=')
        overrides[k.strip()] = yaml.safe_load(v)
    sim_cfg = None
    if a.sim_params:
        with open(os.path.expanduser(a.sim_params), encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        sim_cfg = data.get('lite3_sim', {}).get('ros__parameters', data)
    run = HeadlessRun(a.maze, flat, sim_cfg, verbose=a.verbose, overrides=overrides)
    print(run.maze.summary())
    res = run.run(a.max_time)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
