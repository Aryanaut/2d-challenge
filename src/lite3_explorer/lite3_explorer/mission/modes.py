"""The preconfigured mission modes (no ROS). Configure them under modes.<name> (see config.py).

  idle       nothing moves. Leaves by itself once the robot is ready (auto_start), or on set_active / a mode command.
  scanning   builds the map around the robot: places the tile grid on the first scan, then turns in place
             (opening sweep -90..+90 deg the first time, a full turn later) so neighbouring tiles get classified.
  exploring  asks its exploration profiles (WHERE: tiles, optionally frontiers) for a reachable target and drives
             there with a traversal profile (HOW), then hands over to scanning. Checks the end condition at every
             decision.
  scene      a scene tile (scanned, mostly not traversable) shares a side with the current tile: face it, ask the AI
             to describe it, mark it explored and go back to exploring. The robot never enters it.
  curtain_scan  curtains: true and no reachable tile: turn to face each edge of the current tile with an unvisited tile
             behind it and look (the curtain detector), then exploring again (a curtain found -> curtain mode).
  stairs     a staircase tile (depth camera, stairs_core.py) shares a side with the current tile: square up to it, turn
             the robot's AI mode on, climb over it into the tile beyond, AI mode off, wall the staircase off, exploring.
  roller     a roller tile (depth camera + 3D lidar, roller_core.py) shares a side with the current tile: square up to
             it, h-step gait on, cross it into the tile beyond, terrain gait again, mark the roller tile explored.
  backtracking  no exploration target left but tiles are still unexplored: drive back to the last split tile (a tile
             that had two or more tiles to travel to), then on to its option not travelled to yet; then exploring again.
  recovery   the robot is stuck (or no target has a Nav2 path): drive to the nearest tile centre, then rescan.
  ending     the end condition ending_tile found the ending tile: drive to its centre, then complete.
  complete   the end condition is met. /exploration_complete is published.

Add a mode type with @register_mode('name') on a Mode subclass and list an instance in mission.modes.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Type

import numpy as np

from lite3_explorer import approach as A
from lite3_explorer import costmap_escape as CE
from lite3_explorer import curtain_core as CC
from lite3_explorer import roller_core as RC
from lite3_explorer import stairs_core as SC
from lite3_explorer import tiles as T
from lite3_explorer.mission.core import Mode
from lite3_explorer.mission.traversal import FAILED, SUCCEEDED, TraversalResult
from lite3_explorer.robot_state import BASIC_AI, BASIC_STANDING

MODE_TYPES: Dict[str, Type[Mode]] = {}


def register_mode(type_name: str):
    def deco(cls):
        cls.type_name = type_name
        MODE_TYPES[type_name] = cls
        return cls
    return deco


# --------------------------------------------------------------------------------------------- idle
@register_mode('idle')
class IdleMode(Mode):
    def __init__(self, name, machine, cfg):
        super().__init__(name, machine, cfg)
        self.auto_pending = bool(cfg.get('auto_start', True))

    def targets(self):
        return [self.machine.start_mode]

    def enter(self, reason, **kw):
        if reason != 'startup':
            self.auto_pending = False       # paused on purpose: only a command starts it again
        self.ctx.set_target(None, None, None)
        self.set_phase('waiting_start' if self.auto_pending else 'paused')

    def tick(self):
        if not self.auto_pending:
            return None
        ok, missing = self.ctx.ready_for_start()
        timed_out = self.in_phase_s() > float(self.cfg.get('start_wait_s', 60.0))
        if ok or timed_out:
            self.auto_pending = False
            if not ok:
                self.ctx.log('warn', f'Starting without: {", ".join(missing)} (steps that need them are skipped).')
            return self.go(self.machine.start_mode, 'auto start', opening=not self.ctx.tiles.ready)
        return None


# ----------------------------------------------------------------------------------------- scanning
@register_mode('scanning')
class ScanningMode(Mode):
    """Scan turn(s) in place. Phases: wait_map -> settle -> turning -> (next mode)."""

    def __init__(self, name, machine, cfg):
        super().__init__(name, machine, cfg)
        self.queue, self.opening, self.retries, self.pending_deg = [], False, 0, None
        self.start_yaw, self.returned, self.setpoints = None, False, False
        self.spun, self.early = False, False
        self.turned, self._last_yaw = 0.0, None
        self.back_tries = 0

    def targets(self):
        out = [self.cfg.get('next_mode', 'exploring')]
        if self.cfg.get('opening_stairs', False):
            out.append(self.cfg.get('stairs_mode', 'stairs'))
        return out

    def _stair_detection(self, on):
        f = getattr(self.ctx, 'stair_detection', None)
        if f is not None:
            f(on)

    def enter(self, reason, opening=False, **kw):
        self.opening = bool(opening) or not self.ctx.tiles.ready
        # tile the stairs / curtain mode put the robot on (odom rebased there): checked against the scan
        self.after_stairs = kw.get('after_stairs') if kw.get('after_stairs') is not None else kw.get('after_curtain')
        self.relocate_what = 'curtain' if kw.get('after_curtain') is not None else 'stairs'
        self.slam_waited = False
        if self.opening and self.cfg.get('opening_stairs', False):
            self._stair_detection(True)          # stair detector on for the opening scan (off again in exit)
        self.retries, self.queue, self.pending_deg = 0, [], None
        self.start_yaw, self.returned, self.setpoints = None, False, False
        self.spun, self.early = False, False
        self.turned, self._last_yaw = 0.0, None
        self.back_tries = 0
        self.ctx.set_target(None, None, None)
        if not self.ctx.tiles.ready:
            self.set_phase('wait_map')
        else:
            self._begin_turns()

    def _begin_turns(self):
        cfg = self.cfg
        if not cfg.get('turn', True) or not self.ctx.scanner.ready:
            if cfg.get('turn', True):
                self.ctx.log('warn', 'Nav2 spin not available: scanning without turning.')
            self.set_phase('done')
            return
        if self.opening:
            self.queue = [float(v) for v in cfg.get('opening_setpoints_deg', [0.0, 80.0, -80.0])]
            self.setpoints = True            # queue = yaw setpoints relative to the starting heading
            if not self.queue:
                self.set_phase('done')
                return
            if self.start_yaw is None:
                pose = self.ctx.pose()
                self.start_yaw = pose[2] if pose else None     # the heading to come back to after the sweep
            self.ctx.gait_hold('flat')
        else:
            deg = float(cfg.get('turn_deg', 360.0))
            if abs(deg) < 1.0:
                self.set_phase('done')
                return
            if cfg.get('check_room', True) and not self.ctx.scanner.has_room():
                self.ctx.log('info', 'No room to turn here: skipping the scan turn.')
                self.set_phase('done')
                return
            self.queue = [deg]
            if cfg.get('use_crawl', False):
                self.ctx.gait_hold('crawl')          # else: no hold, the gait the robot is already in stays
        self.set_phase('settle')

    def tick(self):
        cfg = self.cfg
        if self.phase == 'wait_map':
            pose = self.ctx.pose()
            if pose is not None and self.start_yaw is None:
                self.start_yaw = pose[2]                  # the ORIGINAL heading: the robot comes back to it after the spin
                self.ctx.initial_heading = pose[2]
                self.ctx.log('info', f'Original heading saved: {math.degrees(pose[2]):+.0f} deg.')
            if self.opening and not self.spun and pose is not None:
                self.spun = True
                self._begin_turns()                       # ALWAYS spin first; the tile grid is placed once it is over
                return None
            if pose is not None and self.ctx.local_map_ready(pose):
                self._establish(pose, 'wall-snapped')
            elif self.in_phase_s() > float(cfg.get('map_wait_s', 30.0)):
                self.ctx.log('warn', 'Local map still sparse; placing the tile grid with what is mapped.')
                self._establish(pose or [0.0, 0.0, 0.0], 'map sparse (timeout)')
            return None
        if self.phase == 'settle':
            if self.in_phase_s() >= float(cfg.get('settle_s', 1.0)):
                self._next_turn()
            return None
        if self.phase == 'turning':
            if self.opening and not self.returned and not self.early and not self.ctx.tiles.ready:
                pose = self.ctx.pose()
                if pose is not None:                      # how far the robot has turned so far (sum of |yaw change|)
                    if self._last_yaw is not None:
                        d = pose[2] - self._last_yaw
                        self.turned += abs(math.atan2(math.sin(d), math.cos(d)))
                    self._last_yaw = pose[2]
                if (pose is not None and math.degrees(self.turned) >= float(cfg.get('opening_min_turn_deg', 160.0))
                        and self.ctx.local_map_ready(pose)):
                    # the map is no longer sparse: the opening spin has done its job
                    self.early = True
                    self.ctx.log('info', 'Map no longer sparse: stopping the opening spin.')
                    self.ctx.scanner.cancel()
                    self.queue = []
                    self.set_phase('settle')              # lets the cancelled spin stop, then back to the original heading
                    return None
            status = self.ctx.scanner.poll()
            timed_out = self.in_phase_s() > float(cfg.get('time_allowance_s', 20.0)) + 5.0
            if status == SUCCEEDED and self.queue:
                self._next_turn()
                return None
            if status == SUCCEEDED and self.opening and not self.returned and self._turn_back():
                return None
            if status is None and not timed_out:
                return None
            if timed_out and status is None:
                self.ctx.scanner.cancel()
            if status != SUCCEEDED and self.opening and self.retries < int(cfg.get('opening_retries', 2)):
                # The opening scan must finish before navigating: run the whole sweep again (the setpoints are
                # relative to the starting heading, so it does not matter where the robot stopped).
                self.retries += 1
                self.ctx.log('warn', f'Opening spin did not finish ({status or "timeout"}); retry {self.retries}.')
                self.queue = [float(v) for v in cfg.get('opening_setpoints_deg', [0.0, 80.0, -80.0])]
                self.returned = False
                self.set_phase('settle')
                return None
            if status != SUCCEEDED and self.opening and self.back_tries < 2:
                # Sweep retries used up (e.g. Nav2 'Collision Ahead' in a tight start tile): still face the starting
                # heading before exploring. First the short way, then the long way round (the other side may be clear).
                self.back_tries += 1
                self.returned = False
                if self._turn_back(other_way=self.back_tries > 1):
                    return None
            if status != SUCCEEDED:
                self.ctx.log('warn', 'The scan turn did not finish; continuing with the map so far.')
            self.set_phase('done')
        if self.phase == 'done' and self.opening and not self.ctx.tiles.ready:
            self.ctx.gait_hold(None)
            self.set_phase('wait_map')                    # spin over and heading restored: now place the tile grid
            return None
        if (self.phase == 'done' and self.opening and self.cfg.get('opening_stairs', False)
                and float(cfg.get('opening_stairs_wait_s', 2.0)) > 0.0):
            self.set_phase('stairs_wait')                 # stand still: the stair detector confirms what it sees now
            return None
        if self.phase == 'stairs_wait':
            if self.in_phase_s() < float(cfg.get('opening_stairs_wait_s', 2.0)):
                return None
            self.set_phase('done')
        if (self.phase == 'done' and getattr(self, 'after_stairs', None) is not None and not self.slam_waited
                and float(cfg.get('relocate_wait_s', 3.0)) > 0.0):
            # After the stairs the pose is the odom rebase until slam_toolbox has matched a scan taken AFTER it: the scan
            # reaches SLAM ~1.1 s late, then has to be matched. Bag fullrun_25_11_183742: the scan turn was skipped (no
            # room) and the check ran 0.17 s after the rebase, so it only read the rebase back ('as assumed').
            self.slam_waited = True
            self.set_phase('slam_wait')
            return None
        if self.phase == 'slam_wait':
            if self.in_phase_s() < float(cfg.get('relocate_wait_s', 3.0)):
                return None
            self.set_phase('done')
        if self.phase == 'done':
            self.ctx.gait_hold(None)
            self._relocate_after_stairs()
            self.ctx.opening_done = True      # virtual walls and tile calculation start only now
            if self.ctx.tiles.current is not None:
                self.machine.scanned_tile = tuple(self.ctx.tiles.current)
            self.ctx.tiles.invalidate()
            self.ctx.tiles.update()
            hop = self._opening_stairs() if self.opening else None
            return hop or self.go(cfg.get('next_mode', 'exploring'), 'scan finished')
        return None

    def _relocate_after_stairs(self):
        """After a stairs climb (or a curtain crossing) the tile the robot was put on came from dead reckoning (odom
        rebased there). The scan has let SLAM place the robot in the map: if its pose is in another tile, that tile is
        where it is."""
        assumed, self.after_stairs = getattr(self, 'after_stairs', None), None
        what = getattr(self, 'relocate_what', 'stairs')
        tiles = self.ctx.tiles
        pose = self.ctx.pose()
        if assumed is None or pose is None or tiles.grid is None:
            return
        actual = tiles.relocate(assumed, pose, f'{what}: placed by the scan')
        if actual is None:
            return
        gx, gy = tiles.grid.center(tuple(assumed))
        self.ctx.event(f'{what}_relocated', assumed=T.tile_label(tuple(assumed)), actual=T.tile_label(actual),
                       pose=[round(float(v), 3) for v in pose[:3]],
                       offset_m=round(math.hypot(pose[0] - gx, pose[1] - gy), 3))
        after = 'the stairs' if what == 'stairs' else 'the curtain'
        if tuple(actual) == tuple(assumed):
            self.ctx.log('info', f'Scan after {after}: SLAM puts the robot on tile {T.tile_label(actual)}, as assumed.')
        else:
            self.ctx.log('warn', f'Scan after {after}: SLAM puts the robot on tile {T.tile_label(actual)}, not '
                                 f'{T.tile_label(tuple(assumed))} (dead reckoning was off); the tile is corrected.')

    def _opening_stairs(self):
        """opening_stairs: the stair detector ran during the opening scan. A staircase tile sharing a side with the start
        tile (steps going up away from it, stairs_core.stair_crossing) -> straight into the stairs mode. A staircase seen
        further away is only logged (the stairs mode starts from the tile in front of it; exploring climbs it when it gets
        there with use_stairs)."""
        cfg, tiles = self.cfg, self.ctx.tiles
        mode = cfg.get('stairs_mode', 'stairs')
        if not cfg.get('opening_stairs', False) or mode not in self.machine.modes:
            return None
        if not tiles.stairs or tiles.grid is None or tiles.current is None:
            self.ctx.log('info', 'Opening scan: no staircase seen.')
            return None
        crossing = SC.stair_crossing(tiles.grid, tuple(tiles.current), tiles.stairs,
                                     skip=self.machine.stairs_tried | tiles.stairs_crossed,
                                     min_facing=float(cfg.get('stair_min_facing', 0.5)))
        seen = ', '.join(T.tile_label(t) for t in sorted(tiles.stairs))
        if crossing is not None and not tiles.stairs_ready():
            self.ctx.log('info', f'Opening scan: staircase on tile {T.tile_label(crossing["neighbour"])} beside the start '
                                 f'tile, but this side is explored first (the stairs are one way).')
            return None
        if crossing is None:
            self.ctx.log('info', f'Opening scan: staircase tile(s) {seen}, but none beside start tile '
                                 f'{T.tile_label(tiles.current)} to climb from here.')
            return None
        here, first = T.tile_label(tiles.current), T.tile_label(crossing['neighbour'])
        self.ctx.log('info', f'Opening scan: staircase on tile {first}, along the {crossing["side"]} side of tile {here}: '
                             f'stairs mode, climbing over it into tile {T.tile_label(crossing["beyond"])}.')
        return self.go(mode, f'staircase on tile {first} seen in the opening scan', crossing=crossing)

    def _turn_back(self, other_way=False):
        """Opening scan: turn back to the heading the robot started with before exploring. True if a turn started.
        other_way: go the long way round (the short way was blocked)."""
        self.returned = True
        pose = self.ctx.pose()
        if self.start_yaw is None or pose is None:
            return False
        delta = math.degrees(math.atan2(math.sin(self.start_yaw - pose[2]), math.cos(self.start_yaw - pose[2])))
        if abs(delta) < float(self.cfg.get('return_tolerance_deg', 3.0)):
            return False
        if other_way:
            delta -= math.copysign(360.0, delta)
        self.ctx.log('info', f'Opening scan done: turning {delta:+.0f} deg back to the starting heading.')
        self.queue = []
        self.pending_deg = None
        self.ctx.scanner.start(delta, float(self.cfg.get('time_allowance_s', 20.0)))
        self.set_phase('turning')
        return True

    def _establish(self, pose, how):
        if self.start_yaw is not None:
            pose = [pose[0], pose[1], self.start_yaw]     # the wall 'behind the robot' is behind the ORIGINAL heading
        start = self.ctx.tiles.establish(pose, how)
        grid = self.ctx.tiles.grid
        self.ctx.log('info', f'Tile grid placed ({how}): yaw {math.degrees(grid.yaw):.1f} deg, start tile '
                             f'{start[0]},{start[1]}.')
        self.ctx.event('tile_grid', how=how, grid=grid.as_dict(), original_heading_deg=
                       None if self.start_yaw is None else round(math.degrees(self.start_yaw), 1))
        self.set_phase('done')

    def _next_turn(self):
        tol = float(self.cfg.get('return_tolerance_deg', 3.0))
        while self.queue:
            deg = self.queue.pop(0)
            if self.opening and self.setpoints and self.start_yaw is not None:
                pose = self.ctx.pose()
                if pose is None:
                    continue
                # yaw setpoint relative to the starting heading -> the turn from where the robot is now. Both are
                # measured in (-180, 180] from the starting heading and the turn is their plain difference (NOT the
                # shortest way): it never crosses the +-180 deg point, so the sweep stays in front of the robot instead
                # of swinging round through the wall behind it (shortest way from +80 to -160 is +120, through the back).
                rel = math.degrees(math.atan2(math.sin(pose[2] - self.start_yaw), math.cos(pose[2] - self.start_yaw)))
                deg = max(-180.0, min(180.0, deg)) - rel
                if abs(deg) < tol:
                    continue
            self.ctx.scanner.start(deg, float(self.cfg.get('time_allowance_s', 20.0)))
            self.set_phase('turning')
            return
        # nothing (left) to turn: back to the starting heading, then on
        if not (self.opening and not self.returned and self._turn_back()):
            self.set_phase('done')

    def exit(self, next_mode, reason):
        if self.phase in ('settle', 'turning'):
            self.ctx.scanner.cancel()
        self.ctx.gait_hold(None)
        if self.opening and self.cfg.get('opening_stairs', False):
            self._stair_detection(False)         # stays on with use_stairs (mission manager)


# ---------------------------------------------------------------------------------------- exploring
@register_mode('exploring')
class ExploringMode(Mode):
    """Strategy-driven exploration. Phases: decide -> checking_paths -> driving -> (scanning | decide).

    WHERE to go comes from the exploration profiles in cfg['exploration'] (mission/exploration.py), tried in order
    until one yields a reachable target; HOW to drive comes from the traversal profiles (mission/traversal.py).
    """

    def __init__(self, name, machine, cfg):
        super().__init__(name, machine, cfg)
        self.exploration_profiles = [p for p in (cfg.get('exploration') or ['tiles']) if p]
        self.target: Optional[dict] = None
        self.strategy = None
        self.traversal = None
        self.look = None              # terrain look at the target tile: None | {'since': t} (asked) | 'done'
        self.unseen = set()           # tiles whose floor could not be seen when asked (asked once, then driven to anyway)
        self.pending = None           # (strategy index, candidates) while the Nav2 path checks run
        self.index = 0                # which strategy is being asked in this decision
        self.unreached = []           # labels of candidates that had no path in this decision
        self.via_only = []            # labels Nav2 had a path to, but only through other tiles (this decision)
        self.curtain = None           # curtain edge seen this decision (curtain_last: another way is taken first)
        self.avoid = set()            # goals that just failed: the next decision picks a different tile
        self.no_path_streak = 0       # decisions in a row without any Nav2 path (reset when a target is chosen)
        self.stuck_rounds = 0         # retry rounds with nothing reachable left (reset when a new tile is visited)
        self.stairs_backtracks = 0    # backtracks started towards the tile in front of a forced staircase
        self._stuck_visited = None    # tiles.visited_count() when stuck_rounds was last counted
        self.nearest_tried = set()    # (tile, unvisited tile) splits already made by _split_at_nearest_unvisited
        self._nearest_visited = None  # tiles.visited_count() when nearest_tried was last reset
        self.valid_checked_at = 0.0
        self.tile_traversals = self._overrides('tile_traversals', lambda k: k.replace(' ', ''))
        self.kind_traversals = self._overrides('kind_traversals', str.strip)

    def _overrides(self, key, norm):
        out = {}
        for item in self.cfg.get(key, []) or []:
            if item and ':' in item:
                k, profile = item.rsplit(':', 1)
                out[norm(k.strip())] = profile.strip()
        return out

    def targets(self):
        c = self.cfg
        out = [c.get('scan_mode', 'scanning'), self.machine.complete_mode]
        if c.get('on_no_reachable_tile', 'recovery') == 'recovery' or c.get('on_goal_failed') == 'recovery':
            out.append(self.machine.recovery_mode)
        if c.get('quickscan_on_edge', False):
            out.append(c.get('quickscan_mode', 'quickscan'))
        if c.get('scenes', False) and c.get('scene_mode', 'scene') in self.machine.modes:
            out.append(c.get('scene_mode', 'scene'))
        if c.get('curtains', False):
            out.append(c.get('curtain_mode', 'curtain'))
            if c.get('curtain_scan_when_stuck', True) and c.get('curtain_scan_mode', 'curtain_scan') in self.machine.modes:
                out.append(c.get('curtain_scan_mode', 'curtain_scan'))   # optional: unused when not in mission.modes
        if c.get('stairs', False):
            out.append(c.get('stairs_mode', 'stairs'))
        if c.get('rollers', False):
            out.append(c.get('roller_mode', 'roller'))
        if c.get('backtracking', False) and c.get('backtracking_mode', 'backtracking') in self.machine.modes:
            out.append(c.get('backtracking_mode', 'backtracking'))     # optional: unused when not in mission.modes
        return out

    @property
    def is_driving(self):
        return self.phase == 'driving'

    @property
    def target_label(self):
        """The TILE the target is in (frontier targets too), for the recovery mode's exclusions."""
        return self.target.get('tile') if self.target else None

    def enter(self, reason, **kw):
        self.target, self.strategy, self.traversal, self.pending = None, None, None, None
        self.avoid = set()
        self._stairs_wait_logged = None
        self._new_decision()

    def _new_decision(self):
        self.index, self.unreached, self.via_only = 0, [], []
        self.curtain = None                         # a curtain seen this decision, left for later if another way exists
        self.set_phase('decide')

    def tick(self):
        if self.phase == 'decide':
            return self._decide()
        if self.phase == 'checking_paths':
            return self._poll_paths()
        if self.phase == 'driving':
            return self._poll_drive()
        if self.phase in ('wait_traversal', 'wait_planner') and self.in_phase_s() >= 2.0:
            self._new_decision()
        if self.phase == 'retry_wait' and self.in_phase_s() >= self._retry_period():
            self._retry_tiles()
            if self.cfg.get('retry_rescan', True):
                return self.go(self.cfg.get('scan_mode', 'scanning'), 'rescanning before retrying the unexplored tiles')
            self._new_decision()
        return None

    # -- decide: scene / stairs beside the current tile, end condition, then ask the strategies in order
    def _decide(self):
        pose = self.ctx.pose()
        if pose is None:
            return None
        edge = self._curtain_on_edge(pose) if self.index == 0 else None
        if self.index == 0:
            self._note_discovered_splits()
            self._note_late_curtains()
            # scenes first: the scene mode keeps the robot on this tile, but a stairs crossing is one way (the staircase
            # is walled off behind it), so a scene beside this tile could never be looked at afterwards.
            # Both before the end condition: the lidar cannot see past a staircase's riser, so the tiles behind it are
            # not identified yet and 'all identified tiles explored' can be true while a staircase is still to climb.
            # A roller is the same: one way for Nav2 (no route back over it), and the map shows a wall across the half
            # tile, so nothing behind it is identified until it is crossed.
            # the end condition found the ending tile (the robot entered the outside of the maze): straight back to it,
            # no scene / roller / stairs out there first
            hop = self.machine.goal_hop() or self._scene_beside(pose) or self._roller_ahead(pose) \
                or self._stairs_beside(pose)
            if hop is None and self.ctx.tiles.stairs_forced:
                # this side was given up on for the staircase: backtrack to the tile in front of it (no Nav2 path checks
                # to the tiles left, so no recovery); arriving there starts the stairs mode
                hop = self.stairs_backtrack('going for the staircase')
            if hop is not None:
                return hop
            # a curtain on an edge of the tile the robot stands on (e.g. the opening tile) hides what is behind it: the map
            # shows a wall there, so the end condition must not see the mission as finished before the curtain is checked
            end = None if edge is not None else self.machine.end_hop()
            if end is not None:
                return end
            if edge is not None:
                self.ctx.log('info', f'Curtain on the {edge["side"]} edge of the current tile: end condition not checked yet.')
                if not self._can_defer_curtain():
                    return self._through_curtain(edge)
                self.curtain = edge                  # look for another way first (_choose)
        while self.index < len(self.exploration_profiles):
            strat = self.machine.exploration(self.exploration_profiles[self.index])
            strat.avoid = frozenset(self.avoid)      # a goal that just failed: pick another tile when there is one
            cands = strat.candidates(pose)
            if not cands:
                self.index += 1
                continue
            if not strat.check_paths:
                return self._choose(strat, strat.select([(c, [tuple(pose[:2]), tuple(c['goal_xy'])]) for c in cands],
                                                        pose))
            if not self.ctx.paths.start([c['goal_xy'] for c in cands]):
                self.ctx.log('warn', 'Nav2 planner (compute_path_to_pose) not available: waiting.')
                self.set_phase('wait_planner')
                return None
            self.pending = (self.index, cands)
            self.set_phase('checking_paths')
            return None
        # every strategy asked: nothing reachable this decision
        if self.curtain is not None:
            return self._through_curtain(self.curtain, 'no other way to take first')
        if self.via_only:
            # Nav2 HAS a path, it just runs through other tiles: not a transient planner failure, so no retries and no
            # recovery (that only re-centres on this tile). Go back to the last split and take its untravelled way.
            hop = self._backtrack(f'only paths through other tiles to {", ".join(self.via_only)}')
            if hop is not None:
                return hop
            self.unreached += [v for v in self.via_only if v not in self.unreached]   # no split: as before
        # no reachable tile: a curtain is a wall to the 2D lidar, so look at this tile's edges for one first
        hop = self._curtain_scan('no reachable tile')
        if hop is not None:
            return hop
        if self.unreached:
            self.ctx.log('warn', f'No Nav2 path to any candidate ({", ".join(self.unreached)}).')
            self.ctx.event('no_path', targets=self.unreached)
            return self._on_no_path([u for u in self.unreached if not u.startswith('frontier')])
        return self._nothing_left('no exploration target left')

    def _nothing_left(self, why):
        pose = self.ctx.pose()
        end = None if pose is not None and self._curtain_on_edge(pose) is not None else self.machine.end_hop()
        if end is not None:
            return end
        tiles = self.ctx.tiles
        if tiles.ready and (tiles.unexplored() or tiles.unreachable or tiles.curtains_left() or self.machine.stairs_left()):
            hop = self._backtrack(why)
            if hop is not None:
                return hop
            if self._force_stairs(why):
                self._new_decision()
                return None
            # unexplored tiles are left: never finish, look for new Nav2 paths to them later
            self.ctx.log('warn', f'{why}, but {len(tiles.unexplored()) + len(tiles.unreachable)} tile(s) are still '
                                 f'unexplored: retrying for Nav2 paths in {self._retry_period():.0f} s.')
            self.set_phase('retry_wait')
            return None
        idle = self.machine._first_of_type('idle')
        self.ctx.log('warn', f'{why}, but the end condition is not met ({self.machine.end.describe()}): idling.')
        return self.go(idle, f'{why}; end condition not met') if idle else None

    def _force_stairs(self, why):
        """Nothing reachable is left on this side but the staircase is still to climb: after stairs_after_retries retry
        rounds without a new tile, stop waiting for the tiles left (they hold the stairs back via tiles.stairs_ready,
        and each retry restores the given-up ones) and go for the stairs: the exploring decision then targets the tile
        in front of the staircase (TileExploration._stairs_approach) and the stairs mode climbs from there."""
        tiles = self.ctx.tiles
        if (not self.cfg.get('stairs', False) or not self.cfg.get('stairs_when_stuck', True) or tiles.stairs_forced
                or not self.machine.stairs_left()):
            return False
        visited = tiles.visited_count()
        if visited != self._stuck_visited:
            self._stuck_visited, self.stuck_rounds = visited, 0
        rounds = int(self.cfg.get('stairs_after_retries', 1))
        if self.stuck_rounds < rounds:
            self.stuck_rounds += 1
            return False
        left = sorted(set(tiles.unexplored()) | set(tiles.unreachable))
        tiles.stairs_forced = True
        for label in tiles.unexplored():               # given up on: no longer candidates in front of the stairs
            tiles.unreachable[label] = 'given up for the staircase'
        tiles.invalidate()
        tiles.update()
        stair = T.tile_label(tiles.staircase_tile)
        self.ctx.log('warn', f'{why} after {self.stuck_rounds} retry round(s): giving up on {", ".join(left) or "nothing"} '
                             f'and going for the staircase on tile {stair}.')
        self.ctx.event('stairs_forced', stairs=stair, given_up=left, why=why)
        return True

    def stairs_backtrack(self, why):
        """stairs_forced: make the tile in front of the staircase the last split (its option: the staircase) and
        backtrack there along the moves already made (tiles.known_route, tile centre to tile centre); at the split
        tile the backtracking mode starts the stairs mode. Used instead of recovery while the stairs are forced (also by
        the machine when the robot gets stuck on the way). None when there is nothing to do or stairs_backtrack_attempts
        backtracks were started already (then the usual handling, recovery included, takes over)."""
        tiles = self.ctx.tiles
        mode = self.cfg.get('backtracking_mode', 'backtracking')
        if (not tiles.stairs_forced or not self.machine.stairs_left() or mode not in self.machine.modes
                or tiles.grid is None or tiles.current is None):
            return None
        limit = int(self.cfg.get('stairs_backtrack_attempts', 3))
        if limit and self.stairs_backtracks >= limit:
            if self.stairs_backtracks == limit:
                self.stairs_backtracks += 1          # log once
                self.ctx.log('warn', f'Stairs: {limit} backtracks to the staircase did not get there; '
                                     f'back to the usual handling.')
            return None
        approach = self._stairs_approach_tile()
        if approach is None:
            return None
        self.stairs_backtracks += 1
        stair, here = T.tile_label(tiles.staircase_tile), T.tile_label(approach)
        tiles.mark_split(approach, [stair])
        self.ctx.log('info', f'Stairs: {why}; tile {here} in front of the staircase {stair} is the last split: '
                             f'backtracking there (attempt {self.stairs_backtracks}).')
        self.ctx.event('stairs_backtrack', split=here, stairs=stair, why=why, attempt=self.stairs_backtracks)
        return self.go(mode, f'{why}: backtracking to tile {here} in front of the staircase {stair}', split=here)

    def _stairs_approach_tile(self):
        """The tile the staircase can be climbed from (stairs_core.stair_crossing), the one with the shortest way back
        along the moves already made first (then the nearest), or None."""
        tiles = self.ctx.tiles
        stair = tuple(tiles.staircase_tile)
        here = tuple(tiles.current)
        best = None
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (stair[0] + di, stair[1] + dj)
            if tiles.tiles.get(v, {}).get('status') not in ('visited', 'to_visit', 'unreachable') or tiles.is_closed(v):
                continue
            if SC.stair_crossing(tiles.grid, v, tiles.stairs, skip=self.machine.stairs_tried | tiles.stairs_crossed,
                                 min_facing=float(self.cfg.get('stair_min_facing', 0.5))) is None:
                continue
            route = tiles.known_route(here, v)
            key = (route is None, len(route) if route else 0, math.dist(tiles.grid.center(v), tiles.grid.center(here)))
            if best is None or key < best[0]:
                best = (key, v)
        return best[1] if best else None

    def _retry_period(self):
        return float(self.cfg.get('retry_period_s', 15.0))

    def _retry_tiles(self):
        """Give every tile that was eliminated / given up on another chance: the map, the virtual walls and the costmap
        have changed since, so Nav2 may find a path now."""
        tiles = self.ctx.tiles
        back = sorted(tiles.unreachable)
        tiles.unreachable.clear()
        tiles.failures.clear()
        tiles.invalidate()
        tiles.update()
        for strat in self.machine._explorations.values():
            getattr(strat, 'strikes', {}).clear()
        self.machine.recoveries = 0
        self.avoid.clear()
        self.nearest_tried.clear()
        if self.cfg.get('retry_clear_costmaps', True):
            self.ctx.clear_costmaps()
        self.ctx.log('info', f'Retrying for Nav2 paths: {len(tiles.unexplored())} tile(s) to visit'
                             + (f' (restored {", ".join(back)})' if back else '') + '.')
        self.ctx.event('retry', restored=back, to_visit=tiles.unexplored())

    # -- Nav2 reachability
    def _poll_paths(self):
        results = self.ctx.paths.poll()
        timed_out = self.in_phase_s() > float(self.cfg.get('path_check_timeout_s', 10.0))
        if results is None and not timed_out:
            return None
        if results is None:
            self.ctx.paths.cancel()
            results = []
        index, cands = self.pending
        self.pending = None
        strat = self.machine.exploration(self.exploration_profiles[index])
        checked = [(c, results[i] if i < len(results) else None) for i, c in enumerate(cands)]
        reachable = strat.select(checked, self.ctx.pose() or [0.0, 0.0, 0.0])
        if reachable:
            return self._choose(strat, reachable)
        via = set(getattr(strat, 'via_rejected', ()) or ())
        self.via_only += [c['label'] for c, _ in checked if c['label'] in via]
        self.unreached += [c['label'] for c, _ in checked if c['label'] not in via]
        self.ctx.log('info', f'Exploration {strat.name!r}: no reachable candidate; trying the next profile.')
        self.index = index + 1
        self.set_phase('decide')
        return None

    def _on_no_path(self, labels):
        action = self.cfg.get('on_no_reachable_tile', 'recovery')
        self.no_path_streak += 1
        patience = int(self.cfg.get('no_path_retries', 2))
        if action == 'recovery' and self.no_path_streak <= patience and self.ctx.tiles.ready:
            # a missing path is often transient (virtual walls / costmap not updated yet): wait, rescan, retry first
            self.ctx.log('info', f'No Nav2 path ({self.no_path_streak}/{patience}): retrying before recovery.')
            self.set_phase('retry_wait')
            return None
        if self.ctx.tiles.stairs_forced:
            hop = self.stairs_backtrack('no Nav2 path to any exploration target')   # never recovery for the stairs
            if hop is not None:
                return hop
        if action == 'recovery' and self.ctx.tiles.ready and self.cfg.get('backtracking', False):
            # a split with an untravelled way is left (or the robot is boxed in behind a roller): go back there rather
            # than recover, which only re-centres on a nearby tile and gives up on the targets after a few rounds
            hop = self._backtrack('no Nav2 path to any exploration target')
            if hop is not None:
                return hop
        if action == 'recovery':
            return self.go(self.machine.recovery_mode, 'no Nav2 path to any exploration target', blocked=labels)
        if action == 'scanning':
            return self.go(self.cfg.get('scan_mode', 'scanning'), 'no Nav2 path to any exploration target')
        return self._nothing_left('no Nav2 path to any exploration target')

    def _choose(self, strat, reachable):
        if not reachable:
            self.index += 1
            self.set_phase('decide')
            return None
        if self._chosen_scene_props(reachable):
            self._new_decision()                    # again: the scene mode may look at it now, or another tile is chosen
            return None
        if self.curtain is not None:
            return self._defer_curtain(strat, reachable)
        self._note_split(reachable)
        return self._drive_to(strat, self.choose_target(reachable))

    def _chosen_scene_props(self, reachable) -> bool:
        """tiles.chosen_prop_check: run the tile that would be driven to (choose_target) against the perception
        checker (scene-prop sightings placed on the map, prop_votes.py). Anything scene-related seen on it -> it is a scene
        tile (looked at from beside, never entered) and the robot does not drive there. True = a tile was marked."""
        tcfg = self.ctx.cfg.get('tiles', {})
        tiles = self.ctx.tiles
        if not tcfg.get('chosen_prop_check', True) or not tiles.ready or not hasattr(tiles, 'check_chosen_props'):
            return False
        target = self.choose_target(reachable)
        if target.get('kind', 'tile') != 'tile' or not target.get('tile'):
            return False
        tile = tuple(int(v) for v in target['tile'].split(','))
        e = tiles.check_chosen_props(tile, int(tcfg.get('chosen_prop_min_frames', 1)),
                                     float(tcfg.get('chosen_prop_min_score', 0.0)))
        if e is None:
            return False
        self.ctx.log('info', f'Chosen tile {target["tile"]}: scene props seen on it ({e["frames"]} sighting(s), '
                             f'{e["labels"]}): a scene tile, not driven into.')
        self.ctx.event('chosen_prop_scene', tile=target['tile'], frames=e['frames'], score=e['score'], labels=e['labels'])
        return True

    def _adjacent(self, label):
        ci, cj = self.ctx.tiles.current
        return sum(abs(a - b) for a, b in zip(map(int, label.split(',')), (ci, cj))) == 1

    def _defer_curtain(self, strat, reachable):
        """A curtain on an edge of the current tile and another reachable tile sharing a side with it: take that tile
        first and make the current tile the last split (the curtain's tile one of its options), so backtracking comes
        back here and walks through the curtain once the other way is explored."""
        edge, self.curtain = self.curtain, None
        tiles = self.ctx.tiles
        behind = T.tile_label(edge['neighbour'])
        others = [t for t in reachable if t.get('kind') == 'tile' and t.get('tile') and t['tile'] != behind
                  and self._adjacent(t['tile'])]
        if not others:
            return self._through_curtain(edge, 'no other reachable tile beside it')
        here = T.tile_label(tiles.current)
        options = sorted({t['tile'] for t in others} | {behind})
        tiles.defer_curtain(tiles.current, edge['neighbour'], edge['side'])
        tiles.mark_split(tiles.current, options)
        target = self.choose_target(others)
        self.ctx.log('info', f'Curtain along the {edge["side"]} edge of tile {here} (tile {behind} behind it): '
                             f'exploring tile {target["tile"]} first; tile {here} is the last split '
                             f'({", ".join(options)}), backtracking walks through the curtain later.')
        self.ctx.event('curtain_deferred', tile=here, side=edge['side'], neighbour=behind, first=target['tile'])
        self.ctx.event('split', tile=here, options=options)
        return self._drive_to(strat, target)

    # -- split tiles and backtracking
    def _note_split(self, reachable):
        """Two or more reachable tiles sharing a side with the current tile: the current tile is the last split."""
        tiles = self.ctx.tiles
        if not self.cfg.get('backtracking', False) or not tiles.ready or tiles.current is None:
            return
        ci, cj = tiles.current
        options = sorted({t['tile'] for t in reachable if t.get('kind') == 'tile' and t.get('tile')
                          and sum(abs(a - b) for a, b in zip(map(int, t['tile'].split(',')), (ci, cj))) == 1})
        if len(options) < 2:
            return
        if tiles.mark_split(tiles.current, options):
            label = T.tile_label(tiles.current)
            self.ctx.log('info', f'Tile {label} is the last split: {len(options)} tiles to travel to '
                                 f'({", ".join(options)}).')
            self.ctx.event('split', tile=label, options=options)

    def _note_discovered_splits(self):
        """New untravelled tiles found beside tiles visited earlier: those tiles become splits (backtracking returns to
        them), not only the tile the robot stands on."""
        tiles = self.ctx.tiles
        if not (self.cfg.get('backtracking', False) and self.cfg.get('split_on_discovery', True)) or not tiles.ready:
            return
        for s in tiles.note_discovered_splits():
            self.ctx.log('info', f'Tile {s["tile"]} is a split again: new tile(s) {", ".join(s["new"])} beside it '
                                 f'({len(s["options"])} to travel to: {", ".join(s["options"])}).')
            self.ctx.event('split', tile=s['tile'], options=s['options'], discovered=s['new'])

    def _note_late_curtains(self):
        """curtains: true. A curtain confirmed only after the robot left its tile (the camera saw it while driving on: sim
        bag run_20261126_133942, curtain on the N edge of 3,-2 confirmed on the way to 4,-2) is on an edge of a visited
        tile, not the current one, so _curtain_on_edge never sees it and the robot idled with the tile behind it
        unexplored. Such a curtain is left for later from that tile (tiles.defer_curtain) and the tile becomes a split
        with the tile behind as an option: backtracking comes back and walks through it once nothing else is left."""
        tiles = self.ctx.tiles
        if (not self.cfg.get('curtains', False) or not self.cfg.get('backtracking', False)
                or not hasattr(self.ctx, 'curtain_points') or tiles.grid is None or not tiles.ready):
            return
        pts = self.ctx.curtain_points()
        if pts is None or len(pts) == 0:
            return
        pts = np.asarray(pts, float)
        pts = pts[tiles.not_curtain_mask(pts)]
        if len(pts) == 0:
            return
        here = tuple(tiles.current) if tiles.current is not None else None
        for tile in dict.fromkeys(reversed([tuple(t) for t in tiles.trail])):     # most recently visited first
            if tile == here or tiles.is_closed(tile):
                continue
            edge = self._edge_curtain(tile, pts)
            if edge is None:
                continue
            behind = tuple(edge['neighbour'])
            if behind in tiles.deferred_curtains or tiles.is_closed(behind):
                continue
            label = T.tile_label(behind)
            old = next((s for s in tiles.splits if s['tile'] == tile), None)
            tiles.defer_curtain(tile, behind, edge['side'])
            options = sorted(set(tiles.split_remaining(old, include_unreachable=True) if old else []) | {label})
            tiles.mark_split(tile, options)
            self.ctx.log('info', f'Curtain along the {edge["side"]} edge of tile {T.tile_label(tile)} (tile {label} '
                                 f'behind it), seen after leaving that tile: backtracking walks through it later.')
            self.ctx.event('curtain_deferred', tile=T.tile_label(tile), side=edge['side'], neighbour=label, late=True)
            self.ctx.event('split', tile=T.tile_label(tile), options=options)

    def _backtrack(self, why):
        """No exploration target left but tiles are still unexplored -> backtracking mode, back to the last split tile
        (instead of waiting retry_period_s for new Nav2 paths). Options given up on (unreachable) still count."""
        tiles = self.ctx.tiles
        mode = self.cfg.get('backtracking_mode', 'backtracking')
        if not self.cfg.get('backtracking', False) or mode not in self.machine.modes or tiles.current is None:
            return None
        split = tiles.last_split(exclude=tiles.current, include_unreachable=True)
        if split is None:
            split = self._split_at_roller()
        if split is None:
            split = self._split_at_nearest_unvisited()
        if split is None:
            return None
        label = T.tile_label(split['tile'])
        left = len(tiles.unexplored()) + len(tiles.unreachable)
        self.ctx.log('info', f'{why}, but {left} tile(s) are still unexplored: backtracking to the last split tile '
                             f'{label} (untravelled: {", ".join(tiles.split_remaining(split, include_unreachable=True))}).')
        return self.go(mode, f'{why}; {left} tile(s) still unexplored', split=label)

    def _split_at_nearest_unvisited(self):
        """No split is left (every split's options were travelled to / dropped), yet tiles are still unvisited: the
        nearest reachable unvisited tile becomes the last split's option. Its split tile is a visited tile sharing an
        open side with it (not the current tile: exploring already had its chance from here) that the robot can get
        back to along moves already made (tiles.known_route); the shortest way back wins (to_visit before given-up
        tiles, then straight-line distance). Each (tile, unvisited tile) pair is tried once until a new tile is
        visited or the retry round restores the given-up tiles, so a failed backtrack never loops."""
        tiles = self.ctx.tiles
        if not self.cfg.get('backtrack_to_nearest_unvisited', True) or tiles.current is None or tiles.grid is None:
            return None
        visited = tiles.visited_count()
        if visited != self._nearest_visited:
            self._nearest_visited, self.nearest_tried = visited, set()
        here = tuple(tiles.current)
        trail = set(tiles.trail)
        best = None
        for tile, d in tiles.tiles.items():
            status = d.get('status')
            if status not in ('to_visit', 'unreachable') or not tiles._drivable(T.tile_label(tile)):
                continue
            for side in T.SIDES:
                prev = T._step(tile, side)
                if prev == here or prev not in trail or tiles.is_closed(prev) or (prev, tile) in self.nearest_tried \
                        or not T._joined(tiles.tiles, tile, side):
                    continue
                route = tiles.known_route(here, prev)
                if route is None:
                    continue
                key = (len(route), status == 'unreachable',
                       math.dist(tiles.grid.center(tile), tiles.grid.center(here)))
                if best is None or key < best[0]:
                    best = (key, prev, tile)
        if best is None:
            return None
        _, prev, tile = best
        self.nearest_tried.add((prev, tile))
        option = T.tile_label(tile)
        tiles.mark_split(prev, [option])
        self.ctx.log('info', f'No split left: tile {T.tile_label(prev)} beside the nearest reachable unvisited tile '
                             f'{option} ({best[0][0] - 1} tile(s) back) is the last split.')
        self.ctx.event('split', tile=T.tile_label(prev), options=[option], nearest_unvisited=True)
        return next((x for x in tiles.splits if x['tile'] == prev), None)

    def _split_at_roller(self):
        """The robot crossed a roller and the tiles still to visit lie on its other side (Nav2 sees the roller as a wall, so
        it is boxed in: no path, recovery cannot help). _note_roller_split only registers a split when the untravelled
        neighbours were already known at the crossing; here they were found afterwards. Make a tile beside a crossed
        roller (other than the current one) a split with its untravelled neighbours as options, so backtracking crosses
        back over the roller (to_roller) to reach them."""
        tiles = self.ctx.tiles
        if tiles.current is None or not getattr(tiles, 'roller_crossings', None):
            return None
        here = tuple(tiles.current)
        for (a, b), roller in reversed(list(tiles.roller_crossings.items())):
            for tile in (a, b):
                if tile == here:
                    continue
                ci, cj = tile
                options = [T.tile_label(n) for n in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1))
                           if n != tuple(roller) and n != here
                           and tiles.tiles.get(n, {}).get('status') in ('to_visit', 'unreachable')
                           and tiles._drivable(T.tile_label(n))]
                if not options:
                    continue
                tiles.mark_split(tile, options)
                self.ctx.log('info', f'Tile {T.tile_label(tile)} is a split behind the roller on tile {T.tile_label(roller)}: '
                                     f'untravelled {", ".join(options)}.')
                self.ctx.event('split', tile=T.tile_label(tile), options=sorted(options), via_roller=T.tile_label(roller))
                return next((x for x in tiles.splits if x['tile'] == tile), None)
        return None

    def choose_target(self, reachable):
        """Hook for other selection strategies (e.g. an AI choice): default is the strategy's top-ranked target.

        With the classifier's elevation (ground plane) option on and prefer_flat_tiles set, the tile whose elevation
        differs least from the current tile's (up or down) wins; ties (within flat_tile_tolerance_m) keep the strategy's
        ranking. Tiles with no elevation yet count as the start floor (0 m), like the current tile."""
        if not self.ctx.cfg.get('prefer_flat_tiles', True):
            return reachable[0]
        tiles = self.ctx.tiles
        if not tiles.ready or tiles.current is None or not any(
                tiles.elevation(l) for l in {T.tile_label(tiles.current)} | {t['tile'] for t in reachable if t.get('tile')}):
            return reachable[0]

        def elev(label):
            e = tiles.elevation(label) if label else None
            return e['elevation_m'] if e else 0.0
        here = elev(T.tile_label(tiles.current))
        tol = float(self.ctx.cfg.get('flat_tile_tolerance_m', 0.03))
        best = min(abs(elev(t.get('tile')) - here) if t.get('kind') == 'tile' else 0.0 for t in reachable)
        for t in reachable:
            diff = abs(elev(t.get('tile')) - here) if t.get('kind') == 'tile' else 0.0
            if diff <= best + tol:
                if t is not reachable[0]:
                    self.ctx.log('info', f'Elevation: tile {t.get("tile")} chosen (change {diff:.2f} m from here).')
                return t
        return reachable[0]

    def traversal_for(self, target) -> str:
        """Which traversal profile drives to this target: per-tile override, then per-kind, then the mode's default."""
        if target.get('tile') in self.tile_traversals:
            return self.tile_traversals[target['tile']]
        return self.kind_traversals.get(target.get('kind', 'tile'), self.cfg.get('traversal', 'route'))

    # -- driving
    def _current_tile_centre(self):
        """Centre of the tile the robot is in (the robot's own position when no tile grid exists yet)."""
        tiles = self.ctx.tiles
        if tiles.grid is not None and tiles.current is not None:
            return tuple(tiles.grid.center(tiles.current))
        pose = self.ctx.pose() or [0.0, 0.0, 0.0]
        return (pose[0], pose[1])

    def _drive_to(self, strat, target):
        self.no_path_streak = 0
        profile = self.traversal_for(target)
        try:
            trav = self.ctx.traversal(profile)
        except KeyError as exc:
            self.ctx.log('error', str(exc))
            return self._nothing_left('bad traversal profile')
        if not trav.available():
            self.ctx.log('warn', f'Traversal {profile!r} is not available yet (Nav2 down?): waiting.')
            self.set_phase('wait_traversal')
            return None
        grid_yaw = self.ctx.tiles.grid.yaw if self.ctx.tiles.grid is not None else 0.0
        route = None
        face_away = bool(self.cfg.get('face_away_from_walls', True)) and target.get('kind') == 'tile'
        sides = (self.ctx.tiles.tiles.get(tuple(int(v) for v in target['tile'].split(',')), {}).get('sides')
                 if face_away and target.get('tile') else None)
        if trav.uses_route and target.get('path_xy'):
            # N waypoints (traversals.<profile>.waypoints, default 3) evenly spaced, then the goal: along the straight
            # line from the current tile centre to the target (route_mode 'straight') or along the Nav2 path ('path')
            route_path = target['path_xy']
            if trav.cfg.get('route_mode', 'straight') == 'straight':
                route_path = [self._current_tile_centre(), tuple(target['goal_xy'][:2])]
            route = A.default_route(route_path, target['goal_xy'], grid_yaw,
                                    n=int(trav.cfg.get('waypoints', A.N_WAYPOINTS)),
                                    skip_start_m=float(trav.cfg.get('skip_start_m', A.SKIP_START_M)),
                                    sides=sides, face_away=face_away)
            dropped = []
            if trav.cfg.get('drop_backward_waypoints', False):
                route, dropped = A.filter_route(route, route_path[0])
            # a waypoint off the straight line to the tile centre makes an indirect route: drop it
            route, off_line = A.filter_deviating(route, route_path[0],
                                                 float(trav.cfg.get('max_waypoint_deviation_m', 0.25)))
            dropped += off_line
            if dropped:
                self.ctx.log('info', 'Route filter dropped: ' + '; '.join(dropped))
        if route:
            yaw = route[-1][2]
        else:
            pose = self.ctx.pose() or [0.0, 0.0, 0.0]
            yaw = A.snap_to_axis(math.atan2(target['goal_xy'][1] - pose[1], target['goal_xy'][0] - pose[0]), grid_yaw)
            if face_away:
                yaw = A.yaw_away_from_walls(sides, grid_yaw, yaw)
        self.target, self.strategy, self.traversal = target, strat, trav
        if not trav.start(target['goal_xy'], yaw, route):
            return self._failed('the traversal could not start')
        self.ctx.set_target(target.get('tile'), target['goal_xy'], route)
        why = f', {target["adjacency_note"]}' if target.get('adjacency_note') else ''
        if target.get('centre_offset_m'):
            why += f', goal {target["centre_offset_m"]:.2f} m off the centre (wall)'
        if route:
            why += f', {len(route) - 1} waypoints'
        self.ctx.log('info', f'Next: {target["kind"]} {target["label"]} ({target.get("path_m", "?")} m path, '
                             f'{target["clearance_m"]:.2f} m from walls{why}) via {strat.name} / {profile}.')
        self.ctx.event('target', target_kind=target['kind'], label=target['label'], tile=target.get('tile'),
                       goal_xy=target['goal_xy'], path_m=target.get('path_m'), adjacency=target.get('adjacency'),
                       exploration=strat.name, traversal=profile, route=A.route_json(route) if route else None)
        self.valid_checked_at = self.ctx.now()
        self.look = None
        self.set_phase('driving')
        return None

    def _poll_drive(self):
        res = self.traversal.poll()
        if not res.done:
            if self.in_phase_s() > float(self.cfg.get('goal_timeout_s', 120.0)):
                self.traversal.cancel()
                return self._failed('timed out')
            dropped = self._classify_target()
            if dropped:
                return None
            if self.ctx.now() - self.valid_checked_at >= 1.0:
                self.valid_checked_at = self.ctx.now()
                if not self.strategy.still_valid(self.target):
                    self.traversal.cancel()
                    scene = self._scene_while_driving()
                    if scene is not None:
                        return scene
                    self.ctx.log('info', f'{self.target["label"]} is no longer worth visiting: re-deciding.')
                    self.ctx.event('target_dropped', label=self.target['label'])
                    self._clear_target()
                    self._new_decision()
                    return None
            return self._edge_quickscan()
        if res.state == SUCCEEDED:
            t = self.target
            if t.get('kind') == 'tile':
                self.ctx.tiles.mark_goal_reached(tuple(int(v) for v in t['tile'].split(',')), t['goal_xy'],
                                                 self.ctx.pose())
            self.strategy.on_reached(t)
            self.avoid.clear()
            self.unseen.clear()
            self.ctx.event('reached', target_kind=t['kind'], label=t['label'], tile=t.get('tile'))
            self._clear_target()
            if self.cfg.get('scan_after_reach', True):
                return self.go(self.cfg.get('scan_mode', 'scanning'), f'{t["kind"]} {t["label"]} reached')
            self._new_decision()
            return None
        return self._failed(res.reason or 'failed')

    def _scene_while_driving(self):
        """scenes: true. The tile being driven to turned out to be a scene tile on the way (the scene detector keeps running
        while driving): recovery drives back to the tile the robot was last on (never into the scene tile), then the scene
        mode looks at it from there. Returns the transition, or None (not a scene tile / no recovery or scene mode / already
        looked at: the caller just re-decides)."""
        tiles, t = self.ctx.tiles, self.target
        scene_mode = self.cfg.get('scene_mode', 'scene')
        if (not self.cfg.get('scenes', False) or not tiles.skip_scene or t is None or t.get('kind') != 'tile'
                or tiles.current is None or scene_mode not in self.machine.modes
                or self.machine.recovery_mode not in self.machine.modes):
            return None
        label = t['tile']
        tile = tuple(int(v) for v in label.split(','))
        if label in self.machine.scenes_done or not tiles.is_scene(tile):
            return None
        if tiles.wall_between(tiles.current, tile):
            return None          # a wall between: looked at later from a neighbour across an open side (scene_viewpoints)
        back = T.tile_label(tiles.current)
        self.ctx.log('warn', f'Tile {label} turned out to be a scene tile while driving to it: recovering to tile {back}, '
                             f'then looking at it.')
        self.ctx.event('scene_while_driving', tile=label, back_to=back)
        self._clear_target()
        return self.go(self.machine.recovery_mode, f'tile {label} is a scene tile: back to tile {back}',
                       exclude=[label], return_to=back, then_mode=scene_mode, then_kwargs={'tile': label})

    def _scene_beside(self, pose):
        """scenes: true. At a decision, standing on a tile that shares an open side with a scene tile not looked at yet
        (also one that only became a scene tile since the robot arrived: the scan finished it) -> the scene mode faces
        it, has the AI describe it and marks it explored; the robot stays on its tile. Scene tiles are detected
        anywhere, but the scene mode only starts for one that shares a side with the current tile. Each scene tile once
        (machine.scenes_done)."""
        tiles = self.ctx.tiles
        if (not self.cfg.get('scenes', False) or tiles.grid is None
                or self.cfg.get('scene_mode', 'scene') not in self.machine.modes):
            return None
        here = tuple(tiles.grid.tile_of(pose[0], pose[1]))
        if tiles.current is None or here != tuple(tiles.current):
            return None
        for tile in tiles.scene_neighbours(here):
            label = T.tile_label(tile)
            if label in self.machine.scenes_done:
                continue
            self.ctx.log('info', f'Scene tile {label} shares a side with tile {T.tile_label(here)}: looking at it.')
            return self.go(self.cfg.get('scene_mode', 'scene'), f'scene tile {label} beside tile {T.tile_label(here)}',
                           tile=label)
        return None

    def _stairs_beside(self, pose):
        """stairs: true. At a decision, standing on a tile that shares a side with a staircase tile the steps go up away
        from (stairs_core.stair_crossing) -> the stairs mode climbs over it into the tile beyond. Each staircase once
        (machine.stairs_tried); one already climbed is walled off (tiles.stairs_crossed)."""
        tiles = self.ctx.tiles
        if (not self.cfg.get('stairs', False) or not tiles.stairs or tiles.grid is None or tiles.current is None
                or self.cfg.get('stairs_mode', 'stairs') not in self.machine.modes
                or tuple(tiles.grid.tile_of(pose[0], pose[1])) != tuple(tiles.current)):
            return None
        crossing = SC.stair_crossing(tiles.grid, tuple(tiles.current), tiles.stairs,
                                     skip=self.machine.stairs_tried | tiles.stairs_crossed,
                                     min_facing=float(self.cfg.get('stair_min_facing', 0.5)))
        if crossing is None:
            return None
        here, first = T.tile_label(tiles.current), T.tile_label(crossing['neighbour'])
        if not tiles.stairs_ready():
            # the stairs are one way: every traversable tile on this side is explored (scene tiles looked at) first
            if self._stairs_wait_logged != here:
                self._stairs_wait_logged = here
                self.ctx.log('info', f'Staircase on tile {first} beside tile {here}: not yet, '
                                     f'{len(tiles.unexplored())} tile(s) to visit / scene tile(s) to look at first.')
            return None
        self.ctx.log('info', f'Staircase on tile {first}, along the {crossing["side"]} side of tile {here}: climbing '
                             f'over it into tile {T.tile_label(crossing["beyond"])}.')
        return self.go(self.cfg.get('stairs_mode', 'stairs'), f'staircase on tile {first} beside tile {here}',
                       crossing=crossing)

    def _roller_ahead(self, pose):
        """rollers: true. At a decision, standing on a tile that shares a side with a roller tile whose roller lies across
        the way (roller_core.roller_crossing), or on a roller tile in front of its roller -> the roller mode squares up and
        crosses it in the h-step gait into the tile beyond. Each roller tile once (machine.rollers_tried)."""
        tiles = self.ctx.tiles
        if (not self.cfg.get('rollers', False) or not tiles.rollers or tiles.grid is None or tiles.current is None
                or self.cfg.get('roller_mode', 'roller') not in self.machine.modes
                or tuple(tiles.grid.tile_of(pose[0], pose[1])) != tuple(tiles.current)):
            return None
        crossing = RC.roller_crossing(tiles.grid, tuple(tiles.current), tiles.rollers,
                                      skip=self.machine.rollers_tried | tiles.rollers_crossed,
                                      min_facing=float(self.cfg.get('roller_min_facing', 0.8)), robot_xy=pose[:2])
        if crossing is None:
            return None
        here, roller = T.tile_label(tiles.current), T.tile_label(crossing['roller'])
        if crossing['roller'] != tuple(tiles.current):
            self._note_roller_split(crossing)
        where = 'this tile' if crossing['roller'] == tuple(tiles.current) else f'tile {roller}'
        self.ctx.log('info', f'Roller on {where} ({crossing["spots"]} spots, {crossing["lidar"]} confirmed by the lidar), '
                             f'across the {crossing["side"]} side of tile {T.tile_label(crossing["from_tile"])}: crossing '
                             f'it into tile {T.tile_label(crossing["beyond"])} in the h-step gait.')
        return self.go(self.cfg.get('roller_mode', 'roller'), f'roller on tile {roller} (robot on tile {here})',
                       crossing=crossing)

    def _note_roller_split(self, crossing):
        """Crossing a roller from a tile with other untravelled tiles beside it: the tile is the last split (the roller tile
        one of its options), so backtracking comes back here (over the roller) once the far side is explored."""
        tiles = self.ctx.tiles
        if not self.cfg.get('backtracking', False) or tiles.current is None:
            return
        ci, cj = tiles.current
        others = [T.tile_label(n) for n in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1))
                  if n != tuple(crossing['roller']) and tiles.tiles.get(n, {}).get('status') == 'to_visit'
                  and tiles._drivable(T.tile_label(n)) and tiles._open_between(tiles.current, n)]
        if not others:
            return
        options = sorted(others + [T.tile_label(crossing['roller'])])
        if tiles.mark_split(tiles.current, options):
            label = T.tile_label(tiles.current)
            self.ctx.log('info', f'Tile {label} is the last split: crossing the roller first, then back for '
                                 f'{", ".join(others)}.')
            self.ctx.event('split', tile=label, options=options)

    def _curtain_on_edge(self, pose):
        """curtains: true. A detected curtain (depth camera) along an edge of the tile the robot stands on, with an
        unvisited tile behind it -> its edge (the curtain mode aligns with it and walks through), else None."""
        tiles = self.ctx.tiles
        if (not self.cfg.get('curtains', False) or not hasattr(self.ctx, 'curtain_points') or tiles.grid is None
                or tiles.current is None or tuple(tiles.grid.tile_of(pose[0], pose[1])) != tuple(tiles.current)):
            return None
        pts = self.ctx.curtain_points()
        if pts is None or len(pts) == 0:
            return None
        pts = np.asarray(pts, float)
        pts = pts[tiles.not_curtain_mask(pts)]         # no spots on a rejected edge nor on the initial tile's edges
        if len(pts) == 0:
            return None
        return self._edge_curtain(tuple(tiles.current), pts)

    def _edge_curtain(self, tile, pts):
        """CC.edge_curtain for `tile` with this mode's thresholds (curtain points already masked)."""
        tiles = self.ctx.tiles
        edge = CC.edge_curtain(tiles.grid, tuple(tile), pts, visited=tiles.trail,
                               skip=self.machine.curtains_tried,
                               tol_m=float(self.cfg.get('curtain_edge_tol_m', 0.30)),
                               min_span_m=float(self.cfg.get('curtain_min_span_m', 0.40)),
                               min_camera_spots=int(self.cfg.get('curtain_min_camera_spots', 3)),
                               lidar_only=bool(self.cfg.get('curtain_lidar_only', True)),
                               camera_tol_m=float(self.cfg.get('curtain_camera_tol_m', 0.15)),
                               camera_min_span_m=float(self.cfg.get('curtain_camera_min_span_m', 0.30)))
        return edge

    def _curtain_scan(self, why):
        """curtains: true. Nothing reachable from here while tiles are left: go to curtain_scan_mode, which faces each
        edge of the current tile with an unvisited tile behind it so the curtain detector gets a look at it. Once per
        tile until a new tile is visited (retries / recoveries on the same tile do not scan again). None = no scan."""
        c, tiles = self.cfg, self.ctx.tiles
        mode = c.get('curtain_scan_mode', 'curtain_scan')
        if (not c.get('curtains', False) or not c.get('curtain_scan_when_stuck', True) or mode not in self.machine.modes
                or tiles.grid is None or tiles.current is None or not tiles.ready):
            return None
        if not (self.unreached or tiles.unexplored() or tiles.unreachable):
            return None
        key = (tuple(tiles.current), tiles.visited_count())
        if key in self.machine.curtain_scans:
            return None
        sides = curtain_scan_sides(tiles, self.machine.curtains_tried)
        if not sides:
            return None
        self.machine.curtain_scans.add(key)
        here = T.tile_label(tiles.current)
        self.ctx.log('info', f'{why}: looking for a curtain on the {", ".join(sides)} edge(s) of tile {here} first.')
        self.ctx.event('curtain_scan', tile=here, sides=sides, why=why)
        return self.go(mode, f'{why}: curtain scan of tile {here}', sides=sides)

    def _can_defer_curtain(self):
        """curtain_last: a curtain is walked through only after the other ways from its tile (needs backtracking)."""
        return (bool(self.cfg.get('curtain_last', True)) and bool(self.cfg.get('backtracking', False))
                and self.cfg.get('backtracking_mode', 'backtracking') in self.machine.modes)

    def _through_curtain(self, edge, why=''):
        tiles = self.ctx.tiles
        tiles.deferred_curtains.pop(tuple(edge['neighbour']), None)
        self.ctx.log('info', f'Curtain along the {edge["side"]} edge of tile {T.tile_label(tiles.current)} '
                             f'({edge["spots"]} spots, {edge.get("camera_spots", "?")} by camera, {edge["span_m"]} m): walking through it'
                             + (f' ({why}).' if why else '.'))
        return self.go(self.cfg.get('curtain_mode', 'curtain'),
                       f'curtain on the {edge["side"]} edge of tile {T.tile_label(tiles.current)}', edge=edge)

    def _classify_target(self):
        """Terrain of the tile being driven to: asked as soon as the robot points at it (the classifier looks at the floor
        straight ahead). The tile gets the answer. When the classifier could not see the floor there (walls in the way,
        too few points) the tile is dropped and another one chosen; asked only once per tile, so a tile that is never
        visible is still driven to when nothing else is left. True when the target was dropped."""
        t = self.target
        if (not self.cfg.get('classify_target', True) or not t or t.get('kind') != 'tile' or self.look == 'done'
                or not hasattr(self.ctx, 'classify_tile')):
            return False
        label = t['label']
        if label in self.unseen:
            self.look = 'done'
            return False
        if self.look is None:
            pose = self.ctx.pose()
            if pose is None or len(pose) < 3:
                return False
            bearing = math.atan2(t['goal_xy'][1] - pose[1], t['goal_xy'][0] - pose[0])
            off = abs(math.atan2(math.sin(bearing - pose[2]), math.cos(bearing - pose[2])))
            if off > math.radians(float(self.cfg.get('classify_max_angle_deg', 20.0))):
                return False                                  # not pointing at the tile yet
            self.ctx.classify_tile(label)
            self.look = {'since': self.ctx.now()}
            self.ctx.event('classify', label=label, off_deg=round(math.degrees(off), 1))
            return False
        status = self.ctx.terrain_status(label)
        if status == 'seen':
            self.look = 'done'
            return False
        if status != 'unseen':
            return False                                      # still waiting for the answer
        self.look = 'done'
        self.unseen.add(label)
        self.traversal.cancel()
        self.ctx.log('info', f'Tile {label}: its floor could not be seen from here: choosing another tile.')
        self.ctx.event('target_unseen', label=label)
        self.avoid.add(label)
        self._clear_target()
        self._new_decision()
        return True

    def _edge_quickscan(self):
        """quickscan_on_edge: the robot has just crossed into a new tile it has not quickscanned yet -> quickscan."""
        if not self.cfg.get('quickscan_on_edge', False) or self.ctx.tiles.grid is None:
            return None
        pose = self.ctx.pose()
        if pose is None:
            return None
        grid = self.ctx.tiles.grid
        tile = grid.tile_of(pose[0], pose[1])
        label = T.tile_label(tile)
        prev = getattr(self.machine, 'last_tile_label', None)
        self.machine.last_tile_label = label
        # only when the robot has just entered a different tile (not at the start, not while still in the old one)
        if prev is None or prev == label or label in self.machine.quickscanned:
            return None
        self.machine.quickscanned.add(label)
        return self.go(self.cfg.get('quickscan_mode', 'quickscan'), f'on the edge of tile {label}', tile=label)

    def _failed(self, reason):
        t = self.target or {}
        label = t.get('label', '?')
        gave_up = self.strategy.on_failed(t, reason) if self.strategy and t else False
        self.ctx.log('warn', f'{t.get("kind", "target")} {label}: {reason}' + ('; giving up on it.' if gave_up else '.'))
        self.ctx.event('goal_failed', target_kind=t.get('kind'), label=label, reason=reason, gave_up=gave_up)
        if t:
            self.avoid.add(label)
        self._clear_target()
        if self.cfg.get('on_goal_failed', 'decide') == 'recovery':
            return self.go(self.machine.recovery_mode, f'{label} {reason}', exclude=[t['tile']] if t.get('tile') else [])
        self._new_decision()
        return None

    def _clear_target(self):
        self.target, self.strategy, self.traversal = None, None, None
        self.ctx.set_target(None, None, None)

    def exit(self, next_mode, reason):
        if self.phase == 'checking_paths':
            self.ctx.paths.cancel()
        if self.phase == 'driving' and self.traversal is not None:
            self.traversal.cancel()
            if self.target is not None and self.strategy is not None and next_mode == self.machine.recovery_mode:
                self.strategy.on_failed(self.target, reason)
        self.pending = None
        self._clear_target()


# ----------------------------------------------------------------------------------------- mppi
@register_mode('mppi')
class MppiMode(Mode):
    """Stuck in the costmap obstacle zone (Nav2 cannot plan out of an inflated wall): move the robot out by hand.

    Each tick the global costmap around the robot gives the direction that reaches clear floor soonest without crossing
    a wall (costmap_escape.escape_direction); it is sent as linear x / y velocity in the robot frame (the Lite3 takes
    vx, vy and wz). The heading the robot has on entering is the heading to maintain: wz is a P-controller that keeps the
    robot on it while it slides out (hold_heading: false = wz 0, open loop). Exit to next_mode once the robot's cell has cost < clear_cost for clear_hold_s.
    No way out, no costmap, a timeout or moving further than max_travel_m: fail_mode (recovery)."""

    def targets(self):
        return [self.cfg.get('next_mode', 'exploring'), self.cfg.get('fail_mode', 'recovery')]

    @property
    def is_driving(self):
        return False

    def enter(self, reason, **kw):
        self.clear_since, self.start = None, self.ctx.pose()
        self.heading = self.start[2] if self.start is not None and len(self.start) >= 3 else None   # the heading to maintain
        self.ctx.set_target(None, None, None)
        self.ctx.clear_costmaps()                  # marks from before the robot moved can be what keeps the zone lethal
        self.set_phase('adjust')

    def _finish(self, ok, why):
        self.ctx.stop()
        self.ctx.log('info' if ok else 'warn', f'mppi: {why}')
        key = 'next_mode' if ok else 'fail_mode'
        return self.go(self.cfg.get(key, 'exploring' if ok else 'recovery'), f'mppi: {why}')

    def tick(self):
        c = self.cfg
        pose, cm = self.ctx.pose(), self.ctx.costmap_snapshot()
        if pose is None or len(pose) < 3 or cm is None:
            return self._finish(False, 'no pose / costmap') if self.in_phase_s() > 3.0 else None
        grid, info = cm
        here = CE.cost_at(grid, info, pose[0], pose[1])
        if here is not None and here < float(c.get('clear_cost', 70)):
            self.ctx.stop()
            if self.clear_since is None:
                self.clear_since = self.ctx.now()
            if self.ctx.now() - self.clear_since >= float(c.get('clear_hold_s', 0.6)):
                return self._finish(True, f'clear of the costmap (cost {here})')
            return None
        self.clear_since = None
        if self.in_phase_s() > float(c.get('timeout_s', 20.0)):
            return self._finish(False, 'timed out')
        if self.start is not None and math.hypot(pose[0] - self.start[0], pose[1] - self.start[1]) > float(c.get('max_travel_m', 1.0)):
            return self._finish(False, 'moved too far without getting clear')
        way = CE.escape_direction(grid, info, pose[0], pose[1], clear_cost=int(c.get('clear_cost', 70)),
                                  max_m=float(c.get('search_m', 1.0)))
        if way is None:
            return self._finish(False, 'no way out in the costmap')
        ux, uy, _ = way
        cy, sy = math.cos(pose[2]), math.sin(pose[2])
        v = float(c.get('speed_mps', 0.15))
        wz = 0.0
        if c.get('hold_heading', True) and self.heading is not None:
            err = math.atan2(math.sin(self.heading - pose[2]), math.cos(self.heading - pose[2]))
            wlim = float(c.get('wz_max', 0.3))
            wz = max(-wlim, min(wlim, float(c.get('heading_kp', 1.5)) * err))
        self.ctx.drive(v * (ux * cy + uy * sy), wz, v * (-ux * sy + uy * cy))      # map -> robot frame
        return None

    def exit(self, next_mode, reason):
        self.ctx.stop()


# ----------------------------------------------------------------------------------------- curtain
class _ViewsCall:
    """The AI calls of a scene tile's views (left / centre / right) as one call: done when all are, ok when any is.
    The result is the centre view's with all the descriptions joined; the scene type is a majority vote of the views:
    a scene only when more views named a scene type than said traversable (a tie goes to the side with the higher summed
    confidence, traversable when that ties too), then the commonest scene type named (ties: summed confidence). One view
    catching the edge of a neighbouring scene no longer outvotes two that saw none (bags 104357: traversable / kitchen /
    traversable -> kitchen, 175813: traversable / traversable / workshop -> workshop)."""
    ORDER = ('centre', 'left', 'right')

    def __init__(self, calls):
        self.calls = calls                                  # {view: call}

    @property
    def done(self):
        return all(c.done for c in self.calls.values())

    @property
    def ok(self):
        return any(c.ok for c in self.calls.values())

    def failure(self):
        return '; '.join(f'{v}: {c.failure()}' for v, c in self.calls.items() if not c.ok) or 'no views'

    @property
    def result(self):
        good = {v: c.result for v, c in self.calls.items() if c.ok}
        views = [v for v in self.ORDER if v in good]
        out = dict(good[views[0]])
        out['scene_description'] = ' | '.join(f'{v.capitalize()}: {good[v].get("scene_description", "")}'
                                              for v in ('left', 'centre', 'right') if v in good)
        out['images'] = {v: good[v]['image'] for v in ('left', 'centre', 'right') if good[v].get('image')}
        votes = [(good[v]['scene_type'], self._confidence(good[v])) for v in views if good[v].get('scene_type')]
        if votes:
            out['scene_type'] = self.vote(votes)
            # agreement x confidence of the winning answer: two tiles named the same scene -> the reporter keeps the higher
            win = [c for t, c in votes if t == out['scene_type']]
            out['scene_confidence'] = round(sum(win) / len(votes), 3)
            out['scene_votes'] = {v: good[v]['scene_type'] for v in views if good[v].get('scene_type')}
        return out

    @staticmethod
    def _confidence(result) -> float:
        try:
            return float(result.get('confidence') or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def vote(votes) -> str:
        """[(scene_type, confidence)] of the views -> the tile's scene type (see the class docstring)."""
        named = [(t, c) for t, c in votes if t != 'traversable']
        trav = [c for t, c in votes if t == 'traversable']
        if not named:
            return 'traversable'
        named_conf = sum(c for _, c in named)
        if len(named) < len(trav) or (len(named) == len(trav) and named_conf <= sum(trav)):
            return 'traversable'
        score = {}
        for t, c in named:
            n, conf = score.get(t, (0, 0.0))
            score[t] = (n + 1, conf + c)
        return max(sorted(score), key=lambda t: score[t])


@register_mode('scene')
class SceneMode(Mode):
    """A scene tile shares a side with the tile the robot stands on: look at it, never enter it.

    Phases: rest (stand still: a spin just before, e.g. the scan, is still coasting) -> center (drive with vx / vy, the
    heading held, to the centre of the tile the robot stands on, within center_tol_m; not there after center_timeout_s:
    go on from where it is) -> align (only now: turn in place to face the scene tile by the shorter way: face: back_wall
    squares up to the straight wall behind the scene tile in the lidar map, so the left / centre / right views spread
    evenly over it; no such wall, or face: centre, faces the scene tile's centre. Either way the shorter way, and
    keep that direction while it is almost behind the robot: right at 180 deg the shorter way flips with every bit of
    coasting) -> settle (stand still) -> describe (the AI is asked
    what is on the tile; retried llm_attempts times) -> next_mode (exploring), with the scene tile marked explored
    whatever the AI said (machine.scenes_done: never twice). The robot does not leave its tile.

    background: true: once the pictures are taken the robot does not wait for the AI. The call goes to the machine
    (describe_in_background), the scene tile is marked explored and the robot goes on exploring; the answer is recorded
    when it comes (a 'traversable' answer then reopens the tile). No retry: a new picture would be taken from wherever
    the robot is by then, so a failed answer is recorded as failed."""

    def __init__(self, name, machine, cfg):
        super().__init__(name, machine, cfg)
        self.label, self.tile, self.yaw, self.call, self.attempt, self.pose0 = None, None, 0.0, None, 0, None
        self.turn_dir = 0                                  # +1 / -1: the direction the turn started in
        self.centre, self.hold_yaw = None, None            # the current tile's centre; heading held driving there
        self.views = {}                                    # view name -> AI call, while the scenescan animation plays

    def targets(self):
        return [self.cfg.get('next_mode', 'exploring')]

    def enter(self, reason, tile=None, **kw):
        self.label, self.call, self.attempt, self.turn_dir = tile, None, 0, 0
        self.looking_sent = False                          # told the reporter we face the scene tile (settle .. leaving)
        self.views = {}
        self.ctx.set_target(None, None, None)
        pose, tiles = self.ctx.pose(), self.ctx.tiles
        if not tile or pose is None or tiles.grid is None:
            self.tile = None
            self.set_phase('abort')
            return
        self.tile = tuple(int(v) for v in tile.split(','))
        here = tiles.current if tiles.current is not None else tiles.grid.tile_of(pose[0], pose[1])
        if tiles.wall_between(here, self.tile):
            # a wall between this tile and the scene tile: the camera would only see the wall. Not looked at (and not
            # marked done): exploration brings the robot to a neighbour across an open side (scene_viewpoints).
            self.ctx.log('warn', f'Scene tile {tile}: a wall between it and tile {T.tile_label(here)}; not looking at it from here.')
            self.ctx.event('scene_skipped', tile=tile, from_tile=T.tile_label(here), why='wall between')
            self.set_phase('walled')
            return
        self.machine.scenes_done.add(tile)
        self.centre = tuple(tiles.grid.center(tuple(here)))
        self.yaw = self._facing(self.centre)
        self.ctx.event('scene', tile=tile, from_tile=T.tile_label(here), yaw_deg=round(math.degrees(self.yaw), 1))
        self.set_phase('rest')

    def _facing(self, xy):
        """Heading from xy to the scene tile's centre."""
        cx, cy = self.ctx.tiles.grid.center(self.tile)
        return math.atan2(cy - xy[1], cx - xy[0])

    def _back_wall(self):
        """face: back_wall. The scene tile's back wall in the lidar map (T.back_wall_heading) -> its dict, or None (no
        map / no straight wall behind the scene)."""
        c, tiles = self.cfg, self.ctx.tiles
        if str(c.get('face', 'back_wall')) != 'back_wall' or tiles.map is None or tiles.grid is None:
            return None
        here = tiles.current if tiles.current is not None else tiles.grid.tile_of(*self.centre)
        grid, info = tiles.map
        return T.back_wall_heading(grid, info, tiles.grid, tuple(here), tuple(self.tile),
                                   max_deg=float(c.get('back_wall_max_deg', 20.0)),
                                   beyond_m=float(c.get('back_wall_beyond_m', 0.4)),
                                   min_coverage=float(c.get('back_wall_min_coverage', 0.8)))

    def _start_align(self, pose):
        self.ctx.stop()
        self.yaw = self._facing(self.centre)               # yaw setpoint: along the line from this tile's centre to the scene tile's
        wall = self._back_wall()
        if wall is not None:                               # square to the scene's back wall: the views spread evenly over it
            self.yaw = wall['yaw']
            self.ctx.log('info', f'Scene tile {self.label}: facing its back wall ({wall["offset_deg"]:+.1f} deg off the '
                                 f'grid, {wall["distance_m"]:.2f} m past its centre, {wall["coverage"]:.0%} covered).')
            self.ctx.event('scene_back_wall', tile=self.label, **wall)
        elif str(self.cfg.get('face', 'back_wall')) == 'back_wall':
            self.ctx.log('info', f'Scene tile {self.label}: no straight back wall; facing its centre.')
        self.turn_dir = 0
        self.set_phase('align')

    def _finish(self, why):
        self.ctx.stop()
        if self.tile is not None:
            if tuple(self.tile) not in self.ctx.tiles.traversable:      # the AI called it traversable: it stays to be explored
                self.ctx.tiles.mark_described(self.tile)
            self.ctx.event('scene_done', tile=self.label, why=why)
        self.ctx.log('info', f'Scene tile {self.label}: {why}; marked explored.')
        return self.go(self.cfg.get('next_mode', 'exploring'), f'scene tile {self.label}: {why}')

    def _ask(self):
        self.attempt += 1
        self.call = self.ctx.describe_scene(self.label, self.pose0)
        self._describe()

    def _describe(self):
        """The pictures are taken (self.call holds the AI call). background: hand the call to the machine, which records
        the answer when it comes, and go on exploring now; otherwise wait for it here (describe phase, with retries)."""
        if self.cfg.get('background', False) and not self.call.done:
            self.machine.describe_in_background(self.label, self.pose0, self.call)
            return self.set_phase('handed_off')
        self.set_phase('describe')

    def tick(self):
        c = self.cfg
        pose = self.ctx.pose()
        if self.phase == 'abort':
            return self._finish('nothing to look at')
        if self.phase == 'walled':
            self.ctx.stop()
            return self.go(self.cfg.get('next_mode', 'exploring'), f'scene tile {self.label}: wall between, not looked at')
        if pose is None or len(pose) < 3:
            self.ctx.stop()
            return self._finish('no pose') if self.in_phase_s() > 3.0 else None
        if self.phase == 'rest':
            self.ctx.stop()
            if self.in_phase_s() >= float(c.get('rest_s', 1.5)):
                self.hold_yaw = pose[2]
                self.set_phase('center')
            return None
        if self.phase == 'center':
            dx, dy = self.centre[0] - pose[0], self.centre[1] - pose[1]
            dist = math.hypot(dx, dy)
            if dist <= float(c.get('center_tol_m', 0.06)):
                self._start_align(pose)
                return None
            if self.in_phase_s() > float(c.get('center_timeout_s', 15.0)):
                self.ctx.log('warn', f'Scene tile {self.label}: could not reach the tile centre ({dist:.2f} m off); '
                                     f'facing it from here.')
                self._start_align(pose)
                return None
            v = max(float(c.get('center_min_speed_mps', 0.05)),
                    min(float(c.get('center_speed_mps', 0.15)), float(c.get('center_gain', 1.0)) * dist))
            ux, uy = v * dx / dist, v * dy / dist                   # map frame -> robot frame vx / vy
            cs, sn = math.cos(pose[2]), math.sin(pose[2])
            herr = math.atan2(math.sin(self.hold_yaw - pose[2]), math.cos(self.hold_yaw - pose[2]))
            wz = max(-0.3, min(0.3, float(c.get('heading_gain', 1.0)) * herr))
            self.ctx.drive(ux * cs + uy * sn, wz, -ux * sn + uy * cs)
            return None
        if self.phase == 'align':
            err = math.atan2(math.sin(self.yaw - pose[2]), math.cos(self.yaw - pose[2]))
            if self.turn_dir and abs(err) > math.radians(float(c.get('commit_deg', 150.0))) \
                    and (err > 0) != (self.turn_dir > 0):
                err -= math.copysign(2 * math.pi, err)         # almost behind: carry on the way it started, do not flip
            elif abs(err) > math.radians(float(c.get('align_tol_deg', 6.0))):
                self.turn_dir = 1 if err > 0 else -1
            if self.in_phase_s() > float(c.get('align_timeout_s', 15.0)):
                self.ctx.log('warn', f'Scene tile {self.label}: could not turn to face it; describing from here.')
                self.ctx.stop()
                self.set_phase('settle')
                return None
            if self.turn_towards(err, float(c.get('align_tol_deg', 6.0))):
                self.set_phase('settle')
                return None
        elif self.phase == 'settle':
            if not self.looking_sent:
                # facing the scene tile from here on: what perception reports now is on THAT tile (its room / position)
                self.looking_sent = True
                getattr(self.ctx, 'scene_looking', lambda _l: None)(self.label)
            self.ctx.stop()
            if self.in_phase_s() >= float(c.get('settle_s', 1.0)):
                self.pose0 = pose
                if self.attempt == 0 and c.get('quickscan', True) and self.ctx.animation_play(c.get('animation', 'scenescan')):
                    self.ctx.animation_markers()           # drop stale ones
                    self.set_phase('quickscan')
                else:
                    self._ask()
        elif self.phase == 'quickscan':
            # a picture (+ AI call) at each marker the animation reaches: left, centre, right
            for view in self.ctx.animation_markers():
                if view not in self.views:
                    self.views[view] = self.ctx.describe_scene(self.label, self.pose0, view=view)
            state = self.ctx.animation_state()
            t = self.in_phase_s()
            if state == 'pending' and t > float(c.get('start_timeout_s', 3.0)):
                self.ctx.log('warn', 'Scene quickscan: animation_player did not start it (is it running?); one picture.')
                self.ctx.animation_cancel()
            elif state != 'done' and t > float(c.get('timeout_s', 30.0)):
                self.ctx.log('warn', 'Scene quickscan: the animation timed out; cancelling it.')
                self.ctx.animation_cancel()
            elif state != 'done':
                return None
            if self.views:
                self.attempt += 1
                self.call = _ViewsCall(self.views)
                self.ctx.event('scene_quickscan', tile=self.label, views=sorted(self.views))
                self._describe()
            else:
                self._ask()                                # no picture was taken: the single one from here
        elif self.phase == 'handed_off':
            return self._finish('pictures taken, the AI describes it in the background')
        elif self.phase == 'describe':
            if not self.call.done:
                return None
            if self.call.ok:
                self.ctx.record_scene(self.label, self.pose0, self.call)
                return self._finish('described')
            self.ctx.log('warn', f'Scene tile {self.label}: AI description failed ({self.call.failure()}), '
                                 f'try {self.attempt}/{int(c.get("llm_attempts", 2))}.')
            if self.attempt >= int(c.get('llm_attempts', 2)):
                self.ctx.record_scene(self.label, self.pose0, self.call)
                return self._finish('AI description failed')
            self._ask()
        return None

    def exit(self, next_mode, reason):
        if self.phase == 'quickscan' and self.ctx.animation_state() != 'done':
            self.ctx.animation_cancel()                    # leave pose mode if another mode takes over mid-animation
        if getattr(self, 'looking_sent', False):
            self.looking_sent = False
            getattr(self.ctx, 'scene_looking', lambda _l: None)(None)
        self.ctx.stop()


@register_mode('curtain')
class CurtainMode(Mode):
    """A curtain hangs along the edge of the current tile: get to the tile's centre, face the curtain squarely, then walk
    through it with forward (vx) commands (vy keeps the robot on the line from the tile centre through the edge).

    Phases: center (turn towards the centre of the tile in front of the curtain, drive there with vx / wz only: no vy) -> align (turn in
    place until square to the edge) -> gait_on (hstep: true, the default: ctx.hstep holds rug_hstep; stand gait_settle_s)
    -> cross (vx forward, small wz / vy to hold the line) until the robot is `through_m` past the edge line (the centre of
    the next tile) -> gait_off (h-step released, stand gait_settle_s). Curtains are only ever crossed in rug_hstep: the
    gait is released on a failed try and on leaving the mode however it is left. A cross that stalls (blocked: progress under stall_m for stall_s,
    judged only start_grace_s after the cross starts: the robot needs a moment to get walking from standing),
    times out, or an align that times out counts as a failed try: back to center -> align -> cross, up to
    max_attempts tries. Through: next_mode (or `then`). Given up: back to the mode that was running before the curtain
    mode (backtracking gets curtain_ok=False and goes on its way without this curtain).
    The tile behind is never detected twice (machine.curtains_tried), whatever happened. Nav2 is idle meanwhile: the
    velocity goes out through ctx.drive(vx, wz, vy) (the mission manager publishes it on /cmd_vel), ctx.stop() ends it.
    Backtracking uses it too (then=<mode>): afterwards it hands back to that mode instead of next_mode."""

    then, then_kwargs, from_tile, came_from = None, {}, None, None

    def targets(self):
        out = [self.cfg.get('next_mode', 'exploring'), self.cfg.get('low_move_mode', 'recovery')]
        if self.cfg.get('scan_after', True) and self.cfg.get('scan_mode', 'scanning') in self.machine.modes:
            out.append(self.cfg.get('scan_mode', 'scanning'))
        return out

    @property
    def is_driving(self):
        return self.phase in ('center', 'cross')

    def _low_movement(self, pose):
        """True when the robot has barely moved for low_move_s while the mode is moving it (0 = off): less than
        low_move_m of position AND less than low_move_deg of heading since the reference pose, which is renewed whenever
        it moves more, at every phase change and while the first low_move_grace_s of a phase run (a standing robot takes
        ~1.5 s to get walking). Moving includes turning on the spot, so the align / turn phases do not trip it."""
        c = self.cfg
        limit = float(c.get('low_move_s', 2.0))
        if limit <= 0 or self.phase not in ('center', 'align', 'cross'):
            self._lm_ref = None
            return False
        now = self.ctx.now()
        ref = getattr(self, '_lm_ref', None)
        dyaw = abs(math.atan2(math.sin(pose[2] - ref[2]), math.cos(pose[2] - ref[2]))) if ref else 0.0
        if (ref is None or ref[4] != self.phase or math.hypot(pose[0] - ref[0], pose[1] - ref[1]) >= float(c.get('low_move_m', 0.05))
                or dyaw >= math.radians(float(c.get('low_move_deg', 3.0)))):
            self._lm_ref = (pose[0], pose[1], pose[2], now, self.phase)
            return False
        still_since = max(ref[3], self.phase_since + float(c.get('low_move_grace_s', 2.0)))
        return now - still_since > limit

    def enter(self, reason, edge=None, then=None, then_kwargs=None, **kw):
        self.edge = edge
        self._lm_ref = None
        self.then, self.then_kwargs = then, dict(then_kwargs or {})   # backtracking: hand back to it afterwards
        prev = self.machine.previous
        self.came_from = then or (prev if prev and prev != self.name else None) or self.cfg.get('next_mode', 'exploring')
        self.from_tile = tuple(self.ctx.tiles.current) if self.ctx.tiles.current is not None else None
        self.machine.curtains_tried.add(tuple(edge['neighbour']))
        self.start_pose = self.ctx.pose()
        self.attempts = 0
        self.hstep_on, self.result = False, None
        cx, cy = edge['centre_xy']
        nx, ny = edge['normal']
        self.tile_centre = (cx - nx * edge['through_m'], cy - ny * edge['through_m'])   # tile in front of the curtain
        self.ctx.set_target(None, None, None)
        self.ctx.event('curtain', side=edge['side'], neighbour=list(edge['neighbour']), centre_xy=edge['centre_xy'],
                       spots=edge['spots'], camera_spots=edge.get('camera_spots'))
        self._start_center()

    def _start_center(self):
        pose = self.ctx.pose()
        self.hold_yaw = pose[2] if pose is not None and len(pose) >= 3 else self.edge['yaw']
        self.closest = None                                # nearest the robot has been to the tile centre this try
        self.set_phase('center')

    def _yaw_error(self, pose, yaw=None):
        d = (self.edge['yaw'] if yaw is None else yaw) - pose[2]
        return math.atan2(math.sin(d), math.cos(d))

    def _progress(self, pose):
        """Metres past the edge line along the way through it (negative while still inside the tile)."""
        cx, cy = self.edge['centre_xy']
        nx, ny = self.edge['normal']
        return (pose[0] - cx) * nx + (pose[1] - cy) * ny

    def _lateral(self, pose):
        """Metres to the left of the line from the tile centre through the middle of the edge."""
        nx, ny = self.edge['normal']
        return (pose[0] - self.tile_centre[0]) * -ny + (pose[1] - self.tile_centre[1]) * nx

    def _drive_map(self, pose, ux, uy, wz):
        """Velocity (ux, uy) in the map frame -> robot frame vx / vy."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        self.ctx.drive(ux * c + uy * s, wz, -ux * s + uy * c)

    def _curtain_hstep(self, on):
        """Hold (on) / release the h-step gait (rug_hstep, via ctx.hstep) for the cross."""
        if on != getattr(self, 'hstep_on', False):
            self.hstep_on = on
            if hasattr(self.ctx, 'hstep'):
                self.ctx.hstep(on)
            self.ctx.log('info', f'Curtain: h-step gait {"ON" if on else "released"}.')
            self.ctx.event('curtain_gait', on=on)

    def _imu(self):
        """(nose_up_deg, yaw_rad, gyro_z_rad_s) from /imu/data, or None (no IMU in this context, or stale)."""
        f = getattr(self.ctx, 'imu', None)
        r = f() if f is not None else None
        if r is None or r[3] > float(self.cfg.get('imu_max_age_s', 0.5)):
            return None
        pitch, yaw, gz, _ = r
        return float(self.cfg.get('nose_up_sign', -1.0)) * math.degrees(pitch), yaw, gz

    def _start_cross(self, pose):
        # the robot stands still when the cross starts and takes ~1.5 s to get walking: the stall window only
        # opens start_grace_s later (else a curtain right in front is called a wall before the first step)
        # the squared-up heading in the IMU's own frame: the odom rebase after the cross adds the IMU's yaw drift to it
        imu = self._imu()
        self.cross_imu_ref = None if imu is None else imu[1] + self._yaw_error(pose)
        self.best = self._progress(pose)
        self.best_at = self.ctx.now() + float(self.cfg.get('start_grace_s', 2.0))
        self.set_phase('cross')

    def _reject(self):
        """This edge is no curtain (timed out / blocked walking through it): unmark it, so detection goes on elsewhere
        (tiles.not_curtain_mask drops the spots on it; the mission manager drops it as the curtain wall)."""
        if self.from_tile is None:
            return
        self.ctx.tiles.reject_curtain(self.from_tile, self.edge['side'])
        self.ctx.event('curtain_rejected', tile=T.tile_label(self.from_tile), side=self.edge['side'],
                       neighbour=list(self.edge['neighbour']))

    def _finish(self, why, through=False, reject=False):
        self.ctx.stop()
        self._curtain_hstep(False)
        if reject and not through:
            self._reject()
            why += '; unmarked as a curtain'
        self.ctx.log('info' if through else 'warn', f'Curtain: {why}')
        if through and self.from_tile is not None:
            self.ctx.tiles.add_curtain_edge(self.from_tile, self.edge['neighbour'])   # backtracking may walk it again
        if through:
            beyond = self._relocalise_after_cross()
            if self.then:
                return self.go(self.then, f'curtain: {why}', curtain_ok=True, **self.then_kwargs)
            scan = self.cfg.get('scan_mode', 'scanning')
            if beyond is not None and self.cfg.get('scan_after', True) and scan in self.machine.modes:
                # as after the stairs: a scan lets SLAM place the robot in the map; the tile it stands on is corrected
                return self.go(scan, f'curtain: {why}; scanning to place the robot', after_curtain=beyond)
            return self.go(self.cfg.get('next_mode', 'exploring'), f'curtain: {why}')
        self.ctx.event('curtain_failed', neighbour=list(self.edge['neighbour']), why=why, attempts=self.attempts,
                       back_to=self.came_from)
        if self.came_from == self.then:
            return self.go(self.then, f'curtain: {why}', curtain_ok=False, **self.then_kwargs)
        return self.go(self.came_from, f'curtain: {why}')

    def _relocalise_after_cross(self):
        """Through the curtain: re-anchor odom where the cross was meant to end, the way the stairs mode does after a
        climb. The robot squared up to the edge normal at the centre of the tile in front and walked straight along
        it (vy held the line, wz the heading), so it is on that line at the centre of the tile behind; its heading is
        the squared-up one plus the IMU's yaw drift since. Walking through the fabric (it drapes over the lidar and
        the body, the scans see it move) can throw slam_toolbox's match off; the rebase plus the scan after it
        (scan_after, tiles.relocate) puts the robot back. Returns the tile behind (marked reached), or None when
        rebase_odom is off / there is no grid."""
        tiles, pose = self.ctx.tiles, self.ctx.pose()
        if not self.cfg.get('rebase_odom', True) or tiles.grid is None or not hasattr(self.ctx, 'rebase_odom'):
            return None
        nx, ny = self.edge['normal']
        cx, cy = self.edge['centre_xy']
        bx, by = cx + nx * self.edge['through_m'], cy + ny * self.edge['through_m']   # on the line, behind the curtain
        yaw = math.atan2(ny, nx)
        imu, ref = self._imu(), getattr(self, 'cross_imu_ref', None)
        if imu is not None and ref is not None:
            yaw = T.wrap(yaw + T.wrap(imu[1] - ref))
        self.ctx.rebase_odom(bx, by, yaw)
        beyond = tuple(tiles.grid.tile_of(bx, by))
        tiles.mark_goal_reached(beyond, tiles.grid.center(beyond), [bx, by, yaw])
        self.ctx.log('info', f'Curtain: odom re-anchored on tile {T.tile_label(beyond)} at ({bx:.2f}, {by:.2f}, '
                             f'{math.degrees(yaw):+.0f} deg).')
        self.ctx.event('odom_rebase', tile=T.tile_label(beyond), reference='curtain',
                       pose=[round(bx, 3), round(by, 3), round(yaw, 3)],
                       was=[round(float(v), 3) for v in pose] if pose is not None else None)
        return beyond

    def _failed_try(self, why, reject=False):
        """One try did not get through: back to the tile centre and try again, or give up (reject: the edge is
        unmarked as a curtain when given up)."""
        self.attempts += 1
        tries = int(self.cfg.get('max_attempts', 3))
        if self.attempts >= tries:
            return self._finish(f'{why}; gave up after {self.attempts} tries', reject=reject)
        self.ctx.stop()
        self._curtain_hstep(False)
        self.ctx.log('warn', f'Curtain: {why} (try {self.attempts}/{tries}); back to the tile centre to try again.')
        self.ctx.event('curtain_retry', neighbour=list(self.edge['neighbour']), why=why, attempt=self.attempts)
        self._start_center()
        return None

    def tick(self):
        pose = self.ctx.pose()
        if pose is None or len(pose) < 3:
            self.ctx.stop()
            return self._finish('no pose') if self.in_phase_s() > 3.0 else None
        c = self.cfg
        if self._low_movement(pose):
            self.ctx.stop()
            why = (f'barely moved for {float(c.get("low_move_s", 2.0)):g} s while {self.phase}ing'
                   if self.phase != 'cross' else f'barely moved for {float(c.get("low_move_s", 2.0)):g} s while crossing')
            self.ctx.log('warn', f'Curtain: {why}; recovery.')
            self.ctx.event('curtain_failed', neighbour=list(self.edge['neighbour']), why=why, attempts=self.attempts,
                           back_to=c.get('low_move_mode', 'recovery'))
            return self.go(c.get('low_move_mode', 'recovery'), f'curtain: {why}')
        if self.phase in ('gait_on', 'gait_off'):            # standing still while the gait switches
            self.ctx.stop()
            if self.in_phase_s() < float(c.get('gait_settle_s', 2.0)):
                return None
            if self.phase == 'gait_off':
                return self._finish(*self.result)
            return self._start_cross(pose)
        if self.phase == 'center':
            dx, dy = self.tile_centre[0] - pose[0], self.tile_centre[1] - pose[1]
            dist = math.hypot(dx, dy)
            tol = float(c.get('center_tol_m', 0.10))
            self.closest = dist if self.closest is None else min(self.closest, dist)
            # a legged robot cannot hold the centre to a few cm: close enough, or already closest and drifting off again
            if dist <= tol or (self.closest <= 2.0 * tol and dist > self.closest + float(c.get('center_drift_m', 0.04))):
                self.ctx.stop()
                self.prev_err = None
                self.set_phase('align')
                return None
            if self.in_phase_s() > float(c.get('center_timeout_s', 15.0)):
                self.ctx.log('warn', f'Curtain: could not reach the tile centre ({dist:.2f} m off); aligning from here.')
                self.ctx.stop()
                self.set_phase('align')
                return None
            v = max(float(c.get('center_min_speed_mps', 0.05)),
                    min(float(c.get('center_speed_mps', 0.15)), float(c.get('center_gain', 1.0)) * dist))
            # no omni motion: turn towards the centre, then drive there with vx and wz only (the way a Nav2 goal is driven)
            bearing = self._yaw_error(pose, math.atan2(dy, dx))
            if abs(bearing) > math.radians(float(c.get('center_turn_deg', 30.0))):
                w = max(float(c.get('min_turn_rad_s', 0.2)), min(float(c.get('max_turn_rad_s', 0.4)),
                                                                  float(c.get('turn_gain', 1.0)) * abs(bearing)))
                self.ctx.drive(0.0, math.copysign(w, bearing))
            else:
                wz = max(-0.6, min(0.6, float(c.get('center_heading_gain', 1.5)) * bearing))
                self.ctx.drive(v * math.cos(bearing), wz)
            return None
        err = self._yaw_error(pose)
        if self.phase == 'align':
            prev, self.prev_err = getattr(self, 'prev_err', None), err
            # square within the tolerance, or the heading swept through the target (the pose lags the turn: overshoot)
            crossed = prev is not None and prev * err < 0 and abs(err) < math.radians(float(c.get('align_cross_deg', 30.0)))
            if self.in_phase_s() > float(c.get('align_timeout_s', 15.0)):
                return self._failed_try('could not square up to it')
            if crossed:
                self.ctx.stop()
            if crossed or self.turn_towards(err, float(c.get('align_tol_deg', 5.0))):
                if c.get('hstep', True):                 # curtains are crossed in rug_hstep only: switch, stand, then walk
                    self._curtain_hstep(True)
                    self.set_phase('gait_on')
                    return None
                self._start_cross(pose)
                return None
            return None
        if self.phase == 'cross':
            prog = self._progress(pose)
            if prog >= self.edge['through_m'] + float(c.get('extra_m', 0.0)):
                if self.hstep_on:                            # release the h-step gait and stand before driving on
                    self.ctx.stop()
                    self.result = ('through the curtain', True)
                    self._curtain_hstep(False)
                    self.set_phase('gait_off')
                    return None
                return self._finish('through the curtain', through=True)
            if prog > self.best + float(c.get('stall_m', 0.05)):
                self.best, self.best_at = prog, self.ctx.now()
            if self.ctx.now() - self.best_at > float(c.get('stall_s', 3.0)):
                if c.get('blocked_gives_up', True):      # a curtain gives way: a robot that stalls is pushing on a wall
                    return self._finish(f'blocked {prog:+.2f} m from the edge line (a wall, not a curtain)', reject=True)
                return self._failed_try(f'blocked {prog:+.2f} m from the edge line')
            if self.in_phase_s() > float(c.get('cross_timeout_s', 20.0)):
                if c.get('timeout_gives_up', True):      # timed out: no curtain here, look for it elsewhere
                    return self._finish('timed out walking through it (not a curtain)', reject=True)
                return self._failed_try('timed out walking through it', reject=True)
            wz = max(-0.3, min(0.3, float(c.get('heading_gain', 1.0)) * err))
            nx, ny = self.edge['normal']
            lim = float(c.get('lateral_max_mps', 0.1))
            lat = max(-lim, min(lim, -float(c.get('lateral_gain', 1.0)) * self._lateral(pose)))
            v = float(c.get('speed_mps', 0.25))
            self._drive_map(pose, v * nx - lat * ny, v * ny + lat * nx, wz)
        return None

    def exit(self, next_mode, reason):
        self.ctx.stop()
        self._curtain_hstep(False)                         # left mid-crossing (low movement, operator command): release
        if next_mode != self.then:
            self.ctx.tiles.open_curtains = False


# ----------------------------------------------------------------------------------------- stairs
def _cmd_code(value) -> int:
    """A /simple_cmd code from the config: an int, or a string such as '0x21010528'."""
    return int(value, 0) if isinstance(value, str) else int(value)


@register_mode('stairs')
class StairsMode(CurtainMode):
    """A staircase tile shares a side with the current tile: climb over it in the robot's AI mode.

    Phases: center (vx / vy to the centre of the current tile, heading held) -> align (turn in place until square to the
    side the staircase is on) -> ai_on (AI mode on: MotionSimpleCMD ai_on_code on /simple_cmd; stand still ai_settle_s)
    -> cross (vx forward, wz holds the heading, vy the line through the tile centres) until the robot is over the
    staircase tile(s), at the centre of the tile beyond -> ai_off (stop, AI mode off with ai_off_code, stand still
    ai_settle_s) -> next_mode (exploring). The staircase is then walled off (tiles.mark_stairs_crossed: virtual walls on
    all four edges of its tiles) so the robot never enters it again; the tile beyond counts as reached.

    Giving up before AI mode is on (could not square up, no pose) hands back to the mode that was running before. Once it
    is on, a climb that stalls or times out stops the robot, turns AI mode off, walls the staircase off as well (no
    second try) and goes to fail_mode. Leaving the mode any other way (an operator mode command, shutdown) also turns
    AI mode off. Each staircase is gone for once (machine.stairs_tried). While AI mode is on, the mission manager also
    publishes /ai_mode true: the gait switcher sends no gait and the stair detector pauses.

    The centring / squaring-up / line-following helpers are the curtain mode's: the geometry is the same (walk straight
    through one edge of the current tile and on to the centre of a tile further along)."""

    def targets(self):
        return [self.cfg.get('next_mode', 'exploring'), self.cfg.get('fail_mode', 'recovery'),
                self.cfg.get('scan_mode', 'scanning')]

    @property
    def is_driving(self):
        return self.phase in ('center', 'cross')

    def enter(self, reason, crossing=None, **kw):
        self.edge, self.then, self.then_kwargs = crossing, None, {}
        prev = self.machine.previous
        prev_type = self.machine.modes[prev].type_name if prev in self.machine.modes else None
        # straight from the opening scan (opening_stairs): giving up goes on to exploring, not into another scan
        self.came_from = (prev if prev and prev != self.name and prev_type != 'scanning'
                          else self.cfg.get('next_mode', 'exploring'))
        self.ai_on, self.climbed, self.attempts, self.pose_lost_at = False, False, 0, None
        self.closest = None
        self.ai_req, self.ai_attempts, self.ai_sent_t, self.ai_ok_since, self.ai_failed = None, 0, 0.0, None, False
        self.ref = None
        self.ctx.set_target(None, None, None)
        if not crossing:
            self.set_phase('abort')
            return
        self.machine.stairs_tried.update(map(tuple, crossing['stairs']))
        self.from_tile = tuple(self.ctx.tiles.current) if self.ctx.tiles.current is not None else None
        cx, cy = crossing['centre_xy']
        nx, ny = crossing['normal']
        half = 0.5 * self.ctx.tiles.grid.size
        self.tile_centre = (cx - nx * half, cy - ny * half)          # centre of the tile in front of the staircase
        # the reference tile beyond the staircase is predicted only once the robot has squared up to it (_predict_beyond)
        self.ctx.event('stairs', side=crossing['side'], stairs=[T.tile_label(t) for t in crossing['stairs']],
                       beyond=T.tile_label(crossing['beyond']), facing=crossing.get('facing'),
                       through_m=crossing['through_m'])
        self._start_center()

    def _predict_beyond(self):
        """The reference tile beyond the staircase, along the staircase's own direction (not the grid's): odom is
        rebased to it after the climb. Predicted after squaring up at the centre of the tile in front of the stairs, so
        it uses the staircase as the detector sees it from there (the steps right ahead), not as first seen from
        further away. Squaring up again (it coasted off while standing still) predicts it again."""
        tiles, crossing = self.ctx.tiles, self.edge
        if self.ref is not None:                                     # squared up again: replace the earlier prediction
            tiles.stair_refs[:] = [r for r in tiles.stair_refs if r is not self.ref]
        self.ref = SC.beyond_reference(tiles.grid, crossing['stairs'], tiles.stairs, fallback_dir=crossing['normal'])
        tiles.stair_refs.append(self.ref)
        tiles.version += 1                                           # publishers: draw the new reference tile
        rx, ry = self.ref['centre_xy']
        pose = self.ctx.pose()
        self.ctx.log('info', f'Stairs: squared up; the tile beyond the staircase is predicted at ({rx:.2f}, {ry:.2f}), '
                             f'{math.degrees(T.wrap(self.ref["yaw"] - crossing["yaw"])):+.1f} deg off the grid '
                             f'(grid tile {T.tile_label(self.ref["grid_tile"])}).')
        self.ctx.event('stairs_reference', stairs=[T.tile_label(t) for t in crossing['stairs']],
                       centre_xy=[round(rx, 3), round(ry, 3)], yaw_deg=round(math.degrees(self.ref['yaw']), 1),
                       off_grid_deg=round(math.degrees(T.wrap(self.ref['yaw'] - crossing['yaw'])), 1),
                       grid_tile=T.tile_label(self.ref['grid_tile']),
                       pose=[round(float(v), 3) for v in pose] if pose is not None else None)

    def _labels(self):
        return ', '.join(T.tile_label(t) for t in self.edge['stairs']) if self.edge else '?'

    def _ai(self, on):
        code = _cmd_code(self.cfg.get('ai_on_code' if on else 'ai_off_code', 0x21010528 if on else 0x2101052B))
        if getattr(self, 'ai_req', None) != on:
            self.ai_req, self.ai_attempts, self.ai_failed = on, 0, False
        self.ai_attempts += 1
        self.ai_sent_t, self.ai_ok_since = self.ctx.now(), None
        heard = self.ctx.ai_mode(on, code)
        self.ai_on = on
        self.climbed |= on
        again = f', attempt {self.ai_attempts}' if self.ai_attempts > 1 else ''
        self.ctx.log('info' if heard and not again else 'warn',
                     f'Stairs: AI mode {"ON" if on else "OFF"} (0x{code:08X}{again})'
                     + ('' if heard else ': nobody is listening on /simple_cmd, the robot did not get it') + '.')
        self.ctx.event('ai_mode', on=on, cmd_code=f'0x{code:08X}', heard=heard, attempt=self.ai_attempts)

    def _robot_state(self):
        """(basic, gait, policy, motion) from /robot_state (0x0901 report), or None (no bridge / stale / ai_verify off)."""
        if not self.cfg.get('ai_verify', True):
            return None
        get = getattr(self.ctx, 'robot_state', None)
        return get() if get is not None else None

    def _ai_settled(self, on):
        """True once the robot reports the AI state asked for (on: (16, ai_policy_state, *), the AI basic gait; off:
        basic state 6, torque control in a normal gait), held ai_confirm_s. Until then the AI command is sent again
        every ai_retry_s (keeps trying unless ai_max_attempts is set). With no robot state: ai_settle_s, as before."""
        c = self.cfg
        settle = float(c.get('ai_settle_s', 2.0))
        st = self._robot_state() if getattr(self, 'ai_req', None) == on else None
        if st is None:
            return self.in_phase_s() >= settle
        basic, _gait, policy, _motion = st[:4]
        want_policy = int(c.get('ai_policy_state', 0))
        ok = (basic == BASIC_AI and (want_policy < 0 or policy == want_policy)) if on else basic == BASIC_STANDING
        now = self.ctx.now()
        if ok:
            self.ai_ok_since = now if self.ai_ok_since is None else self.ai_ok_since
            if now - self.ai_ok_since >= float(c.get('ai_confirm_s', 0.3)):
                self.ctx.log('info', f'Stairs: robot reports AI mode {"ON" if on else "OFF"} {tuple(st[:4])} '
                                     f'(attempt {self.ai_attempts}).')
                return True
            return False
        self.ai_ok_since = None
        if now - self.ai_sent_t >= max(float(c.get('ai_retry_s', 3.0)), settle):
            limit = int(c.get('ai_max_attempts', 0))
            if limit and self.ai_attempts >= limit:
                self.ctx.log('error', f'Stairs: AI mode {"ON" if on else "OFF"} not reached after {self.ai_attempts} '
                                      f'attempts (robot state {tuple(st[:4])}).')
                self.ai_failed = True
                return True
            self.ctx.log('warn', f'Stairs: robot state {tuple(st[:4])}, not AI mode {"ON" if on else "OFF"} yet: '
                                 'sending it again.')
            self._ai(on)
        return False

    def _finish(self, why, through=False):
        self.ctx.stop()
        if self.ai_on:
            self._ai(False)
        tiles = self.ctx.tiles
        if self.edge is not None and self.climbed:
            tiles.mark_stairs_crossed(self.edge['stairs'])           # never again: virtual walls all round it
        if through:
            beyond = tuple(self.edge['beyond'])
            pose = self.ctx.pose()
            if self.climbed and self.cfg.get('rebase_odom', True):
                # AI mode throws the leg odometry off: the robot is where the crossing was meant to end, the centre of
                # the reference tile beyond the staircase (along the steps, SC.beyond_reference), facing up the stairs
                # (rebase_yaw 'stairs') or the way it squared up to the grid edge ('grid'). Re-anchor odom there; SLAM
                # matches the remainder.
                if self.ref is None:                     # never squared up (cannot happen once AI mode was on)
                    self._predict_beyond()
                ref = self.ref
                # Where the robot really is: it squared up to the edge normal (not the staircase's own direction, which
                # the grid can be turned off by) and walked straight along it from the centre of the tile in front of
                # the staircase, heading held on the IMU. So: on that line, as far along as the reference centre; heading
                # the squared-up one plus whatever the IMU yaw drifted since. (Snapping to the reference centre / the
                # stairs yaw put the pose off the line and turned it by the grid-to-stairs angle.)
                nx, ny = self.edge['normal']
                if self.cfg.get('rebase_yaw', 'heading') == 'stairs':
                    yaw = ref['yaw']
                    bx, by = ref['centre_xy']
                else:
                    yaw = math.atan2(ny, nx)
                    imu, yref = self._imu(), getattr(self, 'imu_yaw_ref', None)
                    if imu is not None and yref is not None:
                        yaw = T.wrap(yaw + T.wrap(imu[1] - yref))
                    rx, ry = ref['centre_xy']
                    d = (rx - self.tile_centre[0]) * nx + (ry - self.tile_centre[1]) * ny
                    bx, by = self.tile_centre[0] + nx * d, self.tile_centre[1] + ny * d
                self.ctx.rebase_odom(bx, by, yaw)
                ref['used'] = True
                tiles.version += 1
                ref_tile = tuple(tiles.grid.tile_of(bx, by))
                if ref_tile in set(map(tuple, self.edge['stairs'])):    # steep angle off the grid: stay off the staircase
                    ref_tile = beyond
                self.ctx.event('odom_rebase', tile=T.tile_label(ref_tile), reference='stairs',
                               pose=[round(bx, 3), round(by, 3), round(yaw, 3)],
                               grid_beyond=T.tile_label(beyond),
                               was=[round(float(v), 3) for v in pose] if pose is not None else None)
                if ref_tile != beyond:
                    self.ctx.log('warn', f'Stairs: the reference tile beyond the staircase lies in grid tile '
                                         f'{T.tile_label(ref_tile)}, not {T.tile_label(beyond)} (grid turned '
                                         f'{math.degrees(T.wrap(ref["yaw"] - self.edge["yaw"])):+.0f} deg off the stairs).')
                beyond = ref_tile
                pose = [bx, by, yaw]
            tiles.mark_goal_reached(beyond, tiles.grid.center(beyond), pose)
            self.ctx.log('info', f'Stairs: {why}; staircase tile(s) {self._labels()} walled off, now on tile '
                                 f'{T.tile_label(beyond)}.')
            self.ctx.event('stairs_crossed', stairs=[T.tile_label(t) for t in self.edge['stairs']],
                           beyond=T.tile_label(beyond))
            scan = self.cfg.get('scan_mode', 'scanning')
            if (self.climbed and self.cfg.get('new_map', False) and scan in self.machine.modes
                    and hasattr(self.ctx, 'start_new_map')):
                # new_map: save the map up to here under reports/, clear it, and start a new one from this tile (the
                # scan mode's opening scan places the new grid; the wall behind the robot is against the staircase)
                self.after_new_map = (scan, why, tuple(beyond))
                self.ctx.start_new_map(self.cfg.get('new_map_dir') or None, float(self.cfg.get('new_map_timeout_s', 60.0)))
                self.set_phase('new_map')
                return None
            if self.cfg.get('scan_after', True) and scan in self.machine.modes:
                # a scan lets SLAM place the robot in the map; the tile it stands on is corrected from that
                return self.go(scan, f'stairs: {why}; scanning to place the robot', after_stairs=tuple(beyond))
            return self.go(self.cfg.get('next_mode', 'exploring'), f'stairs: {why}')
        back = self.cfg.get('fail_mode', 'recovery') if self.climbed else self.came_from
        if self.climbed:
            self._release_odom()
        self.ctx.log('warn', f'Stairs: {why}; giving up on staircase tile(s) {self._labels()}'
                             + (' (walled off)' if self.climbed else '') + '.')
        self.ctx.event('stairs_failed', stairs=self._labels(), why=why, ai_was_on=self.climbed, back_to=back)
        return self.go(back, f'stairs: {why}')

    def _climb_over(self, why, through=False):
        """Done climbing (over it, or given up): stop, AI mode off, stand still, then _finish."""
        self.ctx.stop()
        self.result = (why, through)
        if self.ai_on:
            self._ai(False)
        self.set_phase('ai_off')

    def tick(self):
        c = self.cfg
        if self.phase == 'new_map':
            state = self.ctx.new_map_state()
            if state == 'working':
                self.ctx.stop()
                return None
            scan, why, beyond = self.after_new_map
            if state == 'done':
                return self.go(scan, f'stairs: {why}; new map, scanning from the tile beyond the stairs')
            self.ctx.log('warn', 'Stairs: the new map could not be started; carrying on with the old one.')
            if c.get('scan_after', True):
                return self.go(scan, f'stairs: {why}; scanning to place the robot', after_stairs=beyond)
            return self.go(c.get('next_mode', 'exploring'), f'stairs: {why}')
        if self.phase == 'abort':
            return self._finish('no staircase to climb')
        if self.phase == 'ai_on':
            self.ctx.drive(0.0, 0.0)                 # FIX 2: zero /cmd_vel every tick while AI mode settles (stop() sends one)
            if self._ai_settled(True):
                if self.ai_failed:
                    return self._climb_over('AI mode did not come on')
                pose = self.ctx.pose()
                self.best = self._progress(pose) if pose is not None and len(pose) >= 3 else -math.inf
                self.best_at = self.ctx.now()
                self.xs = {'phase': 'to_stairs', 't': self.ctx.now(), 'down': False,   # FIX 3: IMU crossing state
                           # chosen once here: the whole crossing is followed on the IMU (the map pose is frozen in AI
                           # mode), or, with no IMU at all, on the map pose
                           'imu': bool(c.get('imu_cross', True)) and self._imu() is not None, 'imu_lost_at': None}
                self.set_phase('cross')
            return None
        if self.phase == 'cross' and getattr(self, 'xs', {}).get('imu'):
            if self._imu() is not None:
                self.xs['imu_lost_at'] = None
                return self._cross_imu()             # FIX 3: the map pose is frozen in AI mode: IMU only
            # A late IMU message is no reason to judge the climb on the frozen map pose (bag multiterrain_25_11_115442:
            # /imu/data at ~2 Hz, one stale tick fell through to the pose check, which saw no progress for stall_s and
            # called the robot stuck after it had crossed). Keep walking straight; give up only after imu_lost_s.
            now = self.ctx.now()
            lost_at = self.xs['imu_lost_at'] = self.xs['imu_lost_at'] if self.xs['imu_lost_at'] is not None else now
            if now - lost_at > float(c.get('imu_lost_s', 3.0)):
                over = self.xs['phase'] == 'clear'
                self._climb_over(f'no IMU for {now - lost_at:.1f} s in {self.xs["phase"]}'
                                 + (': over the staircase' if over else ''), through=over)
                return None
            self.ctx.drive(float(c.get('speed_mps', 0.20)), 0.0)
            return None
        if self.phase == 'ai_off':
            self.ctx.drive(0.0, 0.0)                 # FIX 2
            if self._ai_settled(False):
                why, through = self.result
                return self._finish(why, through=through)
            return None
        pose = self.ctx.pose()
        if pose is None or len(pose) < 3:
            if self.phase != 'cross':
                self.ctx.stop()
                return self._finish('no pose') if self.in_phase_s() > 3.0 else None
            # SLAM can lose the robot on the stairs: keep walking straight for a moment
            self.pose_lost_at = self.pose_lost_at if self.pose_lost_at is not None else self.ctx.now()
            if self.ctx.now() - self.pose_lost_at > float(c.get('pose_lost_s', 3.0)):
                self._climb_over('lost the robot pose on the stairs')
                return None
            self.ctx.drive(float(c.get('speed_mps', 0.20)), 0.0)
            return None
        self.pose_lost_at = None
        if self.phase == 'center':
            dx, dy = self.tile_centre[0] - pose[0], self.tile_centre[1] - pose[1]
            dist = math.hypot(dx, dy)
            tol = float(c.get('center_tol_m', 0.10))
            self.closest = dist if getattr(self, 'closest', None) is None else min(self.closest, dist)
            # the climb (align -> AI mode) starts only from the centre of the tile in front of the staircase: close
            # enough, or already closest and drifting off again (a legged robot cannot hold a few cm)
            if dist <= tol or (self.closest <= 2.0 * tol and dist > self.closest + float(c.get('center_drift_m', 0.04))):
                self.ctx.stop()
                self.set_phase('align')
                return None
            if self.in_phase_s() > float(c.get('center_timeout_s', 30.0)):
                # never climb from off-centre: give up before AI mode is on (back to the mode before)
                return self._finish(f'could not reach the tile centre ({dist:.2f} m off)')
            v = max(float(c.get('center_min_speed_mps', 0.05)),
                    min(float(c.get('center_speed_mps', 0.15)), float(c.get('center_gain', 1.0)) * dist))
            # no omni motion: turn towards the centre, then drive there with vx and wz only (the way a Nav2 goal is driven)
            bearing = self._yaw_error(pose, math.atan2(dy, dx))
            if abs(bearing) > math.radians(float(c.get('center_turn_deg', 30.0))):
                w = max(float(c.get('min_turn_rad_s', 0.2)), min(float(c.get('max_turn_rad_s', 0.4)),
                                                                  float(c.get('turn_gain', 1.0)) * abs(bearing)))
                self.ctx.drive(0.0, math.copysign(w, bearing))
            else:
                wz = max(-0.6, min(0.6, float(c.get('center_heading_gain', 1.5)) * bearing))
                self.ctx.drive(v * math.cos(bearing), wz)
            return None
        err = self._yaw_error(pose)
        if self.phase == 'still':
            # FIX 1: zero /cmd_vel every tick until the robot has really stopped turning (IMU gyro), THEN Enter AI.
            # The bags of 25 Nov: Enter AI went out mid-turn and the robot kept turning ~40 deg in AI mode.
            self.ctx.drive(0.0, 0.0)
            tol = math.radians(float(c.get('align_tol_deg', 5.0)) + float(c.get('still_drift_deg', 2.0)))
            if abs(err) > tol:
                self.set_phase('align')                  # it coasted off while stopping: square up again
                return None
            imu = self._imu()
            calm = imu is None or abs(imu[2]) < float(c.get('still_gyro_rad_s', 0.05))
            if self.in_phase_s() >= float(c.get('still_s', 1.5)) and calm:
                # FIX 3: the heading to hold on the stairs, in the IMU's own frame (the map pose freezes from here)
                self.imu_yaw_ref = None if imu is None else imu[1] + err
                self._ai(True)
                self.set_phase('ai_on')
            elif self.in_phase_s() > float(c.get('still_timeout_s', 6.0)):
                return self._finish('could not stand still before AI mode')
            return None
        if self.phase == 'align':
            if self.in_phase_s() > float(c.get('align_timeout_s', 15.0)):
                return self._finish(f'could not square up to it ({math.degrees(err):+.0f} deg off)')
            if self.turn_towards(err, float(c.get('align_tol_deg', 5.0))):
                self._predict_beyond()                   # squared up: only now predict the tile beyond the stairs
                self.set_phase('still')                  # FIX 1: stand still first, Enter AI only then
            return None
        if self.phase == 'cross':
            prog = self._progress(pose)
            if prog >= self.edge['through_m'] + float(c.get('extra_m', 0.0)):
                self._climb_over('over the staircase', through=True)
                return None
            if prog > self.best + float(c.get('stall_m', 0.05)):
                self.best, self.best_at = prog, self.ctx.now()
            if self.ctx.now() - self.best_at > float(c.get('stall_s', 10.0)):
                self._climb_over(f'stuck {prog:+.2f} m past the edge of the staircase')
                return None
            if self.in_phase_s() > float(c.get('cross_timeout_s', 60.0)):
                self._climb_over(f'timed out {prog:+.2f} m past the edge of the staircase')
                return None
            wz = max(-0.3, min(0.3, float(c.get('heading_gain', 1.0)) * err))
            nx, ny = self.edge['normal']
            lim = float(c.get('lateral_max_mps', 0.08))
            lat = max(-lim, min(lim, -float(c.get('lateral_gain', 0.5)) * self._lateral(pose)))
            v = float(c.get('speed_mps', 0.20))
            self._drive_map(pose, v * nx - lat * ny, v * ny + lat * nx, wz)
        return None

    # ---- FIX 3 / 4: the crossing in AI mode from the IMU (pitch + yaw), never from the frozen map pose
    def _held(self, cond, key, hold_s):
        """True once `cond` has been true for hold_s without a break."""
        now = self.ctx.now()
        if not cond:
            self.xs.pop('held_' + key, None)
            return False
        return now - self.xs.setdefault('held_' + key, now) >= hold_s

    def _cross_imu(self):
        """to_stairs (walk at it until the nose tilts) -> on_stairs (up; nose-down = going down) -> level again -> clear
        (a little further) -> AI mode off. Heading held on the IMU yaw. AI mode goes off only on level ground."""
        c = self.cfg
        nose, yaw, _ = self._imu()
        now, xs = self.ctx.now(), self.xs
        level = abs(nose) < float(c.get('level_deg', 4.0))
        if self.in_phase_s() > float(c.get('cross_timeout_s', 60.0)) and (level or self.in_phase_s() > 2.0 * float(
                c.get('cross_timeout_s', 60.0))):
            # 'clear': it went up, (down) and is level again beyond the stairs: the crossing is done, only the short walk
            # on has not finished. Bag stairs-rebasing_25_11_114524: timed out 0.1 s into that walk, counted as a
            # failure, so odom was never rebased and the robot thought it was still on the start tile.
            self._climb_over(f'timed out in {xs["phase"]} (nose {nose:+.0f} deg)'
                             + (': over the staircase' if xs['phase'] == 'clear' else ''),
                             through=xs['phase'] == 'clear')
            return None
        if xs['phase'] == 'to_stairs':
            if self._held(abs(nose) > float(c.get('climb_start_deg', 8.0)), 'tilt', 0.3):
                xs.update(phase='on_stairs', t=now)
                self.ctx.log('info', f'Stairs: on the stairs (nose {nose:+.0f} deg).')
            elif now - xs['t'] > float(c.get('reach_s', 10.0)):
                self._climb_over('no climb tilt: never reached the stairs')     # still on flat ground: AI off is safe
                return None
        elif xs['phase'] == 'on_stairs':
            if self._held(nose < -float(c.get('descend_deg', 12.0)), 'down', 0.3) and not xs['down']:
                xs['down'] = True
                self.ctx.log('info', f'Stairs: going down (nose {nose:+.0f} deg).')
            if xs['down']:
                done = self._held(nose > -float(c.get('level_deg', 4.0)), 'lvl', float(c.get('flat_after_down_s', 1.0)))
            else:            # never nose-down: only a long level spell, late enough
                done = (self._held(level, 'flat', float(c.get('flat_any_s', 3.0)))
                        and now - xs['t'] >= float(c.get('min_cross_s', 12.0)))
            if done:
                xs.update(phase='clear', t=now)
                self.ctx.log('info', 'Stairs: level again after the stairs.')
        elif xs['phase'] == 'clear' and now - xs['t'] >= float(c.get('clear_s', 2.0)):
            self._climb_over('over the staircase (level again)', through=True)
            return None
        nlog = xs.setdefault('nlog', [])
        nlog.append((now, nose))
        del nlog[:-400]
        speed = float(c.get('speed_mps', 0.20))
        if xs['phase'] in ('on_stairs', 'clear') and nose < float(c.get('level_deg', 4.0)):
            # level or going down (never while climbing: the risers ahead are in the scan band): a wall beyond the stairs.
            # Bag fullrun_25_11_183742: down the stairs, a wall ~0.5 m past the last step; level -> clear walked on at
            # 0.34 m/s for 2 s into it (the scan already showed the nose 2 cm off it), AI off pressed against the wall.
            r = self._wall_gap()
            if r is not None:
                gap, capture_nose = r
                stop, slow = float(c.get('wall_stop_m', 0.15)), float(c.get('wall_slow_m', 0.60))
                # stop only on a scan TAKEN level: nose down, the floor below the steps is in the scan band ahead
                if gap <= stop and level and abs(capture_nose) <= float(c.get('wall_capture_level_deg', 8.0)):
                    self._climb_over(f'over the staircase (wall {max(gap, 0.0):.2f} m ahead)', through=True)
                    return None
                if gap < slow:                         # nose still down: creep off the last step, then stop when level
                    creep = float(c.get('wall_creep_mps', 0.10))
                    speed = max(min(creep, speed), min(speed, speed * (gap - stop) / max(slow - stop, 1e-3)))
        ref = getattr(self, 'imu_yaw_ref', None)
        e = 0.0 if ref is None else math.atan2(math.sin(ref - yaw), math.cos(ref - yaw))
        lim = float(c.get('max_wz', 0.25))
        self._drive_logged(speed, max(-lim, min(lim, float(c.get('heading_gain', 1.0)) * e)))
        return None

    def _drive_logged(self, vx, wz):
        """ctx.drive, remembering (t, vx): _wall_gap works out how far the robot walked since a scan was taken."""
        self.ctx.drive(vx, wz)
        log = self.xs.setdefault('vlog', [])
        log.append((self.ctx.now(), float(vx)))
        del log[:-200]

    def _walked_since(self, t0):
        """Distance walked (commanded vx integrated) since time t0 (ctx.now() clock)."""
        log, now, d = self.xs.get('vlog', []), self.ctx.now(), 0.0
        for i, (t, v) in enumerate(log):
            t_end = log[i + 1][0] if i + 1 < len(log) else now
            lo = max(t, t0)
            if t_end > lo:
                d += v * (t_end - lo)
        return d

    def _wall_gap(self):
        """Gap (m) between the nose and the nearest scan return in the body's corridor ahead, as it is NOW: the scan is
        ~1 s old when it arrives (bag fullrun_25_11_183742) and the robot walked on since, so that walk is taken off.
        Closer than the scan's range_min the wall drops out of the scan (inf): an empty corridor right after a gap of
        ~0 counts as touching. Returns (gap_m, nose_deg when the scan was taken), or None (no scan: the check is off)."""
        f = getattr(self.ctx, 'front_gap', None)
        r = f(float(self.cfg.get('wall_half_width_m', 0.20)), float(self.cfg.get('wall_nose_m', 0.305))) \
            if f is not None else None
        if r is None or r[1] > float(self.cfg.get('wall_scan_max_age_s', 2.5)):
            return None
        raw, age = r
        now = self.ctx.now()
        taken = now - age
        if raw is None:
            touch = self.xs.get('wall_touch')
            if touch is not None and now - touch[0] < float(self.cfg.get('wall_touch_hold_s', 3.0)):
                return 0.0, touch[1]
            return None
        capture_nose = self._nose_at(taken)
        if raw < float(self.cfg.get('wall_touch_m', 0.08)):
            self.xs['wall_touch'] = (now, capture_nose)
        return raw - self._walked_since(taken), capture_nose

    def _nose_at(self, t):
        """IMU nose angle (deg) logged by the crossing closest to time t (the oldest one if t is before the log)."""
        log = self.xs.get('nlog') or [(t, 0.0)]
        for tt, n in reversed(log):
            if tt <= t:
                return n
        return log[0][1]

    def _release_odom(self):
        """odom_tf_bridge holds odom->base_link after AI mode until a rebase arrives (so slam_toolbox never matches a scan
        from beyond the stairs at the pre-climb pose). No rebase on this path: rebase to where the pose says the robot
        is, which only lets go of the hold."""
        pose = self.ctx.pose()
        if pose is not None and self.cfg.get('rebase_odom', True):
            self.ctx.rebase_odom(pose[0], pose[1], pose[2])

    def exit(self, next_mode, reason):
        self.ctx.stop()
        if self.ai_on:                                   # left mid-climb (operator command, shutdown): AI mode off
            self._ai(False)
            self._release_odom()


# ----------------------------------------------------------------------------------------- roller
@register_mode('roller')
class RollerMode(CurtainMode):
    """A roller tile shares a side with the current tile (or the robot stands in front of the roller on its tile): cross
    it in the h-step gait.

    Phases: center (turn towards start_xy, the centre of the tile in front of the roller, and drive there with vx / wz)
    -> align (turn in place until square to the roller: the robot faces across it) -> gait_on (/step_up true through
    ctx.hstep: the gait switcher sends rug_hstep; stand still gait_settle_s) -> cross (vx forward, wz holds the heading,
    vy the line through the tile centres) until the robot is at the centre of the tile beyond the roller -> gait_off
    (stop, h-step released, stand still gait_settle_s) -> next_mode (exploring). The roller tile is then marked explored
    (tiles.mark_roller_crossed: on the trail, tile report written) and the tile beyond counts as reached.

    A cross that stalls (stall_m in stall_s) backs off along the line to where it squared up (phase back), squares up
    again and crosses again, up to max_attempts tries. Given up before the robot got past the edge (could not square
    up): back to the mode that was running before; past it: fail_mode (recovery). The h-step gait is released whenever
    the mode is left. Each roller tile once from exploring (machine.rollers_tried).
    Backtracking uses it too (then=<mode>, crossing back over a roller crossed before): afterwards it hands back to that
    mode with roller_ok=True / False instead of next_mode (given up past the edge: still fail_mode)."""

    def targets(self):
        return [self.cfg.get('next_mode', 'exploring'), self.cfg.get('fail_mode', 'recovery')]

    @property
    def is_driving(self):
        return self.phase in ('center', 'cross', 'back', 'recenter')

    def enter(self, reason, crossing=None, then=None, then_kwargs=None, **kw):
        self.edge, self.then, self.then_kwargs = crossing, then, dict(then_kwargs or {})
        prev = self.machine.previous
        self.came_from = then or (prev if prev and prev != self.name else None) or self.cfg.get('next_mode', 'exploring')
        self.gait_on, self.attempts, self.pose_lost_at, self.result = False, 0, None, None
        self.recentred = False                                       # omni recentre at the edge after the roller tile
        self.restore_yaw = self._original_heading(crossing)
        self.ctx.set_target(None, None, None)
        if not crossing:
            self.set_phase('abort')
            return
        self.machine.rollers_tried.add(tuple(crossing['roller']))
        self.from_tile = tuple(crossing['from_tile'])
        self.tile_centre = tuple(crossing['start_xy'])               # where it squares up (the line through the centres)
        self.ctx.event('roller', side=crossing['side'], roller=T.tile_label(crossing['roller']),
                       beyond=T.tile_label(crossing['beyond']), spots=crossing['spots'], lidar=crossing['lidar'],
                       facing=crossing.get('facing'), through_m=crossing['through_m'])
        self._start_center()

    def _label(self):
        return T.tile_label(self.edge['roller']) if self.edge else '?'

    def _original_heading(self, crossing):
        """The yaw setpoint for the midpoint between the roller tile and the tile beyond: the mission's original heading
        (ctx.initial_heading), else square to the roller. None = off (restore_heading false), or when it points more than
        restore_max_deg away from the way across (backtracking over a roller the other way round: never walk it backwards)."""
        if not self.cfg.get('restore_heading', True) or not crossing:
            return None
        yaw = getattr(self.ctx, 'initial_heading', None)
        if yaw is None:
            return float(crossing['yaw'])
        d = abs(math.atan2(math.sin(yaw - crossing['yaw']), math.cos(yaw - crossing['yaw'])))
        return float(yaw) if d <= math.radians(float(self.cfg.get('restore_max_deg', 100.0))) else None

    def _yaw_error(self, pose, yaw=None):
        """The heading to hold: square to the roller, except on the approach (before the roller tile's edge line,
        approach_until_m) where the body stays approach_angle_deg off the roller plane while vx / vy (omni, in
        _hold_line) still walk the straight line through the tile centres. Past the edge it squares up to cross."""
        if yaw is None and self.restore_yaw is not None and (
                self.recentred or self.phase == 'recenter'
                or (self.phase == 'cross' and self._progress(pose) >= self.edge['through_m'] / 1.5)):
            yaw = self.restore_yaw                   # the midpoint between the roller tile and the next: original heading
        angle = float(self.cfg.get('approach_angle_deg', 0.0))
        if (yaw is None and angle and not self.recentred and self.phase in ('align', 'cross', 'back')
                and self._progress(pose) < float(self.cfg.get('approach_until_m', 0.0))):
            yaw = self.edge['yaw'] + math.radians(angle)
        return super()._yaw_error(pose, yaw)

    def _hstep(self, on):
        if on != self.gait_on:
            self.gait_on = on
            if hasattr(self.ctx, 'hstep'):
                self.ctx.hstep(on)
            self.ctx.log('info', f'Roller: h-step gait {"ON" if on else "released"}.')
            self.ctx.event('roller_gait', on=on)

    def _done(self, why, through=False):
        """Crossed (or given up after stepping on it): stop, release the h-step gait, stand still, then _finish."""
        self.ctx.stop()
        self.result = (why, through)
        self._hstep(False)
        self.set_phase('gait_off')

    def _finish(self, why, through=False):
        self.ctx.stop()
        self._hstep(False)
        tiles = self.ctx.tiles
        if through:
            roller, beyond = tuple(self.edge['roller']), tuple(self.edge['beyond'])
            tiles.mark_roller_crossed(roller, beyond, self.ctx.pose(), from_tile=self.from_tile)
            self.ctx.log('info', f'Roller: {why}; roller tile {self._label()} explored, now on tile {T.tile_label(beyond)}.')
            self.ctx.event('roller_crossed', roller=self._label(), beyond=T.tile_label(beyond), attempts=self.attempts + 1,
                           back_to=self.then)
            if self.then:
                return self.go(self.then, f'roller: {why}', roller_ok=True, **self.then_kwargs)
            return self.go(self.cfg.get('next_mode', 'exploring'), f'roller: {why}')
        pose = self.ctx.pose()
        past = pose is not None and len(pose) >= 3 and self._progress(pose) > 0.0
        back = self.cfg.get('fail_mode', 'recovery') if past else self.came_from
        self.ctx.log('warn', f'Roller: {why}; giving up on roller tile {self._label()}.')
        self.ctx.event('roller_failed', roller=self._label(), why=why, attempts=self.attempts, back_to=back)
        if back == self.then:
            return self.go(self.then, f'roller: {why}', roller_ok=False, **self.then_kwargs)
        return self.go(back, f'roller: {why}')

    def _stuck(self, why):
        """A crossing try did not get over the roller: back off and try again, or give up."""
        self.attempts += 1
        tries = int(self.cfg.get('max_attempts', 2))
        if self.attempts >= tries:
            return self._done(f'{why}; gave up after {self.attempts} tries')
        self.ctx.stop()
        self.ctx.log('warn', f'Roller: {why} (try {self.attempts}/{tries}); backing off to cross again.')
        self.ctx.event('roller_retry', roller=self._label(), why=why, attempt=self.attempts)
        self.recentred = False
        self.set_phase('back')
        return None

    def _turn(self, err):
        c = self.cfg
        w = max(float(c.get('min_turn_rad_s', 0.2)), min(float(c.get('max_turn_rad_s', 0.4)),
                                                          float(c.get('turn_gain', 1.0)) * abs(err)))
        self.ctx.drive(0.0, math.copysign(w, err))

    def _edge_waypoint(self):
        """Midpoint of the edge between the roller tile and the tile beyond it, on the line through the tile centres
        (the roller tile spans progress 0..size, the beyond tile's centre is through_m = 1.5 size)."""
        nx, ny = self.edge['normal']
        d = self.edge['through_m']            # start tile's centre (progress -size/2) to the edge (+size) = 1.5 size
        return (self.tile_centre[0] + d * nx, self.tile_centre[1] + d * ny)

    def _recenter(self, pose, err):
        """Omni (vx / vy / wz) towards the edge waypoint; True once within recenter_tol_m (or timed out)."""
        c = self.cfg
        wx, wy = self._edge_waypoint()
        dx, dy = wx - pose[0], wy - pose[1]
        nx, ny = self.edge['normal']
        along = dx * nx + dy * ny
        if along < 0.0:                              # already past the edge (the stop takes a step): fix the sideways
            dx, dy = dx - along * nx, dy - along * ny    # offset only, never back up
        dist = math.hypot(dx, dy)
        yaw_ok = self.restore_yaw is None or abs(err) <= math.radians(float(c.get('recenter_yaw_tol_deg', 5.0)))
        if dist <= float(c.get('recenter_tol_m', 0.06)) and yaw_ok:
            return True
        if self.in_phase_s() > float(c.get('recenter_timeout_s', 8.0)):
            self.ctx.log('warn', f'Roller: could not recentre / turn on the edge waypoint ({dist:.2f} m, '
                                 f'{math.degrees(err):+.0f} deg off); crossing on.')
            return True
        v = min(float(c.get('recenter_speed_mps', 0.08)), max(0.03, float(c.get('recenter_gain', 0.8)) * dist)) \
            if dist > float(c.get('recenter_tol_m', 0.06)) else 0.0
        wz = max(-0.3, min(0.3, float(c.get('heading_gain', 1.0)) * err))
        self._drive_map(pose, v * dx / max(dist, 1e-9), v * dy / max(dist, 1e-9), wz)
        return False

    def _hold_line(self, pose, v, err):
        """vx along the way across (negative: backwards), wz holding the heading, vy holding the centre line."""
        c = self.cfg
        wz = max(-0.3, min(0.3, float(c.get('heading_gain', 1.0)) * err))
        nx, ny = self.edge['normal']
        lim = float(c.get('lateral_max_mps', 0.08))
        lat = max(-lim, min(lim, -float(c.get('lateral_gain', 0.5)) * self._lateral(pose)))
        self._drive_map(pose, v * nx - lat * ny, v * ny + lat * nx, wz)

    def tick(self):
        c = self.cfg
        if self.phase == 'abort':
            return self._finish('no roller to cross')
        if self.phase in ('gait_on', 'gait_off'):
            self.ctx.stop()
            if self.in_phase_s() < float(c.get('gait_settle_s', 2.0)):
                return None
            if self.phase == 'gait_off':
                why, through = self.result
                return self._finish(why, through=through)
            pose = self.ctx.pose()
            self.best = self._progress(pose) if pose is not None and len(pose) >= 3 else -math.inf
            self.best_at = self.ctx.now()
            self.set_phase('cross')
            return None
        pose = self.ctx.pose()
        if pose is None or len(pose) < 3:
            if self.phase != 'cross':
                self.ctx.stop()
                if self.in_phase_s() > 3.0:
                    return self._done('no pose') if self.gait_on else self._finish('no pose')
                return None
            # keep walking straight over the roller for a moment
            self.pose_lost_at = self.pose_lost_at if self.pose_lost_at is not None else self.ctx.now()
            if self.ctx.now() - self.pose_lost_at > float(c.get('pose_lost_s', 3.0)):
                return self._done('lost the robot pose on the roller')
            self.ctx.drive(float(c.get('speed_mps', 0.20)), 0.0)
            return None
        self.pose_lost_at = None
        if self.phase == 'center':
            dx, dy = self.tile_centre[0] - pose[0], self.tile_centre[1] - pose[1]
            dist = math.hypot(dx, dy)
            tol = float(c.get('center_tol_m', 0.10))
            if dist <= tol or self.in_phase_s() > float(c.get('center_timeout_s', 15.0)):
                if dist > tol:
                    self.ctx.log('warn', f'Roller: could not reach the tile centre ({dist:.2f} m off); squaring up '
                                         f'from here.')
                self.ctx.stop()
                self.set_phase('align')
                return None
            v = max(float(c.get('center_min_speed_mps', 0.05)),
                    min(float(c.get('center_speed_mps', 0.15)), float(c.get('center_gain', 1.0)) * dist))
            bearing = self._yaw_error(pose, math.atan2(dy, dx))
            if abs(bearing) > math.radians(float(c.get('center_turn_deg', 30.0))):
                self._turn(bearing)
            else:
                wz = max(-0.6, min(0.6, float(c.get('center_heading_gain', 1.5)) * bearing))
                self.ctx.drive(v * math.cos(bearing), wz)
            return None
        err = self._yaw_error(pose)
        if self.phase == 'align':
            if self.in_phase_s() > float(c.get('align_timeout_s', 15.0)):
                why = f'could not square up to it ({math.degrees(err):+.0f} deg off)'
                return self._done(why) if self.gait_on else self._finish(why)
            if self.turn_towards(err, float(c.get('align_tol_deg', 4.0))):
                if self.gait_on:                         # squared up again after backing off: straight on
                    self.best, self.best_at = self._progress(pose), self.ctx.now()
                    self.set_phase('cross')
                else:
                    self._hstep(True)
                    self.set_phase('gait_on')
            return None
        if self.phase == 'back':
            start = self._progress(tuple(self.tile_centre) + (0.0,))      # where it squared up, past the edge line
            if self._progress(pose) <= start + 0.05 or self.in_phase_s() > float(c.get('back_timeout_s', 10.0)):
                self.ctx.stop()
                self.set_phase('align')
                return None
            self._hold_line(pose, -float(c.get('back_speed_mps', 0.12)), err)
            return None
        if self.phase == 'recenter':
            if self._recenter(pose, err):
                self.ctx.stop()
                self.recentred = True
                self.best, self.best_at = self._progress(pose), self.ctx.now()      # not a stall
                self.set_phase('cross')
            return None
        if self.phase == 'cross':
            prog = self._progress(pose)
            if prog >= self.edge['through_m'] + float(c.get('extra_m', 0.0)):
                return self._done('over the roller', through=True)
            if c.get('recenter', True) and not self.recentred and prog >= self.edge['through_m'] / 1.5:
                self.ctx.stop()                      # at the edge after the roller tile: recentre there first
                self.ctx.log('info', 'Roller: at the edge after the roller tile; recentring on its midpoint (omni).')
                self.ctx.event('roller_recenter', waypoint=[round(v, 3) for v in self._edge_waypoint()])
                self.set_phase('recenter')
                return None
            if prog > self.best + float(c.get('stall_m', 0.05)):
                self.best, self.best_at = prog, self.ctx.now()
            if self.ctx.now() - self.best_at > float(c.get('stall_s', 6.0)):
                return self._stuck(f'stuck {prog:+.2f} m past the edge of the roller tile')
            if self.in_phase_s() > float(c.get('cross_timeout_s', 40.0)):
                return self._stuck(f'timed out {prog:+.2f} m past the edge of the roller tile')
            self._hold_line(pose, float(c.get('speed_mps', 0.20)), err)
        return None

    def exit(self, next_mode, reason):
        self.ctx.stop()
        self._hstep(False)                                 # left mid-crossing (operator command, shutdown): release
        if next_mode != self.then:
            self.ctx.tiles.open_curtains = False           # backtracking lifted them; nobody hands back to it now


# ----------------------------------------------------------------------------------------- curtain scan
def curtain_scan_sides(tiles, tried=()):
    """Sides of the current tile a curtain could hide a tile behind: the neighbour is not visited, not closed (below
    a climbed staircase), not a staircase / roller tile and not already walked into through a curtain (tried). None on
    the initial tile while its edges hold no curtain (tiles.no_curtain_on_initial, tiles.not_curtain_mask)."""
    cur = tuple(tiles.current)
    initial = getattr(tiles, 'initial_tile', None) if getattr(tiles, 'no_curtain_on_initial', True) else None
    if initial is not None and tuple(initial) == cur:
        return []
    out = []
    for side, (di, dj) in CC.SIDE_STEPS.items():
        nb = (cur[0] + di, cur[1] + dj)
        status = tiles.tiles.get(nb, {}).get('status')
        if (status in ('visited', 'staircase', 'roller') or nb in tiles.trail or tiles.is_closed(nb) or nb in tried
                or (initial is not None and nb == tuple(initial))):
            continue
        out.append(side)
    return out


@register_mode('curtain_scan')
class CurtainScanMode(Mode):
    """No reachable tile from here (exploring): a hanging curtain is a wall to the 2D lidar, and the camera may never
    have looked at it (bag explorer_26_11_221940: boxed in on tile 0,-1 facing an OSB wall, the scan turn skipped for
    'no room to turn', the curtain detector saw nothing all run while the robot cycled retry / recovery for a minute).
    So face each edge of the current tile with an unvisited tile behind it (curtain_scan_sides), the smallest turn
    first, and stand still dwell_s there so the curtain detector gets frames (it needs confirm_hits per spot). Then back
    to next_mode (exploring), whose decision checks the current tile's edges for a curtain before anything else.

    Turns in place with ctx.drive wz (Mode.turn_towards), not Nav2 Spin: Spin and the scan's room check refuse in a
    1 m corridor, where this matters most. Phases: turning -> dwell, per side -> (next mode)."""

    def targets(self):
        return [self.cfg.get('next_mode', 'exploring')]

    def enter(self, reason, sides=None, **kw):
        tiles = self.ctx.tiles
        self.tile = tuple(tiles.current) if tiles.current is not None else None
        self.sides = list(sides) if sides is not None else (
            curtain_scan_sides(tiles, self.machine.curtains_tried) if self.tile is not None else [])
        self.side, self.yaw, self.looked = None, None, []
        self.ctx.set_target(None, None, None)
        self._next_side()

    def _side_yaw(self, side):
        g = self.ctx.tiles.grid
        di, dj = CC.SIDE_STEPS[side]
        (cx, cy), (nx, ny) = g.center(self.tile), g.center((self.tile[0] + di, self.tile[1] + dj))
        return math.atan2(ny - cy, nx - cx)

    def _next_side(self):
        """The side with the smallest turn from the current heading next, or 'done'."""
        pose = self.ctx.pose()
        if not self.sides or pose is None or self.ctx.tiles.grid is None:
            self.side = None
            self.set_phase('done')
            return
        self.side = min(self.sides, key=lambda s: abs(T.wrap(self._side_yaw(s) - pose[2])))
        self.sides.remove(self.side)
        self.yaw = self._side_yaw(self.side)
        self.set_phase('turning')

    def tick(self):
        c = self.cfg
        if self.phase == 'turning':
            pose = self.ctx.pose()
            if pose is None:
                return None
            if self.turn_towards(T.wrap(self.yaw - pose[2]), float(c.get('align_tol_deg', 10.0))):
                self.set_phase('dwell')
            elif self.in_phase_s() > float(c.get('turn_timeout_s', 12.0)):
                self.ctx.stop()
                self.ctx.log('warn', f'Curtain scan: the turn to the {self.side} edge did not finish; looking from here.')
                self.set_phase('dwell')
            return None
        if self.phase == 'dwell':
            if self.in_phase_s() < float(c.get('dwell_s', 3.0)):
                return None
            self.looked.append(self.side)
            self._next_side()
            return None
        if self.phase == 'done':
            here = T.tile_label(self.tile) if self.tile is not None else '?'
            self.ctx.event('curtain_scan_done', tile=here, looked=self.looked)
            return self.go(c.get('next_mode', 'exploring'),
                           f'curtain scan of tile {here} done ({", ".join(self.looked) or "no edge"})')
        return None

    def exit(self, next_mode, reason):
        if self.phase == 'turning':
            self.ctx.stop()


# ----------------------------------------------------------------------------------------- quickscan
@register_mode('quickscan')
class QuickscanMode(Mode):
    """Stand still on a tile edge and play a robot animation (the quickscan pose animation), then go on.

    Phases: settle -> playing -> (next mode). The animation_player node enters pose mode and, when the animation
    ends, move mode again; this mode only starts it and waits for it to finish.
    """

    def targets(self):
        return [self.cfg.get('next_mode', 'exploring')]

    def enter(self, reason, tile=None, **kw):
        self.tile = tile
        self.set_phase('settle')

    def tick(self):
        nxt = self.cfg.get('next_mode', 'exploring')
        if self.phase == 'settle':
            if self.in_phase_s() < float(self.cfg.get('settle_s', 1.0)):
                return None
            if not self.ctx.animation_play(self.cfg.get('animation', 'quickscan')):
                self.ctx.log('warn', 'quickscan: the animation could not be started; skipping it.')
                return self.go(nxt, 'quickscan skipped')
            self.set_phase('playing')
            return None
        if self.phase == 'playing':
            state = self.ctx.animation_state()
            if state == 'done':
                self.ctx.event('quickscan', tile=self.tile)
                return self.go(nxt, f'quickscan of tile {self.tile} done')
            if state == 'pending' and self.in_phase_s() > float(self.cfg.get('start_timeout_s', 3.0)):
                self.ctx.log('warn', 'quickscan: animation_player did not start the animation (is it running?); skipping.')
                self.ctx.animation_cancel()
                return self.go(nxt, 'quickscan skipped')
            if self.in_phase_s() > float(self.cfg.get('timeout_s', 30.0)):
                self.ctx.log('warn', 'quickscan: the animation timed out; cancelling it.')
                self.ctx.animation_cancel()
                return self.go(nxt, 'quickscan timed out')
        return None

    def exit(self, next_mode, reason):
        if self.phase == 'playing' and self.ctx.animation_state() != 'done':
            self.ctx.animation_cancel()           # leave pose mode if another mode takes over mid-animation


# ----------------------------------------------------------------------------------------- backtracking
@register_mode('backtracking')
class BacktrackingMode(Mode):
    """No exploration target left (tiles still unexplored): go back to the last split tile (a tile that had two or more tiles to travel to) and on to the option
    the robot has not travelled to yet, then return to exploring.

    Phases: to_split (drive back to the split tile) -> checking_paths (Nav2 path to each untravelled option) ->
    to_option (drive to the nearest reachable one) -> scan_mode / next_mode. When the split tile cannot be reached or
    none of its options has a path, the split is forgotten and exploring decides as usual.

    Curtains: while backtracking the virtual walls on curtain edges (edges the robot walked through a curtain on) are
    lifted (tiles.open_curtains). The way back follows moves the robot already made (tiles.known_route); when it crosses
    a curtain edge, Nav2 (which sees the curtain as a wall) only drives to the tile before it (to_curtain), the curtain
    mode walks through it and hands back here, and the rest of the way is planned from the far side. When the curtain
    mode gives up (curtain_ok=False) backtracking carries on: that crossing is left out (self.blocked) and Nav2 is asked
    for the way back to the split tile without it; only if that fails too is the split dropped.

    Rollers: a roller the robot crossed (tiles.roller_between) is one hop of the way back, from the tile on one side to
    the tile on the other. Nav2 (which sees the roller as a wall) drives to the tile before it (to_roller), the roller
    mode crosses it back in the h-step gait and hands back here (roller_ok), exactly like a curtain.

    The way back is driven through the tile CENTRES of that tile route: for each hop from one tile to the next,
    waypoints_per_tile waypoints evenly spaced on the straight line between the two centres, then the next centre
    Yaw setpoints per tile (align_at_tiles): the hops are driven one at a time; before each, at the tile centre, the
    robot turns in place (phase align, ctx.drive wz) until it faces the next tile within align_tol_deg, and every pose
    of the hop faces along it (align_at_tiles: false = the whole route in one go; a traversal that takes no route only
    gets the goal).
    """

    def __init__(self, name, machine, cfg):
        super().__init__(name, machine, cfg)
        self.split, self.target, self.traversal, self.options, self.why = None, None, None, [], ''
        self.curtain_options = []                 # split options behind a curtain left for later (tiles.deferred_curtains)
        self.crossing = None                      # (tile before, tile behind, 'curtain' | roller tile) on the way back
        self.hops, self.drive_phase, self.align_yaw = [], None, 0.0   # route hops still to drive, one at a time
        self.blocked = set()                      # crossings the curtain / roller mode did not get through (this backtrack)

    def targets(self):
        out = [self.cfg.get('next_mode', 'exploring'), self.cfg.get('scan_mode', 'scanning')]
        if self.cfg.get('stairs_mode', 'stairs') in self.machine.modes:
            out.append(self.cfg.get('stairs_mode', 'stairs'))
        if self.cfg.get('curtain_mode', 'curtain') in self.machine.modes:
            out.append(self.cfg.get('curtain_mode', 'curtain'))
        if self.cfg.get('roller_mode', 'roller') in self.machine.modes:
            out.append(self.cfg.get('roller_mode', 'roller'))
        return out

    @property
    def is_driving(self):
        return self.phase in ('to_split', 'to_curtain', 'to_roller', 'to_option')

    @property
    def target_label(self):
        return self.target['tile'] if self.target else None

    def _label(self):
        return T.tile_label(self.split) if self.split else '?'

    def enter(self, reason, split=None, curtain_ok=None, roller_ok=None, **kw):
        tiles = self.ctx.tiles
        self.target, self.traversal, self.options, self.why = None, None, [], ''
        tiles.open_curtains = True                 # curtain edges are no virtual walls while backtracking
        crossing, self.crossing = self.crossing, None
        if False in (curtain_ok, roller_ok) and crossing is not None:   # the curtain / roller mode did not get through
            self.blocked.add(crossing[:2])
            what = 'curtain' if crossing[2] == 'curtain' else f'roller on tile {T.tile_label(crossing[2])}'
            self.ctx.log('warn', f'Backtracking: could not get through the {what} between tiles '
                                 f'{T.tile_label(crossing[0])} and {T.tile_label(crossing[1])}; looking for another way.')
        elif curtain_ok is None and roller_ok is None:
            self.blocked = set()                   # a fresh backtrack (not a hand-back from the curtain mode)
        entry = None
        if tiles.ready:
            entry = (next((s for s in tiles.splits if T.tile_label(s['tile']) == split), None) if split
                     else tiles.last_split(exclude=tiles.current))
        if entry is None or not tiles.split_remaining(entry, include_unreachable=True):
            self.split = None
            return self._abort('no split tile with an untravelled option')
        self.split = entry['tile']
        self.ctx.event('backtracking', split=self._label(), remaining=tiles.split_remaining(entry, include_unreachable=True))
        pose = self.ctx.pose()
        here = tuple(tiles.grid.tile_of(pose[0], pose[1])) if pose is not None else tiles.current
        if here == self.split:
            self._check_options()
            return
        self._head_back(here)

    def _goal(self, tile):
        tiles = self.ctx.tiles
        return tiles.goal_point(tile, float(self.cfg.get('min_clearance_m', 0.25))) \
            or [round(v, 3) for v in tiles.grid.center(tile)]

    def _head_back(self, here):
        """Drive towards the split tile through the tile centres of the way back: all the way, or first to the tile
        before a curtain on the way."""
        tiles = self.ctx.tiles
        route = tiles.known_route(here, self.split) if here is not None else None
        for k, (a, b) in enumerate(zip(route or [], (route or [])[1:])):
            kind = 'curtain' if tiles.is_curtain_edge(a, b) else tiles.roller_between(a, b)
            if kind is None or (a, b) in self.blocked:
                continue
            self.crossing = (a, b, kind)
            what = 'curtain' if kind == 'curtain' else f'roller on tile {T.tile_label(kind)}'
            self.ctx.log('info', f'Backtracking: the way back to split tile {self._label()} goes through the {what} '
                                 f'between tiles {T.tile_label(a)} and {T.tile_label(b)}.')
            if a == here:
                self.set_phase('cross')
                return
            # the tile centre, facing the curtain / roller: the curtain / roller mode starts from there (it re-centres
            # anyway, Nav2's goal tolerance is loose)
            towards = b if kind == 'curtain' else kind
            edge = CC.edge_crossing(tiles.grid, a, T.side_towards(a, towards))
            self._start({'tile': T.tile_label(a), 'goal_xy': [round(v, 3) for v in tiles.grid.center(a)],
                         'yaw': edge['yaw']}, 'to_curtain' if kind == 'curtain' else 'to_roller', via=route[:k + 1])
            return
        if route and any(tiles.is_curtain_edge(a, b) or tiles.roller_between(a, b) for a, b in zip(route, route[1:])):
            route = None                           # the only known way crosses a blocked curtain / roller: let Nav2 find one
        self._start({'tile': self._label(), 'goal_xy': self._goal(self.split)}, 'to_split', via=route)

    def _centre_route(self, via, goal_xy, final_yaw):
        """Poses through the centres of the tiles `via` (first = the tile the robot is on): per hop
        waypoints_per_tile waypoints on the straight line between the two centres, then the next centre; the last pose
        is goal_xy with final_yaw. Yaw setpoints per tile: the waypoints of a hop face along it (towards the tile it
        goes to), a tile centre on the way faces the NEXT tile. None when there is no hop."""
        if not via or len(via) < 2:
            return None
        n = max(0, int(self.cfg.get('waypoints_per_tile', A.N_WAYPOINTS)))
        centres = [self.ctx.tiles.grid.center(t) for t in via]
        pts = []
        for (ax, ay), (bx, by) in zip(centres, centres[1:]):
            for i in range(1, n + 2):              # n waypoints, then the next centre
                f = i / (n + 1)
                pts.append((ax + f * (bx - ax), ay + f * (by - ay)))
        pts[-1] = (float(goal_xy[0]), float(goal_xy[1]))   # the goal (the centre, or the clear point nearest it)
        hop_yaws = [math.atan2(by - ay, bx - ax) for (ax, ay), (bx, by) in zip(centres, centres[1:])]
        route = []
        for i, (x, y) in enumerate(pts):
            h, k = divmod(i, n + 1)
            if i == len(pts) - 1:
                yaw = final_yaw
            elif k == n:                           # a tile centre on the way: face the next tile
                yaw = hop_yaws[h + 1]
            else:
                yaw = hop_yaws[h]
            route.append((round(x, 3), round(y, 3), yaw))
        return route

    def _next_hop(self):
        """Turn in place at the tile the robot is on until it faces the next tile (align), then drive that hop."""
        self.align_yaw = self.hops[0][0][2]
        self._realigned = False
        self.set_phase('align')

    def _drive_hop(self):
        hop = self.hops.pop(0)
        x, y, yaw = hop[-1]
        if hasattr(self.ctx, 'set_step_tile'):     # step up (h-step gait) per hop: this hop's tile, not only the target
            self.ctx.set_step_tile(T.tile_label(self.ctx.tiles.grid.tile_of(x, y)))
        if not self.traversal.start([x, y], yaw, hop):
            return self._abort(f'could not start driving to tile {self.target["tile"]}')
        self.set_phase(self.drive_phase)
        return None
    def _cross(self):
        """Hand over to the curtain mode for the curtain edge ahead (or the roller mode for the roller ahead); it comes
        back here (curtain_ok / roller_ok says how it went)."""
        a, b, kind = self.crossing
        if kind != 'curtain':
            return self._cross_roller(a, b, kind)
        mode = self.cfg.get('curtain_mode', 'curtain')
        if mode not in self.machine.modes:
            self.ctx.tiles.drop_split(self.split)
            self._abort('a curtain is in the way back but there is no curtain mode')
            return None
        edge = CC.edge_crossing(self.ctx.tiles.grid, a, T.side_towards(a, b))
        return self.go(mode, f'backtracking through the curtain into tile {T.tile_label(b)}', edge=edge,
                       then=self.name, then_kwargs={'split': self._label()})

    def _cross_roller(self, a, b, roller):
        tiles = self.ctx.tiles
        mode = self.cfg.get('roller_mode', 'roller')
        crossing = RC.crossing_between(tiles.grid, a, roller, tiles.rollers.get(tuple(roller)))
        if mode not in self.machine.modes or crossing is None or tuple(crossing['beyond']) != tuple(b):
            self.blocked.add((a, b))
            self.ctx.log('warn', f'Backtracking: cannot cross the roller on tile {T.tile_label(roller)} back; looking '
                                 f'for another way.')
            self._head_back(a)
            return None
        return self.go(mode, f'backtracking over the roller on tile {T.tile_label(roller)} into tile {T.tile_label(b)}',
                       crossing=crossing, then=self.name, then_kwargs={'split': self._label()})

    def _through_deferred_curtain(self):
        """At the split tile, the other ways explored: walk through the curtain that was left for later. Through it:
        the curtain mode goes on to exploring; given up: back here, where that option no longer counts."""
        tiles = self.ctx.tiles
        label = self.curtain_options[0]
        behind = tuple(int(v) for v in label.split(','))
        info = tiles.deferred_curtains.pop(behind, None)
        mode = self.cfg.get('curtain_mode', 'curtain')
        if info is None or mode not in self.machine.modes:
            self._abort(f'cannot walk through the curtain into tile {label}')
            return None
        edge = CC.edge_crossing(tiles.grid, self.split, info['side'])
        self.ctx.log('info', f'Backtracking: at split tile {self._label()}; walking through the curtain left for later '
                             f'into tile {label}.')
        self.ctx.event('backtracked', split=self._label(), tile=label, curtain=True)
        return self.go(mode, f'backtracking: through the curtain into tile {label}', edge=edge)

    def _abort(self, why):
        self.why = why
        self.set_phase('abort')

    def _start(self, target, phase, via=None):
        try:
            trav = self.ctx.traversal(self.cfg.get('traversal', 'direct'))
        except KeyError as exc:
            self.ctx.log('error', str(exc))
            return self._abort('bad traversal profile')
        if not trav.available():
            return self._abort('traversal not available (Nav2 down?)')
        pose = self.ctx.pose() or [0.0, 0.0, 0.0]
        gx, gy = target['goal_xy'][:2]
        grid_yaw = self.ctx.tiles.grid.yaw
        yaw = target.get('yaw', A.snap_to_axis(math.atan2(gy - pose[1], gx - pose[0]), grid_yaw))
        if phase == 'to_option' and self.cfg.get('face_away_from_walls', True):
            sides = self.ctx.tiles.tiles.get(tuple(int(v) for v in target['tile'].split(',')), {}).get('sides')
            yaw = A.yaw_away_from_walls(sides, grid_yaw, yaw)
        route = self._centre_route(via, (gx, gy), yaw) if trav.uses_route else None
        self.target, self.traversal, self.drive_phase, self.hops = target, trav, phase, []
        k = max(0, int(self.cfg.get('waypoints_per_tile', A.N_WAYPOINTS))) + 1
        if route and self.cfg.get('align_at_tiles', True):
            self.hops = [route[i:i + k] for i in range(0, len(route), k)]   # one tile-to-tile hop at a time
        elif not trav.start([gx, gy], yaw, route):
            return self._abort(f'could not start driving to tile {target["tile"]}')
        self.ctx.set_target(target['tile'], [gx, gy], route)
        where = {'to_split': 'back to the split tile', 'to_curtain': 'to the tile before the curtain',
                 'to_roller': 'to the tile before the roller'}.get(
            phase, 'to the untravelled tile')
        how = (f' through the centres of {" -> ".join(T.tile_label(t) for t in via)} ({len(route)} poses)'
               if route else '')
        self.ctx.log('info', f'Backtracking: driving {where} {target["tile"]}{how}.')
        self.ctx.event('backtracking_route', phase=phase, tile=target['tile'],
                       via=[T.tile_label(t) for t in via] if route else None, route=A.route_json(route) if route else None)
        if self.hops:
            return self._next_hop()
        self.set_phase(phase)

    def _to_stairs(self, label):
        """At the split tile in front of a forced staircase (ExploringMode.stairs_backtrack): start the stairs mode
        from here. The split is dropped either way (the stairs mode goes for a staircase once)."""
        tiles = self.ctx.tiles
        tiles.drop_split(self.split)
        mode = self.cfg.get('stairs_mode', 'stairs')
        crossing = SC.stair_crossing(tiles.grid, self.split, tiles.stairs,
                                     skip=self.machine.stairs_tried | tiles.stairs_crossed,
                                     min_facing=float(self.cfg.get('stair_min_facing', 0.5)))
        if mode not in self.machine.modes or crossing is None:
            return self._abort(f'cannot climb the staircase on tile {label} from split tile {self._label()}')
        self.ctx.log('info', f'Backtracking: at split tile {self._label()} in front of the staircase {label}: '
                             f'starting the stairs mode.')
        self.ctx.event('backtracked', split=self._label(), tile=label, stairs=True)
        self.stairs_crossing = crossing
        self.set_phase('to_stairs_mode')
        return None

    def _option_via(self, option):
        """Split tile -> untravelled option: the straight line between their centres."""
        return [self.split, tuple(int(v) for v in option['tile'].split(','))]

    def _check_options(self):
        """At the split tile: Nav2 path to each option not travelled to yet."""
        tiles = self.ctx.tiles
        entry = next((s for s in tiles.splits if s['tile'] == self.split), None)
        remaining = tiles.split_remaining(entry, include_unreachable=True) if entry else []
        stairs = [lab for lab in remaining if tuple(int(v) for v in lab.split(',')) in tiles.stairs]
        if stairs:
            return self._to_stairs(stairs[0])
        restored = [lab for lab in remaining if lab in tiles.unreachable]
        for lab in restored:                       # given up on from elsewhere: try again from the split tile
            tiles.unreachable.pop(lab, None)
            tiles.failures.pop(lab, None)
        if restored:
            tiles.invalidate()
            tiles.update()
            self.ctx.log('info', f'Backtracking: retrying tile(s) {", ".join(restored)} from split tile {self._label()}.')
        clearance = float(self.cfg.get('min_clearance_m', 0.25))
        self.options, self.curtain_options = [], []
        for label in remaining:
            behind = tiles.deferred_curtain(label)
            if behind is not None and behind['tile'] == self.split:
                self.curtain_options.append(label)  # walked through with the curtain mode, not driven to with Nav2
                continue
            goal = tiles.goal_point(tuple(int(v) for v in label.split(',')), clearance)
            if goal is not None:
                self.options.append({'tile': label, 'goal_xy': goal})
        if not self.options:
            if self.curtain_options:
                self.set_phase('through_curtain')
                return None
            tiles.drop_split(self.split)
            return self._abort(f'split tile {self._label()} has no untravelled option left')
        if not self.ctx.paths.start([o['goal_xy'] for o in self.options]):
            self.options.sort(key=lambda o: math.dist(o['goal_xy'], (self.ctx.pose() or [0, 0])[:2]))
            return self._start(self.options[0], 'to_option', via=self._option_via(self.options[0]))   # nearest first
        self.set_phase('checking_paths')

    def tick(self):
        c = self.cfg
        if self.phase == 'abort':
            self.ctx.log('info', f'Backtracking: {self.why}; back to exploring.')
            return self.go(c.get('next_mode', 'exploring'), f'backtracking: {self.why}')
        if self.phase == 'cross':
            return self._cross()
        if self.phase == 'through_curtain':
            return self._through_deferred_curtain()
        if self.phase == 'to_stairs_mode':
            crossing, self.stairs_crossing = self.stairs_crossing, None
            return self.go(c.get('stairs_mode', 'stairs'),
                           f'backtracked to tile {self._label()} in front of the staircase', crossing=crossing)
        if self.phase == 'align':
            pose = self.ctx.pose()
            err = A.wrap(self.align_yaw - pose[2]) if pose is not None and len(pose) >= 3 else 0.0
            if self.in_phase_s() > float(c.get('align_timeout_s', 15.0)):
                self.ctx.stop()
                self.ctx.log('warn', f'Backtracking: could not turn to face the next tile ({math.degrees(err):+.0f} deg '
                                     f'off); driving on.')
                return self._drive_hop()
            if self.turn_towards(err, float(c.get('align_tol_deg', 5.0))):
                self.set_phase('align_settle')
            return None
        if self.phase == 'align_settle':
            # stand still before Nav2 drives the hop: it used to take over mid-coast and the robot swung 25-70 deg past
            # the hop heading (bag fullrun_26_11_210206, 92 s and 124 s)
            self.ctx.drive(0.0, 0.0)
            if self.in_phase_s() < float(c.get('align_settle_s', 0.6)):
                return None
            pose = self.ctx.pose()
            err = A.wrap(self.align_yaw - pose[2]) if pose is not None and len(pose) >= 3 else 0.0
            tol = float(c.get('align_tol_deg', 5.0)) + float(c.get('still_drift_deg', 5.0))
            if abs(err) > math.radians(tol) and not getattr(self, '_realigned', False):
                self._realigned = True                   # coasted off: square up once more (once per hop)
                self.set_phase('align')
                return None
            self.ctx.stop()
            return self._drive_hop()
        if self.phase == 'checking_paths':
            results = self.ctx.paths.poll()
            if results is None and self.in_phase_s() <= float(c.get('path_check_timeout_s', 10.0)):
                return None
            if results is None:
                self.ctx.paths.cancel()
                results = []
            from lite3_explorer.mission.exploration import path_length
            ok = [(path_length([tuple(p[:2]) for p in r]), o) for o, r in zip(self.options, results) if r]
            if not ok and self.curtain_options:
                return self._through_deferred_curtain()
            if not ok:
                self.ctx.tiles.drop_split(self.split)
                self._abort(f'no Nav2 path to the untravelled tile(s) of split tile {self._label()}')
                return None
            ok.sort(key=lambda x: x[0])
            self._start(ok[0][1], 'to_option', via=self._option_via(ok[0][1]))
            return None
        if self.phase in ('to_split', 'to_curtain', 'to_roller', 'to_option'):
            tiles = self.ctx.tiles
            if self.phase == 'to_option' and tiles.skip_scene and self.target is not None \
                    and tiles.is_scene(tuple(int(v) for v in self.target['tile'].split(','))):
                self.traversal.cancel()                           # turned out to be a scene tile on the way: never entered
                self.ctx.event('target_dropped', label=self.target['tile'], why='scene tile')
                self._abort(f'tile {self.target["tile"]} turned out to be a scene tile')
                return None
            res = self.traversal.poll()
            if not res.done:
                if self.in_phase_s() > float(c.get('goal_timeout_s', 120.0)):
                    self.traversal.cancel()
                    res = TraversalResult(FAILED, 'timed out')
                else:
                    return None
            if res.state != SUCCEEDED:
                self.ctx.log('warn', f'Backtracking to tile {self.target["tile"]} failed: {res.reason}.')
                self.ctx.event('backtracking_failed', split=self._label(), tile=self.target['tile'], reason=res.reason)
                if self.phase == 'to_option':
                    self.ctx.tiles.record_failure(self.target['tile'], f'backtracking: {res.reason}')
                self.ctx.tiles.drop_split(self.split)              # forget the split: never backtrack there in a loop
                self._abort(f'could not reach tile {self.target["tile"]}')
                return None
            if self.hops:                                         # at a tile centre on the way: face the next tile
                self._next_hop()
                return None
            if self.phase in ('to_curtain', 'to_roller'):
                self.traversal, self.target = None, None
                self.ctx.set_target(None, None, None)
                return self._cross()
            if self.phase == 'to_split':
                self.traversal, self.target = None, None
                self.ctx.set_target(None, None, None)
                self._check_options()
                return None
            t = self.target
            self.ctx.tiles.mark_goal_reached(tuple(int(v) for v in t['tile'].split(',')), t['goal_xy'], self.ctx.pose())
            self.ctx.event('backtracked', split=self._label(), tile=t['tile'])
            why = f'backtracked from split tile {self._label()} to tile {t["tile"]}'
            if c.get('scan_after_reach', True):
                return self.go(c.get('scan_mode', 'scanning'), why)       # scanning hands over to exploring
            return self.go(c.get('next_mode', 'exploring'), why)
        return None

    def exit(self, next_mode, reason):
        if self.phase == 'checking_paths':
            self.ctx.paths.cancel()
        if self.phase in ('align', 'align_settle'):
            self.ctx.stop()
        self.hops = []
        if self.phase in ('to_split', 'to_curtain', 'to_roller', 'to_option') and self.traversal is not None:
            self.traversal.cancel()
        if next_mode not in (self.cfg.get('curtain_mode', 'curtain'), self.cfg.get('roller_mode', 'roller')):
            self.ctx.tiles.open_curtains = False       # the curtain mode hands back here: keep them open meanwhile
        self.target, self.traversal = None, None
        self.ctx.set_target(None, None, None)


# ----------------------------------------------------------------------------------------- recovery
@register_mode('recovery')
class RecoveryMode(Mode):
    """Stuck: move to the centre of the nearest previously explored tile (one the robot has stood on). Phases: clearing -> driving (candidate by candidate) -> next mode.

    Each candidate is driven to with `traversal` (default recovery_mppi: Nav2 with the holonomic RecoveryMPPI controller,
    vx / vy / wz). When that fails, `fallback_traversal` (default omni: straight linear x / y velocity, heading held, no
    Nav2) takes the SAME tile centre before the next candidate is tried ('' = no fallback)."""

    def __init__(self, name, machine, cfg):
        super().__init__(name, machine, cfg)
        self.candidates, self.target, self.traversal, self.exclude = [], None, None, []
        self.profile = None                       # traversal profile driving the current candidate
        self.return_to, self.then_mode, self.then_kwargs = None, None, {}   # scene found while driving: back to a tile, then a mode

    def targets(self):
        out = [self.cfg.get('next_mode', 'scanning'), self.cfg.get('fail_mode', 'scanning'), self.machine.complete_mode]
        scene_mode = self.cfg.get('scene_mode', 'scene')
        if scene_mode in self.machine.modes:
            out.append(scene_mode)                # exploring: a scene tile found while driving (return_to / then_mode)
        return out

    @property
    def is_driving(self):
        return self.phase == 'driving'

    @property
    def target_label(self):
        return self.target['tile'] if self.target else None

    def enter(self, reason, exclude=(), blocked=(), return_to=None, then_mode=None, then_kwargs=None, **kw):
        # never recover INTO the tile we got stuck on, nor into tiles Nav2 just failed to plan to
        self.exclude = [e for e in list(exclude) + list(blocked) if e]
        self.candidates, self.target, self.traversal = [], None, None
        self.return_to, self.then_mode, self.then_kwargs = return_to, then_mode, dict(then_kwargs or {})
        tiles = self.ctx.tiles
        if then_mode:                             # a planned return, not a failure: no give-up counting
            if self.cfg.get('clear_costmaps', True):
                self.ctx.clear_costmaps()
            self.set_phase('clearing')
            return
        self.machine.recoveries += 1
        give_up = int(self.cfg.get('give_up_after', 3))
        if give_up and self.machine.recoveries > give_up:
            # Recovered give_up_after times in a row without reaching a new tile: the tiles we keep failing on
            # are out of reach. Give up on them so the end condition can finish the mission.
            doomed = [b for b in (list(blocked) or self.exclude) if b] or tiles.unexplored()
            for label in doomed:
                tiles.mark_unreachable(label, f'still unreachable after {give_up} recoveries')
            self.ctx.log('warn', f'{give_up} recoveries without progress: giving up on tile(s) {", ".join(doomed)}.')
            self.ctx.event('unreachable', tiles=doomed, why='recoveries exhausted')
            self.machine.recoveries = 0
            self.set_phase('give_up')
            return
        if self.cfg.get('clear_costmaps', True):
            self.ctx.clear_costmaps()
        self.set_phase('clearing')

    def tick(self):
        if self.phase == 'give_up':
            return self._leave(self.cfg.get('fail_mode', 'scanning'), 'gave up on unreachable tiles')
        if not self.then_mode and not self.ctx.tiles.unexplored():
            return self._leave(self.cfg.get('next_mode', 'scanning'), 'no unexplored tile left')
        if self.phase == 'clearing':
            if self.in_phase_s() < float(self.cfg.get('clear_settle_s', 1.0)):
                return None
            pose = self.ctx.pose()
            if pose is None:
                return None
            if self.return_to:
                # the tile the robot was just in, and only that one (min_distance 0: standing on it already is fine)
                found = self.ctx.tiles.recovery_targets(
                    pose[:2], limit=50, min_distance_m=0.0,
                    min_clearance_m=float(self.cfg.get('min_clearance_m', 0.25)), exclude=self.exclude)
                self.candidates = [c for c in found if c['tile'] == self.return_to]
                self.ctx.event('recovery', candidates=[c['tile'] for c in self.candidates], exclude=self.exclude,
                               return_to=self.return_to)
                return self._next_candidate()
            # always previously explored tiles (the robot stood on them), nearest first
            self.candidates = self.ctx.tiles.recovery_targets(
                pose[:2], limit=int(self.cfg.get('candidates', 3)), min_distance_m=float(self.cfg.get('min_distance_m', 0.25)),
                min_clearance_m=float(self.cfg.get('min_clearance_m', 0.25)), exclude=self.exclude,
                adjacent_only=bool(self.cfg.get('adjacent_only', True)))
            self.ctx.event('recovery', candidates=[c['tile'] for c in self.candidates], exclude=self.exclude)
            return self._next_candidate()
        if self.phase == 'wait_traversal':
            if self.traversal.available():
                return self._drive(self.profile)
            if self.in_phase_s() > float(self.cfg.get('attempt_timeout_s', 45.0)):
                return self._leave(self.cfg.get('fail_mode', 'scanning'), 'traversal not available')
            return None
        if self.phase == 'driving':
            if self._target_became_scene():
                # the candidate passed the scene test when it was picked, but the scan on the way finished it (bag
                # reporter3_25_11_180333: 1,-1 at free/known 0.88 when picked, 0.84 < scene_max_free 5 s later, the omni
                # fallback drove on into it and, walked on, it never became a scene tile). Never recover INTO a scene
                # tile: stop, no fallback onto it, next candidate. The scene mode looks at it from a neighbour later.
                self.traversal.cancel()
                label = self.target['tile']
                self.ctx.log('warn', f'Recovery target tile {label} turned out to be a scene tile: not entering it.')
                self.ctx.event('target_dropped', label=label, why='scene tile', mode='recovery')
                return self._next_candidate()
            res = self.traversal.poll()
            if not res.done:
                if self.in_phase_s() > float(self.cfg.get('attempt_timeout_s', 45.0)):
                    self.traversal.cancel()
                    self.ctx.log('warn', f'Recovery to tile {self.target["tile"]} ({self.profile}) timed out.')
                    return self._fallback_or_next('timed out')
                return None
            if res.state == SUCCEEDED:
                self.ctx.event('recovered', tile=self.target['tile'], traversal=self.profile)
                return self._leave(self.cfg.get('next_mode', 'scanning'),
                                   f'recovered at the centre of tile {self.target["tile"]} ({self.profile})')
            self.ctx.log('warn', f'Recovery to tile {self.target["tile"]} ({self.profile}) failed: {res.reason}.')
            return self._fallback_or_next(res.reason)
        return None

    def _target_became_scene(self) -> bool:
        tiles = self.ctx.tiles
        if self.target is None or not tiles.skip_scene:
            return False
        return tiles.is_scene(tuple(int(v) for v in self.target['tile'].split(',')))

    def _fallback_or_next(self, why):
        """The main traversal failed: the omni x / y fallback tries the same tile centre, then the next candidate."""
        fallback = self.cfg.get('fallback_traversal', 'omni')
        if fallback and self.profile != fallback and self.target is not None:
            self.ctx.event('recovery_fallback', tile=self.target['tile'], failed=self.profile, why=why, traversal=fallback)
            return self._drive(fallback)
        return self._next_candidate()

    def _next_candidate(self):
        if not self.candidates:
            return self._leave(self.cfg.get('fail_mode', 'scanning'), 'no tile centre could be reached')
        self.target = self.candidates.pop(0)
        return self._drive(self.cfg.get('traversal', 'recovery_mppi'))

    def _drive(self, profile):
        """Drive to the current candidate's centre with this traversal profile."""
        self.profile = profile
        try:
            self.traversal = self.ctx.traversal(profile)
        except KeyError as exc:
            self.ctx.log('error', str(exc))
            return self._leave(self.cfg.get('fail_mode', 'scanning'), 'bad traversal profile')
        pose = self.ctx.pose() or [0.0, 0.0, 0.0]
        gx, gy = self.target['goal_xy']
        grid_yaw = self.ctx.tiles.grid.yaw if self.ctx.tiles.grid is not None else 0.0
        yaw = A.snap_to_axis(math.atan2(gy - pose[1], gx - pose[0]), grid_yaw)
        if not self.traversal.available():
            self.ctx.log('warn', f'Recovery traversal {profile!r} not available yet (Nav2 down?): waiting.')
            self.set_phase('wait_traversal')
            return None
        self.ctx.log('info', f'Recovery: driving to the centre of tile {self.target["tile"]} '
                             f'({self.target["distance_m"]:.2f} m away, {self.target["status"]}) with {profile}.')
        if not self.traversal.start(self.target['goal_xy'], yaw, None):
            return self._fallback_or_next('could not start')
        self.ctx.set_target(self.target['tile'], self.target['goal_xy'], None)
        self.set_phase('driving')
        return None

    def _leave(self, mode, reason):
        if self.then_mode:                        # on to the scene mode whatever happened: it centres on the tile it stands on
            return self.go(self.then_mode, reason, **self.then_kwargs)
        if mode == self.machine.complete_mode or not self.ctx.tiles.unexplored():
            end = self.machine.end_hop()
            if end is not None:
                return end
        return self.go(mode, reason)

    def exit(self, next_mode, reason):
        if self.phase == 'driving' and self.traversal is not None:
            self.traversal.cancel()
        self.target, self.traversal = None, None
        self.ctx.set_target(None, None, None)


# ----------------------------------------------------------------------------------------- ending
@register_mode('ending')
class EndingMode(Mode):
    """The end condition ending_tile found the ending tile: drive to its centre (`traversal`, then `fallback_traversal`
    straight to the same centre when that fails or stops off-centre), then the end condition is checked and the mission
    completes. Not reached: rescan (fail_mode) and exploring hands over here again; after give_up_after failed visits
    (0 = never) the mission completes anyway, saying so."""

    def __init__(self, name, machine, cfg):
        super().__init__(name, machine, cfg)
        self.tile, self.traversal, self.profile = None, None, None

    def targets(self):
        return [self.machine.complete_mode, self.cfg.get('fail_mode', 'scanning')]

    @property
    def is_driving(self):
        return self.phase == 'driving'

    @property
    def target_label(self):
        return T.tile_label(self.tile) if self.tile is not None else None

    def enter(self, reason, tile=None, **kw):
        self.tile = tuple(tile) if tile is not None else None
        self.traversal, self.profile = None, None
        if self.tile is None or not self.ctx.tiles.ready:
            self.set_phase('no_tile')
            return
        if self.cfg.get('clear_costmaps', False):
            self.ctx.clear_costmaps()
        self._drive(self.cfg.get('traversal', 'direct'))

    def _goal(self):
        return list(self.ctx.tiles.grid.center(self.tile))

    def _drive(self, profile):
        self.profile = profile
        try:
            self.traversal = self.ctx.traversal(profile)
        except KeyError as exc:
            self.ctx.log('error', str(exc))
            return self._fail('bad traversal profile')
        if not self.traversal.available():
            self.ctx.log('warn', f'Ending traversal {profile!r} not available yet (Nav2 down?): waiting.')
            self.set_phase('wait_traversal')
            return None
        pose = self.ctx.pose() or [0.0, 0.0, 0.0]
        gx, gy = self._goal()
        yaw = A.snap_to_axis(math.atan2(gy - pose[1], gx - pose[0]), self.ctx.tiles.grid.yaw)
        self.ctx.log('info', f'Ending: driving to the centre of ending tile {self.target_label} with {profile}.')
        if not self.traversal.start([gx, gy], yaw, None):
            return self._fallback_or_fail('could not start')
        self.ctx.set_target(self.target_label, [gx, gy], None)
        self.set_phase('driving')
        return None

    def tick(self):
        if self.phase == 'no_tile':
            return self.go(self.cfg.get('fail_mode', 'scanning'), 'no ending tile to drive to')
        if self.phase == 'wait_traversal':
            if self.traversal.available():
                return self._drive(self.profile)
            if self.in_phase_s() > float(self.cfg.get('attempt_timeout_s', 60.0)):
                return self._fail('traversal not available')
            return None
        if self.phase == 'driving':
            res = self.traversal.poll()
            if not res.done:
                if self.in_phase_s() > float(self.cfg.get('attempt_timeout_s', 60.0)):
                    self.traversal.cancel()
                    return self._fallback_or_fail('timed out')
                return None
            if res.state == SUCCEEDED:
                # Nav2 stops within its goal tolerance (0.3-0.4 m): off by more than centre_tolerance_m, the straight
                # fallback (omni) finishes the move to the centre
                fallback = self.cfg.get('fallback_traversal', 'omni')
                pose = self.ctx.pose()
                if pose is not None and fallback and self.profile != fallback:
                    gx, gy = self._goal()
                    off = math.hypot(pose[0] - gx, pose[1] - gy)
                    if off > float(self.cfg.get('centre_tolerance_m', 0.10)):
                        self.ctx.log('info', f'Ending tile {self.target_label}: {off:.2f} m off centre, centring with '
                                             f'{fallback}.')
                        return self._drive(fallback)
                self.machine.ending_driven = self.tile
                end = self.machine.check_end()
                if end:
                    self.machine.ending_failures = 0
                    self.ctx.event('ending_reached', tile=self.target_label, traversal=self.profile)
                    return self.go(self.machine.complete_mode, end)
                return self._fallback_or_fail('stopped off the centre')
            return self._fallback_or_fail(res.reason)
        return None

    def _fallback_or_fail(self, why):
        self.ctx.log('warn', f'Ending tile {self.target_label} ({self.profile}): {why}.')
        fallback = self.cfg.get('fallback_traversal', 'omni')
        if fallback and self.profile != fallback:
            return self._drive(fallback)
        return self._fail(why)

    def _fail(self, why):
        self.machine.ending_failures += 1
        give_up = int(self.cfg.get('give_up_after', 0))
        self.ctx.event('ending_failed', tile=self.target_label, why=why, attempt=self.machine.ending_failures)
        if give_up and self.machine.ending_failures >= give_up:
            return self.go(self.machine.complete_mode, f'ending tile {self.target_label}: centre not reached after '
                                                       f'{self.machine.ending_failures} attempt(s) ({why})')
        return self.go(self.cfg.get('fail_mode', 'scanning'), f'ending tile {self.target_label} not reached ({why})')

    def exit(self, next_mode, reason):
        if self.phase == 'driving' and self.traversal is not None:
            self.traversal.cancel()
        self.traversal = None
        self.ctx.set_target(None, None, None)


# ----------------------------------------------------------------------------------------- complete
@register_mode('complete')
class CompleteMode(Mode):
    def enter(self, reason, **kw):
        self.ctx.set_target(None, None, None)
        self.ctx.on_complete(reason)
        self.set_phase('done')
