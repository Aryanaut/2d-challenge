"""The preconfigured mission modes (no ROS). Configure them under modes.<name> (see config.py).

  idle       nothing moves. Leaves by itself once the robot is ready (auto_start), or on set_active / a mode command.
  scanning   builds the map around the robot: places the tile grid on the first scan, then turns in place
             (opening sweep -90..+90 deg the first time, a full turn later) so neighbouring tiles get classified.
  exploring  asks its exploration profiles (WHERE: tiles, optionally frontiers) for a reachable target and drives
             there with a traversal profile (HOW), then hands over to scanning. Checks the end condition at every
             decision.
  recovery   the robot is stuck (or no target has a Nav2 path): drive to the nearest tile centre, then rescan.
  complete   the end condition is met. /exploration_complete is published.

Add a mode type with @register_mode('name') on a Mode subclass and list an instance in mission.modes.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Type

from lite3_explorer import approach as A
from lite3_explorer import tiles as T
from lite3_explorer.mission.core import Mode
from lite3_explorer.mission.traversal import FAILED, SUCCEEDED

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

    def targets(self):
        return [self.cfg.get('next_mode', 'exploring')]

    def enter(self, reason, opening=False, **kw):
        self.opening = bool(opening) or not self.ctx.tiles.ready
        self.retries, self.queue, self.pending_deg = 0, [], None
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
            half = abs(float(cfg.get('opening_sweep_deg', 90.0)))
            if half < 1.0:
                self.set_phase('done')
                return
            self.queue = [-half, 2 * half]
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
            self.ctx.gait_hold('crawl' if cfg.get('use_crawl', True) else None)
        self.set_phase('settle')

    def tick(self):
        cfg = self.cfg
        if self.phase == 'wait_map':
            pose = self.ctx.pose()
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
            status = self.ctx.scanner.poll()
            timed_out = self.in_phase_s() > float(cfg.get('time_allowance_s', 20.0)) + 5.0
            if status == SUCCEEDED and self.queue:
                self._next_turn()
                return None
            if status is None and not timed_out:
                return None
            if timed_out and status is None:
                self.ctx.scanner.cancel()
            if status != SUCCEEDED and self.opening and self.retries < int(cfg.get('opening_retries', 2)):
                # The opening scan must finish before navigating: retry the sweep the other way round.
                self.retries += 1
                half = abs(float(cfg.get('opening_sweep_deg', 90.0))) * (1 if self.retries % 2 else -1)
                self.ctx.log('warn', f'Opening spin did not finish ({status or "timeout"}); retry {self.retries} '
                                     f'turning {half:+.0f} deg first.')
                self.queue = [half, -2 * half]
                self.set_phase('settle')
                return None
            if status != SUCCEEDED:
                self.ctx.log('warn', 'The scan turn did not finish; continuing with the map so far.')
            self.set_phase('done')
        if self.phase == 'done':
            self.ctx.gait_hold(None)
            self.ctx.tiles.invalidate()
            self.ctx.tiles.update()
            return self.go(cfg.get('next_mode', 'exploring'), 'scan finished')
        return None

    def _establish(self, pose, how):
        start = self.ctx.tiles.establish(pose, how)
        grid = self.ctx.tiles.grid
        self.ctx.log('info', f'Tile grid placed ({how}): yaw {math.degrees(grid.yaw):.1f} deg, start tile '
                             f'{start[0]},{start[1]}.')
        self.ctx.event('tile_grid', how=how, grid=grid.as_dict())
        self._begin_turns()

    def _next_turn(self):
        deg = self.queue.pop(0)
        self.ctx.scanner.start(deg, float(self.cfg.get('time_allowance_s', 20.0)))
        self.set_phase('turning')

    def exit(self, next_mode, reason):
        if self.phase in ('settle', 'turning'):
            self.ctx.scanner.cancel()
        self.ctx.gait_hold(None)


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
        self.pending = None           # (strategy index, candidates) while the Nav2 path checks run
        self.index = 0                # which strategy is being asked in this decision
        self.unreached = []           # labels of candidates that had no path in this decision
        self.avoid = set()            # goals that just failed: the next decision picks a different tile
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
        self._new_decision()

    def _new_decision(self):
        self.index, self.unreached = 0, []
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
        return None

    # -- decide: end condition, then ask the strategies in order
    def _decide(self):
        if self.index == 0:
            end = self.machine.check_end()
            if end:
                return self.go(self.machine.complete_mode, end)
        pose = self.ctx.pose()
        if pose is None:
            return None
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
        if self.unreached:
            self.ctx.log('warn', f'No Nav2 path to any candidate ({", ".join(self.unreached)}).')
            self.ctx.event('no_path', targets=self.unreached)
            return self._on_no_path([u for u in self.unreached if not u.startswith('frontier')])
        return self._nothing_left('no exploration target left')

    def _nothing_left(self, why):
        end = self.machine.check_end()
        if end:
            return self.go(self.machine.complete_mode, end)
        idle = self.machine._first_of_type('idle')
        self.ctx.log('warn', f'{why}, but the end condition is not met ({self.machine.end.describe()}): idling.')
        return self.go(idle, f'{why}; end condition not met') if idle else None

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
        self.unreached += [c['label'] for c, _ in checked]
        self.ctx.log('info', f'Exploration {strat.name!r}: no reachable candidate; trying the next profile.')
        self.index = index + 1
        self.set_phase('decide')
        return None

    def _on_no_path(self, labels):
        action = self.cfg.get('on_no_reachable_tile', 'recovery')
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
        return self._drive_to(strat, self.choose_target(reachable))

    def choose_target(self, reachable):
        """Hook for other selection strategies (e.g. an AI choice): default is the strategy's top-ranked target."""
        return reachable[0]

    def traversal_for(self, target) -> str:
        """Which traversal profile drives to this target: per-tile override, then per-kind, then the mode's default."""
        if target.get('tile') in self.tile_traversals:
            return self.tile_traversals[target['tile']]
        return self.kind_traversals.get(target.get('kind', 'tile'), self.cfg.get('traversal', 'route'))

    # -- driving
    def _drive_to(self, strat, target):
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
            # N waypoints (traversals.<profile>.waypoints, default 3) evenly along the Nav2 path, then the goal
            route = A.default_route(target['path_xy'], target['goal_xy'], grid_yaw,
                                    n=int(trav.cfg.get('waypoints', A.N_WAYPOINTS)),
                                    skip_start_m=float(trav.cfg.get('skip_start_m', A.SKIP_START_M)),
                                    sides=sides, face_away=face_away)
            if trav.cfg.get('drop_backward_waypoints', False):
                route, dropped = A.filter_route(route, target['path_xy'][0])
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
        self.set_phase('driving')
        return None

    def _poll_drive(self):
        res = self.traversal.poll()
        if not res.done:
            if self.in_phase_s() > float(self.cfg.get('goal_timeout_s', 120.0)):
                self.traversal.cancel()
                return self._failed('timed out')
            if self.ctx.now() - self.valid_checked_at >= 1.0:
                self.valid_checked_at = self.ctx.now()
                if not self.strategy.still_valid(self.target):
                    self.traversal.cancel()
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
            self.ctx.event('reached', target_kind=t['kind'], label=t['label'], tile=t.get('tile'))
            self._clear_target()
            if self.cfg.get('scan_after_reach', True):
                return self.go(self.cfg.get('scan_mode', 'scanning'), f'{t["kind"]} {t["label"]} reached')
            self._new_decision()
            return None
        return self._failed(res.reason or 'failed')

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


# ----------------------------------------------------------------------------------------- recovery
@register_mode('recovery')
class RecoveryMode(Mode):
    """Stuck: move to the nearest tile centre. Phases: clearing -> driving (candidate by candidate) -> next mode."""

    def __init__(self, name, machine, cfg):
        super().__init__(name, machine, cfg)
        self.candidates, self.target, self.traversal, self.exclude = [], None, None, []

    def targets(self):
        return [self.cfg.get('next_mode', 'scanning'), self.cfg.get('fail_mode', 'scanning'), self.machine.complete_mode]

    @property
    def is_driving(self):
        return self.phase == 'driving'

    @property
    def target_label(self):
        return self.target['tile'] if self.target else None

    def enter(self, reason, exclude=(), blocked=(), **kw):
        # never recover INTO the tile we got stuck on, nor into tiles Nav2 just failed to plan to
        self.exclude = [e for e in list(exclude) + list(blocked) if e]
        self.candidates, self.target, self.traversal = [], None, None
        tiles = self.ctx.tiles
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
        if not self.ctx.tiles.unexplored():
            return self._leave(self.cfg.get('next_mode', 'scanning'), 'no unexplored tile left')
        if self.phase == 'clearing':
            if self.in_phase_s() < float(self.cfg.get('clear_settle_s', 1.0)):
                return None
            pose = self.ctx.pose()
            if pose is None:
                return None
            self.candidates = self.ctx.tiles.recovery_targets(
                pose[:2], statuses=tuple(self.cfg.get('tile_statuses', ['visited', 'to_visit'])),
                limit=int(self.cfg.get('candidates', 3)), min_distance_m=float(self.cfg.get('min_distance_m', 0.25)),
                min_clearance_m=float(self.cfg.get('min_clearance_m', 0.25)), exclude=self.exclude)
            self.ctx.event('recovery', candidates=[c['tile'] for c in self.candidates], exclude=self.exclude)
            return self._next_candidate()
        if self.phase == 'wait_traversal':
            if self.traversal.available():
                self.candidates.insert(0, self.target)
                return self._next_candidate()
            if self.in_phase_s() > float(self.cfg.get('attempt_timeout_s', 45.0)):
                return self._leave(self.cfg.get('fail_mode', 'scanning'), 'traversal not available')
            return None
        if self.phase == 'driving':
            res = self.traversal.poll()
            if not res.done:
                if self.in_phase_s() > float(self.cfg.get('attempt_timeout_s', 45.0)):
                    self.traversal.cancel()
                    self.ctx.log('warn', f'Recovery to tile {self.target["tile"]} timed out.')
                    return self._next_candidate()
                return None
            if res.state == SUCCEEDED:
                return self._leave(self.cfg.get('next_mode', 'scanning'), f'recovered at the centre of tile {self.target["tile"]}')
            self.ctx.log('warn', f'Recovery to tile {self.target["tile"]} failed: {res.reason}.')
            return self._next_candidate()
        return None

    def _next_candidate(self):
        if not self.candidates:
            return self._leave(self.cfg.get('fail_mode', 'scanning'), 'no tile centre could be reached')
        self.target = self.candidates.pop(0)
        try:
            self.traversal = self.ctx.traversal(self.cfg.get('traversal', 'direct'))
        except KeyError as exc:
            self.ctx.log('error', str(exc))
            return self._leave(self.cfg.get('fail_mode', 'scanning'), 'bad traversal profile')
        pose = self.ctx.pose() or [0.0, 0.0, 0.0]
        gx, gy = self.target['goal_xy']
        grid_yaw = self.ctx.tiles.grid.yaw if self.ctx.tiles.grid is not None else 0.0
        yaw = A.snap_to_axis(math.atan2(gy - pose[1], gx - pose[0]), grid_yaw)
        if not self.traversal.available():
            self.ctx.log('warn', 'Recovery traversal not available yet (Nav2 down?): waiting.')
            self.set_phase('wait_traversal')
            return None
        self.ctx.log('info', f'Recovery: driving to the centre of tile {self.target["tile"]} '
                             f'({self.target["distance_m"]:.2f} m away, {self.target["status"]}).')
        if not self.traversal.start(self.target['goal_xy'], yaw, None):
            return self._next_candidate()
        self.ctx.set_target(self.target['tile'], self.target['goal_xy'], None)
        self.set_phase('driving')
        return None

    def _leave(self, mode, reason):
        if mode == self.machine.complete_mode or not self.ctx.tiles.unexplored():
            end = self.machine.check_end()
            if end:
                return self.go(self.machine.complete_mode, end)
        return self.go(mode, reason)

    def exit(self, next_mode, reason):
        if self.phase == 'driving' and self.traversal is not None:
            self.traversal.cancel()
        self.target, self.traversal = None, None
        self.ctx.set_target(None, None, None)


# ----------------------------------------------------------------------------------------- complete
@register_mode('complete')
class CompleteMode(Mode):
    def enter(self, reason, **kw):
        self.ctx.set_target(None, None, None)
        self.ctx.on_complete(reason)
        self.set_phase('done')
