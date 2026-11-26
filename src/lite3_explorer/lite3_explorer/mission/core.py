"""Mission manager core: modes, transitions and the supervisor that switches between them (no ROS).

A MODE is one behaviour (idle, scanning, exploring, recovery, complete, or your own). Exactly one runs at a
time. Each tick the machine
  1. checks the CONTINUOUS end conditions (e.g. a time limit),
  2. runs the STUCK supervisor for watched modes: stuck (no motion / in the global costmap obstacle zone) while
     unexplored tiles remain -> the recovery mode (drive to the nearest tile centre),
  3. ticks the current mode, which may return a Transition to another mode.
Checkpoint end conditions (all tiles explored) are evaluated by modes at their decision points via check_end() /
end_hop(). A condition with a goal (ending_tile: drive to the ending tile's centre) makes end_hop() hand over to the
mode that goes there.

The context object (`ctx`, the ROS node in production, a fake in tests) gives modes everything they need:
  now() log(level, msg) pose() -> [x, y, yaw] | None      tiles (TileTracker)
  local_map_ready(pose) ready_for_start() -> (ok, missing)  gait_hold('crawl' | 'flat' | None)
  scanner: start(deg, time_allowance_s) poll() -> None | 'succeeded' | 'failed'  cancel()  has_room()  ready
  paths:   start([goal_xy, ...]) -> bool  poll() -> None | [path_or_None, ...]  cancel()
  traversal(profile) -> Traversal (HOW to drive; exploration profiles, WHERE to go, live in the machine)
  costmap_cost_at_robot() -> int | None   clear_costmaps()
  animation_play(name) -> bool  animation_state() -> 'pending' | 'running' | 'done'  animation_cancel()
  set_target(label, goal_xy, route)       event(kind, **data)   on_mode_change(mode, phase)   on_complete(reason)
  curtain_points() -> Nx4 (x, y, ux, uy) or Nx5 (+ 1 camera / 0 lidar-only spot) | None    (only with curtains: true)
  drive(vx, wz, vy=0.0)  stop()  costmap_snapshot() -> (grid, GridInfo) | None      (curtain, stairs and mppi modes)
  ai_mode(on, cmd_code) -> bool   robot AI mode on / off via /simple_cmd; False = no robot listening   (stairs mode)
  robot_state() -> (basic, gait, policy, motion) | None   /robot_state (0x0901 report); optional   (stairs mode: AI on / off
                           is sent again until the robot reports it)
  rebase_odom(x, y, yaw)   the robot is here (global frame) after the AI-mode climb: odom is re-anchored   (stairs mode)
  (staircase tiles come in through tiles.set_stair_points(); the mission manager feeds it /stairs/points)
  hstep(on)   hold the h-step gait (/step_up) while crossing a roller / curtain    (roller, curtain modes)
  (roller tiles come in through tiles.set_roller_points(); the mission manager feeds it /roller/points)
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Dict, Optional

from lite3_explorer import tiles as T
from lite3_explorer.mission.config import mode_cfg
from lite3_explorer.mission.end_conditions import MissionSnapshot, build_end_condition
from lite3_explorer.mission.stuck import StuckDetector


@dataclass
class Transition:
    mode: str
    reason: str = ''
    kwargs: Dict = field(default_factory=dict)


class Mode:
    type_name = 'base'

    def __init__(self, name: str, machine: 'ModeMachine', cfg: dict):
        self.name, self.machine, self.cfg, self.ctx = name, machine, cfg, machine.ctx
        self.phase, self.phase_since, self.entered_at, self.reason = '', 0.0, 0.0, ''

    # -- override these
    @property
    def is_driving(self) -> bool:
        """True while the robot is supposed to be moving (used by the no_motion stuck trigger)."""
        return False

    @property
    def target_label(self) -> Optional[str]:
        return None

    def targets(self):
        """Names of the modes this mode may switch to (validated at start-up)."""
        return []

    def enter(self, reason: str, **kwargs):
        pass

    def tick(self) -> Optional[Transition]:
        return None

    def exit(self, next_mode: str, reason: str):
        pass

    # -- helpers
    def set_phase(self, phase: str):
        self.phase, self.phase_since = phase, self.ctx.now()
        self.ctx.on_mode_change(self.name, phase)

    def in_phase_s(self) -> float:
        return self.ctx.now() - self.phase_since

    def turn_towards(self, err: float, tol_deg: float) -> bool:
        """One tick of an in-place turn (ctx.drive wz) towards the heading error err (rad). True = done: it stopped.

        The dog keeps turning for ~0.5 s after /cmd_vel drops (bag fullrun_26_11_210206: in-place aligns at 0.6 rad/s
        coasted 20-25 deg past the target, the Nav2 spins 7-10 deg). So the turn counts the coast as already turned:
        the remaining error is |err| - turn_lead_s * |last wz|, and both the stop (remaining <= tol) and the slow-down
        (wz = turn_gain * remaining) use it. The lead fades in over the first turn_lead_s of a turn, while the robot is
        still getting going (a short turn would otherwise stop before it ever moved)."""
        c = self.cfg
        now = self.ctx.now()
        prev = getattr(self, '_turn_state', None)
        if prev is None or prev[0] != self.phase_since:
            prev = (self.phase_since, 0.0, now)          # a new turn: (phase, last wz, started)
        _, w_prev, started = prev
        lead_s = float(c.get('turn_lead_s', 0.0))   # 0 = off (sim / tests: no lag); mission.yaml sets the robot's
        remaining = abs(err)
        if w_prev and (w_prev > 0) == (err > 0) and lead_s > 0:
            remaining -= lead_s * abs(w_prev) * min(1.0, (now - started) / lead_s)
        if remaining <= math.radians(tol_deg):
            self.ctx.stop()
            self._turn_state = None
            return True
        w = max(float(c.get('min_turn_rad_s', 0.2)), min(float(c.get('max_turn_rad_s', 0.4)),
                                                          float(c.get('turn_gain', 1.0)) * remaining))
        w = math.copysign(w, err)
        self.ctx.drive(0.0, w)
        self._turn_state = (self.phase_since, w, started)
        return False

    def go(self, mode: str, reason: str, **kwargs) -> Transition:
        return Transition(mode, reason, kwargs)

    def status(self) -> dict:
        return {'mode': self.name, 'type': self.type_name, 'phase': self.phase, 'reason': self.reason,
                'target': self.target_label}


class ModeMachine:
    def __init__(self, ctx, cfg: dict, mode_types: Optional[dict] = None):
        from lite3_explorer.mission.modes import MODE_TYPES
        self.ctx, self.cfg = ctx, cfg
        types = mode_types or MODE_TYPES
        mission = cfg['mission']
        self.modes: Dict[str, Mode] = {}
        for name in [n for n in mission['modes'] if n]:
            mc = mode_cfg(cfg, name)
            if mc['type'] not in types:
                raise ValueError(f'mode {name!r}: unknown type {mc["type"]!r} (known: {sorted(types)})')
            self.modes[name] = types[mc['type']](name, self, mc)
        self.initial_mode = mission['initial_mode']
        self.start_mode = mission['start_mode']
        self.complete_mode = mission['complete_mode']
        self.recovery_mode = mission['recovery_mode']
        st = cfg['stuck']
        self.stuck_watch = set(n for n in st.get('watch_modes', []) if n)
        self.stuck_cooldown_s = float(st.get('cooldown_s', 5.0))
        self.stuck_per_mode = {n: dict(v) for n, v in (st.get('per_mode') or {}).items() if isinstance(v, dict)}
        self.stuck = StuckDetector(st['triggers'], st['window_s'], st['min_motion_m'], int(st['costmap_cost']),
                                   st['costmap_hold_s']) if st.get('enabled', True) and st.get('triggers') else None
        needed = {self.initial_mode, self.start_mode, self.complete_mode}
        if self.stuck is not None and self.stuck_watch:
            needed.add(self.recovery_mode)
        for mode in self.modes.values():
            needed.update(t for t in mode.targets() if t)
        self.end = build_end_condition(cfg['end_condition'])
        needed.update(c.mode for c in self.end.conditions if getattr(c, 'mode', None))   # e.g. ending_tile -> ending
        missing = sorted(needed - set(self.modes))
        if missing:
            raise ValueError(f'mode(s) {missing} are referenced but not in mission.modes {list(self.modes)}')
        self._explorations = {}
        for mode in self.modes.values():          # fail at start-up on a bad exploration profile, not mid-mission
            for name in getattr(mode, 'exploration_profiles', []):
                self.exploration(name)
        self.scenes_done = set()           # scene tiles already looked at (never twice)
        self.scene_calls = {}              # scene tile label -> (pose, AI call): described in the background (scene mode)
        self.curtains_tried = set()        # tiles behind a curtain that was already walked into (never twice)
        self.stairs_tried = set()          # staircase tiles the stairs mode already went for (never twice)
        self.scanned_tile = None           # tile the last scan finished on (the end condition waits for a scan on the last tile)
        self.rollers_tried = set()         # roller tiles the roller mode already went for (never twice)
        self.current: Optional[Mode] = None
        self.previous: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_reason: Optional[str] = None
        self.recoveries = 0                       # consecutive recoveries without reaching a new tile
        self.ending_failures = 0                  # ending mode: attempts at the ending tile's centre that failed
        self._ending_announced = None
        self.ending_driven = None                 # ending tile the ending mode drove to (centred on) already
        self._visited_seen = 0
        self.transitions = []
        self.quickscanned = set()                 # tile labels already quickscanned on their edge

    def reset_for_new_map(self):
        """A new map (and tile grid) starts: everything remembered by tile label belongs to the old grid."""
        self.scenes_done, self.curtains_tried, self.stairs_tried = set(), set(), set()
        if self.scene_calls:
            self.ctx.log('warn', f'New map: dropping {len(self.scene_calls)} scene description(s) still pending '
                                 f'({", ".join(sorted(self.scene_calls))}): their tile labels belong to the old grid.')
        self.scene_calls = {}
        self.rollers_tried, self.quickscanned = set(), set()
        self.scanned_tile = None
        self.recoveries = 0
        self.ending_failures, self._ending_announced, self.ending_driven = 0, None, None
        self.end.reset()
        self._visited_seen = 0

    # ------------------------------------------------------------- lifecycle
    def start(self):
        self._switch(Transition(self.initial_mode, 'startup'))

    def activate(self, active: bool, reason: str = 'set_active'):
        """Resume (from idle / complete) or pause (to idle)."""
        if active:
            if self.current.type_name in ('idle', 'complete'):
                self.finished_reason = None
                self._switch(Transition(self.start_mode, reason, {'opening': not self.ctx.tiles.ready}))
            return f'active ({self.current.name})'
        idle = self._first_of_type('idle')
        if idle is None:
            return 'no idle mode configured'
        self._switch(Transition(idle, reason))
        return 'idle'

    def request(self, name: str, reason: str = 'command', **kwargs):
        if name not in self.modes:
            return False, f'unknown mode {name!r}; modes: {list(self.modes)}'
        if self.modes[name].type_name != 'idle' and not self.ctx.tiles.ready \
                and self.modes[name].type_name != 'scanning':
            return False, 'no tile grid yet: start with the scanning mode (or set_active true)'
        self._switch(Transition(name, reason, kwargs))
        return True, f'switched to {name}'

    # ------------------------------------------------------------- background scene descriptions
    def describe_in_background(self, label, pose, call):
        """Scene mode took its pictures and does not wait for the AI: record the answer when it comes (poll_scene_calls)."""
        self.scene_calls[label] = (pose, call)
        self.ctx.event('scene_background', tile=label, pending=sorted(self.scene_calls))

    def poll_scene_calls(self):
        for label, (pose, call) in list(self.scene_calls.items()):
            if not call.done:
                continue
            del self.scene_calls[label]
            if not call.ok:
                self.ctx.log('warn', f'Scene tile {label}: AI description failed in the background ({call.failure()}).')
            self.ctx.record_scene(label, pose, call)
            self.ctx.event('scene_described', tile=label, ok=bool(call.ok), background=True)
            if self.current is not None and self.current.type_name == 'complete' \
                    and tuple(int(v) for v in label.split(',')) in self.ctx.tiles.traversable:
                self.ctx.log('warn', f'Scene tile {label}: the AI called it traversable after the mission finished; '
                                     f'it is not explored.')

    # ------------------------------------------------------------- the loop
    def tick(self):
        self.poll_scene_calls()
        mode = self.current
        now = self.ctx.now()
        tiles = self.ctx.tiles
        if tiles.ready and tiles.visited_count() > self._visited_seen:
            self._visited_seen = tiles.visited_count()
            self.recoveries = 0
        if mode.type_name not in ('idle', 'complete') and self.started_at is not None:
            reason = self._unless_centring(self.end.check(self.snapshot(), checkpoint=False))
            if reason:
                return self._finish(reason)
        if self.stuck is not None and mode.name in self.stuck_watch:
            own = self.stuck_per_mode.get(mode.name, {})
            if now - mode.entered_at >= float(own.get('cooldown_s', self.stuck_cooldown_s)):
                hold = own.get('costmap_hold_s')
                why = self.stuck.update(now, self.ctx.pose(), mode.is_driving, self.ctx.costmap_cost_at_robot(),
                                        costmap_hold_s=None if hold is None else float(hold))
                if why:
                    return self._on_stuck(mode, why)
        t = mode.tick()
        if t is not None:
            self._switch(t)

    def _on_stuck(self, mode, why):
        self.ctx.log('warn', f'STUCK in {mode.name}: {why}')
        self.ctx.event('stuck', mode=mode.name, why=why, target=mode.target_label)
        nudge = self.cfg['stuck'].get('costmap_mode', '')
        if nudge and nudge in self.modes and getattr(self.stuck, 'trigger', None) == 'costmap':
            # inside the costmap obstacle zone Nav2 cannot plan out: move the robot out directly (x / y velocity)
            self._switch(Transition(nudge, f'stuck: {why}'))
            return
        explorer = self._first_of_type('exploring')
        if self.ctx.tiles.stairs_forced and explorer is not None and mode.type_name in ('exploring', 'backtracking'):
            # going for a forced staircase: no recovery, backtrack to the tile in front of it again from here
            hop = self.modes[explorer].stairs_backtrack(f'stuck: {why}')
            if hop is not None:
                self._switch(hop)
                return
        if self.ctx.tiles.unexplored():
            exclude = [mode.target_label] if mode.target_label else []
            self._switch(Transition(self.recovery_mode, f'stuck: {why}', {'exclude': exclude}))
            return
        reason = self.check_end()
        if reason:
            self._finish(reason)
        else:
            self._switch(Transition(self.start_mode, f'stuck with no unexplored tile ({why}): rescanning'))

    # ------------------------------------------------------------- end
    def snapshot(self) -> MissionSnapshot:
        tiles = self.ctx.tiles
        elapsed = self.ctx.now() - self.started_at if self.started_at is not None else 0.0
        if not tiles.ready or not getattr(self.ctx, 'opening_done', True):
            return MissionSnapshot({}, 0, elapsed_s=elapsed, tiles_ready=False)
        tiles.update()
        return MissionSnapshot(tiles.counts(), tiles.visited_count(),
                               sorted(set(tiles.unexplored()) | set(tiles.curtains_left()) | set(self.stairs_left())),
                               sorted(tiles.unreachable), elapsed, map=tiles.map, tiles=tiles.tiles,
                               trail=list(tiles.trail), pose=self.ctx.pose(), grid=tiles.grid)

    def stairs_left(self):
        """The staircase, while it is still to be climbed (found, not crossed, not tried yet, an exploring mode climbs
        stairs): the mission is not over before it is (the tiles beyond it are closed until then)."""
        tiles = self.ctx.tiles
        stair = tiles.staircase_tile
        if stair is None or tiles.stairs_crossed or tuple(stair) in self.stairs_tried:
            return []
        if not any(m.type_name == 'exploring' and m.cfg.get('stairs', False) for m in self.modes.values()):
            return []
        return [T.tile_label(stair)]

    def check_end(self) -> Optional[str]:
        """Checkpoint: evaluate every end condition. Returns the reason to finish, or None."""
        return self.end.check(self.snapshot(), checkpoint=True)

    def end_hop(self) -> Optional[Transition]:
        """Checkpoint: the transition the end conditions ask for. To complete_mode when they are met; else, when a
        condition first needs the robot somewhere (ending_tile: the ending tile's centre), to the mode that drives there;
        else None (keep exploring)."""
        reason = self._unless_centring(self.check_end())
        goal = self.end.goal()
        if reason:
            return Transition(self.complete_mode, reason)
        if goal is None or goal['mode'] not in self.modes:
            return None
        if self._ending_announced != goal['tile']:
            self._ending_announced = goal['tile']
            self.ctx.log('info', goal['reason'])
            self.ctx.event('ending_tile', tile=T.tile_label(goal['tile']), reason=goal['reason'])
        return Transition(goal['mode'], goal['reason'], {'tile': goal['tile']})

    def _unless_centring(self, reason: Optional[str]) -> Optional[str]:
        """None instead of a condition's own 'met' while its goal mode has not driven to the goal tile yet: the ending
        tile counts as reached within reach_radius_m, but the ending mode still centres the robot on it first. Other
        conditions (time_limit, ...) finish at once."""
        goal = self.end.goal()
        if (reason and goal is not None and goal['mode'] in self.modes and self.ending_driven != goal['tile']
                and reason.startswith(goal.get('condition', '') + ':')):
            return None
        return reason

    def goal_hop(self) -> Optional[Transition]:
        """Checkpoint, before anything else a mode would do here (scenes, stairs, rollers): once a condition has a goal
        (ending_tile: the robot has left the maze) it takes precedence. None while no condition has one."""
        if self.end.goal() is None:
            self.check_end()                       # may find one now
            if self.end.goal() is None:
                return None
        return self.end_hop()

    def _finish(self, reason):
        self._switch(Transition(self.complete_mode, reason))

    # ------------------------------------------------------------- switching
    def _switch(self, t: Transition):
        if t.mode not in self.modes:
            self.ctx.log('error', f'transition to unknown mode {t.mode!r} ({t.reason}); staying in {self.current.name}')
            return
        prev = self.current
        tiles = self.ctx.tiles
        if (t.mode == self.complete_mode and str(t.reason).startswith('all_tiles_explored') and prev is not None
                and prev.type_name != 'scanning' and self.start_mode in self.modes and tiles.ready
                and tiles.current is not None and self.scanned_tile != tuple(tiles.current)):
            # the last tile was reached without a scan (recovery / roller / backtracking hand over without one): scan
            # once here before the end condition counts, in the gait the robot is already in
            self.ctx.log('info', f'Last tile reached without a scan: scanning once before finishing ({t.reason}).')
            t = Transition(self.start_mode, 'scanning the last tile before the end condition')
        if prev is not None:
            prev.exit(t.mode, t.reason)
        nxt = self.modes[t.mode]
        if prev is not None and prev.type_name == 'idle' and nxt.type_name != 'idle' and self.started_at is None:
            self.started_at = self.ctx.now()
        if self.stuck is not None:
            self.stuck.reset()
        self.ctx.log('info', f'MODE {prev.name if prev else "-"} -> {t.mode}: {t.reason}')
        self.transitions.append((self.ctx.now(), prev.name if prev else None, t.mode, t.reason))
        self.ctx.event('mode', previous=prev.name if prev else None, mode=t.mode, reason=t.reason)
        self.previous = prev.name if prev else None    # the mode that handed over (e.g. the curtain mode returns there)
        self.current = nxt
        nxt.reason, nxt.entered_at = t.reason, self.ctx.now()
        nxt.phase = ''
        if nxt.type_name == 'complete':
            self.finished_reason = t.reason
        nxt.enter(t.reason, **t.kwargs)
        if not nxt.phase:
            nxt.set_phase('')

    def exploration(self, name):
        """The exploration strategy instance for a profile (one per profile, shared by every mode that uses it)."""
        from lite3_explorer.mission.exploration import build_exploration
        if name not in self._explorations:
            self._explorations[name] = build_exploration(self.ctx, name, self.cfg.get('explorations', {}))
            self._explorations[name].machine = self
        return self._explorations[name]

    def _first_of_type(self, type_name):
        if self.initial_mode in self.modes and self.modes[self.initial_mode].type_name == type_name:
            return self.initial_mode
        return next((n for n, m in self.modes.items() if m.type_name == type_name), None)

    def status(self) -> dict:
        tiles = self.ctx.tiles
        return {**self.current.status(), 'recoveries': self.recoveries, 'finished': self.finished_reason,
                'end_condition': self.end.describe(),
                'tiles': tiles.counts() if tiles.ready else None,
                'explorations': {n: e.describe() for n, e in self._explorations.items()},
                'unexplored': tiles.unexplored() if tiles.ready else None}
