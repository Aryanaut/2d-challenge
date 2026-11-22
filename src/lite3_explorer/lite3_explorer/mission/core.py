"""Mission manager core: modes, transitions and the supervisor that switches between them (no ROS).

A MODE is one behaviour (idle, scanning, exploring, recovery, complete, or your own). Exactly one runs at a
time. Each tick the machine
  1. checks the CONTINUOUS end conditions (e.g. a time limit),
  2. runs the STUCK supervisor for watched modes: stuck (no motion / in the global costmap obstacle zone) while
     unexplored tiles remain -> the recovery mode (drive to the nearest tile centre),
  3. ticks the current mode, which may return a Transition to another mode.
Checkpoint end conditions (all tiles explored) are evaluated by modes at their decision points via check_end().

The context object (`ctx`, the ROS node in production, a fake in tests) gives modes everything they need:
  now() log(level, msg) pose() -> [x, y, yaw] | None      tiles (TileTracker)
  local_map_ready(pose) ready_for_start() -> (ok, missing)  gait_hold('crawl' | 'flat' | None)
  scanner: start(deg, time_allowance_s) poll() -> None | 'succeeded' | 'failed'  cancel()  has_room()  ready
  paths:   start([goal_xy, ...]) -> bool  poll() -> None | [path_or_None, ...]  cancel()
  traversal(profile) -> Traversal (HOW to drive; exploration profiles, WHERE to go, live in the machine)
  costmap_cost_at_robot() -> int | None   clear_costmaps()
  animation_play(name) -> bool  animation_state() -> 'pending' | 'running' | 'done'  animation_cancel()
  set_target(label, goal_xy, route)       event(kind, **data)   on_mode_change(mode, phase)   on_complete(reason)
  curtain_points() -> Nx4 (x, y, ux, uy) | None    (only with curtains: true)
  drive(vx, wz, vy=0.0)  stop()  costmap_snapshot() -> (grid, GridInfo) | None      (curtain and mppi modes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

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
        self.stuck = StuckDetector(st['triggers'], st['window_s'], st['min_motion_m'], int(st['costmap_cost']),
                                   st['costmap_hold_s']) if st.get('enabled', True) and st.get('triggers') else None
        needed = {self.initial_mode, self.start_mode, self.complete_mode}
        if self.stuck is not None and self.stuck_watch:
            needed.add(self.recovery_mode)
        for mode in self.modes.values():
            needed.update(t for t in mode.targets() if t)
        missing = sorted(needed - set(self.modes))
        if missing:
            raise ValueError(f'mode(s) {missing} are referenced but not in mission.modes {list(self.modes)}')
        self.end = build_end_condition(cfg['end_condition'])
        self._explorations = {}
        for mode in self.modes.values():          # fail at start-up on a bad exploration profile, not mid-mission
            for name in getattr(mode, 'exploration_profiles', []):
                self.exploration(name)
        self.scenes_done = set()           # scene tiles already looked at (never twice)
        self.curtains_tried = set()        # tiles behind a curtain that was already walked into (never twice)
        self.current: Optional[Mode] = None
        self.previous: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_reason: Optional[str] = None
        self.recoveries = 0                       # consecutive recoveries without reaching a new tile
        self._visited_seen = 0
        self.transitions = []
        self.quickscanned = set()                 # tile labels already quickscanned on their edge

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

    # ------------------------------------------------------------- the loop
    def tick(self):
        mode = self.current
        now = self.ctx.now()
        tiles = self.ctx.tiles
        if tiles.ready and tiles.visited_count() > self._visited_seen:
            self._visited_seen = tiles.visited_count()
            self.recoveries = 0
        if mode.type_name not in ('idle', 'complete') and self.started_at is not None:
            reason = self.end.check(self.snapshot(), checkpoint=False)
            if reason:
                return self._finish(reason)
        if self.stuck is not None and mode.name in self.stuck_watch:
            if now - mode.entered_at >= self.stuck_cooldown_s:
                why = self.stuck.update(now, self.ctx.pose(), mode.is_driving, self.ctx.costmap_cost_at_robot())
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
        return MissionSnapshot(tiles.counts(), tiles.visited_count(), sorted(set(tiles.unexplored()) | set(tiles.curtains_left())),
                               sorted(tiles.unreachable), elapsed, map=tiles.map)

    def check_end(self) -> Optional[str]:
        """Checkpoint: evaluate every end condition. Returns the reason to finish, or None."""
        return self.end.check(self.snapshot(), checkpoint=True)

    def _finish(self, reason):
        self._switch(Transition(self.complete_mode, reason))

    # ------------------------------------------------------------- switching
    def _switch(self, t: Transition):
        if t.mode not in self.modes:
            self.ctx.log('error', f'transition to unknown mode {t.mode!r} ({t.reason}); staying in {self.current.name}')
            return
        prev = self.current
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
