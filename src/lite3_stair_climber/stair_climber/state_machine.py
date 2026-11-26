"""Stair-crossing state machine (pure Python, no ROS needed).

The ROS node (stair_climber_node.py) collects sensor data into an
`Observation`, calls `StairStateMachine.update()` ~20 times per second and
executes the returned `Command`. Keeping the logic here means you can unit
test the whole climb on your laptop (see test/test_state_machine.py).

States
------
IDLE                  disabled; not driving
SEARCH                walk forward slowly (or stand still) and watch for stairs
CONFIRM               stopped; waiting for the camera/VLM to confirm "stairs"
ALIGN                 square up to the first step, centred, at a stand-off distance
SWITCH_TO_STAIR_GAIT  send the RUG gait code, wait for the robot to change gait
APPROACH              walk forward blind (lidar can't see the step this close)
                      until the body pitches nose-up
CLIMB_UP              climb at constant slow speed, holding heading
SWITCH_TO_TOP_GAIT    (optional) nose-up has peaked and dropped = front legs on top;
                      stop and change gait for crossing the top
TOP_PLATFORM          walk forward until it pitches nose-down
SWITCH_TO_DESCEND_GAIT (optional) stop and change gait for going down
DESCEND               go down slowly
CLEAR_BOTTOM          level again; walk a bit further so the rear legs clear the stairs
SWITCH_TO_FLAT_GAIT   send the flat gait code, wait
DONE                  finished; hand control back to navigation

AI-gait path (use_ai_gait: true; arena rule: AI mode only from 1 cell before
the stairs to 1 cell after):
ALIGN -> WALK_TO_AI_ZONE  walk (flat gait) until ai_enter_distance from the first step
      -> ENTER_AI         send Enter AI, wait
      -> CROSS_AI         just drive straight forward (the AI gait does the climbing).
                          Uses ONLY the IMU and time, not odometry: once the body has
                          tilted and is flat again, drive ai_clear_time more
      -> EXIT_AI          send Exit AI, wait
      -> SWITCH_TO_FLAT_GAIT -> DONE
NOT_STAIRS            the VLM said it wasn't stairs; hand control back
ABORT                 something went wrong; stand still and hold control
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .robot_state import checkable, describe, target_reached

from .stair_detector import StairObservation


class State(str, Enum):
    IDLE = 'IDLE'
    SEARCH = 'SEARCH'
    CONFIRM = 'CONFIRM'
    ALIGN = 'ALIGN'
    SWITCH_TO_STAIR_GAIT = 'SWITCH_TO_STAIR_GAIT'
    APPROACH = 'APPROACH'
    CLIMB_UP = 'CLIMB_UP'
    SWITCH_TO_TOP_GAIT = 'SWITCH_TO_TOP_GAIT'
    TOP_PLATFORM = 'TOP_PLATFORM'
    SWITCH_TO_DESCEND_GAIT = 'SWITCH_TO_DESCEND_GAIT'
    DESCEND = 'DESCEND'
    CLEAR_BOTTOM = 'CLEAR_BOTTOM'
    SWITCH_TO_FLAT_GAIT = 'SWITCH_TO_FLAT_GAIT'
    WALK_TO_AI_ZONE = 'WALK_TO_AI_ZONE'
    ENTER_AI = 'ENTER_AI'
    CROSS_AI = 'CROSS_AI'
    EXIT_AI = 'EXIT_AI'
    DONE = 'DONE'
    NOT_STAIRS = 'NOT_STAIRS'
    ABORT = 'ABORT'


# States in which this node owns /cmd_vel
ACTIVE_STATES = {
    State.SEARCH, State.CONFIRM, State.ALIGN, State.SWITCH_TO_STAIR_GAIT,
    State.APPROACH, State.CLIMB_UP, State.SWITCH_TO_TOP_GAIT, State.TOP_PLATFORM,
    State.SWITCH_TO_DESCEND_GAIT, State.DESCEND,
    State.CLEAR_BOTTOM, State.SWITCH_TO_FLAT_GAIT, State.ABORT,
    State.WALK_TO_AI_ZONE, State.ENTER_AI, State.CROSS_AI, State.EXIT_AI,
}
# States where the robot should be making forward progress (stuck check)
MOVING_STATES = {
    State.APPROACH, State.CLIMB_UP, State.TOP_PLATFORM, State.DESCEND,
    State.CLEAR_BOTTOM, State.WALK_TO_AI_ZONE,
}   # CROSS_AI is not here: odometry may not update in AI mode, so no odometry stuck check


@dataclass
class SMParams:
    # --- pitch sign ---------------------------------------------------------
    # nose_up_pitch = nose_up_sign * imu_pitch. In ROS, positive pitch is nose
    # DOWN, so the default is -1. VERIFY on the robot: tilt it nose-up by hand
    # and check /stair_climber/debug shows a positive nose_up value.
    nose_up_sign: float = -1.0

    # --- search / detection -------------------------------------------------
    search_speed: float = 0.0          # m/s; 0 = stand still and let Nav2 bring us here
    engage_distance: float = 1.2       # start the stair sequence when the edge is this close
    confirm_detections: int = 5        # consecutive lidar detections required
    use_vlm: bool = False              # ask the scene-understanding server to confirm
    vlm_timeout: float = 10.0
    vlm_accept_on_timeout: bool = True  # server down/slow -> trust the lidar
    vlm_max_rejections: int = 2

    # --- alignment ----------------------------------------------------------
    align_standoff: float = 0.45       # stop this far from the first step edge (m)
    align_yaw_tol_deg: float = 3.0
    align_center_tol: float = 0.05
    align_dist_tol: float = 0.05
    align_hold_time: float = 0.5       # must be inside tolerance this long
    align_k_yaw: float = 1.0
    align_k_dist: float = 0.6
    align_k_lat: float = 0.8
    align_max_vx: float = 0.15
    align_max_vy: float = 0.10
    align_max_wz: float = 0.4
    # The robot ignores very small commands, so outside tolerance never send less than this
    align_min_vx: float = 0.0
    align_min_vy: float = 0.0
    align_min_wz: float = 0.0
    align_lost_timeout: float = 2.0    # detection lost this long -> back to SEARCH
    align_timeout: float = 25.0

    # --- gaits (codes for /simple_cmd; 0 = not configured) ------------------
    stair_gait_code: int = 0           # for climbing up
    flat_gait_code: int = 0
    top_gait_code: int = 0             # for crossing the top (0 = keep the stair gait)
    descend_gait_code: int = 0         # for going down (0 = keep the current gait)
    require_gait_codes: bool = True    # refuse to climb if codes are 0
    gait_switch_wait: float = 2.0       # no robot state (/robot_state): just wait this long after a gait code
    # With /robot_state (0x0901 report): after each gait / Enter AI / Exit AI code, wait until the robot reports the
    # matching state; not there gait_retry_s after the last send -> send the code again (Move Mode first for gaits).
    verify_gait: bool = True
    gait_retry_s: float = 3.0           # (at least gait_switch_wait / ai_switch_wait)
    gait_max_attempts: int = 0          # give up (ABORT) after this many sends; 0 = keep trying
    gait_confirm_s: float = 0.3         # the target state must hold this long
    ai_policy_state: int = 0            # robot_policy_state expected after Enter AI (0 = AI basic); -1 = any

    # --- motion on the stairs ----------------------------------------------
    heading_kp: float = 1.2
    max_wz: float = 0.3
    approach_speed: float = 0.12
    approach_max_distance: float = 1.0  # walked this far and never pitched up -> abort
    approach_timeout: float = 12.0
    climb_start_pitch_deg: float = 8.0  # nose-up beyond this = climbing
    climb_speed: float = 0.12
    climb_timeout: float = 30.0
    level_pitch_deg: float = 4.0        # |pitch| below this = level
    level_hold_time: float = 1.0
    use_odom_height: bool = True        # also require /leg_odom2 z to have risen
    min_height_gain: float = 0.15
    expect_descent: bool = True         # stairs go up AND down (platform in between)
    platform_speed: float = 0.10
    platform_max_distance: float = 2.0
    platform_timeout: float = 20.0
    descend_start_pitch_deg: float = 8.0
    descend_speed: float = 0.08
    descend_timeout: float = 30.0
    clear_distance: float = 0.5
    clear_speed: float = 0.12
    clear_timeout: float = 10.0
    pitch_hold_time: float = 0.3        # pitch must exceed threshold this long
    crest_min_peak_deg: float = 20.0    # nose-up must have reached this while climbing...
    crest_pitch_deg: float = 15.0       # ...then dropped below this = front legs on top
    crest_hold_time: float = 1.0        # ...for this long (run2 had a brief dip to ~20 deg mid-climb)
    pulse_move_s: float = 0.0           # stop-and-go on the stairs: move this long,
    pulse_pause_s: float = 0.0          # then stand still this long (0 = continuous)
    max_reclimbs: int = 2               # false crest (slipped back): go back to climbing this often

    # --- AI gait (Enter AI / Exit AI on /simple_cmd) -------------------------
    use_ai_gait: bool = False
    ai_enter_code: int = 0x21010528
    ai_exit_code: int = 0x2101052B
    ai_switch_wait: float = 3.0
    ai_pre_enter_wait: float = 0.0      # stand still this long before Enter AI (it was sent with the legs still moving)
    ai_enter_distance: float = 0.6      # Enter AI this far from the first step (< 1 cell)
    ai_walk_speed: float = 0.06         # flat-gait walk from the align point to ai_enter_distance
    ai_walk_timeout: float = 40.0
    ai_speed: float = 0.1               # cmd_vel while crossing in AI mode (climbing needs speed)
    ai_down_speed: float = 0.0          # once nose-down / flat again: this speed (0 = ai_speed)
    ai_max_wz: float = 0.3              # heading correction limit in AI mode (0 = no turning)
    ai_reach_timeout: float = 20.0      # no tilt at all within this long -> abort
    ai_flat_after_down_s: float = 1.0   # flat this long after nose-down = off the stairs
    ai_flat_any_s: float = 3.0          # or flat this long after any tilt (AI keeps body level)...
    ai_min_cross_time: float = 12.0     # ...but only once this long since the first tilt
    ai_clear_time: float = 3.0          # then keep walking this long before Exit AI
    ai_cross_timeout: float = 90.0
    # Lidar wall check (24 Nov): before, the node walked blind from WALK_TO_AI_ZONE to DONE. Only used in
    # CROSS_AI once the nose has gone down (on the way up, posts and the stairs themselves look like a wall).
    wall_stop: bool = True
    wall_stop_distance: float = 0.6     # tall obstacle this close (m from base_link, ~0.3 m from the nose):
                                        # Exit AI as soon as the body is off the steps (not nose-down)...
    wall_hold_distance: float = 0.45    # ...but this close while still going down: stand still on the step

    # --- safety -------------------------------------------------------------
    max_roll_deg: float = 20.0
    max_pitch_deg: float = 40.0
    stuck_check: bool = True
    stuck_window: float = 5.0
    stuck_min_progress: float = 0.03
    sensor_timeout: float = 0.5         # IMU/odom older than this -> abort


@dataclass
class Observation:
    t: float                       # seconds
    enabled: bool = True
    imu_age: float = 0.0           # seconds since last IMU message
    odom_age: float = 0.0
    roll: float = 0.0              # rad, from IMU
    pitch: float = 0.0             # rad, from IMU (ROS sign: + = nose down)
    yaw: float = 0.0               # rad, from odometry
    x: float = 0.0                 # odometry position (m)
    y: float = 0.0
    z: float = 0.0
    stair: Optional[StairObservation] = None   # latest detection (None = stale)
    stair_seq: int = 0             # increments on each new lidar result
    vlm: Optional[bool] = None     # VLM verdict since the last request
    wall_dist: float = math.inf    # lidar: nearest tall obstacle straight ahead (inf = none / unknown)
    ai_on_already: bool = False    # the mission already put the robot in AI mode (/stair_climber/ai_on): no Enter AI / Move Mode
    robot_state: Optional[tuple] = None   # (basic, gait, policy, motion) from /robot_state; None = none / stale


@dataclass
class Command:
    state: State
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    active: bool = False               # True = we own /cmd_vel
    gait_code: Optional[int] = None    # publish this once on /simple_cmd
    raw_code: bool = False             # True = don't send Move Mode first (AI enter/exit)
    request_vlm: bool = False          # ask the VLM node for a verdict now
    note: str = ''


def _clamp(v, lim):
    return max(-lim, min(lim, v))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class _Hold:
    """True once a condition has been continuously true for `duration` seconds."""

    def __init__(self):
        self.since = None

    def update(self, cond: bool, t: float, duration: float) -> bool:
        if not cond:
            self.since = None
            return False
        if self.since is None:
            self.since = t
        return t - self.since >= duration

    def reset(self):
        self.since = None


@dataclass
class _Ctx:
    """Values remembered across ticks."""
    state_t0: float = 0.0
    entry_pending: bool = True
    yaw_ref: float = 0.0
    start_xy: tuple = (0.0, 0.0)
    start_z: float = 0.0
    top_z: float = 0.0
    peak_nose: float = 0.0
    reclimbs: int = 0
    align_dist: float = 0.0
    phase: str = ''
    phase_t: float = 0.0
    seen_down: bool = False
    wall_hold: bool = False
    state_xy: tuple = (0.0, 0.0)
    detections: int = 0
    last_seq: int = -1
    last_seen_t: float = 0.0
    vlm_rejections: int = 0
    gait_attempts: int = 0
    gait_sent_t: float = 0.0
    progress_hist: list = field(default_factory=list)


class StairStateMachine:

    def __init__(self, params: SMParams = SMParams()):
        self.p = params
        self.state = State.IDLE
        self.c = _Ctx()
        self.h_pitch = _Hold()
        self.h_level = _Hold()
        self.h_align = _Hold()
        self.h_gait = _Hold()
        self.abort_reason = ''
        self.transitions = []          # [(t, from, to, why)] for logging/tests
        self._forced_abort = None

    # ------------------------------------------------------------------ API

    def force_abort(self, reason: str):
        """Called by the node for problems it detects (e.g. can't send gait)."""
        self._forced_abort = reason

    def update(self, o: Observation) -> Command:
        p = self.p

        if not o.enabled:
            if self.state != State.IDLE:
                self._go(State.IDLE, o, 'disabled')
            return Command(State.IDLE, note='disabled')

        if self.state == State.IDLE:
            self._go(State.SEARCH, o, 'enabled')

        # ---- global safety checks (only while we're driving) -------------
        if self.state in ACTIVE_STATES and self.state != State.ABORT:
            why = self._safety(o)
            if why:
                self.abort_reason = why
                self._go(State.ABORT, o, why)

        handler = getattr(self, '_s_' + self.state.value.lower())
        cmd = handler(o)
        cmd.active = self.state in ACTIVE_STATES or cmd.active
        return cmd

    # -------------------------------------------------------------- helpers

    def _go(self, new: State, o: Observation, why: str):
        self.transitions.append((o.t, self.state, new, why))
        self.state = new
        self.c.state_t0 = o.t
        self.c.entry_pending = True
        self.c.state_xy = (o.x, o.y)
        self.c.progress_hist = []
        self.h_pitch.reset()
        self.h_level.reset()
        self.h_align.reset()
        self.h_gait.reset()

    def _entered(self) -> bool:
        """True exactly once, on the first tick of a state."""
        if self.c.entry_pending:
            self.c.entry_pending = False
            return True
        return False

    def _elapsed(self, o):
        return o.t - self.c.state_t0

    def _dist_in_state(self, o):
        return math.hypot(o.x - self.c.state_xy[0], o.y - self.c.state_xy[1])

    def _nose_up_deg(self, o):
        return math.degrees(self.p.nose_up_sign * o.pitch)

    def _heading_wz(self, o):
        return _clamp(self.p.heading_kp * _wrap(self.c.yaw_ref - o.yaw), self.p.max_wz)

    def _drive(self, o, vx, note=''):
        return Command(self.state, vx=vx, wz=self._heading_wz(o), note=note)

    def _drive_pulsed(self, o, vx, note=''):
        """Stop-and-go: move pulse_move_s, pause pulse_pause_s, repeat."""
        p = self.p
        if p.pulse_move_s > 0 and p.pulse_pause_s > 0:
            phase = self._elapsed(o) % (p.pulse_move_s + p.pulse_pause_s)
            if phase >= p.pulse_move_s:
                return Command(self.state, note=f'{note} (pause)'.strip())
        return self._drive(o, vx, note)

    def _to_top(self, o, why):
        nxt = State.SWITCH_TO_TOP_GAIT if self.p.top_gait_code else State.TOP_PLATFORM
        self._go(nxt, o, why)

    def _to_descend(self, o, why):
        nxt = State.SWITCH_TO_DESCEND_GAIT if self.p.descend_gait_code else State.DESCEND
        self._go(nxt, o, why)

    def _stop(self, note=''):
        return Command(self.state, note=note)

    def _safety(self, o) -> str:
        p = self.p
        if self._forced_abort:
            r, self._forced_abort = self._forced_abort, None
            return r
        if o.imu_age > p.sensor_timeout:
            return f'IMU stale ({o.imu_age:.2f}s)'
        if o.odom_age > p.sensor_timeout:
            return f'odometry stale ({o.odom_age:.2f}s)'
        if abs(math.degrees(o.roll)) > p.max_roll_deg:
            return f'roll {math.degrees(o.roll):.0f} deg too large'
        if abs(math.degrees(o.pitch)) > p.max_pitch_deg:
            return f'pitch {math.degrees(o.pitch):.0f} deg too large'
        if p.stuck_check and self.state in MOVING_STATES:
            h = self.c.progress_hist
            h.append((o.t, o.x, o.y))
            while h and o.t - h[0][0] > p.stuck_window:
                h.pop(0)
            if self._elapsed(o) > p.stuck_window and len(h) > 2:
                moved = math.hypot(h[-1][1] - h[0][1], h[-1][2] - h[0][2])
                if moved < p.stuck_min_progress:
                    return f'stuck: moved {moved:.2f} m in {p.stuck_window:.0f}s'
        return ''

    def _timeout(self, o, limit) -> bool:
        if self._elapsed(o) > limit:
            self.abort_reason = f'{self.state.value} timed out after {limit:.0f}s'
            self._go(State.ABORT, o, self.abort_reason)
            return True
        return False

    def _gait_switch(self, o, code, next_state, label, raw=False, wait=None):
        p = self.p
        wait = p.gait_switch_wait if wait is None else wait
        if self._entered():
            if not code:
                if p.require_gait_codes:
                    self.abort_reason = f'{label} gait code not configured'
                    self._go(State.ABORT, o, self.abort_reason)
                    return self._stop(self.abort_reason)
                return self._stop(f'{label} gait code not set, skipping')
            self.c.gait_attempts, self.c.gait_sent_t = 1, o.t
            return Command(self.state, gait_code=code, raw_code=raw,
                           note=f'switch to {label} gait')
        if p.verify_gait and code and checkable(code) and o.robot_state is not None:
            now = describe(o.robot_state)
            ok = target_reached(code, o.robot_state, p.ai_policy_state)
            if self.h_gait.update(ok, o.t, p.gait_confirm_s):
                self._go(next_state, o, f'{label} gait confirmed: robot state {now} '
                                        f'(attempt {self.c.gait_attempts})')
                return self._stop(f'{label} gait confirmed')
            if not ok and o.t - self.c.gait_sent_t >= max(p.gait_retry_s, wait):
                if p.gait_max_attempts and self.c.gait_attempts >= p.gait_max_attempts:
                    return self._abort(o, f'{label} gait not reached after {self.c.gait_attempts} attempts '
                                          f'(robot state {now})')
                self.c.gait_attempts += 1
                self.c.gait_sent_t = o.t
                return Command(self.state, gait_code=code, raw_code=raw,
                               note=f'{label} gait not reached (robot state {now}): '
                                    f'sending again, attempt {self.c.gait_attempts}')
            return self._stop(f'waiting for {label} gait, robot state {now}')
        if self._elapsed(o) >= wait:   # no robot state: the old timed wait
            self._go(next_state, o, f'{label} gait settled'
                                    + (' (no robot state, not checked)' if p.verify_gait and checkable(code) else ''))
        return self._stop('waiting for gait change')

    # --------------------------------------------------------------- states

    def _s_search(self, o):
        p = self.p
        if self._entered():
            self.c.yaw_ref = o.yaw
            self.c.detections = 0
            self.c.vlm_rejections = 0
            self.c.reclimbs = 0
        s = o.stair
        if o.stair_seq != self.c.last_seq:          # a fresh lidar result
            self.c.last_seq = o.stair_seq
            if s and s.detected and s.distance <= p.engage_distance:
                self.c.detections += 1
            else:
                self.c.detections = 0
        if self.c.detections >= p.confirm_detections:
            nxt = State.CONFIRM if p.use_vlm else State.ALIGN
            self._go(nxt, o, f'stairs at {s.distance:.2f} m, step {s.step_height*100:.0f} cm')
            return self._stop('stairs detected')
        if p.search_speed > 0:
            return self._drive(o, p.search_speed, 'searching')
        return self._stop('watching for stairs')

    def _s_confirm(self, o):
        p = self.p
        if self._entered():
            return Command(self.state, request_vlm=True, note='asking VLM')
        if o.vlm is True:
            self._go(State.ALIGN, o, 'VLM confirmed stairs')
        elif o.vlm is False:
            self.c.vlm_rejections += 1
            if self.c.vlm_rejections >= p.vlm_max_rejections:
                self._go(State.NOT_STAIRS, o, 'VLM says not stairs')
            else:
                self._go(State.CONFIRM, o, 'VLM said no, asking again')
        elif self._elapsed(o) > p.vlm_timeout:
            if p.vlm_accept_on_timeout:
                self._go(State.ALIGN, o, 'VLM timeout, trusting lidar')
            else:
                self._go(State.NOT_STAIRS, o, 'VLM timeout')
        return self._stop('waiting for VLM')

    def _s_align(self, o):
        p = self.p
        if self._entered():
            self.c.last_seen_t = o.t
        if self._timeout(o, p.align_timeout):
            return self._stop(self.abort_reason)
        s = o.stair
        if not (s and s.detected):
            if o.t - self.c.last_seen_t > p.align_lost_timeout:
                self._go(State.SEARCH, o, 'lost the stairs while aligning')
            return self._stop('stairs not visible, holding')
        self.c.last_seen_t = o.t

        e_dist = s.distance - p.align_standoff
        e_lat = s.center_y
        e_yaw = s.edge_yaw
        ok = (abs(e_dist) < p.align_dist_tol and abs(e_lat) < p.align_center_tol
              and abs(math.degrees(e_yaw)) < p.align_yaw_tol_deg)
        if self.h_align.update(ok, o.t, p.align_hold_time):
            self.c.yaw_ref = _wrap(o.yaw + e_yaw)    # heading to hold on the stairs
            self.c.start_xy = (o.x, o.y)
            self.c.start_z = o.z
            self.c.align_dist = s.distance
            nxt = State.WALK_TO_AI_ZONE if p.use_ai_gait else State.SWITCH_TO_STAIR_GAIT
            self._go(nxt, o, f'aligned at {s.distance:.2f} m')
            return self._stop('aligned')
        def cmd(err, tol, k, vmin, vmax):
            if abs(err) < tol:
                return 0.0
            v = _clamp(k * err, vmax)
            return math.copysign(max(abs(v), vmin), err)

        return Command(
            self.state,
            vx=cmd(e_dist, p.align_dist_tol, p.align_k_dist, p.align_min_vx, p.align_max_vx),
            vy=cmd(e_lat, p.align_center_tol, p.align_k_lat, p.align_min_vy, p.align_max_vy),
            wz=cmd(e_yaw, math.radians(p.align_yaw_tol_deg), p.align_k_yaw, p.align_min_wz,
                   p.align_max_wz),
            note=f'align d={e_dist:+.2f} y={e_lat:+.2f} yaw={math.degrees(e_yaw):+.1f}deg',
        )

    def _s_switch_to_stair_gait(self, o):
        return self._gait_switch(o, self.p.stair_gait_code, State.APPROACH, 'stair')

    def _s_approach(self, o):
        p = self.p
        self._entered()
        if self._timeout(o, p.approach_timeout):
            return self._stop(self.abort_reason)
        if self.h_pitch.update(self._nose_up_deg(o) > p.climb_start_pitch_deg,
                               o.t, p.pitch_hold_time):
            self._go(State.CLIMB_UP, o, 'nose up: climbing')
            return self._drive(o, p.climb_speed)
        if self._dist_in_state(o) > p.approach_max_distance:
            self.abort_reason = 'walked forward but never started climbing'
            self._go(State.ABORT, o, self.abort_reason)
            return self._stop(self.abort_reason)
        return self._drive(o, p.approach_speed, 'approaching first step')

    def _s_climb_up(self, o):
        p = self.p
        if self._entered():
            self.c.peak_nose = 0.0
        if self._timeout(o, p.climb_timeout):
            return self._stop(self.abort_reason)
        nose = self._nose_up_deg(o)
        self.c.peak_nose = max(self.c.peak_nose, nose)
        level = abs(nose) < p.level_pitch_deg
        gained = (o.z - self.c.start_z) >= p.min_height_gain or not p.use_odom_height
        crested = self.c.peak_nose >= p.crest_min_peak_deg and nose < p.crest_pitch_deg
        if p.expect_descent and self.h_level.update(crested, o.t, p.crest_hold_time):
            # Recorded run (12 Nov): nose-up ~30-38 deg on the steps, then drops to
            # ~10 deg once the front legs are on the top step.
            self.c.top_z = o.z
            self._to_top(o, f'crested (peak {self.c.peak_nose:.0f} deg, now {nose:.0f} deg)')
            return self._stop('crested')
        if not p.expect_descent and self.h_level.update(level and gained, o.t,
                                                        p.level_hold_time):
            self.c.top_z = o.z
            self._go(State.CLEAR_BOTTOM, o, 'level at the top (no descent)')
            return self._stop('top')
        # Fallback: straight from nose-up to nose-down without a detected crest.
        if p.expect_descent and self.h_pitch.update(
                gained and -nose > p.descend_start_pitch_deg, o.t, p.pitch_hold_time):
            self.c.top_z = o.z
            self._to_descend(o, 'nose down straight after climbing')
            return self._stop('nose down')
        return self._drive_pulsed(o, p.climb_speed, f'climbing, nose up {nose:.0f} deg')

    def _s_switch_to_top_gait(self, o):
        return self._gait_switch(o, self.p.top_gait_code, State.TOP_PLATFORM, 'top')

    def _s_switch_to_descend_gait(self, o):
        return self._gait_switch(o, self.p.descend_gait_code, State.DESCEND, 'descend')

    def _s_top_platform(self, o):
        p = self.p
        self._entered()
        if self._timeout(o, p.platform_timeout):
            return self._stop(self.abort_reason)
        if self.h_pitch.update(-self._nose_up_deg(o) > p.descend_start_pitch_deg,
                               o.t, p.pitch_hold_time):
            self._to_descend(o, 'nose down: descending')
            return self._stop('nose down')
        # Steeply nose-up again = we were NOT on top (front feet slipped back, or the
        # crest fired early). Go back to the stair gait and keep climbing.
        if self.h_level.update(self._nose_up_deg(o) > p.crest_min_peak_deg, o.t,
                               p.pitch_hold_time):
            if self.c.reclimbs >= p.max_reclimbs:
                self.abort_reason = 'still climbing after the top was detected'
                self._go(State.ABORT, o, self.abort_reason)
                return self._stop(self.abort_reason)
            self.c.reclimbs += 1
            nxt = State.SWITCH_TO_STAIR_GAIT if p.top_gait_code else State.CLIMB_UP
            self._go(nxt, o, 'nose up again: not on top yet, back to climbing')
            return self._stop('back to climbing')
        if self._dist_in_state(o) > p.platform_max_distance:
            self.abort_reason = 'platform longer than expected, no descent found'
            self._go(State.ABORT, o, self.abort_reason)
            return self._stop(self.abort_reason)
        return self._drive_pulsed(o, p.platform_speed, 'crossing the top')

    def _s_descend(self, o):
        p = self.p
        self._entered()
        if self._timeout(o, p.descend_timeout):
            return self._stop(self.abort_reason)
        level = abs(self._nose_up_deg(o)) < p.level_pitch_deg
        dropped = (self.c.top_z - o.z) >= p.min_height_gain or not p.use_odom_height
        if self.h_level.update(level and dropped, o.t, p.level_hold_time):
            self._go(State.CLEAR_BOTTOM, o, 'level at the bottom')
        return self._drive_pulsed(o, p.descend_speed,
                                  f'descending, nose down {-self._nose_up_deg(o):.0f} deg')

    def _s_clear_bottom(self, o):
        p = self.p
        self._entered()
        if self._timeout(o, p.clear_timeout):
            return self._stop(self.abort_reason)
        if self._dist_in_state(o) >= p.clear_distance:
            self._go(State.SWITCH_TO_FLAT_GAIT, o, 'clear of the stairs')
            return self._stop('clear')
        return self._drive(o, p.clear_speed, 'clearing the stairs')

    def _s_switch_to_flat_gait(self, o):
        return self._gait_switch(o, self.p.flat_gait_code, State.DONE, 'flat')

    # ------------------------------------------------------- AI-gait path

    def _abort(self, o, why):
        self.abort_reason = why
        self._go(State.ABORT, o, why)
        return self._stop(why)

    def _s_walk_to_ai_zone(self, o):
        p = self.p
        self._entered()
        if self._timeout(o, p.ai_walk_timeout):
            return self._stop(self.abort_reason)
        if self.h_pitch.update(self._nose_up_deg(o) > p.climb_start_pitch_deg,
                               o.t, p.pitch_hold_time):
            return self._abort(o, 'reached the stairs before entering AI mode')
        to_go = self.c.align_dist - p.ai_enter_distance - self._dist_in_state(o)
        if to_go <= 0:
            self._go(State.ENTER_AI, o, f'{p.ai_enter_distance:.2f} m from the first step')
            return self._stop('in the AI zone')
        return self._drive(o, p.ai_walk_speed, f'walking to the AI zone, {to_go:.2f} m to go')

    def _s_enter_ai(self, o):
        if o.ai_on_already:                       # the mission sent Enter AI before enabling us
            self._go(State.CROSS_AI, o, 'AI mode already on')
            return self._stop('AI mode already on')
        if self.c.entry_pending:                  # not sent yet: let the walk to the AI zone come to a stop first
            if self._elapsed(o) < self.p.ai_pre_enter_wait:
                return self._stop('standing still before Enter AI')
            self.c.state_t0 = o.t                 # the settle wait counts from the command
        return self._gait_switch(o, self.p.ai_enter_code, State.CROSS_AI, 'AI', raw=True,
                                 wait=self.p.ai_switch_wait)

    def _s_cross_ai(self, o):
        """Drive straight; the AI gait handles the steps. Only IMU + time are used."""
        p = self.p
        if self._entered():
            self.c.phase, self.c.phase_t, self.c.seen_down = 'to_stairs', o.t, False
            self.c.wall_hold = False
        if self._timeout(o, p.ai_cross_timeout):
            return self._stop(self.abort_reason)
        past_top = self.c.seen_down or self.c.phase == 'clear'
        if p.wall_stop and past_top and o.wall_dist < p.wall_stop_distance:
            self.c.wall_hold = True     # latched: right at the wall the lidar may lose it again
        if self.c.wall_hold:
            if self._nose_up_deg(o) > -p.level_pitch_deg:     # off the steps (not nose-down): done here
                self._go(State.EXIT_AI, o, f'wall {o.wall_dist:.2f} m ahead after the stairs')
                return self._stop('wall ahead: stopping')
            if o.wall_dist < p.wall_hold_distance:
                return self._stop(f'wall {o.wall_dist:.2f} m ahead while still going down: holding')
            # still on the last step: keep stepping down slowly (ai_down_speed), Exit AI once level
        nose = self._nose_up_deg(o)
        flat = abs(nose) < p.level_pitch_deg
        ph = self.c.phase
        if ph == 'to_stairs':
            if self.h_pitch.update(abs(nose) > p.climb_start_pitch_deg, o.t, p.pitch_hold_time):
                self.c.phase, self.c.phase_t = 'on_stairs', o.t
                self.h_pitch.reset()
            elif o.t - self.c.phase_t > p.ai_reach_timeout:
                return self._abort(o, f'no tilt after {p.ai_reach_timeout:.0f}s in AI mode: '
                                      'not moving, or not reaching the stairs')
        elif ph == 'on_stairs':
            if self.h_pitch.update(-nose > p.descend_start_pitch_deg, o.t, p.pitch_hold_time):
                self.c.seen_down = True
            if self.c.seen_down:
                # stairs_ai3 (18 Nov): after the descent the nose goes from -10 to +7 deg
                # (rear legs dropping off the last step), and AI mode stands ~+2..3 deg
                # nose-up on flat floor. So "off the stairs" = no longer nose-down.
                done = self.h_level.update(nose > -p.level_pitch_deg, o.t,
                                           p.ai_flat_after_down_s)
            else:   # never nose-down: only trust a long flat spell, late enough
                done = (self.h_level.update(flat, o.t, p.ai_flat_any_s)
                        and o.t - self.c.phase_t >= p.ai_min_cross_time)
            if done:
                self.c.phase, self.c.phase_t = 'clear', o.t
                self.h_level.reset()
        elif ph == 'clear':
            if o.t - self.c.phase_t >= p.ai_clear_time:
                self._go(State.EXIT_AI, o, 'flat again after the stairs')
                return self._stop('past the stairs')
        speed = p.ai_speed
        if p.ai_down_speed > 0 and (self.c.seen_down or self.c.phase == 'clear'):
            speed = p.ai_down_speed     # going down / past the stairs: no momentum needed
        wz = _clamp(self.p.heading_kp * _wrap(self.c.yaw_ref - o.yaw), p.ai_max_wz)
        return Command(self.state, vx=speed, wz=wz,
                       note=f'AI forward [{self.c.phase}], nose {nose:+.0f} deg, '
                            f'roll {math.degrees(o.roll):+.0f} deg')

    def _s_exit_ai(self, o):
        return self._gait_switch(o, self.p.ai_exit_code, State.SWITCH_TO_FLAT_GAIT, 'exit-AI',
                                 raw=True, wait=self.p.ai_switch_wait)

    def _s_done(self, o):
        return Command(self.state, note='stairs crossed')

    def _s_not_stairs(self, o):
        return Command(self.state, note='not stairs')

    def _s_abort(self, o):
        return self._stop(f'ABORT: {self.abort_reason}')

    def _s_idle(self, o):
        return Command(self.state)
