"""Balance pauses while Nav2 drives the robot up onto a higher tile (no ROS).

While /step_up is true (the mission manager: the target tile is higher than the one the robot started from) and the IMU
shows the nose tilting up (the front legs going up the step), the robot is stopped for pause_s at each of
pause_offsets_s, counted from the moment the tilt was seen: e.g. [0.0, 0.7] = once as the front legs go up, once more
as the rear legs follow. This gives the robot time to balance on the step. After those pauses no more are made until
the robot has been level again for rearm_s (a second step on the way to the same tile), or /step_up goes off and on.

step_pause_gate.py (the ROS node) sits between Nav2's velocity smoother and /cmd_vel and sends zero velocity while
StepPause.update() returns True.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepPauseParams:
    pause_offsets_s: List[float] = field(default_factory=lambda: [0.0, 0.7])   # seconds after the climb tilt
    pause_s: float = 0.2                 # length of each pause
    climb_start_deg: float = 6.0         # nose up beyond this ...
    tilt_hold_s: float = 0.1             # ... for this long = climbing the step
    level_deg: float = 3.0               # |nose| below this ...
    rearm_s: float = 1.0                 # ... for this long after a climb = on top: the next tilt is a new step


class StepPause:
    def __init__(self, params: StepPauseParams = StepPauseParams()):
        self.p = params
        self.offsets = sorted(float(x) for x in params.pause_offsets_s)
        self.reset()

    def reset(self):
        self.climb_t0: Optional[float] = None   # time the climb tilt was confirmed
        self.next = 0                           # index of the next pause in offsets
        self.pause_until: Optional[float] = None
        self._tilt_since: Optional[float] = None
        self._level_since: Optional[float] = None
        self.started: Optional[str] = None      # set on the tick a pause starts (for logging), else None

    def paused(self, t: float) -> bool:
        return self.pause_until is not None and t < self.pause_until

    def update(self, t: float, step_up: bool, nose_up_deg: Optional[float], driving: bool = True) -> bool:
        """True = hold the robot still now. nose_up_deg None = no IMU (never pauses). driving False = Nav2 is not
        commanding motion (no pause starts, one already running finishes)."""
        self.started = None
        if not step_up:
            self.reset()
            return False
        if self.paused(t):
            return True
        if nose_up_deg is None:
            self._tilt_since = self._level_since = None
            return False
        p = self.p
        if self.climb_t0 is None:
            if nose_up_deg > p.climb_start_deg:
                self._tilt_since = t if self._tilt_since is None else self._tilt_since
                if t - self._tilt_since >= p.tilt_hold_s:
                    self.climb_t0, self.next, self._level_since = t, 0, None
            else:
                self._tilt_since = None
        else:
            if abs(nose_up_deg) < p.level_deg:
                self._level_since = t if self._level_since is None else self._level_since
                if t - self._level_since >= p.rearm_s:
                    self.climb_t0, self._tilt_since, self._level_since = None, None, None
                    return False
            else:
                self._level_since = None
        if (self.climb_t0 is not None and driving and self.next < len(self.offsets)
                and t - self.climb_t0 >= self.offsets[self.next]):
            self.next += 1
            self.pause_until = t + p.pause_s
            self.started = (f'pause {self.next}/{len(self.offsets)} on the step up '
                            f'({t - self.climb_t0:.1f} s into the climb, nose {nose_up_deg:+.0f} deg)')
            return True
        return False


class GaitWait:
    """Hold the robot while /step_up is true until the robot reports the h-step gait (no ROS).

    The mission manager sets /step_up as soon as it knows the target tile is higher; the gait switcher then sends rug_hstep,
    which the robot takes ~1 s to accept. Without a hold Nav2 keeps driving and the robot can reach the step still in the
    flat gait (fullrun_26_11_120359, tile -2,-2). update() returns True (send zero velocity) from /step_up on until
    /robot_state shows gait hstep_state once; after that it never holds again for this /step_up (a robot state glitch
    mid-step must not stop the robot on the edge). max_wait_s > 0: give up holding after that long (no robot state bridge,
    gait refused) so the mission's stuck / recovery logic can act; 0 = wait for ever.
    """

    def __init__(self, hstep_state: int = 13, max_wait_s: float = 4.0):
        self.hstep_state = int(hstep_state)
        self.max_wait_s = float(max_wait_s)
        self.reset()

    def reset(self):
        self.since: Optional[float] = None      # /step_up went on
        self.accepted = False                   # robot reported h-step since then
        self.gave_up = False
        self.event: Optional[str] = None        # set on the tick something changes (for logging), else None

    def update(self, t: float, step_up: bool, gait_state: Optional[int], driving: bool = True) -> bool:
        """gait_state: robot_gait_state from /robot_state while standing / stepping (basic 6), None = unknown / stale.
        driving False = Nav2 is not commanding motion (curtain / roller modes drive /cmd_vel themselves): no hold."""
        self.event = None
        if not step_up:
            if self.since is not None:
                self.reset()
            return False
        if self.since is None:
            self.since = t
        if self.accepted or self.gave_up:
            return False
        if gait_state == self.hstep_state:
            self.accepted = True
            self.event = f'h-step gait accepted after {t - self.since:.1f} s: driving on'
            return False
        if self.max_wait_s > 0 and t - self.since >= self.max_wait_s:
            self.gave_up = True
            self.event = (f'robot still not in the h-step gait after {self.max_wait_s:.1f} s '
                          f'(gait state {gait_state}): driving on anyway')
            return False
        return driving
