"""Robot animations for the Lite3 (no ROS imports, so it can be unit tested).

An animation is a timed sequence of MotionSimpleCMD {cmd_code, size} commands on /simple_cmd.
There are two categories:

  * MovementAnimation - animations that move the robot around (none defined yet).
  * PoseAnimation     - body-pose animations. They enter pose mode when they start and
                        enter move mode when they end (also when cancelled).

In pose mode the robot holds a pose only while the pose command is being published (like
`ros2 topic pub -r 20 ... lite3_lean_fwd`), so a PoseAnimation returns every pose command it
wants held on every tick and the player republishes them at a steady rate.

Usage: anim = ANIMATIONS['quickscan'](pause_s=0.5); anim.start(now); then every tick
`for code, size in anim.tick(now): publish(...)` until anim.finished.
"""
from typing import Dict, List, Optional, Tuple

# Aliases: lite3_move_mode, lite3_pose_mode, lite3_lean_fwd / lite3_lean_back (size +-20000).
MOVE_MODE = 0x21010D06
POSE_MODE = 0x21010D05
LEAN = 0x21010130               # size [-20000, 20000], positive leans forward
YAW = 0x21010135                # positive rotates to the right; |size| <= 9553 is the dead zone (read as 0)

LEAN_MAX = 100000                # validation ceiling only; the robot decides what it accepts
YAW_DEAD_ZONE = 9553            # axis values inside +-9553 are ignored by the robot
YAW_MAX = 100000                 # validation ceiling only; the robot decides what it accepts

Command = Tuple[int, int]       # (cmd_code, size)

MOVEMENT, POSE = 'movement', 'pose'


class RobotAnimation:
    """Base class: a time-driven animation. tick() returns the commands to publish now."""
    name = ''
    category = ''

    def __init__(self):
        self.started_at: Optional[float] = None
        self.finished = False
        self._markers: List[str] = []           # named moments reached since the last pop_markers()

    @property
    def running(self) -> bool:
        return self.started_at is not None and not self.finished

    def start(self, now: float) -> None:
        self.started_at = now
        self.finished = False

    def tick(self, now: float) -> List[Command]:
        raise NotImplementedError

    def pop_markers(self) -> List[str]:
        """Names of the marker moments (see PoseAnimation.markers) reached since the last call."""
        out, self._markers = self._markers, []
        return out

    def cancel(self, now: float) -> List[Command]:
        """Stop early; returns the commands needed to leave the robot in a safe state."""
        self.finished = True
        return []


class MovementAnimation(RobotAnimation):
    """Animations that move the robot (walk, turn, ...). The robot stays in move mode."""
    category = MOVEMENT


class PoseAnimation(RobotAnimation):
    """Animations that pose the body. Subclasses define keyframes(); the base class handles
    the mode switching: pose mode on start, the pose held while it plays, move mode at the end.

    keyframes() returns [(duration_s, {cmd_code: size}), ...]. Each keyframe ramps linearly from
    the previous pose (all zeros at the start) to its targets; a keyframe whose targets equal the
    previous ones is a pause. Commands absent from a keyframe keep their previous value.
    """
    category = POSE

    def __init__(self, mode_settle_s: float = 1.0, mode_repeats: int = 3,
                 mode_repeat_period_s: float = 0.2, end_settle_s: float = 0.5):
        super().__init__()
        self.mode_settle_s = mode_settle_s          # wait after pose mode before moving the body
        self.mode_repeats = max(1, int(mode_repeats))   # a lone mode command can be lost
        self.mode_repeat_period_s = mode_repeat_period_s
        self.end_settle_s = end_settle_s            # keep holding the neutral pose before move mode
        self._timeline = self._build_timeline()
        self.duration = self._timeline[-1][1] if self._timeline else 0.0
        self._sent_enter = 0
        self._sent_exit = 0
        self._last_mode_t = 0.0
        self._exiting_at: Optional[float] = None
        self._marker_todo: List[Tuple[float, str]] = []

    def keyframes(self) -> List[Tuple[float, Dict[int, int]]]:
        raise NotImplementedError

    def markers(self) -> List[Tuple[float, str]]:
        """[(seconds into the animated part, name)]: moments the owner is told about (e.g. to take a picture)."""
        return []

    def _build_timeline(self):
        """[(t_start, t_end, from_pose, to_pose)] with absolute pose dicts."""
        pose: Dict[int, int] = {}
        t = 0.0
        out = []
        for dur, targets in self.keyframes():
            nxt = {**pose, **targets}
            codes = set(pose) | set(nxt)
            out.append((t, t + dur, {c: pose.get(c, 0) for c in codes}, {c: nxt.get(c, 0) for c in codes}))
            pose, t = nxt, t + dur
        return out

    def pose_at(self, t: float) -> Dict[int, int]:
        """The pose commands to hold t seconds into the animated part (after the pose-mode settle)."""
        if not self._timeline:
            return {}
        if t >= self.duration:
            return dict(self._timeline[-1][3])
        for t0, t1, a, b in self._timeline:
            if t0 <= t < t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                return {c: int(round(a[c] + (b[c] - a[c]) * f)) for c in b}
        return dict(self._timeline[0][2])

    def _mode_burst(self, code: int, sent: int, now: float) -> Tuple[List[Command], int]:
        if sent == 0 or (sent < self.mode_repeats and now - self._last_mode_t >= self.mode_repeat_period_s):
            self._last_mode_t = now
            return [(code, 0)], sent + 1
        return [], sent

    def start(self, now: float) -> None:
        super().start(now)
        self._sent_enter = self._sent_exit = 0
        self._exiting_at = None
        self._marker_todo = sorted(self.markers())
        self._markers = []

    def tick(self, now: float) -> List[Command]:
        if not self.running:
            return []
        t = now - self.started_at
        if self._exiting_at is None:
            cmds, self._sent_enter = self._mode_burst(POSE_MODE, self._sent_enter, now)
            body_t = t - self.mode_settle_s
            if body_t >= 0.0:
                cmds += list(self.pose_at(body_t).items())
                while self._marker_todo and self._marker_todo[0][0] <= body_t:
                    self._markers.append(self._marker_todo.pop(0)[1])
            if body_t >= self.duration + self.end_settle_s:
                self._exiting_at = now
            return cmds
        # Pose done: enter move mode (repeated), then finish.
        cmds, self._sent_exit = self._mode_burst(MOVE_MODE, self._sent_exit, now)
        if self._sent_exit >= self.mode_repeats and now - self._last_mode_t >= self.mode_repeat_period_s:
            self.finished = True
        return cmds

    def cancel(self, now: float) -> List[Command]:
        self.finished = True
        return [(MOVE_MODE, 0)] * self.mode_repeats if self.started_at is not None else []


class QuickScan(PoseAnimation):
    """Yaw the body left and right (level, no lean by default), then back to centre.

    lean > 0 optionally leans forward during the scan. Pauses for pause_s at the start pose and at the right extreme (no pause at the left: the
    look left flows straight into the sweep right).
    """
    name = 'quickscan'

    def __init__(self, pause_s: float = 1.0, move_s: float = 1.0, lean: int = 0, yaw: int = 32767,
                 **kw):
        if not 0 <= abs(lean) <= LEAN_MAX or not YAW_DEAD_ZONE < abs(yaw) <= YAW_MAX:
            raise ValueError(f'lean must be within +-{LEAN_MAX} and yaw beyond the dead zone '
                             f'({YAW_DEAD_ZONE}) up to {YAW_MAX}')
        self.pause_s, self.move_s, self.lean, self.yaw = pause_s, move_s, lean, yaw
        super().__init__(**kw)

    def keyframes(self):
        p, m, lean, yaw = self.pause_s, self.move_s, self.lean, self.yaw
        return [
            (m, {LEAN: lean, YAW: 0}),          # lean forward
            (p, {}),                            # pause
            (m, {YAW: -yaw}),                   # look left (negative = left)
            (2 * m, {YAW: yaw}),                # sweep through centre to the right
            (p, {}),                            # pause at right
            (m, {LEAN: 0, YAW: 0}),             # back up: level and centred
        ]


class SceneScan(PoseAnimation):
    """Yaw to the left, back to the centre, then to the right, stopping at each for pause_s (level, no lean).

    The markers 'left', 'centre' and 'right' are reached at the end of each pause: the scene mode takes a picture
    at each. Ends level and centred.
    """
    name = 'scenescan'

    def __init__(self, pause_s: float = 1.0, move_s: float = 1.0, yaw: int = 20000, **kw):
        if not YAW_DEAD_ZONE < abs(yaw) <= YAW_MAX:
            raise ValueError(f'yaw must be beyond the dead zone ({YAW_DEAD_ZONE}) up to {YAW_MAX}')
        self.pause_s, self.move_s, self.yaw = pause_s, move_s, yaw
        super().__init__(**kw)

    def keyframes(self):
        p, m, yaw = self.pause_s, self.move_s, self.yaw
        return [
            (m, {YAW: -yaw}), (p, {}),          # look left, hold
            (m, {YAW: 0}), (p, {}),             # back to the centre, hold
            (m, {YAW: yaw}), (p, {}),           # look right, hold
            (m, {YAW: 0}),                      # centred again
        ]

    def markers(self):
        p, m = self.pause_s, self.move_s
        return [(m + p, 'left'), (2 * (m + p), 'centre'), (3 * (m + p), 'right')]


ANIMATIONS = {cls.name: cls for cls in (QuickScan, SceneScan)}
