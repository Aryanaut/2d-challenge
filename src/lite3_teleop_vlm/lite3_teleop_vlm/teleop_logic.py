"""Key -> velocity / action logic for the console (pure Python, unit-tested).

Driving is hold-to-move: a movement key sets a velocity that stays active only while the key
keeps repeating. If no movement key arrives for hold_s seconds the command drops to zero
(dead-man). Set latch=True for teleop_twist_keyboard behaviour (keeps going until stop).
"""

from dataclasses import dataclass

# key: (forward, left, turn-left) unit directions
MOVE_KEYS = {
    "w": (1, 0, 0), "s": (-1, 0, 0),
    "a": (0, 0, 1), "d": (0, 0, -1),
    "q": (0, 1, 0), "e": (0, -1, 0),            # strafe (holonomic)
    "UP": (1, 0, 0), "DOWN": (-1, 0, 0), "LEFT": (0, 0, 1), "RIGHT": (0, 0, -1),
}
STOP_KEYS = {" ", "x", "ESC"}            # Esc = emergency stop; quit with Ctrl+C
ACTION_KEYS = {
    "n": "mode_auto", "m": "mode_manual",
    "v": "vlm_front", "b": "vlm_down",
    "p": "prompt_next", "o": "prompt_prev",
    "k": "snapshot", "h": "help", "?": "help",
}
CONFIRM_KEYS = {"t": "stand_sit", "r": "reset"}   # must be pressed twice within confirm_s


@dataclass
class TeleopConfig:
    linear_speed: float = 0.3       # m/s  (lab guidance: about 0.3)
    angular_speed: float = 0.6      # rad/s
    max_linear: float = 0.5
    max_angular: float = 1.2
    step: float = 0.05              # speed change per +/- press
    hold_s: float = 0.6             # dead-man timeout
    latch: bool = False
    confirm_s: float = 1.5


class Teleop:
    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        self.linear = min(cfg.linear_speed, cfg.max_linear)
        self.angular = min(cfg.angular_speed, cfg.max_angular)
        self.cmd = (0.0, 0.0, 0.0)
        self.last_move_t = float("-inf")
        self.pending = {}               # confirm key -> time of first press

    def key(self, k: str, t: float):
        """Handle one key. Returns an action string for the node, or None."""
        lk = k.lower() if len(k) == 1 else k
        if lk in MOVE_KEYS:
            f, l, r = MOVE_KEYS[lk]
            self.cmd = (f * self.linear, l * self.linear, r * self.angular)
            self.last_move_t = t
            return None
        if lk in STOP_KEYS:
            self.stop()
            return "stop"
        if lk in ("+", "="):
            return self._scale(+1)
        if lk in ("-", "_"):
            return self._scale(-1)
        if lk in CONFIRM_KEYS:
            first = self.pending.get(lk)
            if first is not None and t - first <= self.cfg.confirm_s:
                self.pending.pop(lk)
                return CONFIRM_KEYS[lk]
            self.pending[lk] = t
            return "confirm_" + CONFIRM_KEYS[lk]
        return ACTION_KEYS.get(lk)

    def _scale(self, sign):
        ratio = self.angular / self.linear if self.linear > 0 else 2.0
        self.linear = round(min(self.cfg.max_linear, max(self.cfg.step, self.linear + sign * self.cfg.step)), 3)
        self.angular = round(min(self.cfg.max_angular, self.linear * ratio), 3)
        if self.cmd != (0.0, 0.0, 0.0):        # rescale a held command immediately
            f, l, r = (0 if v == 0 else (1 if v > 0 else -1) for v in self.cmd)
            self.cmd = (f * self.linear, l * self.linear, r * self.angular)
        return "speed"

    def stop(self):
        self.cmd = (0.0, 0.0, 0.0)
        self.last_move_t = float("-inf")

    def velocity(self, t: float):
        """Current (vx, vy, wz) command, applying the dead-man timeout."""
        if not self.cfg.latch and t - self.last_move_t > self.cfg.hold_s:
            self.cmd = (0.0, 0.0, 0.0)
        return self.cmd
