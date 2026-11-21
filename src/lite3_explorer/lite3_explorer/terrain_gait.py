"""Terrain state and gait selection for the Lite3 (no ROS imports, so it can be unit tested).

/terrain_type (from lite3_terrain_classifier) is turned into a terrain state, and the state
picks a Lite3 gait. Gaits are switched with a MotionSimpleCMD {cmd_code, type: 0} on
/simple_cmd, the same as the lite3_<gait> shell aliases (names below match them).

Rules:
  * RUGGED switches to the rugged gait at once (safety first).
  * FLAT switches back only after the terrain has been flat for flat_delay_s, so the robot
    keeps the rugged gait until it is fully past the step.
  * UNKNOWN (no sensor data, classifier paused) keeps the current gait.
  * Two commands are never sent closer together than min_switch_interval_s.
  * EVENT-DRIVEN: the gait is only changed after arm() (the robot starts for a new tile, or a hold such as the spin
    ended and the terrain gait must come back) or when the terrain label changes (a new terrain was classified).
    Between those events the gait is left alone, whatever the label says.
"""
from dataclasses import dataclass
from typing import Optional

GAIT_CODES = {
    'flat_slow': 0x21010300,
    'flat_medium': 0x21010307,
    'flat_fast': 0x21010303,
    'flat_crawl': 0x21010406,
    'rug_general': 0x21010401,
    'rug_grip': 0x21010402,
    'rug_hstep': 0x21010407,
}

FLAT, RUGGED, UNKNOWN = 'FLAT', 'RUGGED', 'UNKNOWN'

_LABELS = {'rugged': RUGGED, 'non_rugged': FLAT}


def terrain_from_label(label: str) -> str:
    """Map a /terrain_type label (rugged / non_rugged / unknown) to a terrain state."""
    return _LABELS.get(label.strip().lower(), UNKNOWN)


@dataclass
class GaitCommand:
    gait: str
    cmd_code: int


class TerrainGaitSwitcher:
    def __init__(self, flat_gait: str = 'flat_slow', rugged_gait: str = 'rug_general',
                 flat_delay_s: float = 2.0, min_switch_interval_s: float = 1.0):
        for gait in (flat_gait, rugged_gait):
            if gait not in GAIT_CODES:
                raise ValueError(f'unknown gait {gait!r}; use one of {sorted(GAIT_CODES)}')
        self.flat_gait = flat_gait
        self.rugged_gait = rugged_gait
        self.flat_delay_s = max(0.0, float(flat_delay_s))
        self.min_switch_interval_s = max(0.0, float(min_switch_interval_s))
        self.terrain = UNKNOWN
        self.terrain_since: Optional[float] = None
        self.gait: Optional[str] = None          # last gait sent (None = nothing sent yet)
        self.last_sent_at: Optional[float] = None
        self.armed = False                       # a switch is allowed (see the module docstring)

    def arm(self):
        """Allow the next switch: the robot is attempting a new tile."""
        self.armed = True

    def restore(self):
        """A hold (spin gait, animation) ended: forget the gait sent and bring the terrain gait back."""
        self.gait, self.armed = None, True

    def update_terrain(self, label: str, now: float) -> bool:
        """Feed a /terrain_type label. Returns True when the terrain state changed."""
        terrain = terrain_from_label(label)
        if terrain == self.terrain:
            return False
        self.terrain, self.terrain_since = terrain, now
        self.armed = True                        # a new terrain was classified
        return True

    def desired_gait(self, now: float) -> Optional[str]:
        """Gait the robot should be in now, or None to keep whatever it has."""
        if self.terrain == RUGGED:
            return self.rugged_gait
        if self.terrain == FLAT:
            if self.gait is None or self.gait == self.flat_gait:
                return self.flat_gait
            if now - self.terrain_since >= self.flat_delay_s:
                return self.flat_gait
        return None

    def pending_command(self, now: float) -> Optional[GaitCommand]:
        """The command to send now, or None. Call mark_sent() after publishing it."""
        if not self.armed:
            return None
        gait = self.desired_gait(now)
        if gait is None:
            if self.terrain != FLAT:             # nothing to apply (terrain unknown); flat: still waiting out flat_delay_s
                self.armed = False
            return None
        if gait == self.gait:
            self.armed = False
            return None
        if self.last_sent_at is not None and now - self.last_sent_at < self.min_switch_interval_s:
            return None
        return GaitCommand(gait, GAIT_CODES[gait])

    def mark_sent(self, command: GaitCommand, now: float):
        self.gait, self.last_sent_at, self.armed = command.gait, now, False
