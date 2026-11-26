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
  * OVERRIDE (set_override): a gait that wins over the terrain, e.g. rug_hstep while the robot drives onto a higher tile.
    Setting or clearing it arms a switch; cleared, the terrain decides again. It also wins over a FREEZE (recovery).
"""
from dataclasses import dataclass
from typing import Optional

from lite3_explorer.robot_state import GAIT_STATE_BY_CODE

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
    def __init__(self, flat_gait: str = 'flat_slow', rugged_gait: str = 'rug_hstep',
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
        self.override: Optional[str] = None      # gait that wins over the terrain (None = the terrain decides)
        self.frozen = False                      # keep the gait the robot has, whatever happens (recovery mode)

    def set_override(self, gait: Optional[str]) -> bool:
        """Force `gait` (None = back to the terrain gait). True when it changed."""
        if gait is not None and gait not in GAIT_CODES:
            raise ValueError(f'unknown gait {gait!r}; use one of {sorted(GAIT_CODES)}')
        if gait == self.override:
            return False
        self.override, self.armed = gait, True
        return True

    def freeze(self, on: bool) -> bool:
        """Keep the gait the robot already has (on) / let the terrain, target and override rules switch again (off).
        Terrain changes while frozen are still recorded, but nothing is armed by them: after the freeze the gait changes
        only at the next new tile or new terrain, never the moment the freeze ends. The override (step up onto a higher
        tile) still switches while frozen; cleared while frozen, the robot keeps the gait it has. True when it changed."""
        if bool(on) == self.frozen:
            return False
        self.frozen, self.armed = bool(on), False
        return True

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
        if self.override is not None:
            return self.override
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
        if self.frozen and self.override is None:
            self.armed = False                   # whatever armed it meanwhile is dropped (see freeze)
            return None
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


class GaitConfirm:
    """Re-send a gait until /robot_state reports it (no ROS).

    A gait command can be lost or refused; the burst of command_repeats only covers the first ~0.6 s. While a gait must
    be confirmed (the step-up gait while /step_up is on), update() says when to send it again: every retry_s after the
    last publish, until the robot's robot_gait_state matches (robot_state.GAIT_STATE_BY_CODE). Once it matched, nothing
    more is sent for that gait (a state glitch on the step edge must not re-command the robot); a new gait to confirm, or
    None in between, starts afresh. steptest_26_11_133619: rug_hstep was accepted ~2.3 s after the first publish.
    """

    def __init__(self, retry_s: float = 1.5):
        self.retry_s = float(retry_s)
        self.gait: Optional[str] = None
        self.confirmed = False
        self.retries = 0
        self.event: Optional[str] = None         # set on the tick something changes (for logging), else None

    def update(self, now: float, gait: Optional[str], robot_gait: Optional[int], last_publish_at: Optional[float]) -> bool:
        """gait: the gait that must be confirmed now (None = nothing). robot_gait: robot_gait_state while standing /
        stepping, None = unknown / stale. last_publish_at: time of the last /simple_cmd publish of any gait.
        True = send `gait` again now."""
        self.event = None
        if gait != self.gait:
            self.gait, self.confirmed, self.retries = gait, False, 0
        if gait is None or self.confirmed or self.retry_s <= 0:
            return False
        want = GAIT_STATE_BY_CODE.get(GAIT_CODES.get(gait))
        if want is None:                          # no robot state for this gait (crawl): cannot confirm
            return False
        if robot_gait == want:
            self.confirmed = True
            self.event = f'robot reports {gait} (gait state {want})' + (f' after {self.retries} re-send(s)'
                                                                        if self.retries else '')
            return False
        if last_publish_at is None or now - last_publish_at >= self.retry_s:
            self.retries += 1
            self.event = (f'robot still not in {gait} (gait state {robot_gait}, want {want}): '
                          f're-sending, attempt {self.retries}')
            return True
        return False


class SwitchHold:
    """Gait switches only while the robot stands still (no ROS).

    request(cmd) starts a hold: the robot is sent zero velocity for stop_s first, then the command is sent (update()
    returns it once), and the hold lasts until the robot reports the new gait (robot_state.GAIT_STATE_BY_CODE), but at
    least settle_s after the send (the burst of repeats) and, when the gait cannot be confirmed (no /robot_state, crawl),
    exactly settle_s. The whole hold never lasts more than stop_s + max_wait_s (gait refused / lost: drive on, so stuck
    detection can still act). A request while a hold runs replaces the command; if the robot is already held past stop_s
    it is sent at once (the robot is still), and the hold time limit is not extended.
    """

    def __init__(self, stop_s: float = 0.4, settle_s: float = 1.0, max_wait_s: float = 3.0):
        self.stop_s = max(0.0, float(stop_s))
        self.settle_s = max(0.0, float(settle_s))
        self.max_wait_s = max(self.settle_s, float(max_wait_s))
        self.cancel()

    def cancel(self):
        self.started_at: Optional[float] = None   # hold start (None = robot free)
        self.cmd: Optional[GaitCommand] = None    # waiting for the robot to stop
        self.sent: Optional[GaitCommand] = None
        self.sent_at: Optional[float] = None
        self.event: Optional[str] = None          # set on the tick the hold ends (for logging), else None

    @property
    def holding(self) -> bool:
        return self.started_at is not None

    def request(self, cmd: GaitCommand, now: float):
        if self.started_at is None:
            self.started_at = now
        self.cmd = cmd

    def update(self, now: float, robot_gait: Optional[int]) -> Optional[GaitCommand]:
        """robot_gait: robot_gait_state while standing / stepping, None = unknown / stale. Returns the command to send
        now (once per request), else None; afterwards `holding` says whether to keep the robot still."""
        self.event = None
        if self.started_at is None:
            return None
        if self.cmd is not None:
            if now - self.started_at >= self.stop_s:
                self.sent, self.sent_at, self.cmd = self.cmd, now, None
                return self.sent
            return None
        want = GAIT_STATE_BY_CODE.get(self.sent.cmd_code)
        waited = now - self.sent_at
        if now - self.started_at >= self.stop_s + self.max_wait_s:
            self.event = (f'robot still not in {self.sent.gait} (gait state {robot_gait}) after '
                          f'{now - self.started_at:.1f} s: driving on anyway')
        elif waited < self.settle_s:
            return None
        elif want is not None and robot_gait is not None and robot_gait != want:
            return None
        else:
            self.event = (f'{self.sent.gait} ' + ('confirmed' if want is not None and robot_gait == want
                                                  else 'not confirmable, settle time over')
                          + f' after {now - self.started_at:.1f} s still: driving on')
        self.cancel_keep_event()
        return None

    def cancel_keep_event(self):
        event = self.event
        self.cancel()
        self.event = event
