"""Combines the fast local classifier and the slow VLM into one surface state and a
latched "course complete" decision. Pure Python (no ROS) so it can be unit-tested.

Modes:
  hybrid     (default) local classifier drives the state every frame; the VLM is asked on
             every surface change and periodically, can override a weak local vote, and must
             confirm an exit before course_complete is latched (local-only fallback if the
             gateway gives no answer within vlm_confirm_timeout_s).
  vlm_only   state = latest fresh VLM answer; exit needs 2 consecutive confident "other".
  local_only no network at all.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

COURSE = ("wood", "metal")
LABELS = ("wood", "metal", "other")


@dataclass
class FusionConfig:
    mode: str = "hybrid"
    local_window_s: float = 1.0        # majority vote window for the local classifier
    local_majority: float = 0.6        # share of votes needed to switch state
    vlm_max_age_s: float = 15.0        # ignore VLM answers about older images
    vlm_override_conf: float = 0.85    # VLM may override a weak local vote above this
    vlm_min_interval_s: float = 3.0    # never send requests closer than this
    vlm_heartbeat_s: float = 8.0       # periodic check even without a change
    min_course_time_s: float = 3.0     # must have been on wood/metal this long before an exit counts
    exit_hold_s: float = 2.0           # "other" must persist this long
    vlm_exit_conf: float = 0.6         # VLM confidence needed to confirm / veto an exit
    vlm_confirm_timeout_s: float = 15.0  # after this, exit on local evidence alone
    camera_timeout_s: float = 1.0      # no frames for this long -> state "unknown"


@dataclass
class VlmAnswer:
    surface: str
    confidence: float
    t_img: float      # time the image was captured
    t_rx: float       # time the answer arrived


class SurfaceFusion:
    def __init__(self, cfg: FusionConfig):
        if cfg.mode not in ("hybrid", "vlm_only", "local_only"):
            raise ValueError(f"unknown mode {cfg.mode}")
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.votes = deque()
        self.vlm_last: Optional[VlmAnswer] = None
        self.vlm_history = deque(maxlen=5)
        self.state = "unknown"
        self.state_since = 0.0
        self.on_course_time = 0.0
        self.last_step_t: Optional[float] = None
        self.last_frame_t: Optional[float] = None
        self.exit_candidate_since: Optional[float] = None
        self.course_complete = False
        self.complete_reason = ""
        self.complete_t: Optional[float] = None
        self.urgent = True
        self.vlm_in_flight = False
        self.last_vlm_request_t = float("-inf")
        self.vlm_fail_streak = 0
        self.last_local_label = "unclear"
        self.last_local_frac = 0.0

    # ---------- inputs ----------
    def add_local(self, t: float, label: str):
        self.last_frame_t = t
        self.votes.append((t, label))

    def mark_vlm_requested(self, t: float):
        self.vlm_in_flight = True
        self.last_vlm_request_t = t
        self.urgent = False

    def add_vlm(self, t_rx: float, t_img: float, ok: bool, surface: str = "unclear",
                confidence: float = 0.0):
        self.vlm_in_flight = False
        if not ok:
            self.vlm_fail_streak += 1
            return
        self.vlm_fail_streak = 0
        if surface in LABELS:
            ans = VlmAnswer(surface, confidence, t_img, t_rx)
            self.vlm_last = ans
            self.vlm_history.append(ans)

    # ---------- helpers ----------
    def _local_majority(self, t: float):
        while self.votes and t - self.votes[0][0] > self.cfg.local_window_s:
            self.votes.popleft()
        if not self.votes:
            return None, 0.0
        counts = {}
        for _, lab in self.votes:
            if lab in LABELS:
                counts[lab] = counts.get(lab, 0) + 1
        if not counts:
            return None, 0.0
        lab = max(counts, key=counts.get)
        return lab, counts[lab] / len(self.votes)

    def _vlm_fresh(self, t: float) -> Optional[VlmAnswer]:
        v = self.vlm_last
        if v and t - v.t_img <= self.cfg.vlm_max_age_s:
            return v
        return None

    def should_query_vlm(self, t: float) -> bool:
        if self.cfg.mode == "local_only" or self.vlm_in_flight or self.course_complete:
            return False
        since = t - self.last_vlm_request_t
        if since < self.cfg.vlm_min_interval_s:
            return False
        if self.cfg.mode == "vlm_only":
            return True
        return self.urgent or since >= self.cfg.vlm_heartbeat_s

    def _complete(self, t: float, reason: str):
        self.course_complete = True
        self.complete_reason = reason
        self.complete_t = t

    # ---------- main update ----------
    def step(self, t: float) -> str:
        cfg = self.cfg
        dt = 0.0 if self.last_step_t is None else max(0.0, t - self.last_step_t)
        self.last_step_t = t

        # 1. decide the current surface
        if self.last_frame_t is None or t - self.last_frame_t > cfg.camera_timeout_s:
            new = "unknown"
        elif cfg.mode == "vlm_only":
            v = self._vlm_fresh(t)
            new = v.surface if v else "unknown"
        else:
            lab, frac = self._local_majority(t)
            self.last_local_label, self.last_local_frac = lab or "unclear", frac
            new = lab if (lab and frac >= cfg.local_majority) else self.state
            v = self._vlm_fresh(t)
            if (cfg.mode == "hybrid" and v and v.surface != new
                    and v.confidence >= cfg.vlm_override_conf and frac < 0.8
                    and v.t_img >= self.state_since):
                new = v.surface

        if new != self.state:
            self.state = new
            self.state_since = t
            self.urgent = True      # ask the VLM to look at the new surface

        if self.state in COURSE:
            self.on_course_time += dt

        # 2. exit / course-complete logic (latched)
        if not self.course_complete:
            armed = self.on_course_time >= cfg.min_course_time_s
            if armed and self.state == "other":
                if self.exit_candidate_since is None:
                    self.exit_candidate_since = self.state_since
                    self.urgent = True
                held = t - self.exit_candidate_since >= cfg.exit_hold_s
                if cfg.mode == "local_only":
                    if held:
                        self._complete(t, "local classifier held 'other'")
                elif cfg.mode == "vlm_only":
                    recent = [a for a in self.vlm_history if a.t_img >= self.exit_candidate_since]
                    if (len(recent) >= 2 and all(a.surface == "other" and a.confidence >= cfg.vlm_exit_conf
                                                 for a in recent[-2:])):
                        self._complete(t, "2 consecutive VLM 'other'")
                else:  # hybrid
                    v = self.vlm_last
                    if v and v.t_img >= self.exit_candidate_since and v.confidence >= cfg.vlm_exit_conf:
                        if v.surface == "other" and held:
                            self._complete(t, "local + VLM agree: off course")
                        elif v.surface in COURSE:
                            # VLM vetoes: restart the hold and ask again
                            self.exit_candidate_since = t
                            self.urgent = True
                    elif held and t - self.exit_candidate_since >= cfg.vlm_confirm_timeout_s:
                        self._complete(t, "local only (no VLM confirmation in time)")
            else:
                self.exit_candidate_since = None
        return self.state
