"""mission_manager: the depth fallback for /step_up (step_ahead_detector), with the node stubbed out."""
import json
import types

from std_msgs.msg import String

from lite3_explorer.mission_manager import MissionManager


class Stub:
    _update_step_up = MissionManager._update_step_up
    _update_depth_step = MissionManager._update_depth_step
    _on_step_ahead = MissionManager._on_step_ahead

    def __init__(self, cfg=None):
        self.cfg = {'step_up_gait': True, 'step_up_depth': True, 'step_up_depth_max_age_s': 1.0, **(cfg or {})}
        self.t = 100.0
        self.target = ('1,1', None)
        self._step_tile = None
        self._gait_frozen, self._frozen_step_up = False, True
        self._step_up, self._hstep_hold = None, False
        self._rise_target, self._rise_from, self._rise_max = None, 0.0, None
        self._step_ahead, self._depth_tile, self._depth_step = None, None, None
        self.tiles = types.SimpleNamespace(ready=False)
        self.published, self.events = [], []
        self.step_up_pub = types.SimpleNamespace(publish=lambda m: self.published.append(m.data))

    def now(self):
        return self.t

    def log(self, level, msg):
        pass

    def event(self, kind, **data):
        self.events.append((kind, data))


def step(on=True):
    return String(data=json.dumps({'step': on, 'rise_m': 0.13, 'dist_m': 0.6, 'at_edge': False} if on else {'step': False}))


def test_depth_step_switches_step_up_and_holds_until_the_tile_changes():
    m = Stub()
    m._update_step_up()
    assert m.published == [False]
    m._on_step_ahead(step(True))
    assert m.published == [False, True]
    assert m.events[-1][1]['by'] == 'depth'
    m._on_step_ahead(step(False))            # the step left the camera view: still held for this tile
    m.t += 5.0
    m._update_step_up()
    assert m.published[-1] is True
    m.target = ('2,1', None)                  # next tile: the old detection is too old to count
    m._update_step_up()
    assert m.published[-1] is False


def test_fresh_detection_carries_over_to_a_new_tile():
    m = Stub()
    m._on_step_ahead(step(True))
    m.target = ('2,1', None)
    m.t += 0.5
    m._update_step_up()
    assert m.published[-1] is True


def test_no_target_or_disabled_means_no_depth_step_up():
    m = Stub()
    m.target = None
    m._on_step_ahead(step(True))
    assert True not in m.published
    m = Stub({'step_up_depth': False})
    m._on_step_ahead(step(True))
    m._update_step_up()
    assert True not in m.published
