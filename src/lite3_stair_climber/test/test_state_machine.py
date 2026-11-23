"""State machine tests against a tiny fake world. Run: python3 -m pytest test/

The fake world is a 1-D track: the first step edge is at EDGE metres ahead.
It produces the IMU pitch / odom height a real climb would roughly produce:

    x <  EDGE-0.2          flat floor
    EDGE-0.2 .. EDGE+0.6   climbing   (nose up 25 deg, z rising)
    EDGE+0.6 .. EDGE+1.4   platform   (level, z = 0.36)
    EDGE+1.4 .. EDGE+2.2   descending (nose down 25 deg, z falling)
    beyond                 flat floor
"""

import math

from stair_climber.stair_detector import StairObservation
from stair_climber.state_machine import (Observation, SMParams, State,
                                         StairStateMachine)

EDGE = 1.6
STAIR, FLAT = 0x2101AAAA, 0x21010300   # fake stair code, flat code
DT = 0.05


def world(x):
    """Return (nose_up_deg, z) at robot position x."""
    a = EDGE - 0.2
    if x < a:
        return 0.0, 0.0
    if x < EDGE + 0.6:
        return 25.0, 0.36 * (x - a) / (EDGE + 0.6 - a)
    if x < EDGE + 1.4:
        return 0.0, 0.36
    if x < EDGE + 2.2:
        return -25.0, 0.36 * (1 - (x - EDGE - 1.4) / 0.8)
    return 0.0, 0.0


class Sim:
    def __init__(self, params, vlm=None, stuck_after=None, roll_at=None):
        self.sm = StairStateMachine(params)
        self.x = 0.0
        self.t = 0.0
        self.seq = 0
        self.vlm_answer = vlm          # what the "VLM" answers (None = never answers)
        self.vlm = None
        self.stuck_after = stuck_after
        self.roll_at = roll_at
        self.gaits = []
        self.vlm_requests = 0
        self.view = (0.3, 1.6)         # distances at which the lidar can see the first step

    def detection(self):
        d = EDGE - self.x
        if self.view[0] <= d <= self.view[1]:
            return StairObservation(True, 'ok', distance=d, step_height=0.12,
                                    edge_yaw=0.0, center_y=0.0, width=0.8, num_steps=3)
        return StairObservation(False, 'out of view')

    def step(self):
        nose_up, z = world(self.x)
        # new lidar result every 4 ticks (5 Hz)
        if round(self.t / DT) % 4 == 0:
            self.seq += 1
        roll = math.radians(30) if self.roll_at is not None and self.x > self.roll_at else 0.0
        obs = Observation(t=self.t, roll=roll, pitch=-math.radians(nose_up), x=self.x, z=z,
                          stair=self.detection(), stair_seq=self.seq, vlm=self.vlm)
        cmd = self.sm.update(obs)
        if cmd.gait_code:
            self.gaits.append(cmd.gait_code)
        if cmd.request_vlm:
            self.vlm_requests += 1
            self.vlm = self.vlm_answer
        stuck = self.stuck_after is not None and self.x >= self.stuck_after
        if not stuck:
            self.x += cmd.vx * DT
        self.t += DT
        return cmd

    def run(self, seconds=120, until=(State.DONE, State.ABORT, State.NOT_STAIRS)):
        for _ in range(int(seconds / DT)):
            cmd = self.step()
            if cmd.state in until:
                return cmd
        return cmd


def params(**kw):
    base = dict(search_speed=0.2, stair_gait_code=STAIR, flat_gait_code=FLAT)
    base.update(kw)
    return SMParams(**base)


def states_visited(sim):
    return [t[2] for t in sim.sm.transitions]


def test_full_crossing_reaches_done():
    sim = Sim(params())
    cmd = sim.run()
    assert cmd.state == State.DONE, sim.sm.transitions
    assert states_visited(sim) == [
        State.SEARCH, State.ALIGN, State.SWITCH_TO_STAIR_GAIT, State.APPROACH,
        State.CLIMB_UP, State.TOP_PLATFORM, State.DESCEND, State.CLEAR_BOTTOM,
        State.SWITCH_TO_FLAT_GAIT, State.DONE]
    assert sim.gaits == [STAIR, FLAT]
    assert sim.x > EDGE + 2.2          # actually got past the stairs
    assert not cmd.active              # control handed back


def test_aligns_at_standoff_before_switching_gait():
    sim = Sim(params())
    while sim.sm.state != State.SWITCH_TO_STAIR_GAIT:
        sim.step()
    assert abs((EDGE - sim.x) - 0.45) < 0.06


def test_missing_gait_code_aborts():
    sim = Sim(params(stair_gait_code=0))
    cmd = sim.run()
    assert cmd.state == State.ABORT
    assert 'gait code' in sim.sm.abort_reason
    assert cmd.vx == 0 and cmd.active   # holds still and keeps control


def test_roll_aborts_mid_climb():
    sim = Sim(params(), roll_at=EDGE + 0.2)
    cmd = sim.run()
    assert cmd.state == State.ABORT
    assert 'roll' in sim.sm.abort_reason


def test_stuck_on_stairs_aborts():
    sim = Sim(params(), stuck_after=EDGE + 0.1)
    cmd = sim.run()
    assert cmd.state == State.ABORT
    assert 'stuck' in sim.sm.abort_reason


def test_vlm_confirms():
    sim = Sim(params(use_vlm=True), vlm=True)
    assert sim.run().state == State.DONE
    assert sim.vlm_requests == 1


def test_vlm_rejects_twice_gives_not_stairs():
    sim = Sim(params(use_vlm=True), vlm=False)
    cmd = sim.run()
    assert cmd.state == State.NOT_STAIRS
    assert sim.vlm_requests == 2
    assert not cmd.active


def test_vlm_timeout_falls_back_to_lidar():
    sim = Sim(params(use_vlm=True, vlm_timeout=3.0), vlm=None)
    assert sim.run().state == State.DONE


def test_disable_mid_climb_goes_idle_and_releases():
    sim = Sim(params())
    while sim.sm.state != State.CLIMB_UP:
        sim.step()
    cmd = sim.sm.update(Observation(t=sim.t, enabled=False))
    assert cmd.state == State.IDLE and not cmd.active


def test_stale_imu_aborts():
    sm = StairStateMachine(params())
    sm.update(Observation(t=0.0))
    cmd = sm.update(Observation(t=0.05, imu_age=2.0))
    assert cmd.state == State.ABORT


def test_short_top_step_goes_straight_to_descend(monkeypatch):
    """Arena stairs: the 18 cm top step is shorter than the robot, so the body
    pitches from nose-up straight to nose-down with no level platform."""
    import test_state_machine as me

    def peaked(x):
        a, top, b = EDGE - 0.2, EDGE + 0.4, EDGE + 1.0
        if x < a or x >= b:
            return 0.0, 0.0
        if x < top:
            return 25.0, 0.515 * (x - a) / (top - a)
        return -25.0, 0.515 * (1 - (x - top) / (b - top))

    monkeypatch.setattr(me, 'world', peaked)
    sim = Sim(params())
    cmd = sim.run()
    assert cmd.state == State.DONE, (sim.sm.transitions, sim.sm.abort_reason)
    assert State.TOP_PLATFORM not in states_visited(sim)
    assert State.DESCEND in states_visited(sim)


def test_blind_zone_long_approach():
    """Real robot: lidar sees the floor only beyond ~1.3 m, so align far (1.7 m)
    and walk the rest blind until the body pitches up."""
    global EDGE
    import test_state_machine as me
    old = me.EDGE
    me.EDGE = 3.5
    try:
        sim = Sim(params(engage_distance=2.4, align_standoff=1.7, approach_max_distance=2.3,
                         approach_speed=0.06, climb_speed=0.06, approach_timeout=50.0))
        sim.view = (1.3, 2.6)
        cmd = sim.run(seconds=300)
        assert cmd.state == State.DONE, (sim.sm.transitions, sim.sm.abort_reason)
    finally:
        me.EDGE = old
