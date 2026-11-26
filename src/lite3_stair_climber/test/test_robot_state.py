"""0x0901 robot state parsing and gait / AI switches that are sent again until the robot reports them."""
import struct

from stair_climber.robot_state import (AI_ENTER_CODE, AI_EXIT_CODE, describe, parse, target_reached)
from stair_climber.state_machine import Observation, SMParams, State, StairStateMachine

HSTEP, GENERAL = 0x21010407, 0x21010401


def packet(basic, gait, policy, motion, battery=87.0, packed=False):
    fmt = '<3i18dI?Iidi4?2d' if packed else '<3i4x18dI?3xIidi4?2d'
    body = struct.pack(fmt, basic, gait, policy, *[0.0] * 18, 0, False, 0, motion, battery, 0,
                       False, False, False, False, 0.0, 0.0)
    return struct.pack('<3I', 0x0901, len(body), 1) + body


def test_parse_aligned_and_packed():
    for packed in (False, True):
        s = parse(packet(6, 13, 0, 1, packed=packed))
        assert s.mode == (6, 13, 0, 1) and s.battery == 87.0
    assert len(packet(0, 0, 0, 0)) == 12 + 208
    assert parse(struct.pack('<3I', 0x0902, 4, 1) + b'\0' * 4) is None
    assert describe((16, 3, 0, 1)) == '(16,0,1) AI basic gait'


def test_targets():
    assert target_reached(HSTEP, (6, 13, 0, 1))
    assert not target_reached(HSTEP, (6, 2, 0, 1))
    assert target_reached(AI_ENTER_CODE, (16, 0, 0, 1))
    assert not target_reached(AI_ENTER_CODE, (16, 0, 20, 1))
    assert target_reached(AI_EXIT_CODE, (6, 0, 0, 0))
    assert not target_reached(AI_EXIT_CODE, (16, 0, 0, 1))
    assert target_reached(0x21010D06, (6, 0, 0, 0)) is None          # Move Mode: nothing to check


def switching(code, label_state=State.SWITCH_TO_STAIR_GAIT, **kw):
    sm = StairStateMachine(SMParams(stair_gait_code=code, flat_gait_code=GENERAL, gait_retry_s=3.0, **kw))
    sm.state = label_state
    return sm


def run(sm, state_at, seconds, dt=0.25):
    """state_at(t) -> robot state tuple or None. Returns the times a gait code was sent."""
    sent, t = [], 0.0
    while t < seconds and sm.state not in (State.APPROACH, State.ABORT, State.CROSS_AI, State.SWITCH_TO_FLAT_GAIT):
        cmd = sm.update(Observation(t=t, robot_state=state_at(t)))
        if cmd.gait_code:
            sent.append(round(t, 2))
        t += dt
    return sent


def test_gait_resent_until_robot_reports_it():
    sm = switching(HSTEP)
    sent = run(sm, lambda t: (6, 13, 0, 1) if t >= 7.5 else (6, 2, 0, 1), 30)
    assert sent == [0.0, 3.0, 6.0]                                    # sent again every 3 s until it took
    assert sm.state == State.APPROACH


def test_gait_already_reported_moves_on_without_waiting_the_full_switch_wait():
    sm = switching(HSTEP, gait_switch_wait=5.0)
    sent = run(sm, lambda t: (6, 13, 0, 0), 30)
    assert sent == [0.0] and sm.state == State.APPROACH
    assert sm.transitions[-1][0] < 1.0


def test_no_robot_state_falls_back_to_the_timed_wait():
    sm = switching(HSTEP, gait_switch_wait=2.0)
    assert run(sm, lambda t: None, 30) == [0.0]
    assert sm.state == State.APPROACH and 'not checked' in sm.transitions[-1][3]


def test_max_attempts_aborts():
    sm = switching(HSTEP, gait_max_attempts=2)
    assert run(sm, lambda t: (6, 2, 0, 1), 30) == [0.0, 3.0]
    assert sm.state == State.ABORT and '2 attempts' in sm.abort_reason


def test_enter_ai_resent_until_ai_basic_gait():
    sm = switching(HSTEP, use_ai_gait=True, ai_switch_wait=4.0)
    sm.state = State.ENTER_AI
    sent = run(sm, lambda t: (16, 0, 0, 1) if t >= 9.0 else (6, 0, 0, 0), 30)
    assert sent == [0.0, 4.0, 8.0]                                    # retry period = max(gait_retry_s, ai_switch_wait)
    assert sm.state == State.CROSS_AI


def test_exit_ai_resent_until_back_in_torque_control():
    sm = switching(HSTEP, use_ai_gait=True, ai_switch_wait=3.0)
    sm.state = State.EXIT_AI
    sent = run(sm, lambda t: (6, 0, 0, 0) if t >= 4.0 else (16, 0, 0, 1), 30)
    assert sent == [0.0, 3.0]
    assert sm.state == State.SWITCH_TO_FLAT_GAIT
