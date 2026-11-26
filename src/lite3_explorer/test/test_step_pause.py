"""Balance pauses while stepping up onto a higher tile (step_pause.py, no ROS)."""
from lite3_explorer.step_pause import StepPause, StepPauseParams

DT = 0.02


def run(nose_of_t, step_up=True, t_end=4.0, driving=True, **kw):
    """Tick at 50 Hz; returns the list of (start, end) pause intervals."""
    sp = StepPause(StepPauseParams(**kw))
    pauses, start, t = [], None, 0.0
    while t < t_end:
        held = sp.update(t, step_up(t) if callable(step_up) else step_up, nose_of_t(t), driving)
        if held and start is None:
            start = t
        if not held and start is not None:
            pauses.append((round(start, 2), round(t, 2)))
            start = None
        t += DT
    return pauses


def step(t):
    """Flat, front legs up the step at 1.0 s (nose +12 deg), rear legs up by 2.0 s, level on top after."""
    return 12.0 if 1.0 <= t < 2.0 else 0.0


def test_two_short_pauses_on_one_step_up():
    p = run(step)
    assert len(p) == 2
    (a0, a1), (b0, b1) = p
    assert 1.0 <= a0 <= 1.2                                   # right after the tilt is confirmed
    assert abs((a1 - a0) - 0.2) < 0.03 and abs((b1 - b0) - 0.2) < 0.03
    assert abs((b0 - a0) - 0.7) < 0.03


def test_one_pause_when_configured():
    assert len(run(step, pause_offsets_s=[0.0])) == 1


def test_no_pause_without_step_up_or_imu_or_nav2_driving():
    assert run(step, step_up=False) == []
    assert run(lambda t: None) == []
    assert run(step, driving=False) == []


def test_a_brief_wobble_is_not_a_climb():
    assert run(lambda t: 12.0 if 1.0 <= t < 1.05 else 0.0) == []


def test_a_second_step_on_the_way_pauses_again_once_level_in_between():
    two = lambda t: 12.0 if (1.0 <= t < 2.0 or 3.5 <= t < 4.5) else 0.0   # noqa: E731
    assert len(run(two, t_end=6.0)) == 4


def test_step_up_off_ends_a_pause_and_rearms():
    p = run(step, step_up=lambda t: t < 1.15, pause_s=1.0)
    assert len(p) == 1 and p[0][1] <= 1.16


# ---------------------------------------------------------------- GaitWait: hold until the robot is in h-step
from lite3_explorer.step_pause import GaitWait  # noqa: E402


def hold_times(gait_of_t, step_up=True, t_end=6.0, driving=True, **kw):
    gw = GaitWait(**kw)
    held, t = [], 0.0
    while t < t_end:
        if gw.update(t, step_up(t) if callable(step_up) else step_up, gait_of_t(t), driving):
            held.append(round(t, 2))
        t += DT
    return gw, held


def test_holds_until_hstep_reported():
    gw, held = hold_times(lambda t: 13 if t >= 1.0 else 0)
    assert held and held[0] == 0.0 and max(held) < 1.0
    assert gw.accepted and not gw.gave_up


def test_no_hold_again_after_acceptance():
    # robot state glitches back to flat mid-step: never stop on the edge
    gw, held = hold_times(lambda t: 13 if 1.0 <= t < 2.0 else 0)
    assert max(held) < 1.0


def test_no_hold_without_step_up_or_when_not_driving():
    assert hold_times(lambda t: 0, step_up=False)[1] == []
    assert hold_times(lambda t: 0, driving=False)[1] == []


def test_gives_up_after_max_wait():
    gw, held = hold_times(lambda t: None, max_wait_s=2.0)
    assert gw.gave_up and max(held) < 2.0 and min(held) == 0.0


def test_new_step_up_waits_again():
    gw, held = hold_times(lambda t: 13 if 1.0 <= t < 2.0 else 0, step_up=lambda t: t < 3.0 or t >= 4.0)
    assert any(t >= 4.0 for t in held)
