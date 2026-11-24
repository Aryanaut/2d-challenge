from lite3_explorer.animations import (ANIMATIONS, LEAN, MOVE_MODE, POSE, POSE_MODE, YAW, MovementAnimation,
                                       PoseAnimation, QuickScan)


def run(anim, dt=0.05):
    """Tick an animation to completion; returns [(t, cmds)]."""
    anim.start(0.0)
    log, t = [], 0.0
    while not anim.finished and t < 60:
        log.append((t, anim.tick(t)))
        t += dt
    return log


def test_registry_and_category():
    assert ANIMATIONS['quickscan'] is QuickScan
    assert QuickScan.category == POSE and issubclass(QuickScan, PoseAnimation)
    assert MovementAnimation.category == 'movement'


def test_pose_mode_first_and_move_mode_last():
    log = run(QuickScan())
    flat = [(t, c) for t, cmds in log for c in cmds]
    assert flat[0][1] == (POSE_MODE, 0)
    assert flat[-1][1] == (MOVE_MODE, 0)
    # no pose command after move mode starts, none before pose mode
    first_move = next(i for i, (_, c) in enumerate(flat) if c[0] == MOVE_MODE)
    assert all(c[0] == MOVE_MODE for _, c in flat[first_move:])
    assert all(c[0] != MOVE_MODE for _, c in flat[:first_move])


def test_pose_is_held_every_tick():
    log = run(QuickScan(mode_settle_s=0.0))
    body = [cmds for t, cmds in log if any(c[0] == LEAN for c in cmds)]
    assert all({c[0] for c in cmds if c[0] in (LEAN, YAW)} == {LEAN, YAW} for cmds in body)


def test_pauses_at_forward_left_right_and_configurable():
    for pause in (0.5, 1.5):
        a = QuickScan(pause_s=pause, move_s=1.0, lean=20000, yaw=20000)
        # forward lean reached at t=1, held through t=1+pause
        assert a.pose_at(1.0)[LEAN] == 20000
        assert a.pose_at(1.0 + pause - 0.01) == {LEAN: 20000, YAW: 0}
        left_t = 2.0 + pause
        assert a.pose_at(left_t)[YAW] == -20000 and a.pose_at(left_t + 0.5)[YAW] > -20000   # no hold at left
        right_t = left_t + 2.0
        assert a.pose_at(left_t)[LEAN] == 20000 and a.pose_at(right_t)[LEAN] == 20000   # still leaning
        assert a.pose_at(right_t)[YAW] == 20000 and a.pose_at(right_t + pause - 0.01)[YAW] == 20000
        assert a.pose_at(a.duration) == {LEAN: 0, YAW: 0}


def test_cancel_goes_to_move_mode():
    a = QuickScan()
    a.start(0.0)
    a.tick(0.0)
    cmds = a.cancel(1.0)
    assert cmds and all(c == (MOVE_MODE, 0) for c in cmds) and a.finished


def test_scenescan_markers_left_centre_right_once_each():
    from lite3_explorer.animations import SceneScan
    a = SceneScan(pause_s=1.0, move_s=1.0, yaw=20000, mode_settle_s=0.0)
    assert 'scenescan' in ANIMATIONS and a.pose_at(1.0)[YAW] == -20000 and a.pose_at(5.0)[YAW] == 20000
    assert a.pose_at(a.duration)[YAW] == 0
    a.start(0.0)
    seen, t = [], 0.0
    while not a.finished and t < 60:
        a.tick(t)
        seen += a.pop_markers()
        t += 0.1
    assert seen == ['left', 'centre', 'right']
