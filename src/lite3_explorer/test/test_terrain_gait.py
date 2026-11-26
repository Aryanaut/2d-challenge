from lite3_explorer.terrain_gait import GAIT_CODES, TerrainGaitSwitcher


def test_defaults_match_aliases():
    assert GAIT_CODES['rug_general'] == 0x21010401
    assert GAIT_CODES['flat_medium'] == 0x21010307
    s = TerrainGaitSwitcher()
    assert (s.flat_gait, s.rugged_gait) == ('flat_medium', 'rug_hstep')


def test_rugged_switches_immediately_and_non_rugged_at_start():
    s = TerrainGaitSwitcher(min_switch_interval_s=0.0)
    s.update_terrain('non_rugged', 0.0)
    cmd = s.pending_command(0.0)
    assert cmd and cmd.gait == 'flat_medium' and cmd.cmd_code == 0x21010307
    s.mark_sent(cmd, 0.0)
    s.update_terrain('rugged', 1.0)
    cmd = s.pending_command(1.0)
    assert cmd and cmd.gait == 'rug_hstep' and cmd.cmd_code == 0x21010407


def test_back_to_flat_waits_for_delay():
    s = TerrainGaitSwitcher(flat_delay_s=2.0, min_switch_interval_s=0.0)
    s.update_terrain('rugged', 0.0)
    s.mark_sent(s.pending_command(0.0), 0.0)
    s.update_terrain('non_rugged', 1.0)
    assert s.pending_command(2.0) is None           # 1 s flat: not yet
    cmd = s.pending_command(3.5)                    # 2.5 s flat: switch
    assert cmd and cmd.gait == 'flat_medium'


def test_unknown_keeps_current_gait():
    s = TerrainGaitSwitcher(min_switch_interval_s=0.0)
    s.update_terrain('rugged', 0.0)
    s.mark_sent(s.pending_command(0.0), 0.0)
    s.update_terrain('unknown', 5.0)
    assert s.pending_command(10.0) is None


def test_min_interval_between_commands():
    s = TerrainGaitSwitcher(min_switch_interval_s=1.0)
    s.update_terrain('rugged', 0.0)
    s.mark_sent(s.pending_command(0.0), 0.0)
    s.update_terrain('non_rugged', 0.1)
    assert s.pending_command(0.2) is None           # too soon after the last command


def test_gait_only_changes_on_a_new_tile_or_a_new_terrain():
    s = TerrainGaitSwitcher(flat_delay_s=0.0, min_switch_interval_s=0.0)
    s.update_terrain('rugged', 0.0)
    s.mark_sent(s.pending_command(0.0), 0.0)                 # new terrain classified -> rug_hstep
    assert s.pending_command(1.0) is None and not s.armed     # nothing more happens by itself
    s.gait = 'flat_medium'                                    # (something else changed the gait)
    assert s.pending_command(5.0) is None                     # ... and it is left alone: no event
    s.arm()                                                   # the robot attempts a new tile
    cmd = s.pending_command(6.0)
    assert cmd and cmd.gait == 'rug_hstep'                    # the terrain gait is applied then
    s.mark_sent(cmd, 6.0)
    assert s.pending_command(7.0) is None


def test_same_terrain_again_does_not_switch_and_a_finished_hold_restores():
    s = TerrainGaitSwitcher(flat_delay_s=0.0, min_switch_interval_s=0.0)
    s.update_terrain('non_rugged', 0.0)
    s.mark_sent(s.pending_command(0.0), 0.0)
    assert s.update_terrain('non_rugged', 1.0) is False and s.pending_command(1.0) is None
    s.arm()
    assert s.pending_command(2.0) is None and not s.armed     # already in the right gait: nothing to send
    s.restore()                                               # the spin's crawl hold ended
    cmd = s.pending_command(3.0)
    assert cmd and cmd.gait == 'flat_medium'


def test_unknown_terrain_keeps_the_gait_and_disarms():
    s = TerrainGaitSwitcher(min_switch_interval_s=0.0)
    s.update_terrain('rugged', 0.0)
    s.mark_sent(s.pending_command(0.0), 0.0)
    s.update_terrain('unknown', 1.0)
    assert s.pending_command(1.0) is None and not s.armed


def test_override_wins_over_terrain_and_clearing_returns_the_terrain_gait():
    sw = TerrainGaitSwitcher()
    sw.update_terrain('non_rugged', 0.0)
    cmd = sw.pending_command(0.0)
    sw.mark_sent(cmd, 0.0)
    assert sw.set_override('rug_hstep') and not sw.set_override('rug_hstep')
    cmd = sw.pending_command(5.0)
    assert cmd.gait == 'rug_hstep'
    sw.mark_sent(cmd, 5.0)
    assert sw.set_override(None)
    assert sw.pending_command(10.0).gait == sw.flat_gait


def test_frozen_keeps_the_current_gait_until_the_next_tile_or_terrain():
    # recovery mode: the robot keeps the gait it already has, whatever its recovery targets / new terrain / step up say
    s = TerrainGaitSwitcher(flat_delay_s=0.0, min_switch_interval_s=0.0)
    s.update_terrain('non_rugged', 0.0)
    cmd = s.pending_command(0.0)
    s.mark_sent(cmd, 0.0)
    assert s.gait == s.flat_gait
    assert s.freeze(True)
    s.arm()                                              # a recovery target
    s.update_terrain('rugged', 1.0)                      # a new terrain while recovering
    s.set_override('rug_hstep')                          # a step up
    assert s.pending_command(2.0) is None and s.gait == s.flat_gait
    assert s.freeze(False)
    assert s.pending_command(3.0) is None                # not the moment the freeze ends ...
    s.arm()                                              # ... only at the next tile
    cmd = s.pending_command(4.0)
    assert cmd is not None and cmd.gait == 'rug_hstep'
