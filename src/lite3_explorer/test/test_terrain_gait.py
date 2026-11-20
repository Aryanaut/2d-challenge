from lite3_explorer.terrain_gait import GAIT_CODES, TerrainGaitSwitcher


def test_defaults_match_aliases():
    assert GAIT_CODES['rug_general'] == 0x21010401
    assert GAIT_CODES['flat_medium'] == 0x21010307
    s = TerrainGaitSwitcher()
    assert (s.flat_gait, s.rugged_gait) == ('flat_medium', 'rug_general')


def test_rugged_switches_immediately_and_non_rugged_at_start():
    s = TerrainGaitSwitcher(min_switch_interval_s=0.0)
    s.update_terrain('non_rugged', 0.0)
    cmd = s.pending_command(0.0)
    assert cmd and cmd.gait == 'flat_medium' and cmd.cmd_code == 0x21010307
    s.mark_sent(cmd, 0.0)
    s.update_terrain('rugged', 1.0)
    cmd = s.pending_command(1.0)
    assert cmd and cmd.gait == 'rug_general' and cmd.cmd_code == 0x21010401


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
