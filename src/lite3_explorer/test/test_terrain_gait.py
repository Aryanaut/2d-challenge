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
    # recovery mode: the robot keeps the gait it already has, whatever its recovery targets / new terrain say
    s = TerrainGaitSwitcher(flat_delay_s=0.0, min_switch_interval_s=0.0)
    s.update_terrain('non_rugged', 0.0)
    cmd = s.pending_command(0.0)
    s.mark_sent(cmd, 0.0)
    assert s.gait == s.flat_gait
    assert s.freeze(True)
    s.arm()                                              # a recovery target
    s.update_terrain('rugged', 1.0)                      # a new terrain while recovering
    assert s.pending_command(2.0) is None and s.gait == s.flat_gait
    assert s.freeze(False)
    assert s.pending_command(3.0) is None                # not the moment the freeze ends ...
    s.arm()                                              # ... only at the next tile
    cmd = s.pending_command(4.0)
    assert cmd is not None and cmd.gait == 'rug_hstep'


def test_step_up_still_switches_while_frozen():
    # recovery mode: a higher recovery target switches to the h-step gait even though the gait is frozen
    s = TerrainGaitSwitcher(flat_delay_s=0.0, min_switch_interval_s=0.0)
    s.update_terrain('non_rugged', 0.0)
    s.mark_sent(s.pending_command(0.0), 0.0)
    assert s.freeze(True)
    s.update_terrain('rugged', 1.0)                      # terrain alone: still frozen
    assert s.pending_command(1.0) is None
    s.set_override('rug_hstep')                          # a step up
    cmd = s.pending_command(2.0)
    assert cmd is not None and cmd.gait == 'rug_hstep'
    s.mark_sent(cmd, 2.0)
    s.set_override(None)                                 # step up over, still recovering: keep h-step
    assert s.pending_command(3.0) is None and s.gait == 'rug_hstep'


def test_confirm_resends_the_step_up_gait_until_the_robot_reports_it():
    from lite3_explorer.terrain_gait import GaitConfirm
    c = GaitConfirm(retry_s=1.5)
    assert not c.update(0.0, 'rug_hstep', 0, last_publish_at=0.0)        # just sent: wait
    assert not c.update(1.4, 'rug_hstep', 0, last_publish_at=0.6)        # burst repeats push the timer
    assert c.update(2.1, 'rug_hstep', 0, last_publish_at=0.6)            # still flat: send again
    assert c.retries == 1
    assert not c.update(2.5, 'rug_hstep', None, last_publish_at=2.1)     # state unknown: wait for the timer
    assert c.update(3.6, 'rug_hstep', None, last_publish_at=2.1)         # unknown is not a match either
    assert not c.update(4.0, 'rug_hstep', 13, last_publish_at=3.6)       # H-step reported: done
    assert c.confirmed and 'after 2 re-send' in c.event
    assert not c.update(9.0, 'rug_hstep', 0, last_publish_at=3.6)        # a glitch on the step: never re-commanded


def test_confirm_starts_afresh_for_a_new_step_up_and_idles_without_one():
    from lite3_explorer.terrain_gait import GaitConfirm
    c = GaitConfirm(retry_s=1.0)
    assert not c.update(0.0, 'rug_hstep', 13, last_publish_at=0.0) and c.confirmed
    assert not c.update(5.0, None, 0, last_publish_at=0.0)               # /step_up off: nothing to confirm
    assert c.update(6.0, 'rug_hstep', 0, last_publish_at=0.0)            # next step up: checked again
    assert not GaitConfirm(retry_s=0.0).update(9.0, 'rug_hstep', 0, None)  # 0 = re-sends off
    assert not GaitConfirm().update(9.0, 'flat_crawl', 0, None)           # no robot state for crawl: cannot confirm


def test_switch_hold_stops_robot_before_and_until_gait_reported():
    from lite3_explorer.terrain_gait import GaitCommand, SwitchHold
    h = SwitchHold(stop_s=0.4, settle_s=1.0, max_wait_s=3.0)
    assert not h.holding and h.update(0.0, 4) is None
    hstep = GaitCommand('rug_hstep', GAIT_CODES['rug_hstep'])
    h.request(hstep, 0.0)
    assert h.update(0.0, 4) is None and h.holding               # stopping first: nothing sent yet
    assert h.update(0.39, 4) is None and h.holding
    assert h.update(0.4, 4) == hstep and h.holding              # robot still: send
    assert h.update(0.5, 4) is None                             # sent once only
    assert h.update(1.3, 13) is None and h.holding              # reported, but the settle time is not over
    assert h.update(2.0, 4) is None and h.holding               # old gait still reported: keep holding
    assert h.update(2.1, 13) is None and not h.holding and 'confirmed' in h.event


def test_switch_hold_unknown_gait_and_give_up():
    from lite3_explorer.terrain_gait import GaitCommand, SwitchHold
    h = SwitchHold(stop_s=0.4, settle_s=1.0, max_wait_s=3.0)
    h.request(GaitCommand('flat_crawl', GAIT_CODES['flat_crawl']), 0.0)
    assert h.update(0.4, 4)
    assert h.update(1.3, 4) is None and h.holding
    assert h.update(1.45, 4) is None and not h.holding          # crawl cannot be confirmed: settle time only
    h.request(GaitCommand('rug_hstep', GAIT_CODES['rug_hstep']), 10.0)
    assert h.update(10.45, None)
    assert h.update(11.5, None) is None and not h.holding       # no /robot_state: settle time only
    h.request(GaitCommand('rug_hstep', GAIT_CODES['rug_hstep']), 20.0)
    assert h.update(20.45, 4)
    assert h.update(23.3, 4) is None and h.holding
    assert h.update(23.45, 4) is None and not h.holding and 'anyway' in h.event   # refused: drive on


def test_switch_hold_new_request_while_held_sends_at_once():
    from lite3_explorer.terrain_gait import GaitCommand, SwitchHold
    h = SwitchHold(stop_s=0.4, settle_s=1.0, max_wait_s=3.0)
    flat = GaitCommand('flat_slow', GAIT_CODES['flat_slow'])
    hstep = GaitCommand('rug_hstep', GAIT_CODES['rug_hstep'])
    h.request(flat, 0.0)
    h.request(hstep, 0.2)                                       # replaces the one not sent yet
    assert h.update(0.4, 4) == hstep
    h.request(flat, 0.8)                                        # already still: no new stop time
    assert h.update(0.8, 13) == flat
    assert h.update(1.8, 0) is None and not h.holding
