"""The reporter's tile logic and AI hooks, with a scripted stand-in for the llm_client link (needs ROS only to import)."""
import json
import math
from unittest import mock

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('nav2_msgs')
from builtin_interfaces.msg import Time as TimeMsg   # noqa: E402

from lite3_kitchen_explorer import environment_reporter as E   # noqa: E402
from lite3_kitchen_explorer import tiles as T   # noqa: E402
from lite3_kitchen_explorer.tile_reports import TileReportBook   # noqa: E402

R = E.EnvironmentReporter


class Call:
    def __init__(self, ok=True, result=None, error='', done=True):
        self.done, self.ok, self.result, self._error = done, ok, result or {}, error

    def failure(self):
        return self._error


class FakeLlm:
    """Records requests; answers from a {task: Call} script."""

    def __init__(self, ready=True, script=None):
        self.ready, self.script, self.requests = ready, script or {}, []

    def request(self, task, **kw):
        self.requests.append((task, kw))
        return self.script.get(task, Call(False, error='no script'))

    def poll(self):
        pass


class Clock:
    def now(self):
        return mock.Mock(to_msg=lambda: TimeMsg())


def make(tmp_path, use_ai=True, ready=True, script=None, tiles=True):
    n = R.__new__(R)
    n.use_ai, n.ai_frontier_choice, n.tiles_enabled, n.tile_targets_first = use_ai, False, tiles, True
    n.tile_prompt_choice = n.tile_ai_choice = False            # the AI only writes the tile report by default
    n.complete_pending, n.reports, n.stop, n.state, n.phase = False, [], None, '', ''
    n.llm = FakeLlm(ready, script)
    n.llm_timeout_s, n.llm_report_timeout_s = 120.0, 180.0
    n.run_dir, n.query_count, n.pending, n.report_calls = str(tmp_path / 'run'), 0, None, {}
    n.book = TileReportBook(str(tmp_path / 'reports' / 'tile_report_x.md'), n.run_dir, use_ai, 0.30)
    n.target_min_clearance_m, n.min_wall_clearance_m, n.choice_min_candidates = 0.30, 0.35, 2
    n.tile_size_m, n.tile_reach_radius_m, n.global_frame = 1.0, 0.30, 'map'
    n.turn_at_stop, n.turn_deg, n.turn_time_allowance_s, n.turn_use_crawl, n.crawl_settle_s = True, 360.0, 20.0, True, 1.0
    n.spin_client = mock.Mock(server_is_ready=lambda: True)
    n.turn_check_room, n.local_costmap, n.failed_tiles = False, None, {}      # the room check has its own tests
    n.robot_length_m, n.robot_width_m, n.turn_room_margin_m, n.robot_frame = 0.61, 0.37, 0.0, 'base_link'
    n.crawl_pub, n.pending_spin, n.action_done, n.action_handle, n.action_seq = mock.Mock(), None, None, None, 0
    n.tile_grid = T.TileGrid(0.0, 0.0, 0.0, 1.0)             # tile (i, j) is the square [i, i+1] x [j, j+1]
    n.tiles, n.trail, n.trail_log, n.next_target, n.tiles_version = {(0, 0): {'status': 'visited'}}, [(0, 0)], [], None, 0
    n.grid = n.grid_info = None                               # no map: tile sides come back empty
    n.markers = []
    n.state_pub, n.tile_report_pub, n.marker_pub = mock.Mock(), mock.Mock(), mock.Mock()
    n.get_clock = lambda: Clock()
    log = mock.Mock()
    n.get_logger = lambda: log
    n.log = log
    for name in ('_explorer_active', '_publish_stop_report', '_publish_tile_markers', '_choose_target',
                 '_establish_tiles', '_finish_stop', '_send_route', '_send_preferred', '_send_action'):
        setattr(n, name, mock.Mock())
    n._robot_pose = lambda: [1.0, 2.0, 0.5]
    return n


TARGETS = [{'id': 1, 'tile': '1,0', 'goal_xy': (1.5, 0.5)}, {'id': 2, 'tile': '0,1', 'goal_xy': (0.5, 1.5)}]
CTX = {'pose': [1, 2, 0.5], 'visited': [[0, 0, 0]], 'cands': [], 'tile_summary': 'current tile 0,0'}


def stop(n):
    n.stop = {'stop': 1}


# ------------------------------------------------------------------ the quick scan turn at every stop
def crawl_values(n):
    return [c.args[0].data for c in n.crawl_pub.publish.call_args_list]


def test_a_stop_makes_one_quick_turn_then_decides_and_never_calls_the_ai(tmp_path):
    n = make(tmp_path)
    n._begin_stop('goal reached')
    assert n.state == R.DECIDING and n.phase == 'crawl_wait' and n.stop['reason'] == 'goal reached'
    n._explorer_active.assert_called_once_with(False)
    assert crawl_values(n) == [True] and math.isclose(math.degrees(n.pending_spin.target_yaw), 360.0)
    n._finish_stop.assert_not_called()                       # the decision waits for the turn, not for the AI
    assert n.llm.requests == []
    n._tick_deciding(n.phase_since + 0.5)                    # crawl gait still settling
    n._send_action.assert_not_called()
    n._tick_deciding(n.phase_since + 1.1)
    n._send_action.assert_called_once()
    assert n._send_action.call_args[0][1].target_yaw == pytest.approx(2 * math.pi) and n.phase == 'turning'
    n._tick_deciding(n.phase_since + 3.0)                    # still turning
    n._finish_stop.assert_not_called()
    n.action_done = E.GoalStatus.STATUS_SUCCEEDED
    n._tick_deciding(n.phase_since + 3.1)
    n._finish_stop.assert_called_once()
    assert crawl_values(n) == [True, False]                  # crawl gait released when the turn is over


def test_a_failed_or_stuck_turn_does_not_block_the_next_tile(tmp_path):
    n = make(tmp_path)
    n._begin_stop('start')
    n._tick_deciding(n.phase_since + 1.1)
    n.action_done = E.GoalStatus.STATUS_ABORTED              # e.g. "Collision Ahead - Exiting Spin"
    n._tick_deciding(n.phase_since + 1.5)
    n._finish_stop.assert_called_once()
    assert crawl_values(n)[-1] is False
    n = make(tmp_path)
    n._begin_stop('start')
    n._tick_deciding(n.phase_since + 1.1)                    # spin sent, then no answer at all
    n._tick_deciding(n.phase_since + 26.0)                   # allowance (20 s) + 5 s
    n._finish_stop.assert_called_once()
    assert crawl_values(n)[-1] is False and n.action_seq == 1    # the goal was cancelled


@pytest.mark.parametrize('setup', ['turn off', 'spin server down', 'tiny turn', 'exploration complete'])
def test_no_turn_when_it_is_off_or_impossible(tmp_path, setup):
    n = make(tmp_path)
    if setup == 'turn off':
        n.turn_at_stop = False
    elif setup == 'spin server down':
        n.spin_client = mock.Mock(server_is_ready=lambda: False)
    elif setup == 'tiny turn':
        n.turn_deg = 0.0
    else:
        n.complete_pending = True
    n._begin_stop('goal reached')
    n._finish_stop.assert_called_once()                      # straight to the decision
    assert n.crawl_pub.publish.call_count == 0 and n.pending_spin is None


def test_turn_without_the_crawl_gait_sends_the_spin_at_once(tmp_path):
    n = make(tmp_path)
    n.turn_use_crawl = False
    n._begin_stop('goal reached')
    n._send_action.assert_called_once()
    assert n.phase == 'turning' and n.crawl_pub.publish.call_count == 0


def test_the_tile_is_described_before_the_turn_starts_and_the_turn_does_not_wait_for_it(tmp_path):
    n = make(tmp_path, script={'describe_tile': Call(done=False)})
    n.state = R.EXPLORING
    n.next_target = ((1, 0), (1.5, 0.5))
    at(n, 1.5, 0.6)
    n._on_frontier_reached(mock.Mock(data=True))
    assert [t for t, _ in n.llm.requests] == ['describe_tile']          # requested on arrival (camera frame taken now)
    assert n.phase == 'crawl_wait' and '1,0' in n.report_calls          # ...and still pending while the dog turns
    n._tick_deciding(n.phase_since + 1.1)
    n.action_done = E.GoalStatus.STATUS_SUCCEEDED
    n._tick_deciding(n.phase_since + 4.0)
    n._finish_stop.assert_called_once()
    assert n.report_calls                                                # the stop went on; the answer is still outstanding


def test_the_tile_pipeline_runs_without_any_ai_choice(tmp_path):
    n = make(tmp_path)
    n._finish_stop = lambda: R._finish_stop(n)
    n._publish_stop_report = mock.Mock()
    n._finish_stop()
    assert n.phase == 'choose_wait'                            # tiles on: choose the nearest tile, AI choice or not
    n = make(tmp_path, tiles=False)
    n._finish_stop = lambda: R._finish_stop(n)
    n._finish_stop()
    assert n.phase == 'resume' and n._publish_stop_report.called   # nothing to decide: the explorer carries on


# ------------------------------------------------------------------ a tile is reached at its centre
def at(n, x, y):
    n._robot_pose = lambda: [x, y, 0.0]


def test_a_tile_is_reached_only_near_its_centre(tmp_path):
    n = make(tmp_path)
    at(n, 1.9, 0.5)                                            # inside tile (1, 0) but 0.4 m from its centre (1.5, 0.5)
    n._track_tiles()
    assert n.trail == [(0, 0)] and n.book.entries == {}
    at(n, 1.7, 0.5)                                            # 0.2 m from the centre
    n._track_tiles()
    assert n.trail == [(0, 0), (1, 0)] and list(n.book.entries) == ['1,0']
    assert n.book.entries['1,0']['source'] == 'centre reached'


def test_repeat_visits_do_not_repeat_the_report_and_the_current_tile_is_not_added_twice(tmp_path):
    n = make(tmp_path)
    at(n, 1.5, 0.5); n._track_tiles(); n._track_tiles()        # standing on it for two ticks
    assert n.trail == [(0, 0), (1, 0)]
    at(n, 0.5, 0.5); n._track_tiles()                          # back to the start tile: a revisit
    at(n, 1.5, 0.5); n._track_tiles()                          # and again to (1, 0)
    assert n.trail == [(0, 0), (1, 0), (0, 0), (1, 0)]
    assert list(n.book.entries) == ['1,0']                     # one report per tile, ever


def test_nav2_goal_reached_counts_a_tile_that_is_slightly_off_centre(tmp_path):
    n = make(tmp_path)
    n.next_target = ((1, 0), (1.5, 0.5))                       # goal point beside a wall: 0.4 m off its tile centre
    at(n, 1.5, 0.9)
    n._mark_goal_reached()
    assert n.trail[-1] == (1, 0) and n.book.entries['1,0']['source'] == 'goal reached'
    n.trail = [(0, 0)]; n.book.entries.clear()
    at(n, 2.5, 0.5)                                            # Nav2 stopped somewhere else (another goal): not this tile
    n._mark_goal_reached()
    assert n.trail == [(0, 0)] and n.book.entries == {}


def test_goal_reached_message_marks_the_tile_before_deciding_the_next_one(tmp_path):
    n = make(tmp_path)
    n.state = R.EXPLORING
    n.next_target = ((1, 0), (1.5, 0.5))
    at(n, 1.5, 0.6)
    calls = []
    n._begin_stop = lambda reason: calls.append((reason, list(n.trail)))
    n._on_frontier_reached(mock.Mock(data=True))
    assert calls == [('goal reached', [(0, 0), (1, 0)])]       # the current tile is already (1, 0) when the choice is made


# ------------------------------------------------------------------ the background tile report
def test_reaching_a_tile_asks_the_ai_in_the_background_and_does_not_wait(tmp_path):
    n = make(tmp_path, script={'describe_tile': Call(done=False)})
    n.tiles[(1, 0)] = {'status': 'to_visit'}
    n._tile_sides = lambda tile: {'N': 'wall', 'E': 'open', 'S': 'wall', 'W': 'unknown'}
    at(n, 1.5, 0.5)
    n._track_tiles()
    (task, kw), = n.llm.requests
    assert task == 'describe_tile' and kw['tile'] == '1,0' and kw['save_dir'] == n.run_dir
    assert kw['name'] == 'tile_1_0_q0001' and kw['timeout_s'] == 180.0 and kw['pose'] == [1.5, 0.5, 0.0]
    assert 'reached the centre of tile 1,0' in kw['context'] and 'north wall' in kw['context'] and 'east open' in kw['context']
    assert n.book.entries['1,0']['status'] == 'pending' and '1,0' in n.report_calls     # exploring carries on meanwhile
    assert 'pending' in open(n.book.path, encoding='utf-8').read() or '(waiting for the AI' in open(n.book.path, encoding='utf-8').read()


def test_the_answer_fills_in_the_entry_and_is_published_and_written(tmp_path):
    n = make(tmp_path)
    n.book.reached('1,0', [1.5, 0.5, 0.0], 'centre reached')
    call = Call(True, {'tile': '1,0', 'scene_description': 'ramp section, walls east and west', 'relevant_to_mission': True,
                       'evidence': ['sloped boards'], 'confidence': 0.88, 'image': 'tile_1_0_q0001.jpg'}, done=False)
    n.report_calls['1,0'] = (call, 'tile_1_0_q0001')
    n._poll_tile_reports()
    assert n.report_calls and n.book.entries['1,0']['status'] == 'pending'     # not done yet: nothing changes
    call.done = True
    n._poll_tile_reports()
    e = n.book.entries['1,0']
    assert e['status'] == 'done' and e['description'] == 'ramp section, walls east and west' and not n.report_calls
    assert 'ramp section, walls east and west' in open(n.book.path, encoding='utf-8').read()
    published = json.loads(n.tile_report_pub.publish.call_args[0][0].data)
    assert published['tile'] == '1,0' and published['status'] == 'done'
    assert n.marker_pub.publish.called and n._environment_hint() == 'ramp section, walls east and west'


def test_a_failed_description_is_recorded_without_stopping_anything(tmp_path):
    n = make(tmp_path)
    n.book.reached('1,0', None, 'centre reached')
    n.report_calls['1,0'] = (Call(False, error='HTTP 503'), 'x')
    n._poll_tile_reports()
    assert n.book.entries['1,0']['status'] == 'error' and '(AI query failed: HTTP 503)' in open(n.book.path, encoding='utf-8').read()


@pytest.mark.parametrize('use_ai, ready, why', [(False, True, 'AI disabled'), (True, False, 'llm_client is not running')])
def test_without_the_ai_the_tile_is_still_listed(tmp_path, use_ai, ready, why):
    n = make(tmp_path, use_ai=use_ai, ready=ready)
    at(n, 1.5, 0.5)
    n._track_tiles()
    assert n.llm.requests == [] and n.book.entries['1,0']['status'] == 'no_ai' and why in n.book.entries['1,0']['error']
    assert '1,0' in open(n.book.path, encoding='utf-8').read() and n.report_calls == {}


def test_the_initial_tile_is_reported_at_the_start_without_moving(tmp_path):
    n = make(tmp_path, script={'describe_tile': Call(done=False)})
    n._establish_tiles = R._establish_tiles.__get__(n)
    n._update_tiles, n._after_start_setup = mock.Mock(), mock.Mock()
    n.tile_init = None
    g = T.TileGrid(0.0, 0.0, 0.0, 1.0)
    n._establish_tiles({'id': 'A', 'kind': 'snapped', 'grid': g, 'sides': {}}, 'AI choice disabled', [0.4, 0.3, 0.0])
    (task, kw), = n.llm.requests
    assert task == 'describe_tile' and kw['tile'] == '0,0' and n.book.entries['0,0']['source'] == 'start'
    n._after_start_setup.assert_called_once()
    assert n.llm.requests[0][1]['pose'] == [0.4, 0.3, 0.0]


def test_manual_report_describes_the_tile_the_robot_is_in(tmp_path):
    n = make(tmp_path, script={'describe_tile': Call(done=False)})
    at(n, 1.2, 0.8)
    resp = mock.Mock()
    n._on_report_now(None, resp)
    assert resp.success and n.llm.requests[0][1]['tile'] == '1,0' and n.book.entries['1,0']['source'] == 'manual'


# ------------------------------------------------------------------ choosing the next tile (geometry by default)
def test_next_tile_is_the_nearest_one_without_asking_the_ai(tmp_path):
    n = make(tmp_path)                                         # AI on and ready, but tile_target_ai_choice is off
    stop(n)
    n._offer_choice(CTX, TARGETS)
    n._choose_target.assert_called_once_with(TARGETS[0], 'nearest tile (AI choice disabled)')
    assert n.llm.requests == []


def test_frontier_fallback_is_left_to_the_explorer_without_asking_the_ai(tmp_path):
    n = make(tmp_path)
    stop(n)
    n._offer_choice({**CTX, 'cands': [{'id': 1}, {'id': 2}]}, [])
    assert 'ai_frontier_choice is off' in n.stop['next_frontier']['note'] and n.llm.requests == []
    n = make(tmp_path, use_ai=False)
    stop(n)
    n._offer_choice({**CTX, 'cands': [{'id': 1}, {'id': 2}]}, [])
    assert n.stop['next_frontier']['note'].startswith('AI disabled') and n.llm.requests == []


# ------------------------------------------------------------------ optional: the AI makes the choices (flags on)
def test_offer_choice_sends_the_tile_state_and_waits(tmp_path):
    n = make(tmp_path)
    n.tile_prompt_choice = True
    stop(n)
    n._offer_choice(CTX, TARGETS)
    (task, kw), = n.llm.requests
    assert task == 'choose_target' and kw['mode'] == 'tile' and kw['candidates'] == TARGETS
    assert kw['tiles'] == [{'tile': [0, 0], 'status': 'visited'}] and kw['trail'] == [[0, 0]]
    assert kw['tile_grid']['size_m'] == 1.0 and kw['save_dir'] == n.run_dir and kw['name'] == 'stop001_map'
    assert n.pending[1]['mode'] == 'tile' and n.phase == 'choosing'


def test_frontier_mode_when_no_tile_is_connected(tmp_path):
    n = make(tmp_path)
    n.ai_frontier_choice = True
    stop(n)
    fr = [{'id': 1, 'goal_xy': [1, 1]}, {'id': 2, 'goal_xy': [2, 2]}]
    n._offer_choice({**CTX, 'cands': fr}, [])
    assert n.llm.requests[0][1]['mode'] == 'frontier' and n.llm.requests[0][1]['candidates'] == fr


def _answer(choice, extras=None, text='target 2: open'):
    return Call(True, {'answer': {'scene_description': text, 'evidence': ['clear'], 'confidence': 0.9,
                                  'relevant_to_mission': True}, 'choice': choice, 'extras': extras or {}})


def test_ai_tile_choice_goes_to_choose_target_with_its_extras(tmp_path):
    n = make(tmp_path)
    stop(n)
    extras = {'target_yaw_deg': 90, 'waypoints': [{'x': 0.5, 'y': 1.0, 'yaw_deg': 80}]}
    n._on_frontier_choice(_answer(2, extras), {'mode': 'tile', 'candidates': TARGETS})
    args = n._choose_target.call_args[0]
    assert args[0] == TARGETS[1] and args[1] == 'AI' and args[2]['extras'] == extras
    assert args[2]['scene_description'] == 'target 2: open' and n.phase == 'resume'


def test_invalid_or_failed_ai_answer_falls_back_to_the_nearest_tile(tmp_path):
    n = make(tmp_path)
    stop(n)
    n._on_frontier_choice(_answer(None), {'mode': 'tile', 'candidates': TARGETS})
    assert n._choose_target.call_args[0][:2] == (TARGETS[0], 'nearest tile (AI answer invalid)')
    n._on_frontier_choice(_answer(99), {'mode': 'tile', 'candidates': TARGETS})
    assert n._choose_target.call_args[0][1] == 'nearest tile (AI answer invalid)'
    n._on_frontier_choice(Call(False, error='HTTP 422 invalid_group'), {'mode': 'tile', 'candidates': TARGETS})
    assert 'AI query failed: HTTP 422 invalid_group' in n._choose_target.call_args[0][1]


def test_frontier_choice_is_sent_to_the_explorer_or_left_to_it(tmp_path):
    n = make(tmp_path)
    stop(n)
    fr = [{'id': 1, 'goal_xy': [1, 1], 'clearance_m': 0.5}, {'id': 2, 'goal_xy': [2, 2], 'clearance_m': 0.6}]
    n._on_frontier_choice(_answer(2, text='frontier 2: open'), {'mode': 'frontier', 'candidates': fr})
    n._send_preferred.assert_called_once_with([2, 2])
    assert n.stop['next_frontier']['chosen_by'] == 'AI' and n.stop['next_frontier']['id'] == 2
    n._on_frontier_choice(_answer(None), {'mode': 'frontier', 'candidates': fr})
    assert n.stop['next_frontier']['chosen_by'] == 'explorer'
    n._on_frontier_choice(Call(False, error='down'), {'mode': 'frontier', 'candidates': fr})
    assert 'AI query failed: down' in n.stop['next_frontier']['note']


# ------------------------------------------------------------------ initial tile placement by the AI (tile_ai_choice on)
def _cands():
    g = mock.Mock()
    g.as_dict.return_value = {'origin_xy': [0, 0], 'yaw_deg': 0.0, 'size_m': 1.0}
    return [{'id': 'A', 'kind': 'snapped', 'sides': {'N': 'wall'}, 'grid': g},
            {'id': 'B', 'kind': 'centred', 'sides': {}, 'grid': g}]


def test_initial_tile_request_and_answers(tmp_path):
    n = make(tmp_path)
    n.tile_ai_choice = True
    n.grid, n.grid_info = object(), object()
    cands = _cands()
    with mock.patch.object(E.T, 'initial_candidates', return_value=cands):
        n._begin_tile_init([0.0, 0.0, 0.0])
    task, kw = n.llm.requests[0]
    assert task == 'place_initial_tile' and [c['id'] for c in kw['candidates']] == ['A', 'B']
    assert kw['candidates'][0]['grid'] == {'origin_xy': [0, 0], 'yaw_deg': 0.0, 'size_m': 1.0}
    capture = n.pending[1]
    n._on_tile_choice(Call(True, {'answer': {'x': 1}, 'choice': 'B', 'ai_sides': {'N': 'wall'}}), capture)
    assert n._establish_tiles.call_args[0][:2] == (cands[1], 'chosen by the AI')
    n._on_tile_choice(Call(True, {'answer': {}, 'choice': None, 'ai_sides': {}}), capture)
    assert n._establish_tiles.call_args[0][1] == 'AI answer had no valid tile; default placement'
    n._on_tile_choice(Call(False, error='down'), capture)
    assert 'AI query failed (down)' in n._establish_tiles.call_args[0][1]


def test_initial_tile_without_ai_takes_the_wall_snapped_placement(tmp_path):
    n = make(tmp_path, ready=False)
    n.tile_ai_choice = True
    n.grid, n.grid_info = object(), object()
    cands = _cands()
    with mock.patch.object(E.T, 'initial_candidates', return_value=cands):
        n._begin_tile_init([0.0, 0.0, 0.0])
    assert n._establish_tiles.call_args[0][:2] == (cands[0], 'AI not available') and n.llm.requests == []


# ------------------------------------------------------------------ turning only where Nav2's spin can finish
import numpy as np   # noqa: E402


def corridor_costmap(half_width_m, age_s=0.0):
    """A local costmap (odom frame) of a corridor along y: lethal walls, 2-cell inscribed halo. Origin (-2, -2), 5 cm cells."""
    n, res = 80, 0.05
    g = np.zeros((n, n), np.int16)
    for col in range(n):
        x = -2.0 + (col + 0.5) * res
        if half_width_m - 2 * res <= abs(x) < half_width_m:
            g[:, col] = 99
        elif abs(x) >= half_width_m and abs(x) < half_width_m + 3 * res:
            g[:, col] = 100
    return (g, (-2.0, -2.0), res, 'odom', E.time.monotonic() - age_s)


def at_origin(n):
    t = mock.Mock()
    t.transform.translation.x = t.transform.translation.y = 0.0
    n.tf_buffer = mock.Mock(lookup_transform=lambda *a, **k: t)


def test_no_turn_in_a_corridor_too_narrow_to_rotate_in(tmp_path):
    n = make(tmp_path)
    n.turn_check_room, n.local_costmap = True, corridor_costmap(0.40)
    at_origin(n)
    n._begin_stop('start')
    n._finish_stop.assert_called_once()                      # decides straight away, robot stays aligned with the corridor
    assert n.crawl_pub.publish.call_count == 0 and n.pending_spin is None
    assert any('No room to turn here' in str(c) for c in n.log.info.call_args_list)


def test_turn_in_a_wide_enough_space(tmp_path):
    n = make(tmp_path)
    n.turn_check_room, n.local_costmap = True, corridor_costmap(0.80)
    at_origin(n)
    n._begin_stop('start')
    assert n.phase == 'crawl_wait' and n.pending_spin is not None and n._finish_stop.call_count == 0


@pytest.mark.parametrize('case', ['no costmap yet', 'stale costmap', 'no tf', 'check off'])
def test_when_the_room_cannot_be_judged_nav2_decides(tmp_path, case):
    n = make(tmp_path)
    n.turn_check_room, n.local_costmap = True, corridor_costmap(0.40)
    at_origin(n)
    if case == 'no costmap yet':
        n.local_costmap = None
    elif case == 'stale costmap':
        n.local_costmap = corridor_costmap(0.40, age_s=30.0)
    elif case == 'no tf':
        n.tf_buffer = mock.Mock(lookup_transform=mock.Mock(side_effect=E.TransformException('x')))
    else:
        n.turn_check_room = False
    n._begin_stop('start')
    assert n.phase == 'crawl_wait'                           # the turn is attempted


# ------------------------------------------------------------------ a tile goal that fails goes back to the reporter
def test_a_failed_tile_goal_marks_the_tile_and_decides_again_without_turning(tmp_path):
    n = make(tmp_path)
    n.state = R.EXPLORING
    n.next_target = ((1, 0), (1.5, 0.5))
    calls = []
    n._begin_stop = lambda reason, turn=True: calls.append((reason, turn))
    n._on_goal_failed(mock.Mock(data=json.dumps({'goal': [1.5, 0.52], 'reason': 'made no progress (stuck)'})))
    assert n.failed_tiles == {'1,0': 'made no progress (stuck)'} and n.next_target is None
    assert calls == [('goal failed', False)]                 # decide at once, and do not turn in a place that just failed


def test_a_failed_goal_that_is_not_the_current_tile_target_is_ignored_for_the_tile_list(tmp_path):
    n = make(tmp_path)
    n.state = R.EXPLORING
    n.next_target = ((1, 0), (1.5, 0.5))
    n._begin_stop = mock.Mock()
    n._on_goal_failed(mock.Mock(data=json.dumps({'goal': [3.0, 3.0], 'reason': 'x'})))
    assert n.failed_tiles == {} and n.next_target is not None     # not our target: no tile is marked failed...
    n._begin_stop.assert_called_once()                            # ...but the explorer is holding, so decide anyway
    n._on_goal_failed(mock.Mock(data='not json'))            # garbage does not crash
    n._begin_stop.reset_mock()
    n.state = R.DECIDING                                     # already deciding: just record, no second stop
    n._on_goal_failed(mock.Mock(data=json.dumps({'goal': [1.5, 0.5], 'reason': 'x'})))
    assert n.failed_tiles == {'1,0': 'x'} and n._begin_stop.call_count == 0


def test_a_failed_tile_is_not_offered_again(tmp_path):
    n = make(tmp_path)
    stop(n)
    n.failed_tiles = {'1,0': 'stuck'}
    n.grid, n.grid_info = object(), object()
    n.candidates, n.tile_target_limit, n.tile_targets_first = {'candidates': []}, 6, True
    n.tile_size_m = 1.0
    n._update_tiles, n._start_path_checks, n._offer_choice = mock.Mock(), mock.Mock(), mock.Mock()
    raw = [{'id': 1, 'tile': '1,0', 'goal_xy': (1.5, 0.5)}, {'id': 2, 'tile': '0,1', 'goal_xy': (0.5, 1.5)}]
    with mock.patch.object(E.T, 'nearest_connected_targets', return_value=raw):
        n._ask_frontier_choice()
    offered = n._start_path_checks.call_args[0][0]
    assert [t['tile'] for t in offered] == ['0,1']
    n._start_path_checks.reset_mock()
    with mock.patch.object(E.T, 'nearest_connected_targets', return_value=raw[:1]):    # nothing else left
        n._ask_frontier_choice()
    n._start_path_checks.assert_not_called()
    n._offer_choice.assert_called_once()                     # falls back to frontiers only when no other tile is left
