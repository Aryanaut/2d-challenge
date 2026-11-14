import os

from lite3_kitchen_explorer.tile_reports import TileReportBook, within_reach


def book(tmp_path, ai=True):
    return TileReportBook(str(tmp_path / 'reports' / 'tile_report_x.md'), '/runs/run_1', ai)


def test_entries_start_pending_and_are_filled_in_later(tmp_path):
    b = book(tmp_path)
    b.reached('0,0', [0.2, 0.1, 0.0], 'start', {'N': 'wall', 'E': 'open'})
    b.reached('1,0', [1.1, 0.0, 0.0], 'centre reached', {'N': 'open'})
    assert b.counts() == {'done': 0, 'pending': 2, 'error': 0, 'no_ai': 0}
    b.set_result('0,0', {'scene_description': 'flat corridor, wall to the north', 'relevant_to_mission': True,
                         'evidence': ['plain floor', 'wall panels'], 'confidence': 0.9, 'image': 'tile_0_0_q1.jpg'})
    b.set_error('1,0', 'HTTP 503')
    md = b.render(finished=True)
    assert '| 1 | 0,0 |' in md and 'flat corridor, wall to the north' in md and '| 0.90 |' in md
    assert 'AI query failed' not in md and 'waiting for the AI' not in md and 'exploration complete' in md
    assert '`/runs/run_1/tile_0_0_q1.jpg`' in md and 'Evidence: plain floor; wall panels' in md
    assert '## 1. Tile 0,0 (start)' in md and 'Tile 1,0' not in md


def test_tiles_are_listed_in_the_order_they_were_reached_and_a_rereport_keeps_its_place(tmp_path):
    b = book(tmp_path)
    for label in ('0,0', '1,0', '1,1'):
        b.reached(label, [0, 0, 0], 'centre reached')
    b.set_result('1,0', {'scene_description': 'ramp', 'relevant_to_mission': True, 'evidence': [], 'confidence': 0.8})
    b.reached('1,0', [1, 0, 0], 'manual')                        # manual re-report
    assert [e['order'] for e in b.entries.values()] == [1, 2, 3] and b.entries['1,0']['status'] == 'pending'


def test_ai_off_entries_say_so_and_the_header_reports_the_grid_and_counts(tmp_path):
    b = book(tmp_path, ai=False)
    b.reached('0,0', [0, 0, 0], 'start', {'N': 'wall'})
    b.set_no_ai('0,0', 'AI disabled (use_ai:=false)')
    md = b.render(grid={'origin_xy': [-0.72, -0.4], 'yaw_deg': 4.0, 'size_m': 1.0},
                  tile_counts={'visited': 1, 'to_visit': 2, 'blocked': 0, 'unknown': 5}, to_visit=['1,0', '0,1'])
    assert 'AI descriptions:** off' in md and '(AI disabled (use_ai:=false))' in md
    assert 'yaw 4 deg' in md and '1 visited, 2 to visit' in md and '## Still to visit\n\n1,0, 0,1' in md


def test_table_cells_cannot_break_the_table(tmp_path):
    b = book(tmp_path)
    b.reached('0,0', None, 'start')
    b.set_result('0,0', {'scene_description': 'a | b\nc', 'relevant_to_mission': False, 'evidence': [], 'confidence': 0.1})
    row = [ln for ln in b.render().splitlines() if ln.startswith('| 1 |')][0]
    assert row.count('|') == 8 and 'a / b c' in row


def test_write_creates_the_folder_and_replaces_the_file_atomically(tmp_path):
    b = book(tmp_path)
    b.reached('0,0', [0, 0, 0], 'start')
    path = b.write()
    assert os.path.isfile(path) and path.endswith('tile_report_x.md') and not os.path.exists(path + '.tmp')
    b.reached('1,0', [1, 0, 0], 'centre reached')
    b.set_result('1,0', {'scene_description': 'ramp', 'relevant_to_mission': True, 'evidence': [], 'confidence': 0.8})
    b.write()
    assert 'Tile 1,0' in open(path, encoding='utf-8').read()


def test_within_reach():
    assert within_reach([1.0, 1.0, 0.0], (1.2, 1.1), 0.30) and not within_reach([1.0, 1.0, 0.0], (1.5, 1.0), 0.30)
    assert not within_reach(None, (0, 0), 1.0)
