from lite3_kitchen_explorer.report_utils import render_markdown, summarize_views


def view(desc, ident, conf, heading=0.0, evidence=('a',)):
    return {'heading_deg': heading, 'scene_description': desc, 'relevant_to_mission': ident,
            'evidence': list(evidence), 'confidence': conf, 'image': 'x.jpg'}


def test_most_confident_identified_view_wins():
    s = summarize_views([
        view('unknown', False, 0.9, 0),
        view('Kitchen area with toy stove', True, 0.7, 90, ['stove', 'pots']),
        view('workshop with robots', True, 0.85, 180),
        view('kitchen area with toy stove.', True, 0.6, 270),
    ])
    assert s['environment'] == 'workshop with robots'
    assert s['identified_views'] == 3 and s['total_views'] == 4
    assert s['all_descriptions']['kitchen area with toy stove'] == 2   # normalised + counted


def test_falls_back_to_unidentified_descriptions_then_unknown():
    s = summarize_views([view('plain grey wall', False, 0.4), {'heading_deg': 90, 'error': 'timeout'}])
    assert s['environment'] == 'plain grey wall' and s['identified_views'] == 0 and s['answered_views'] == 1
    s = summarize_views([{'heading_deg': 0, 'error': 'down'}, view('unknown', False, 0.2)])
    assert s['environment'] == 'unknown' and s['confidence'] == 0.0


def test_markdown_lists_every_stop_and_view():
    views = [view('corridor', True, 0.8, 0, ['long walls']), {'heading_deg': 90.0, 'error': 'HTTP 503'}]
    reports = [{'stop': 1, 'reason': 'start', 'wall_time': 't', 'pose': [1.0, 2.0, 1.5708],
                'views': views, 'summary': summarize_views(views)}]
    md = render_markdown(reports, finished=True)
    assert '| 1 | start | (1.00, 2.00) facing 90° | corridor | 0.80 | long walls | 1/2 |' in md
    assert 'query failed (HTTP 503)' in md and 'every frontier visited' in md
