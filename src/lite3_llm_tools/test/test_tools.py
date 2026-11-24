"""LlmTools end to end against a local fake AI server (no ROS)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from lite3_llm_tools.gridmap import GridInfo
from lite3_llm_tools.perception_client import PerceptionClient
from lite3_llm_tools.tools import LlmTools, Settings

ANSWERS = {'view': 'ramp section with sloped boards', 'tile': 'tile B: walls on two sides, sides: N=wall, E=open',
           'target': 'target 2: nearest tile with a clear corridor'}


def _fake_server(fail_first=0):
    """Replies depend on the prompt: a view, an initial tile, or a target choice (with waypoints)."""
    state = {'hits': 0, 'prompts': []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('content-length', 0))).decode('latin-1')
            state['hits'] += 1
            prompt = body.split('name="prompt"')[1].split('\r\n--')[0].strip() if 'name="prompt"' in body else ''
            state['prompts'].append(prompt)
            if state['hits'] <= fail_first:
                self.send_response(422)
                self.end_headers()
                self.wfile.write(b'{"error": "nope"}')
                return
            if 'ROUTE TO THE CHOSEN TILE' in prompt:
                reply = {'scene_description': ANSWERS['target'], 'relevant_to_mission': True, 'evidence': ['open'],
                         'confidence': 0.9, 'target_yaw_deg': 90, 'waypoints': [{'x': 1, 'y': 1, 'yaw_deg': 80}]}
            elif 'the first tile must be placed around the robot' in ' '.join(prompt.split()):
                reply = {'scene_description': ANSWERS['tile'], 'relevant_to_mission': True, 'evidence': [],
                         'confidence': 0.8}
            else:
                reply = {'scene_description': ANSWERS['view'], 'relevant_to_mission': True,
                         'evidence': ['boards'], 'confidence': 0.9}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header('content-length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def _map():
    g = np.full((100, 100), -1, np.int16)
    g[10:90, 10:90] = 0
    g[10, 10:90] = g[89, 10:90] = g[10:90, 10] = g[10:90, 89] = 100
    return g, GridInfo(100, 100, 0.05, 0.0, 0.0)


@pytest.fixture
def tools(tmp_path):
    srv, state = _fake_server()
    client = PerceptionClient(f'http://127.0.0.1:{srv.server_port}', 'g', 'describe this place', connect_timeout_s=1.0)
    t = LlmTools(client, Settings(output_dir=str(tmp_path), error_backoff_s=0.0), get_map=_map,
                 get_scan_text=lambda: 'LIDAR (robot frame): nearest obstacle ahead 0.60 m.')
    t.state, t.dir = state, tmp_path
    yield t
    srv.shutdown()


def test_ping_and_unknown_task(tools):
    assert tools.handle({'id': 'a', 'task': 'ping'})['result'] == {'pong': True}
    r = tools.handle({'id': 'b', 'task': 'dance'})
    assert r['ok'] is False and 'unknown task' in r['error'] and r['id'] == 'b'


def test_capture_then_describe_views_with_lidar_context(tools):
    for n in ('v1', 'v2'):
        tools.store_capture(n, b'\xff\xd8jpeg', str(tools.dir), pose=[0, 0, 0], scan_text='LIDAR: ahead 0.5 m')
    r = tools.handle({'id': 1, 'task': 'describe_views', 'save_dir': str(tools.dir),
                      'views': [{'name': 'v1', 'heading_deg': 0.0}, {'name': 'v2', 'heading_deg': 45.0},
                                {'name': 'missing'}]})
    assert r['ok']
    v = r['result']['views']
    assert v[0]['scene_description'] == ANSWERS['view'] and v[1]['heading_deg'] == 45.0
    assert 'no captured frame' in v[2]['error']                              # asked for a frame that was never kept
    assert (tools.dir / 'v1.json').exists() and (tools.dir / 'v1.jpg').exists()
    assert all('LIDAR: ahead 0.5 m' in p for p in tools.state['prompts'])    # the capture-time lidar is in the prompt


def test_describe_views_retries_after_a_server_error(tmp_path):
    srv, state = _fake_server(fail_first=1)
    client = PerceptionClient(f'http://127.0.0.1:{srv.server_port}', 'g', 'p', connect_timeout_s=1.0)
    t = LlmTools(client, Settings(output_dir=str(tmp_path), query_retries=1, error_backoff_s=0.0), sleep=lambda s: None)
    t.store_capture('v', b'x', str(tmp_path))
    r = t.handle({'task': 'describe_views', 'save_dir': str(tmp_path), 'views': [{'name': 'v'}]})
    assert r['result']['views'][0]['scene_description'] == ANSWERS['view'] and state['hits'] == 2
    srv.shutdown()


def test_describe_views_reports_the_error_when_retries_run_out(tmp_path):
    srv, state = _fake_server(fail_first=99)
    client = PerceptionClient(f'http://127.0.0.1:{srv.server_port}', 'g', 'p', connect_timeout_s=1.0)
    t = LlmTools(client, Settings(output_dir=str(tmp_path), query_retries=1, error_backoff_s=0.0), sleep=lambda s: None)
    t.store_capture('v', b'x', str(tmp_path))
    r = t.handle({'task': 'describe_views', 'save_dir': str(tmp_path), 'views': [{'name': 'v'}]})
    assert r['ok'] and 'HTTP 422' in r['result']['views'][0]['error'] and state['hits'] == 2
    srv.shutdown()


def test_place_initial_tile(tools):
    cands = [{'id': 'A', 'kind': 'snapped', 'sides': {'N': 'wall'}, 'grid': {'origin_xy': [1.5, 1.5], 'yaw_deg': 0}},
             {'id': 'B', 'kind': 'centred', 'sides': {}, 'grid': {'origin_xy': [1.7, 1.6], 'yaw_deg': 4}}]
    r = tools.handle({'id': 2, 'task': 'place_initial_tile', 'save_dir': str(tools.dir), 'pose': [2.0, 2.0, 0.0],
                      'candidates': cands, 'size_m': 1.0})
    assert r['ok'], r
    assert r['result']['choice'] == 'B' and r['result']['ai_sides'] == {'N': 'wall', 'E': 'open'}
    assert (tools.dir / 'initial_tile.png').exists() and (tools.dir / 'initial_tile_prompt.json').exists()


def test_choose_target_tile_mode_returns_choice_and_waypoints(tools):
    targets = [{'id': 1, 'tile': '0,0', 'steps': 5, 'distance_m': 0.9, 'clearance_m': 0.46, 'has_frontier': True,
                'goal_xy': [1.5, 1.5], 'start_xy': [1.0, 1.0], 'hint_xy': [[1.2, 1.2]]},
               {'id': 2, 'tile': '1,0', 'steps': 7, 'distance_m': 1.4, 'clearance_m': 0.5, 'has_frontier': False,
                'goal_xy': [2.5, 1.5], 'start_xy': [1.0, 1.0], 'hint_xy': [[1.6, 1.2], [2.0, 1.4]]}]
    r = tools.handle({'id': 3, 'task': 'choose_target', 'mode': 'tile', 'name': 'stop001_map', 'save_dir': str(tools.dir),
                      'pose': [1.0, 1.0, 0.0], 'visited': [[1.0, 1.0, 0.0]], 'candidates': targets,
                      'tile_grid': {'origin_xy': [0.5, 0.5], 'yaw_deg': 0.0, 'size_m': 1.0},
                      'tiles': [{'tile': [0, 0], 'status': 'visited'}, {'tile': [1, 0], 'status': 'to_visit'}],
                      'trail': [[0, 0]], 'environment': 'ramps', 'stops_so_far': 2, 'tile_summary': 'current tile 0,0'})
    assert r['ok'], r
    res = r['result']
    assert res['choice'] == 2 and res['extras']['target_yaw_deg'] == 90 and res['extras']['waypoints'][0]['x'] == 1
    assert (tools.dir / 'stop001_map.png').exists()
    assert 'LIDAR (robot frame)' in tools.state['prompts'][-1] and '$' not in tools.state['prompts'][-1]


def test_choose_target_without_a_map_or_bad_request(tools, tmp_path):
    tools.get_map = lambda: None
    r = tools.handle({'task': 'choose_target', 'mode': 'frontier', 'candidates': [{'id': 1}], 'save_dir': str(tmp_path)})
    assert r['ok'] is False and 'no map' in r['error']
    r = tools.handle({'task': 'choose_target', 'mode': 'wander', 'candidates': []})
    assert r['ok'] is False and 'needs "mode"' in r['error']


def test_lidar_context_can_be_switched_off(tools):
    tools.s.lidar_context = False
    tools.store_capture('v', b'x', str(tools.dir), scan_text='LIDAR: ahead 0.5 m')
    tools.handle({'task': 'describe_views', 'save_dir': str(tools.dir), 'views': [{'name': 'v'}]})
    assert 'LIDAR' not in tools.state['prompts'][-1]


def test_render_map_task_draws_the_route(tools):
    r = tools.handle({'task': 'render_map', 'name': 'stop001_route', 'save_dir': str(tools.dir), 'pose': [1.0, 1.0, 0.0],
                      'route': [[1.2, 1.2, 0.0], [1.8, 1.5, 1.57]], 'candidates': []})
    assert r['ok'] and (tools.dir / 'stop001_route.png').exists()


def test_images_sent_to_the_ai_fit_the_servers_1024x768_limit(tools):
    import cv2
    g = np.full((400, 400), -1, np.int16)                 # a big explored area: the full-size map render is ~900 px tall
    g[10:390, 10:390] = 0
    g[10, 10:390] = g[389, 10:390] = g[10:390, 10] = g[10:390, 389] = 100
    tools.get_map = lambda: (g, GridInfo(400, 400, 0.05, 0.0, 0.0))
    targets = [{'id': 1, 'tile': '0,0', 'steps': 5, 'distance_m': 0.9, 'clearance_m': 0.46, 'has_frontier': True,
                'goal_xy': [5.0, 5.0], 'start_xy': [4.0, 4.0], 'hint_xy': [[4.5, 4.5]]}]
    r = tools.handle({'task': 'choose_target', 'mode': 'tile', 'name': 'big', 'save_dir': str(tools.dir),
                      'pose': [4.0, 4.0, 0.0], 'candidates': targets})
    assert r['ok'], r
    sent = cv2.imread(str(tools.dir / 'big.png'))
    assert sent.shape[1] <= 1024 and sent.shape[0] <= 768
    full = tools._render({'pose': [4.0, 4.0, 0.0]}, targets)
    assert full.shape[0] > 768                            # it really was too big before


def test_fit_within_only_shrinks_and_keeps_the_aspect_ratio():
    from lite3_llm_tools.image_utils import fit_within
    big = np.zeros((886, 894, 3), np.uint8)
    out = fit_within(big, 1024, 768)
    assert out.shape[0] == 768 and abs(out.shape[1] / out.shape[0] - 894 / 886) < 0.01
    small = np.zeros((480, 640, 3), np.uint8)
    assert fit_within(small, 1024, 768) is small
    assert fit_within(big, 1024, 0).shape == big.shape


def test_describe_tile_uses_the_tile_prompt_with_context_and_lidar(tools):
    tools.store_capture('tile_0_0_q1', b'\xff\xd8jpeg', str(tools.dir), pose=[0.1, 0.2, 0.0], scan_text='LIDAR: ahead 0.5 m')
    r = tools.handle({'id': 'r1', 'task': 'describe_tile', 'name': 'tile_0_0_q1', 'tile': '0,0', 'save_dir': str(tools.dir),
                      'context': 'TILE CONTEXT: the robot has just reached the centre of tile 0,0. Map: N=wall, E=open.'})
    assert r['ok'], r
    res = r['result']
    assert res['tile'] == '0,0' and res['scene_description'] == ANSWERS['view'] and res['image'] == 'tile_0_0_q1.jpg'
    sent = tools.state['prompts'][-1]
    assert 'what THIS tile looks like' in ' '.join(sent.split())          # the tile prompt, not the stop prompt
    assert 'TILE CONTEXT: the robot has just reached the centre of tile 0,0' in sent and 'LIDAR: ahead 0.5 m' in sent
    assert (tools.dir / 'tile_0_0_q1.json').exists()


def test_describe_scene_prompt_lists_the_scene_types_and_returns_the_type(tools):
    tools.store_capture('scene_1_0_q2', b'\xff\xd8jpeg', str(tools.dir))
    r = tools.handle({'task': 'describe_tile', 'prompt': 'describe_scene', 'name': 'scene_1_0_q2', 'tile': '1,0',
                      'save_dir': str(tools.dir)})
    assert r['ok'], r
    sent = tools.state['prompts'][-1]
    assert 'furniture_shop' in sent and 'traversable' in sent
    assert r['result']['scene_type'] == 'traversable'                    # the test answer names no scene type


def test_describe_tile_failures_come_back_as_errors(tools):
    r = tools.handle({'task': 'describe_tile', 'name': 'never-captured', 'tile': '1,1', 'save_dir': str(tools.dir)})
    assert r['ok'] is False and 'no captured frame' in r['error']
    r = tools.handle({'task': 'describe_tile', 'tile': '1,1'})
    assert r['ok'] is False and 'needs "name"' in r['error']
