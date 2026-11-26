"""describe_scene with reference photos: collage upload, prompt, and the scene-type gate (no ROS)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest

from lite3_llm_tools import prompts as P
from lite3_llm_tools import scene_refs as S
from lite3_llm_tools.perception_client import PerceptionClient
from lite3_llm_tools.tools import LlmTools, Settings


def _jpeg(colour, w=640, h=480):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = colour
    return cv2.imencode('.jpg', img)[1].tobytes()


@pytest.fixture
def ref_dir(tmp_path):
    d = tmp_path / 'refs'
    d.mkdir()
    for name in ('kitchen_1', 'kitchen_2', 'cleaning_room_1', 'workshop_1'):
        (d / f'{name}.jpg').write_bytes(_jpeg((40, 80, 120)))
    (d / 'notes.txt').write_text('ignored')
    return d


def _server(reply):
    state = {'prompts': [], 'images': []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('content-length', 0)))
            text = body.decode('latin-1')
            state['prompts'].append(text.split('name="prompt"')[1].split('\r\n--')[0].strip())
            img = body.split(b'Content-Type: image/jpeg\r\n\r\n', 1)[1].rsplit(b'\r\n--', 1)[0]
            state['images'].append(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR))
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


def _ask(tmp_path, ref_dir, reply, min_conf=0.0):
    srv, state = _server(reply)
    try:
        client = PerceptionClient(f'http://127.0.0.1:{srv.server_port}', 'g', 'x', connect_timeout_s=1.0)
        tools = LlmTools(client, Settings(output_dir=str(tmp_path), error_backoff_s=0.0,
                                          scene_references_dir=str(ref_dir) if ref_dir else '',
                                          scene_min_confidence=min_conf),
                         get_scan_text=lambda: None)
        tools.store_capture('scene_1_0_q1', _jpeg((200, 200, 200)), str(tmp_path))
        r = tools.handle({'task': 'describe_tile', 'prompt': 'describe_scene', 'name': 'scene_1_0_q1', 'tile': '1,0',
                          'save_dir': str(tmp_path), 'context': 'SCENE TILE CONTEXT: tile 1,0.'})
        return r, state, tools
    finally:
        srv.shutdown()


def _reply(desc, conf=0.9):
    return {'scene_description': desc, 'relevant_to_mission': True, 'evidence': [], 'confidence': conf}


def test_labels_and_loading(ref_dir):
    assert S.label_of('cleaning_room_2.jpg') == 'cleaning_room' and S.label_of('kitchen.jpg') == 'kitchen'
    refs = S.load_references(str(ref_dir))
    assert S.scene_types(refs) == ['cleaning_room', 'kitchen', 'workshop'] and len(refs) == 4


def test_missing_reference_folder_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.load_references(str(tmp_path / 'nope'))
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        S.load_references(str(tmp_path / 'empty'))


def test_select_shares_slots_between_scene_types():
    img = np.zeros((4, 4, 3), np.uint8)
    refs = [('a', img)] * 10 + [('b', img)] * 2 + [('c', img)] * 10
    picked = [label for label, _ in S.select(refs, 7)]
    assert picked.count('b') == 2 and picked.count('a') + picked.count('c') == 5 and len(picked) == 7
    assert S.capacity() == 16


def test_collage_keeps_the_live_frame_full_size():
    live = np.full((480, 640, 3), 200, np.uint8)
    out = S.build_collage(live, [('kitchen', np.zeros((480, 640, 3), np.uint8))])
    assert out.shape == (768, 1024, 3)
    assert (out[240, 320] == 200).all()                 # live frame, unscaled, top left
    assert out[300, 700].mean() < 60                    # first reference thumbnail to its right


def test_parse_scene_type_with_allowed_set():
    assert P.parse_scene_type('kitchen: toy stove', ['kitchen']) == 'kitchen'
    assert P.parse_scene_type('furniture_shop: chairs', ['kitchen']) is None
    assert P.parse_scene_type('Traversable: wall', ['kitchen']) == 'traversable'
    assert P.parse_scene_type('kitchenette: x', ['kitchen']) is None


def test_scene_with_references_sends_a_collage_and_lists_only_their_types(tmp_path, ref_dir):
    r, state, tools = _ask(tmp_path, ref_dir, _reply('kitchen: white toy kitchen with red knobs'))
    assert r['ok'], r
    res = r['result']
    assert res['scene_type'] == 'kitchen' and 'scene_gate' not in res
    assert res['image'] == 'scene_1_0_q1.jpg' and res['collage'] == 'scene_1_0_q1_collage.jpg'
    assert (tmp_path / 'scene_1_0_q1_collage.jpg').exists()
    assert state['images'][0].shape == (768, 1024, 3)
    sent = state['prompts'][0]
    assert 'The scene types are exactly these: cleaning_room, kitchen, workshop' in sent
    assert 'furniture_shop' not in sent and '$' not in sent and 'SCENE TILE CONTEXT: tile 1,0.' in sent


def test_scene_without_a_reference_photo_becomes_traversable(tmp_path, ref_dir):
    r, _, _ = _ask(tmp_path, ref_dir, _reply('furniture_shop: chairs on display'))
    assert r['result']['scene_type'] == 'traversable'
    assert r['result']['scene_gate'] == 'furniture_shop has no reference photo'


def test_low_confidence_scene_becomes_traversable(tmp_path, ref_dir):
    r, _, _ = _ask(tmp_path, ref_dir, _reply('kitchen: maybe a stove', conf=0.4), min_conf=0.6)
    assert r['result']['scene_type'] == 'traversable' and 'confidence 0.40 < 0.60' in r['result']['scene_gate']


def test_without_references_the_eight_types_still_work(tmp_path):
    r, state, _ = _ask(tmp_path, None, _reply('furniture_shop: chairs on display'))
    assert r['result']['scene_type'] == 'furniture_shop' and 'collage' not in r['result']
    assert state['images'][0].shape == (480, 640, 3)
