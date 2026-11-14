import pytest

from lite3_llm_tools.perception_client import (
    PerceptionClient, extract_mission_result, read_prompt)

GOOD = {'scene_description': 'toy kitchen corner unit', 'relevant_to_mission': True,
        'evidence': ['stovetop with pots', 'oven door', 'microwave'], 'confidence': 0.9}


def test_plain_object():
    r = extract_mission_result(GOOD)
    assert r.relevant_to_mission is True and r.confidence == 0.9 and len(r.evidence) == 3


@pytest.mark.parametrize('payload', [
    {'result': GOOD},
    {'status': 'ok', 'data': {'output': GOOD}},
    {'response': '{"scene_description": "toy kitchen corner unit", "relevant_to_mission": true, '
                 '"evidence": ["stovetop with pots", "oven door", "microwave"], "confidence": 0.9}'},
    {'text': '```json\n{"scene_description": "toy kitchen corner unit", "relevant_to_mission": true,'
             ' "evidence": ["stovetop with pots"], "confidence": 0.9}\n```'},
    'Sure! {"scene_description": "toy kitchen corner unit", "relevant_to_mission": true, '
    '"evidence": [], "confidence": 0.9} hope that helps',
    {'choices': [{'message': {'content': '{"scene_description": "toy kitchen corner unit", '
                                         '"relevant_to_mission": true, "evidence": [], "confidence": 0.9}'}}]},
])
def test_wrapped_payloads(payload):
    r = extract_mission_result(payload)
    assert r.scene_description == 'toy kitchen corner unit' and r.relevant_to_mission


def test_string_bool_and_confidence_clamped():
    r = extract_mission_result({**GOOD, 'relevant_to_mission': 'false', 'confidence': '1.7'})
    assert r.relevant_to_mission is False and r.confidence == 1.0


@pytest.mark.parametrize('payload', [
    {'scene_description': 'x', 'evidence': [], 'confidence': 0.5},          # missing key
    {**GOOD, 'relevant_to_mission': 'maybe'},
    {**GOOD, 'confidence': float('nan')},
    'no json here',
    {'error': 'model overloaded'},
])
def test_bad_payloads(payload):
    with pytest.raises(ValueError):
        extract_mission_result(payload)


def test_client_validation(tmp_path):
    with pytest.raises(ValueError):
        PerceptionClient('http://x', '', 'prompt')
    with pytest.raises(ValueError):
        PerceptionClient('http://x', 'g', '   ')
    with pytest.raises(ValueError):
        PerceptionClient('http://x', 'g', 'p', read_timeout_s=float('inf'))
    assert PerceptionClient('http://host:8000/', 'g', 'p').url == 'http://host:8000/analyze'
    f = tmp_path / 'p.txt'
    f.write_bytes('﻿hello'.encode('utf-8'))
    assert read_prompt(str(f)) == 'hello'


def test_shipped_prompt_keeps_schema():
    import pathlib
    text = read_prompt(str(pathlib.Path(__file__).parents[1] / 'prompts' / 'environment_report.txt'))
    for key in ('scene_description', 'relevant_to_mission', 'evidence', 'confidence'):
        assert f'"{key}"' in text
    flat = ' '.join(text.split())
    assert 'kind of maze environment' in flat and 'block field' in flat


# ---- connection retries, against a real local HTTP server ----
import json as _json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from lite3_llm_tools.perception_client import PerceptionError


def _server(fail_first, status=503):
    """Local /analyze server that answers `status` for the first `fail_first` requests, then GOOD."""
    state = {'hits': 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get('content-length', 0)))
            state['hits'] += 1
            ok = state['hits'] > fail_first
            body = _json.dumps(GOOD if ok else {'error': 'busy'}).encode()
            self.send_response(200 if ok else status)
            self.send_header('content-length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def _client(port, **kw):
    c = PerceptionClient(f'http://127.0.0.1:{port}', 'g', 'prompt', connect_timeout_s=1.0, **kw)
    c.sleeps = []
    c._sleep = c.sleeps.append
    return c


def test_retries_transient_503_then_succeeds():
    srv, state = _server(fail_first=2)
    c = _client(srv.server_port)
    assert c.analyze(b'x').relevant_to_mission is True
    assert state['hits'] == 3 and c.sleeps == [2.0, 4.0]       # exponential backoff
    srv.shutdown()


def test_gives_up_after_connect_retries():
    srv, state = _server(fail_first=99)
    c = _client(srv.server_port, connect_retries=2)
    with pytest.raises(PerceptionError, match='HTTP 503'):
        c.analyze(b'x')
    assert state['hits'] == 3
    srv.shutdown()


def test_400_is_not_retried():
    srv, state = _server(fail_first=99, status=400)
    c = _client(srv.server_port)
    with pytest.raises(PerceptionError, match='HTTP 400'):
        c.analyze(b'x')
    assert state['hits'] == 1 and c.sleeps == []
    srv.shutdown()


def test_connect_error_is_retried_until_server_appears():
    with socket.socket() as s:                                  # reserve a port nobody listens on
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    c = _client(port)
    started = {}

    def sleep(wait):                                            # the server comes up during the 2nd backoff
        c.sleeps.append(wait)
        if len(c.sleeps) == 2:
            class H(BaseHTTPRequestHandler):
                def do_POST(self):
                    self.rfile.read(int(self.headers.get('content-length', 0)))
                    body = _json.dumps(GOOD).encode()
                    self.send_response(200)
                    self.send_header('content-length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *a):
                    pass
            started['srv'] = HTTPServer(('127.0.0.1', port), H)
            threading.Thread(target=started['srv'].serve_forever, daemon=True).start()
    c._sleep = sleep
    assert c.analyze(b'x').confidence == 0.9
    assert c.sleeps == [2.0, 4.0]
    started['srv'].shutdown()


def test_dead_server_raises_connect_error_after_retries():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    c = _client(port, connect_retries=1)
    with pytest.raises(PerceptionError, match='ConnectError'):
        c.analyze(b'x')
    assert c.sleeps == [2.0]
