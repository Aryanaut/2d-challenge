import pytest

from lite3_kitchen_explorer.perception_client import (
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
