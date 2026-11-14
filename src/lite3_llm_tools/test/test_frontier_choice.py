import pathlib

import numpy as np
import pytest

from lite3_kitchen_explorer import frontier_core as fc
from lite3_kitchen_explorer.frontier_choice import build_prompt, parse_choice
from lite3_kitchen_explorer.map_render import render_map
from lite3_kitchen_explorer.perception_client import extract_mission_result, read_prompt

CANDS = [{'id': 1, 'goal_xy': [1.0, 2.5], 'centroid_xy': [1.0, 2.6], 'distance_m': 1.6, 'size_m': 1.2,
          'clearance_m': 0.25, 'cost': 1.0},
         {'id': 2, 'goal_xy': [3.2, 1.0], 'centroid_xy': [3.3, 1.0], 'distance_m': 2.4, 'size_m': 2.0,
          'clearance_m': 0.9, 'cost': 1.4}]


@pytest.mark.parametrize('desc, evidence, expected', [
    ('frontier 2: wide opening, far from walls', [], 2),
    ('Frontier #1 - open floor', [], 1),
    ('none: every candidate hugs a wall', [], None),
    ('frontier 7: not offered', [], None),
    ('best option is open', ['chose frontier 2 because it is open'], 2),
    ('2', [], 2),
])
def test_parse_choice(desc, evidence, expected):
    assert parse_choice(desc, evidence, [1, 2]) == expected


def test_prompt_fills_every_placeholder_and_keeps_schema():
    root = pathlib.Path(__file__).parents[1]
    template = read_prompt(str(root / 'prompts' / 'frontier_choice.txt'))
    prompt = build_prompt(template, CANDS, 'corridor', 0.35, 3)
    assert '$' not in prompt
    assert '2 | 2.4 | 2.0 | 0.90' in prompt and 'at least 0.35 m from' in prompt
    assert 'Valid choices: 1, 2, or none' in prompt
    for key in ('scene_description', 'relevant_to_mission', 'evidence', 'confidence'):
        assert f'"{key}"' in prompt
    # a typical answer in the shared four-key format parses and yields the choice
    r = extract_mission_result({'scene_description': 'frontier 2: open floor', 'relevant_to_mission': True,
                                'evidence': ['far from black walls'], 'confidence': 0.8})
    assert parse_choice(r.scene_description, r.evidence, [1, 2]) == 2


def test_render_map_draws_walls_robot_and_candidates(tmp_path):
    g = np.full((60, 80), -1, dtype=np.int16)
    g[10:50, 10:60] = 0
    g[10, 10:60] = g[49, 10:60] = 100
    g[10:50, 10] = 100
    info = fc.GridInfo(80, 60, 0.05, 0.0, 0.0)
    img = render_map(g, info, (1.5, 1.5, 0.0), [{**CANDS[0], 'goal_xy': [2.9, 1.5]}], visited=[(1.0, 1.0, 0.0)])
    assert img.ndim == 3 and img.shape[0] > 100 and img.shape[1] > 100
    assert (img == 0).all(axis=2).any()                              # black walls present
    assert ((img[:, :, 0] > 200) & (img[:, :, 2] < 50)).any()         # blue robot
    assert ((img[:, :, 0] > 150) & (img[:, :, 1] < 50) & (img[:, :, 2] > 150)).any()  # magenta frontier
    with pytest.raises(ValueError):
        render_map(np.full((5, 5), -1, np.int16), fc.GridInfo(5, 5, 0.05, 0, 0), None, [])
