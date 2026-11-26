import math
import pathlib

import numpy as np
import pytest

from lite3_llm_tools import prompts as P
from lite3_llm_tools.gridmap import GridInfo
from lite3_llm_tools.map_render import render_initial_tile_panels, render_map
from lite3_llm_tools.perception_client import read_prompt
from lite3_llm_tools.tile_view import TileView, parse_tile_label

ROOT = pathlib.Path(__file__).parents[1] / 'prompts'


@pytest.mark.parametrize('desc, evidence, choice, sides', [
    ('tile A: grid follows the walls', ['sides: N=wall, E=wall, S=open, W=frontier'], 'A',
     {'N': 'wall', 'E': 'wall', 'S': 'open', 'W': 'frontier'}),
    ('Tile b - centred', ['N: open, E: wall'], 'B', {'N': 'open', 'E': 'wall'}),
    ('tile D: not offered', [], None, {}),
    ('A', [], 'A', {}),
])
def test_parse_tile_choice(desc, evidence, choice, sides):
    assert P.parse_tile_choice(desc, evidence, {'A', 'B', 'C'}) == (choice, sides)


def test_every_task_has_a_prompt_file():
    assert set(P.PROMPT_FILES) == {'describe_view', 'describe_tile', 'describe_scene', 'describe_scene_refs', 'place_initial_tile',
                                  'choose_tile', 'choose_frontier'}
    for task in P.PROMPT_FILES:
        assert pathlib.Path(P.prompt_path(task)).resolve().is_file()
    assert P.prompt_path('choose_tile', '/x/custom.txt') == '/x/custom.txt'


def test_prompts_fill_and_images_render():
    cands = [{'id': 'A', 'kind': 'snapped', 'sides': {'N': 'wall'}, 'grid': {'origin_xy': [0, 0], 'yaw_deg': 0}},
             {'id': 'B', 'kind': 'centred', 'sides': {}, 'grid': {'origin_xy': [0.2, 0.1], 'yaw_deg': 5}}]
    tile_prompt = P.build_initial_tile_prompt(read_prompt(str(ROOT / 'tile_init.txt')), cands, 1.0)
    assert '$' not in tile_prompt and '"sides: N=' in tile_prompt and 'A | snapped | N=wall' in tile_prompt
    fc_cand = [{'id': 1, 'goal_xy': [1.0, 1.0], 'distance_m': 1.0, 'size_m': 1.0, 'clearance_m': 0.5,
                'tile': '1,0', 'tile_status': 'to_visit'}]
    fp = P.build_prompt(read_prompt(str(ROOT / 'frontier_choice.txt')), fc_cand, 'corridor', 0.35, 2,
                        'current tile 0,0; 1 tiles visited')
    assert '$' not in fp and '| 1,0 | to visit' in fp and 'current tile 0,0' in fp

    g = np.full((100, 100), -1, np.int16)
    g[10:90, 10:90] = 0
    g[10, 10:90] = g[89, 10:90] = g[10:90, 10] = g[10:90, 89] = 100
    info = GridInfo(100, 100, 0.05, 0.0, 0.0)
    views = [{**c, 'grid': TileView.from_dict(c['grid'])} for c in cands]
    sheet = render_initial_tile_panels(g, info, (2.0, 2.0, 0.0), views)
    assert sheet.shape[1] > 400 * len(cands)
    tg = TileView.from_dict({'origin_xy': [1.5, 1.5], 'yaw_deg': 0.0, 'size_m': 1.0})
    img = render_map(g, info, (2.0, 2.0, 0.0), [], tile_grid=tg,
                     tiles={(0, 0): {'status': 'visited'}, (1, 0): {'status': 'to_visit'}}, trail=[(0, 0)])
    assert (img == 0).all(axis=2).any()            # walls still black under the tile tint


def test_tile_prompt_and_target_parsing():
    targets = [{'id': 1, 'tile': '0,-1', 'steps': 1, 'distance_m': 0.9, 'clearance_m': 0.46, 'has_frontier': True},
               {'id': 2, 'tile': '-1,0', 'steps': 1, 'distance_m': 1.0, 'clearance_m': 0.47, 'has_frontier': False}]
    p = P.build_tile_prompt(read_prompt(str(ROOT / 'tile_choice.txt')), targets, 'flat corridor', 0.30, 2,
                            'current tile 0,0')
    flat = ' '.join(p.split())
    assert '$' not in p and '1 | 0,-1 | 1 | 0.9 | 0.46 | yes' in p
    assert 'Nav2 path' in flat and 'Only the tiles in this list are reachable' in flat
    assert '"target_yaw_deg", "waypoints"' in flat and '2-3 evenly spaced points' in flat
    assert 'Valid choices: 1, 2.' in flat and 'do not answer none' in flat
    assert P.parse_choice('target 2: open corridor', [], [1, 2]) == 2
    assert P.parse_choice('none: all blocked', [], [1, 2]) is None


def test_tile_view_geometry_matches_a_rotated_grid():
    tv = TileView.from_dict({'origin_xy': [1.0, 2.0], 'yaw_deg': 90.0, 'size_m': 1.0})
    x, y = tv.center((0, 0))                      # rotated 90 deg: the tile extends along map -x / +y
    assert math.isclose(x, 0.5, abs_tol=1e-9) and math.isclose(y, 2.5, abs_tol=1e-9)
    (x0, y0), (x1, y1) = tv.edge((0, 0), 'S')
    assert math.isclose(math.dist((x0, y0), (x1, y1)), 1.0)
    assert parse_tile_label('1,-2') == (1, -2) and parse_tile_label([3, 4]) == (3, 4) and parse_tile_label('x') is None


def test_legend_wraps_instead_of_running_off_the_edge():
    from lite3_llm_tools.map_render import _legend
    img = np.full((50, 300, 3), 150, np.uint8)
    text = 'black=wall  red=near wall  white=free  grey=unknown  blue=robot  magenta#=candidate  orange=past stop'
    out = _legend(img, text)
    strip = out[:out.shape[0] - 50]
    assert strip.shape[0] > 28                                   # more than one line
    assert (strip[:, -8:] == 255).all()                          # nothing is drawn into the right margin
    assert _legend(np.full((50, 2000, 3), 150, np.uint8), text).shape[0] == 50 + 28    # wide image: still one line


def test_tile_report_prompt_keeps_the_four_key_schema_and_asks_about_the_tile():
    text = read_prompt(str(ROOT / 'tile_report.txt'))
    assert '$' not in text and 'TILE CONTEXT' in text and 'what THIS tile looks like' in ' '.join(text.split())
    for key in ('scene_description', 'relevant_to_mission', 'evidence', 'confidence'):
        assert f'"{key}"' in text


def test_parse_scene_type():
    assert P.parse_scene_type('kitchen: stove and pots') == 'kitchen'
    assert P.parse_scene_type('Furniture_shop: chairs') == 'furniture_shop'
    assert P.parse_scene_type('traversable: only a dark wall panel') == 'traversable'
    assert P.parse_scene_type('wall close-up: a dark panel') is None
    assert P.parse_scene_type('') is None
