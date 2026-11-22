"""Maze file parsing (no ROS)."""
import math

import pytest

from lite3_sim.maze import CURTAIN, SLIT, WALL, MazeError, load_maze_dict, parse_layout
from lite3_sim.maze_preview import render

LAYOUT = """
+---+---+---+
| >     ~   |
+   +===+   +
|   | R1!   |
+---+---+---+
"""


def test_edges_labels_and_start():
    m = load_maze_dict({'layout': LAYOUT})
    assert (m.cols, m.rows, m.tile_size) == (3, 2, 1.02)
    assert m.edge((0, 1), 'N') == WALL and m.edge((0, 1), 'E') is None
    assert m.edge((1, 1), 'E') == CURTAIN and m.edge((2, 1), 'W') == CURTAIN
    assert m.edge((1, 1), 'S') == SLIT and m.edge((1, 0), 'N') == SLIT
    assert m.edge((1, 0), 'E') == SLIT
    assert m.edge((0, 0), 'E') == WALL and m.edge((0, 0), 'N') is None
    assert m.labels == {(1, 0): 'R1'}
    assert m.start[:2] == pytest.approx((0.51, 1.53)) and m.start[2] == pytest.approx(0.0)


def test_slit_panels_leave_gaps_and_segments_are_unique():
    m = load_maze_dict({'layout': LAYOUT, 'slit_m': 0.05})
    slits = [s for s in m.segments if s.kind == SLIT]
    assert len(slits) == 2
    assert all(math.hypot(s.x1 - s.x0, s.y1 - s.y0) == pytest.approx(1.02 - 0.10) for s in slits)
    keys = [tuple(round(v, 3) for v in s.as_tuple()) for s in m.segments]
    assert len(keys) == len(set(keys))


def test_reachability_through_curtains():
    m = load_maze_dict({'layout': LAYOUT})
    assert m.reachable_tiles() == {(0, 1), (1, 1), (0, 0), (2, 1), (2, 0)}
    assert m.reachable_tiles(through_curtains=False) == {(0, 1), (1, 1), (0, 0)}


def test_start_block_overrides_layout():
    m = load_maze_dict({'layout': LAYOUT, 'start': {'tile': [2, 0], 'heading': 'N', 'offset_m': [0.1, 0.0],
                                                    'yaw_offset_deg': 5}})
    assert m.start == pytest.approx((2.5 * 1.02 + 0.1, 0.51, math.radians(95)))


def test_solid_cells_are_boxed_in():
    m = load_maze_dict({'layout': '+---+---+\n| > | # |\n+---+---+\n'})
    assert (1, 0) in m.solid and m.edge((1, 0), 'W') == WALL


def test_errors_are_explained():
    with pytest.raises(MazeError, match='odd number'):
        parse_layout('+---+\n|   |\n')
    with pytest.raises(MazeError, match='not a wall'):
        load_maze_dict({'layout': '+---+---+\n| > x   |\n+---+---+\n'})
    with pytest.raises(MazeError, match='no start'):
        load_maze_dict({'layout': '+---+\n|   |\n+---+\n'})


def test_preview_round_trip():
    layout = LAYOUT.replace(' R1!', ' R !')
    m = load_maze_dict({'layout': layout})
    drawn = render(m).splitlines()[1:]
    assert [ln[4:].rstrip() for ln in drawn] == [ln.rstrip() for ln in layout.strip().splitlines()]
