import math

import numpy as np

from lite3_explorer import map_grid as fc
from lite3_explorer import prop_votes as PV
from lite3_explorer import tiles as T
from lite3_explorer.tile_tracker import TileTracker

# two 1 m tiles side by side: (0,0) open floor where the robot stands, (1,0) open floor with a prop block in the middle
INFO = fc.GridInfo(40, 20, 0.05, 0.0, 0.0)


def _grid(prop=True):
    g = np.zeros((20, 40), dtype=np.int16)
    g[0, :] = g[-1, :] = 100
    g[:, 0] = g[:, -1] = 100
    if prop:
        g[8:12, 26:30] = 100                                   # 20 cm block at x 1.3-1.5, y 0.4-0.6
    return g


def _det(label='food', conf=0.6, u=320):
    return {'label': label, 'confidence': conf, 'bbox_xyxy': [u - 20, 200, u + 20, 260]}


def _tracker(grid, on=True):
    t = TileTracker()
    t.grid = T.TileGrid(0.0, 0.0, 0.0)
    t.start_wall = None
    t.set_map(grid, INFO)
    t.prop_votes = on
    t.update(force=True)
    return t


def test_place_puts_a_prop_straight_ahead_on_the_tile_behind_the_first_obstacle():
    pts = PV.place([_det()], (0.5, 0.5, 0.0), _grid(), INFO, fx=462.6, cx=320.0, cam_x=0.25)
    assert len(pts) == 1
    x, y, conf, lab, rng = pts[0]
    assert T.TileGrid(0.0, 0.0, 0.0).tile_of(x, y) == (1, 0)
    assert lab == 'food' and abs(rng - 0.55) < 0.1


def test_place_ignores_non_props_low_confidence_and_bearing_goes_left_for_small_u():
    dets = [_det('survivor'), _det(conf=0.2), _det(u=40)]       # u=40: far left of the image
    pts = PV.place(dets, (0.5, 0.5, 0.0), _grid(), INFO, fx=462.6, cx=320.0, cam_x=0.25)
    assert len(pts) == 1 and pts[0][1] > 0.5                    # only the left one, and it lands to the left (+y)


def test_enough_votes_keep_a_tile_a_scene_tile_and_switch_off_restores_the_lidar_answer():
    t = _tracker(_grid())
    tile = (1, 0)
    assert not t.is_scene(tile)                                # lidar: mostly free -> no scene
    pts = PV.place([_det()], (0.5, 0.5, 0.0), _grid(), INFO, fx=462.6, cx=320.0, cam_x=0.25)
    assert t.add_prop_points(pts * 7) == []                    # 7 sightings: not enough
    assert t.add_prop_points(pts) == [tile]                    # 8th: enough
    t.skip_scene = True
    assert t.is_scene(tile) and t.is_prop_scene(tile) and not t._drivable("1,0")
    t.prop_votes = False
    t.invalidate(); t.update()
    assert not t.is_scene(tile) and not t.is_prop_scene(tile)


def test_prop_votes_beat_a_traversable_answer_but_never_a_tile_the_robot_stood_on():
    t = _tracker(_grid())
    tile = (1, 0)
    pts = PV.place([_det()], (0.5, 0.5, 0.0), _grid(), INFO, fx=462.6, cx=320.0, cam_x=0.25)
    t.add_prop_points(pts * 8)
    t.traversable.add(tile)
    t.invalidate(); t.update()
    assert t.is_scene(tile)
    t.trail.append(tile)
    t.invalidate(); t.update()
    assert not t.is_scene(tile)


def test_low_confidence_votes_need_the_score_too():
    t = _tracker(_grid())
    pts = PV.place([_det(conf=0.4)], (0.5, 0.5, 0.0), _grid(), INFO, fx=462.6, cx=320.0, cam_x=0.25)
    t.add_prop_points(pts * 9)                                  # 9 x 0.4 = 3.6 < 4.0
    assert not t.is_prop_scene((1, 0))


def _half_scanned_grid():
    """Tile (1,0) only partly scanned (its far third unknown) with a block in the scanned part: a lidar scene at
    scene_min_known 0.55, but scanned well under 0.8 (like the outside-maze tiles beside the start in
    fullrun_26_11_141104)."""
    g = _grid()
    g[1:-1, 34:39] = -1                                         # x 1.70-1.95 unknown
    g[1:-1, 30:34] = -1                                         # x 1.50-1.70 unknown
    return g


def test_partly_scanned_scene_needs_a_prop_sighting():
    g = _half_scanned_grid()
    t = _tracker(g, on=True)
    t.scene_min_known, t.scene_max_free = 0.55, 0.90
    t.invalidate(); t.update()
    tile = (1, 0)
    known = t.tiles[tile]['known']
    assert 0.55 <= known < 0.8 and t.is_scene(tile)             # lidar alone: a scene
    t.scene_lidar_min_known = 0.8
    t.invalidate(); t.update()
    assert not t.is_scene(tile) and t.tiles[tile].get('scene_weak')
    pts = PV.place([_det()], (0.5, 0.5, 0.0), g, INFO, fx=462.6, cx=320.0, cam_x=0.25)
    assert pts and T.TileGrid(0.0, 0.0, 0.0).tile_of(pts[0][0], pts[0][1]) == tile
    t.add_prop_points(pts)                                      # one sighting is enough for a partly scanned one
    assert t.is_scene(tile)
