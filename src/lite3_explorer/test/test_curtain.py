import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lite3_explorer import curtain_core as cc  # noqa: E402
from lite3_explorer.tiles import TileGrid  # noqa: E402
from test_mission import make, modes_seen  # noqa: E402


def depth_image(curtain=True, wall=False):
    """480x640 metres: far background, a slightly slanted curtain, the hem jump, the floor behind it, then the floor
    coming towards the camera."""
    rows = np.arange(480)
    col = np.interp(rows, [240, 300, 460], [1.0, 0.67, 0.36])
    if curtain:
        col[:40] = 1.1
        col[40:230] = np.linspace(0.65, 0.77, 190)
    elif wall:
        col[:40] = 1.1
        col[40:300] = np.linspace(0.65, 0.8, 260)
        col[300:] = np.interp(rows[300:], [300, 460], [0.67, 0.36])
    return np.tile(col[:, None], (1, 640)).astype(np.float32)


def test_curtain_is_detected_and_a_wall_is_not():
    det = cc.detect_curtain(depth_image())
    assert det is not None and len(det.columns) > 100
    assert 0.6 < np.median(det.ranges) < 0.85
    assert cc.detect_curtain(depth_image(curtain=False, wall=True)) is None
    assert cc.detect_curtain(np.zeros((480, 640), np.float32)) is None


def spots(x, y0, y1, ux=1.0, uy=0.0, n=12):
    return np.array([[x, y, ux, uy] for y in np.linspace(y0, y1, n)])


GRID = TileGrid(0.0, 0.0, 0.0, 1.0)


def test_edge_curtain_finds_the_edge_it_lies_along():
    e = cc.edge_curtain(GRID, (0, 0), spots(1.05, 0.1, 0.9))          # along the east edge, seen from inside
    assert e and e['side'] == 'E' and e['neighbour'] == (1, 0) and abs(e['yaw']) < 1e-6
    assert abs(e['centre_xy'][0] - 1.0) < 1e-6 and abs(e['centre_xy'][1] - 0.5) < 1e-6
    n = cc.edge_curtain(GRID, (0, 0), spots(0.1, 1.0, 1.0, 0, 1, n=2))
    assert n is None                                                   # too few spots


def test_camera_spots_beat_more_lidar_only_spots():
    # bag curtaintest_25_11_140529: 15 lidar-only spots on the W wall, 12 colour-camera spots on the curtain (N)
    north = np.hstack([spots(0.1, 1.0, 1.0, 0, 1, n=12)[:, :4], np.ones((12, 1))])
    north[:, 0] = np.linspace(0.05, 0.6, 12)
    west = np.hstack([spots(-0.05, 0.15, 0.65, ux=-1.0, n=15), np.zeros((15, 1))])
    e = cc.edge_curtain(GRID, (0, 0), np.vstack([north, west]))
    assert e['side'] == 'N' and e['camera_spots'] == 12 and e['spots'] == 12
    w = cc.edge_curtain(GRID, (0, 0), np.vstack([north, west])[:, :4])     # no source column: the most spots win
    assert w['side'] == 'W'
    assert cc.edge_curtain(GRID, (0, 0), west)['side'] == 'W'              # lidar only still counts when alone
    assert cc.edge_curtain(GRID, (0, 0), west, lidar_only=False) is None


def test_camera_seen_curtain_never_falls_back_to_a_lidar_wall():
    # the camera saw the curtain on N, but N's neighbour is explored (or skipped): the lidar-only W wall must not be walked into
    north = np.hstack([spots(0.1, 1.0, 1.0, 0, 1, n=12)[:, :4], np.ones((12, 1))])
    north[:, 0] = np.linspace(0.05, 0.6, 12)
    west = np.hstack([spots(-0.05, 0.15, 0.65, ux=-1.0, n=15), np.zeros((15, 1))])
    both = np.vstack([north, west])
    assert cc.edge_curtain(GRID, (0, 0), both, skip=[(0, 1)]) is None
    assert cc.edge_curtain(GRID, (0, 0), both, min_camera_spots=20) is None      # too few camera spots to confirm any edge
    e = cc.edge_curtain(GRID, (0, 0), both)
    assert e['side'] == 'N' and e['confirmed_by'] == 'camera'


def test_edge_curtain_ignores_other_places():
    assert cc.edge_curtain(GRID, (0, 0), spots(0.5, 0.1, 0.9)) is None                # middle of the tile
    assert cc.edge_curtain(GRID, (0, 0), spots(1.05, 0.1, 0.9), visited=[(1, 0)]) is None   # behind it is explored
    assert cc.edge_curtain(GRID, (0, 0), spots(1.05, 0.1, 0.9), skip=[(1, 0)]) is None
    assert cc.edge_curtain(GRID, (0, 0), spots(1.05, 0.1, 0.9, ux=-1.0)) is None         # seen from the far side
    assert cc.edge_curtain(GRID, (0, 0), spots(1.05, 0.1, 0.25)) is None                 # too narrow
    assert cc.edge_curtain(GRID, (0, 0), spots(3.0, 0.1, 0.9)) is None                   # another tile's edge


def test_edge_curtain_with_a_rotated_grid():
    g = TileGrid(2.0, 1.0, math.radians(90), 1.0)                     # local +u is world +y
    pts = np.array([[2.0 - v, 1.0 + 1.05, 0.0, 1.0] for v in np.linspace(0.1, 0.9, 12)])
    e = cc.edge_curtain(g, (0, 0), pts)
    assert e and e['side'] == 'E' and abs(e['yaw'] - math.pi / 2) < 1e-6


# ---------------------------------------------------------------------------------------- the mode
class Drive:
    """Add a robot that obeys drive(vx, wz) to a FakeCtx."""
    def __init__(self, ctx, pts):
        self.ctx, self.pts, self.cmds, self.path = ctx, pts, [], []
        self.wall_x = None
        self.blocked = lambda: False
        ctx.curtain_points = lambda: self.pts
        ctx.drive = self.drive
        ctx.stop = lambda: self.cmds.append('stop')

    def drive(self, vx, wz, vy=0.0):
        self.cmds.append((vx, wz))
        p = self.ctx.pose_
        dt = 1.0
        c, s = math.cos(p[2]), math.sin(p[2])
        nx = p[0] + (vx * c - vy * s) * dt
        ny = p[1] + (vx * s + vy * c) * dt
        p[2] += wz * dt
        if (self.wall_x is None or nx < self.wall_x) and not self.blocked():   # a wall stops it (bumped)
            p[0], p[1] = nx, ny
        self.path.append((p[0], p[1]))


def build(pose_yaw, flat=None, initial_curtain=True):
    """Robot on tile 0,0 of a 4 m corridor, a curtain on its E edge. The initial tile never holds a curtain: these
    tests open that rule (initial_curtain=True) to keep the curtain right beside the start; test_no_curtain_on_the_
    initial_tile keeps it."""
    ctx, m = make({'modes.exploring.curtains': True, 'modes.exploring.scan_after_reach': False,
                   'modes.curtain.speed_mps': 0.25, **(flat or {})}, length_m=4.0)
    ctx.pose_[:] = [0.5, 0.5, pose_yaw]
    d = Drive(ctx, spots(1.05, 0.1, 0.9))
    if initial_curtain:
        establish = ctx.tiles.establish

        def est(*a, **k):
            out = establish(*a, **k)
            ctx.tiles.initial_tile = None
            return out
        ctx.tiles.establish = est
    return ctx, m, d


def test_curtain_on_the_edge_is_aligned_with_and_walked_through():
    ctx, m, d = build(pose_yaw=1.0)                          # not facing the curtain
    for _ in range(80):
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if 'curtain' in modes_seen(m) and ctx.pose_[0] >= 1.5:
            break
    seen = modes_seen(m)
    assert 'curtain' in seen, seen
    assert ctx.pose_[0] >= 1.5 - 1e-6                            # reached the centre of the tile behind the curtain
    assert abs(ctx.pose_[2]) < math.radians(8)                   # squared up
    turns = [c for c in d.cmds if isinstance(c, tuple) and c[0] == 0.0]
    walks = [c for c in d.cmds if isinstance(c, tuple) and c[0] > 0]
    assert turns and walks                                        # turned in place first, then walked with vx
    assert all(abs(w[1]) <= 0.3 for w in walks) and d.cmds[-1] == 'stop' or 'stop' in d.cmds
    assert (1, 0) in m.curtains_tried


def test_curtain_is_crossed_only_in_the_hstep_gait():
    ctx, m, d = build(pose_yaw=0.0)
    hstep = []                                                    # calls to ctx.hstep, in order
    ctx.hstep = lambda on: hstep.append(on)
    seen = []                                                     # (cross-phase forward walk, h-step held) per drive
    real = d.drive

    def drive(vx, wz, vy=0.0):
        if m.current.name == 'curtain' and m.current.phase == 'cross':
            seen.append(bool(hstep and hstep[-1]))
        real(vx, wz, vy)
    ctx.drive = drive
    assert run_until(ctx, m, lambda: 'curtain' in modes_seen(m) and m.current.name != 'curtain', n=200)
    assert seen and all(seen)                                     # every step through the curtain in h-step
    assert hstep == [True, False]                                 # switched on once, released after
    assert ctx.pose_[0] >= 1.5 - 1e-6
    phases = [ph for md, ph in ctx.phases if md == 'curtain']
    assert phases.index('gait_on') < phases.index('cross') < phases.index('gait_off')


def test_curtain_hstep_is_released_when_the_mode_is_left_mid_crossing():
    ctx, m, d = build(pose_yaw=0.0)
    hstep = []
    ctx.hstep = lambda on: hstep.append(on)
    assert run_until(ctx, m, lambda: m.current.name == 'curtain' and m.current.phase == 'cross', n=200)
    m.request('exploring', 'operator')
    ctx.t += 1.0
    m.tick()
    assert hstep == [True, False]


def run_until(ctx, m, stop, n=120):
    for _ in range(n):
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if stop():
            return True
    return False


def test_curtain_mode_goes_to_the_tile_centre_before_walking_through():
    ctx, m, d = build(pose_yaw=1.0)
    ctx.pose_[:] = [0.62, 0.28, 1.0]                             # off the line through the middle of the edge
    at_align = []
    orig = ctx.on_mode_change

    def spy(mode, phase):
        if mode == 'curtain' and phase == 'align':
            at_align.append(list(ctx.pose_))
        return orig(mode, phase)
    ctx.on_mode_change = spy
    assert run_until(ctx, m, lambda: 'curtain' in modes_seen(m) and ctx.pose_[0] >= 1.5)
    assert at_align and math.hypot(at_align[0][0] - 0.5, at_align[0][1] - 0.5) <= 0.15   # centred first (center_tol_m 0.10, the fake robot moves in 1 s steps)
    assert abs(ctx.pose_[1] - 0.5) < 0.05                        # and stayed on the centre line through the curtain


def test_blocked_curtain_is_given_up_at_once_it_is_a_wall():
    ctx, m, d = build(pose_yaw=0.0, flat={'modes.curtain.low_move_s': 0})      # the stall check, not the low-movement timeout
    d.wall_x = 0.8
    assert run_until(ctx, m, lambda: any(k == 'curtain_failed' for k, _ in ctx.events), n=300)
    failed = next(dd for k, dd in ctx.events if k == 'curtain_failed')
    assert failed['attempts'] == 0 and 'a wall' in failed['why'] and failed['back_to'] == 'exploring'
    assert not [1 for k, _ in ctx.events if k == 'curtain_retry']


def test_no_curtain_on_the_initial_tile():
    ctx, m, d = build(pose_yaw=0.0, initial_curtain=False)
    run_until(ctx, m, lambda: 'curtain' in modes_seen(m), n=60)
    assert 'curtain' not in modes_seen(m)
    assert not ctx.tiles.not_curtain_mask(d.pts).any()                  # every spot is on the initial tile's E edge


def test_curtain_timeout_unmarks_the_edge_and_detection_goes_on_elsewhere():
    ctx, m, d = build(pose_yaw=0.0, flat={'modes.curtain.low_move_s': 0, 'modes.curtain.stall_s': 99.0,
                                          'modes.curtain.cross_timeout_s': 5.0})
    d.wall_x = 0.8                                               # never gets through: the cross times out
    assert run_until(ctx, m, lambda: any(k == 'curtain_failed' for k, _ in ctx.events), n=300)
    failed = next(dd for k, dd in ctx.events if k == 'curtain_failed')
    assert 'timed out' in failed['why'] and 'unmarked' in failed['why'] and failed['attempts'] == 0
    assert not [1 for k, _ in ctx.events if k == 'curtain_retry']          # given up at once
    rejected = next(dd for k, dd in ctx.events if k == 'curtain_rejected')
    assert rejected['tile'] == '0,0' and rejected['side'] == 'E'
    assert ctx.tiles.is_not_curtain((0, 0), 'E') and ctx.tiles.is_not_curtain((1, 0), 'W')   # from either side
    assert not ctx.tiles.not_curtain_mask(d.pts).any()                  # its spots no longer count ...
    other = spots(2.05, 0.1, 0.9)
    assert ctx.tiles.not_curtain_mask(other).all()                      # ... spots on another edge still do


def test_curtain_timeout_with_retries_unmarks_the_edge_when_given_up():
    ctx, m, d = build(pose_yaw=0.0, flat={'modes.curtain.low_move_s': 0, 'modes.curtain.stall_s': 99.0,
                                          'modes.curtain.cross_timeout_s': 5.0, 'modes.curtain.timeout_gives_up': False})
    d.wall_x = 0.8
    assert run_until(ctx, m, lambda: any(k == 'curtain_failed' for k, _ in ctx.events), n=400)
    assert sum(1 for k, _ in ctx.events if k == 'curtain_retry') == 2
    assert sum(1 for k, _ in ctx.events if k == 'curtain_rejected') == 1     # only when given up, not per try
    assert ctx.tiles.not_curtains == [((0, 0), 'E')]


def test_rejected_curtain_is_no_longer_deferred():
    from test_mission import corridor
    from lite3_explorer.tile_tracker import TileTracker
    tr = TileTracker()
    tr.align_on_entry = False
    tr.set_map(*corridor(4.0))
    tr.establish([0.5, 0.5, 0.0])
    tr.defer_curtain((1, 0), (2, 0), 'E')
    assert tr.curtains_left() == ['2,0']
    tr.reject_curtain((1, 0), 'E')
    assert tr.curtains_left() == [] and tr.is_not_curtain((2, 0), 'W')


def test_curtain_wall_pick_skips_rejected_and_initial_tile_edges():
    from lite3_explorer.curtain_core import curtain_wall
    from test_mission import corridor
    from lite3_explorer.tile_tracker import TileTracker
    tr = TileTracker()
    tr.align_on_entry = False
    tr.set_map(*corridor(4.0))
    tr.establish([0.5, 0.5, 0.0])
    on = lambda x, n: [[x + dx, y, 1.0] for y in np.linspace(0.2, 0.8, n) for dx in (-0.03, 0.03)]   # noqa: E731
    seen = np.array(on(1.0, 6) + on(2.0, 4) + on(3.0, 2))                 # most on x=1 (the initial tile's E edge)
    w = curtain_wall(tr.grid, seen[tr.not_curtain_mask(seen)])
    assert tr.grid.edge(w['tile'], w['side']) == tr.grid.edge((2, 0), 'W')
    tr.reject_curtain((1, 0), 'E')                                       # x=2 timed out: the next one
    w = curtain_wall(tr.grid, seen[tr.not_curtain_mask(seen)])
    assert tr.grid.edge(w['tile'], w['side']) == tr.grid.edge((3, 0), 'W')


def test_slow_start_of_the_cross_is_not_called_a_wall():
    # 25 Nov bag: standing in front of the curtain, the robot took ~1.5 s to start walking and the 2 s stall check
    # called the curtain a wall before the first step. The stall window opens start_grace_s after the cross starts.
    ctx, m, d = build(pose_yaw=0.0, flat={'modes.curtain.stall_s': 2.0, 'modes.curtain.start_grace_s': 2.0})
    standing = {'left': 3, 'started': False}

    def blocked():                                               # the first 3 s (ticks) of the cross: no motion yet
        if m.current.name == 'curtain' and m.current.phase == 'cross' and standing['left'] > 0:
            standing['left'] -= 1
            return True
        return False
    d.blocked = blocked
    assert run_until(ctx, m, lambda: ctx.pose_[0] >= 1.5 or any(k == 'curtain_failed' for k, _ in ctx.events), n=200)
    assert not [dd for k, dd in ctx.events if k == 'curtain_failed'], ctx.events
    assert standing['left'] == 0 and ctx.pose_[0] >= 1.5 - 1e-6


def test_blocked_curtain_is_retried_then_hands_back_to_the_mode_before_it():
    ctx, m, d = build(pose_yaw=0.0, flat={'modes.curtain.blocked_gives_up': False, 'modes.curtain.low_move_s': 0})
    d.wall_x = 0.8                                               # a wall in front of the curtain: never gets through
    assert run_until(ctx, m, lambda: any(k == 'curtain_failed' for k, _ in ctx.events), n=300)
    retries = [dd for k, dd in ctx.events if k == 'curtain_retry']
    failed = next(dd for k, dd in ctx.events if k == 'curtain_failed')
    assert len(retries) == 2 and failed['attempts'] == 3 and failed['back_to'] == 'exploring'
    ctx.t += 1.0
    m.tick()
    seen = modes_seen(m)
    assert seen[seen.index('curtain') - 1] == 'exploring' and seen[seen.index('curtain') + 1] == 'exploring', seen
    assert not ctx.tiles.curtain_edges                           # never got through: no curtain edge recorded


def test_curtain_is_not_used_when_disabled():
    ctx, m = make({'modes.exploring.scan_after_reach': False}, length_m=4.0)
    Drive(ctx, spots(1.05, 0.1, 0.9))
    for _ in range(10):
        ctx.t += 1.0
        m.tick()
    assert 'curtain' not in modes_seen(m)


# ------------------------------------------------------------------------------------------------ mppi mode
from lite3_explorer import costmap_escape as CE  # noqa: E402
from lite3_explorer.map_grid import GridInfo  # noqa: E402


def wall_costmap(wall_x=1.0):
    """A costmap with a lethal wall at x >= wall_x and Nav2-like inflation: 99 within 0.185 m, decaying to 0 by 0.5 m."""
    res = 0.05
    info = GridInfo(60, 60, res, -1.0, -1.0)
    xs = info.origin_x + (np.arange(60) + 0.5) * res
    dist = np.maximum(0.0, wall_x - xs)
    row = np.where(dist == 0, 100, np.where(dist <= 0.185, 99, (98 * np.exp(-3.0 * (dist - 0.185))).astype(int)))
    row = np.where(dist > 0.5, 0, row)
    return np.tile(row[None, :], (60, 1)).astype(np.int16), info


def test_escape_direction_points_away_from_the_wall():
    grid, info = wall_costmap()
    ux, uy, d = CE.escape_direction(grid, info, 0.9, 0.0, clear_cost=70)
    assert ux < -0.8 and abs(uy) < 0.6 and 0.0 < d < 0.5            # west, away from the wall at x=1.0
    walled = np.full_like(grid, 100)
    assert CE.escape_direction(walled, info, 0.0, 0.0) is None       # lethal all around: no way out


def test_mppi_mode_moves_out_with_x_and_y_then_returns_to_exploring():
    from lite3_explorer.mission.core import Transition
    ctx, m = make({'modes.exploring.scan_after_reach': False}, length_m=4.0)
    grid, info = wall_costmap()
    ctx.pose_[:] = [0.9, 0.0, 0.6]                                  # inside the zone, facing 34 deg off the wall normal
    cmds = []
    ctx.costmap_snapshot = lambda: (grid, info)
    ctx.stop = lambda: cmds.append('stop')

    def drive(vx, wz, vy=0.0):
        cmds.append((vx, vy, wz))
        p, dt = ctx.pose_, 0.5
        p[0] += (vx * math.cos(p[2]) - vy * math.sin(p[2])) * dt
        p[1] += (vx * math.sin(p[2]) + vy * math.cos(p[2])) * dt
    ctx.drive = drive
    m._switch(Transition('mppi', 'test'))
    for _ in range(60):
        ctx.t += 0.5
        m.tick()
        if m.current.name != 'mppi':
            break
    assert m.current.name == 'exploring', modes_seen(m)
    assert CE.cost_at(grid, info, ctx.pose_[0], ctx.pose_[1]) < 70      # clear
    moves = [c for c in cmds if isinstance(c, tuple)]
    assert moves and any(abs(c[1]) > 0.01 for c in moves) and any(abs(c[0]) > 0.01 for c in moves)   # both x and y used
    assert all(c[2] == 0.0 for c in moves) and cmds[-1] == 'stop'      # no turning, ends stopped


def test_mppi_fails_to_recovery_when_there_is_no_way_out():
    from lite3_explorer.mission.core import Transition
    ctx, m = make({'modes.exploring.scan_after_reach': False}, length_m=4.0)
    grid, info = wall_costmap()
    grid[:] = 100
    ctx.costmap_snapshot = lambda: (grid, info)
    ctx.drive, ctx.stop = (lambda *a, **k: None), (lambda: None)
    m._switch(Transition('mppi', 'test'))
    ctx.t += 0.5
    m.tick()
    assert m.current.name == 'recovery'


def test_costmap_stuck_goes_to_the_mppi_mode():
    ctx, m = make({'modes.exploring.scan_after_reach': False, 'stuck.costmap_hold_s': 2.0,
                   'stuck.cooldown_s': 1.0, 'stuck.triggers': ['costmap']}, length_m=4.0)
    grid, info = wall_costmap()
    ctx.costmap_cost_at_robot = lambda: 99
    ctx.costmap_snapshot = lambda: (grid, info)
    ctx.drive, ctx.stop = (lambda *a, **k: None), (lambda: None)
    for _ in range(30):
        ctx.t += 1.0
        m.tick()
    assert 'mppi' in modes_seen(m), modes_seen(m)


def test_mppi_mode_holds_the_heading_it_entered_with():
    from lite3_explorer.mission.core import Transition
    ctx, m = make({'modes.exploring.scan_after_reach': False}, length_m=4.0)
    grid, info = wall_costmap()
    ctx.pose_[:] = [0.9, 0.0, 0.6]
    ctx.costmap_snapshot = lambda: (grid, info)
    ctx.stop = lambda: None
    wzs = []

    def drive(vx, wz, vy=0.0):
        wzs.append(wz)
    ctx.drive = drive
    m._switch(Transition('mppi', 'test'))
    ctx.t += 0.5
    m.tick()
    assert wzs and wzs[-1] == 0.0                       # on the entry heading: no turning
    ctx.pose_[2] = 0.6 + 0.4                            # knocked 0.4 rad to the left: turn back (negative wz)
    ctx.t += 0.5
    m.tick()
    assert wzs[-1] < 0 and abs(wzs[-1]) <= 0.3 + 1e-9
    ctx.pose_[2] = 0.6 - 0.4                            # and the other way
    ctx.t += 0.5
    m.tick()
    assert wzs[-1] > 0


# ---------------------------------------------------------------------------------------- backtracking through a curtain
def test_backtracking_walks_back_through_a_curtain_with_its_virtual_wall_lifted():
    from test_mission import T, tee_world
    ctx, m = make({'explorations.tiles.neighbours_fallback': False, 'modes.curtain.speed_mps': 0.25,
                   'modes.curtain.blocked_gives_up': False},
                  world=tee_world())
    d = Drive(ctx, np.zeros((0, 4)))
    branch = None
    for _ in range(200):                       # until the robot has gone from the split tile into a branch
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        trail = [T.tile_label(t) for t in ctx.tiles.trail]
        if '1,0' in trail and trail[-1] in ('1,1', '2,0'):
            branch = tuple(int(v) for v in trail[-1].split(','))
            break
    assert branch
    ctx.tiles.add_curtain_edge((1, 0), branch)           # it came through a curtain on that edge
    key = next(iter(ctx.tiles.curtain_edges))
    walls_open = []
    for _ in range(400):
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if m.current.type_name in ('backtracking', 'curtain'):
            ctx.tiles.virtual_walls()
            walls_open.append(ctx.tiles.open_curtains)
        if ctx.complete:
            break
    seen = modes_seen(m)
    i = seen.index('backtracking')
    assert seen[i:i + 3] == ['backtracking', 'curtain', 'backtracking'], seen
    assert walls_open and all(walls_open)                 # curtain edges stay open the whole way back
    assert not ctx.tiles.open_curtains                    # ... and close again afterwards
    assert key in ctx.tiles.curtain_edges
    other = '1,1' if branch == (2, 0) else '2,0'
    assert any(k == 'backtracked' and dd['tile'] == other for k, dd in ctx.events)
    assert [c for c in d.cmds if isinstance(c, tuple) and c[0] > 0]   # walked through it with vx
    assert ctx.complete


def test_curtain_edges_are_left_out_of_the_virtual_walls_only_while_open():
    from test_mission import corridor
    from lite3_explorer.tile_tracker import TileTracker
    tr = TileTracker()
    tr.align_on_entry = False
    tr.set_map(*corridor(4.0))
    tr.establish([0.5, 0.5, 0.0])
    tr.start_wall = None
    seg = tr.grid.edge((1, 0), 'N')                      # the corridor's north wall above tile 1,0
    near = lambda segs: any(abs(a[0] - seg[0][0]) + abs(a[1] - seg[0][1]) + abs(b[0] - seg[1][0]) + abs(b[1] - seg[1][1])
                            < 0.05 for a, b in segs)
    assert near(tr.virtual_walls())
    tr.add_curtain_edge((1, 0), (1, 1))
    assert near(tr.virtual_walls())                      # closed: still a wall for Nav2
    tr.open_curtains = True
    assert not near(tr.virtual_walls())                  # backtracking: lifted
    tr.open_curtains = False
    assert near(tr.virtual_walls())                      # and back afterwards (hysteresis kept it)


def test_backtracking_carries_on_when_the_curtain_cannot_be_walked_through():
    from test_mission import T, tee_world
    ctx, m = make({'explorations.tiles.neighbours_fallback': False, 'modes.curtain.speed_mps': 0.25, 'modes.curtain.low_move_s': 0,
                   'modes.curtain.blocked_gives_up': False},
                  world=tee_world())
    d = Drive(ctx, np.zeros((0, 4)))
    branch = None
    for _ in range(200):
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        trail = [T.tile_label(t) for t in ctx.tiles.trail]
        if '1,0' in trail and trail[-1] in ('1,1', '2,0'):
            branch = tuple(int(v) for v in trail[-1].split(','))
            break
    assert branch
    ctx.tiles.add_curtain_edge((1, 0), branch)
    d.blocked = lambda: m.current.name == 'curtain' and m.current.phase == 'cross'   # something behind the curtain
    for _ in range(600):
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if ctx.complete:
            break
    seen = modes_seen(m)
    i = seen.index('curtain')
    assert seen[i - 1:i + 2] == ['backtracking', 'curtain', 'backtracking'], seen
    failed = next(dd for k, dd in ctx.events if k == 'curtain_failed')
    assert failed['back_to'] == 'backtracking' and failed['attempts'] == 3
    assert sum(1 for k, _ in ctx.events if k == 'curtain_retry') == 2
    assert not any('could not walk through the curtain' in r for _, _, _, r in m.transitions)
    assert any('looking for another way' in msg for _, msg in ctx.logs)
    other = '1,1' if branch == (2, 0) else '2,0'
    assert any(k == 'backtracked' and dd['tile'] == other for k, dd in ctx.events)   # got there the other way
    assert ctx.complete


# ---------------------------------------------------------------------------------------- curtain left for later
def test_curtain_at_a_split_is_left_for_later_and_walked_through_when_backtracking():
    from test_mission import T, tee_world
    ctx, m = make({'explorations.tiles.neighbours_fallback': False, 'modes.exploring.curtains': True,
                   'modes.curtain.speed_mps': 0.25}, world=tee_world())
    Drive(ctx, spots(2.05, 0.1, 0.9))                     # a curtain on the east edge of split tile 1,0 (tile 2,0 behind)
    for _ in range(800):
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if ctx.complete:
            break
    deferred = [dd for k, dd in ctx.events if k == 'curtain_deferred']
    assert deferred and deferred[0]['tile'] == '1,0' and deferred[0]['neighbour'] == '2,0' and deferred[0]['first'] == '1,1'
    splits = [dd for k, dd in ctx.events if k == 'split']
    assert splits and splits[0]['tile'] == '1,0' and splits[0]['options'] == ['1,1', '2,0']
    trail = [T.tile_label(t) for t in ctx.tiles.trail]
    assert '1,2' in trail and trail.index('1,2') < trail.index('2,0'), trail   # the branch without the curtain first
    seen = modes_seen(m)
    i = seen.index('curtain')
    assert seen[i - 1] == 'backtracking', seen            # back at the split, then through the curtain
    assert ctx.tiles.curtain_edges and ctx.complete
    assert not ctx.tiles.curtains_left()


def test_curtain_is_walked_through_at_once_without_another_way():
    ctx, m, d = build(pose_yaw=0.0)                       # corridor: the curtain is the only way on
    assert run_until(ctx, m, lambda: 'curtain' in modes_seen(m))
    assert not any(k == 'curtain_deferred' for k, _ in ctx.events)


def _wall_cloud(x, y0, y1, zs, n=40):
    ys = np.linspace(y0, y1, n)
    return np.array([[x, y, z] for y in ys for z in zs])


def test_lidar_curtain_is_floor_level_only_with_a_tall_wall_behind():
    robot = (0.0, 0.5)
    floor_hem = [0.02, 0.05, 0.10, 0.15]
    tall = [0.05, 0.2, 0.35, 0.5, 0.6]
    m = cc.LidarCurtainMap()
    for _ in range(3):
        m.add(np.vstack([_wall_cloud(1.0, 0.0, 1.0, floor_hem), _wall_cloud(1.5, 0.0, 1.0, tall)]), robot)
    pts = m.confirmed()
    assert len(pts) > 10 and np.all(np.abs(pts[:, 0] - 1.0) < 0.1) and np.all(pts[:, 2] > 0.9)
    # a solid 0.6 m panel in the same place is a wall, not a curtain
    w = cc.LidarCurtainMap()
    for _ in range(3):
        w.add(np.vstack([_wall_cloud(1.0, 0.0, 1.0, tall), _wall_cloud(1.5, 0.0, 1.0, tall)]), robot)
    assert len(w.confirmed()) == 0
    # floor-level returns with nothing tall behind them: no curtain either
    f = cc.LidarCurtainMap()
    for _ in range(3):
        f.add(_wall_cloud(1.0, 0.0, 1.0, floor_hem), robot)
    assert len(f.confirmed()) == 0
    assert cc.edge_curtain(GRID, (0, 0), pts) is not None


def test_a_small_patch_of_camera_spots_does_not_confirm_an_edge_lidar_cannot_widen_it():
    # bag stairs_25_11_153409: 3 camera spots in a 9 cm patch + lidar-only spots along the wall started a crossing
    cam = np.array([[1.0, y, 1.0, 0.0, 1.0] for y in (0.53, 0.58, 0.62)])
    lid = np.array([[1.0, y, 1.0, 0.0, 0.0] for y in np.linspace(0.8, 1.1, 14)])
    assert cc.edge_curtain(GRID, (0, 0), np.vstack([cam, lid])) is None
    wide = np.array([[1.0, y, 1.0, 0.0, 1.0] for y in np.linspace(0.3, 0.8, 8)])
    assert cc.edge_curtain(GRID, (0, 0), wide)['side'] == 'E'
    far = np.array([[1.25, y, 1.0, 0.0, 1.0] for y in np.linspace(0.3, 0.8, 8)])     # 0.25 m off the line: not close enough
    assert cc.edge_curtain(GRID, (0, 0), far) is None


def test_curtain_map_needs_two_viewpoints_and_one_hit_per_frame():
    pts = np.array([[1.0, 0.5], [1.0, 0.5], [1.01, 0.51]])          # the same cell three times in one frame = one hit
    m = cc.CurtainMap(confirm_hits=3, min_view_sep_m=0.15)
    for _ in range(2):
        assert not m.add(pts, (0.0, 0.5))
    assert len(m) == 0
    for _ in range(10):                                              # standing still: never confirmed
        m.add(pts, (0.0, 0.5))
    assert len(m) == 0
    assert m.add(pts, (0.0, 0.8)) and len(m) >= 1                    # a second viewpoint 0.3 m away confirms it
    off = cc.CurtainMap(confirm_hits=3, min_view_sep_m=0.0)
    for _ in range(3):
        off.add(pts, (0.0, 0.5))
    assert len(off) >= 1


def test_low_movement_in_the_cross_goes_to_recovery_after_low_move_s():
    ctx, m, d = build(pose_yaw=0.0, flat={'modes.curtain.low_move_s': 2.0})
    d.wall_x = 0.8                                               # bumped: the robot keeps commanding but does not move
    assert run_until(ctx, m, lambda: m.current.name == 'recovery' or any(k == 'curtain_failed' for k, _ in ctx.events), n=300)
    failed = next(dd for k, dd in ctx.events if k == 'curtain_failed')
    assert failed['back_to'] == 'recovery' and 'barely moved for 2 s' in failed['why'], failed
    assert 'recovery' in modes_seen(m) or m.current.name == 'recovery'
    assert d.cmds[-1] == 'stop' or 'stop' in d.cmds


def test_low_movement_timeout_is_configurable_and_can_be_switched_off():
    def seconds_to_fail(low):
        ctx, m, d = build(pose_yaw=0.0, flat={'modes.curtain.low_move_s': low, 'modes.curtain.stall_s': 99.0,
                                              'modes.curtain.cross_timeout_s': 99.0})
        d.wall_x = 0.8
        t0 = ctx.t
        hit = run_until(ctx, m, lambda: any(k == 'curtain_failed' for k, _ in ctx.events), n=60)
        return (ctx.t - t0) if hit else None
    short, long_ = seconds_to_fail(2.0), seconds_to_fail(5.0)
    assert short is not None and long_ is not None and long_ - short >= 2.5     # 3 s more timeout, about 3 s later
    assert seconds_to_fail(0) is None                                            # off: nothing else fails within 60 s


def test_a_turn_in_place_is_not_low_movement():
    ctx, m, d = build(pose_yaw=2.5, flat={'modes.curtain.low_move_s': 2.0, 'modes.curtain.min_turn_rad_s': 0.2,
                                          'modes.curtain.max_turn_rad_s': 0.25})   # a slow turn: ~14 deg per step
    assert run_until(ctx, m, lambda: ctx.pose_[0] >= 1.5 or any(k == 'curtain_failed' for k, _ in ctx.events), n=200)
    assert not [dd for k, dd in ctx.events if k == 'curtain_failed'], ctx.events


# ------------------------------------------------------------------- the one curtain wall (mission manager, RViz yellow)
def test_curtain_wall_is_the_tile_edge_with_the_most_camera_spots():
    from lite3_explorer.curtain_core import curtain_wall
    from lite3_explorer.tiles import TileGrid
    g = TileGrid(0.0, 0.0, 0.0, 1.0)
    curtain = [[2.0 + dx, y, 1.0] for y in (0.2, 0.4, 0.6, 0.8) for dx in (-0.03, 0.04)]   # on x = 2, tile row 0
    lidar_wall = [[x, 1.0, 0.0] for x in np.linspace(0.1, 0.9, 12)]                       # lidar only, on y = 1
    stray = [[0.5, 0.5, 1.0]]                                                             # mid-tile: on no edge
    w = curtain_wall(g, np.array(curtain + lidar_wall + stray))
    assert w['tile'] == (2, 0) and w['side'] == 'W' and w['camera_spots'] == 8
    assert w['segment'] == g.edge((2, 0), 'W')
    assert curtain_wall(g, np.array(lidar_wall)) is None                                  # lidar alone: never a curtain
    assert curtain_wall(g, np.array([[2.0, 0.5, 1.0], [2.0, 0.55, 1.0], [2.0, 0.6, 1.0]])) is None   # too short


def test_curtain_wall_moves_to_an_edge_with_clearly_more_camera_spots():
    # bag fullrun_25_11_215528: a grey wall seen close up gave 6 colour spots at 65 s and was locked as the curtain
    # wall; the real curtain past the stairs reached 16 spots at 224 s and was never shown. The pick is redone and
    # moves at switch_ratio x the current edge's camera spots, not for a similar count (no flicker).
    from lite3_explorer.curtain_core import pick_curtain_wall
    from lite3_explorer.tiles import TileGrid
    g = TileGrid(0.0, 0.0, 0.0, 1.0)
    on_x = lambda x, n: [[x, y, 1.0] for y in np.linspace(0.15, 0.85, n)]   # noqa: E731  camera spots along x = const
    wall = on_x(2.0, 6)
    first = pick_curtain_wall(g, np.array(wall))
    assert (first['tile'], first['side']) == ((2, 0), 'W') and first['camera_spots'] == 6
    cur = (first['tile'], first['side'])
    assert pick_curtain_wall(g, np.array(wall + on_x(4.0, 9)), current=cur) is None             # 9 < 2 x 6: keep
    moved = pick_curtain_wall(g, np.array(wall + on_x(4.0, 16)), current=cur)
    assert (moved['tile'], moved['side']) == ((4, 0), 'W') and moved['camera_spots'] == 16
    assert pick_curtain_wall(g, np.array(wall + on_x(4.0, 16)), current=((1, 0), 'E')) is not None   # same edge, other name
    assert pick_curtain_wall(g, np.array(wall), current=((1, 0), 'E')) is None                  # already shown: keep
    assert pick_curtain_wall(g, np.array(on_x(4.0, 3)), current=cur) is not None                # current lost all spots
