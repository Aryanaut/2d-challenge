"""Roller tiles: detection (roller_core, no ROS), roller tiles, and the roller mode crossing one in the h-step gait."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lite3_explorer import roller_core as rc  # noqa: E402
from lite3_explorer import stairs_core as sc  # noqa: E402
from lite3_explorer.tiles import TileGrid  # noqa: E402
from test_curtain import Drive  # noqa: E402
from test_mission import make, modes_seen  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), 'data', 'roller_frames.npz')
STAIRS = os.path.join(os.path.dirname(__file__), 'data', 'stairs_depth_frames.npz')


# ------------------------------------------------------------------------------------------ synthetic scenes
def plane(x0, x1, y0, y1, z, step=0.01):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=1)


def face(x, y0, y1, z0, z1, step=0.01):
    ys, zs = np.meshgrid(np.arange(y0, y1, step), np.arange(z0, z1, step))
    return np.stack([np.full(ys.size, x), ys.ravel(), zs.ravel()], axis=1)


def roller(x=0.75, top=0.17, radius=0.075, y0=-0.4, y1=0.4, far_floor=True, seed=0):
    """Seen from the origin (camera ~0.3 m up): floor up to the roller, its front half (a cylinder along y), then the
    floor hidden behind it up to ~1.7 m."""
    parts = [plane(0.3, x - radius, y0, y1, 0.0)]
    cz = top - radius
    for a in np.linspace(-math.pi / 2, math.pi / 2, 16):        # front half of the cylinder, facing the camera
        parts.append(face(x - radius * math.cos(a), y0, y1, cz + radius * math.sin(a), cz + radius * math.sin(a) + 0.01))
    if far_floor:
        parts.append(plane(1.75, 2.2, y0, y1, 0.0))
    pts = np.vstack(parts)
    return pts + np.random.default_rng(seed).normal(0, 0.003, pts.shape)


def rotate(pts, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    out = pts.copy()
    out[:, 0], out[:, 1] = c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]
    return out


def test_roller_is_detected_with_its_axis_and_the_way_across():
    det = rc.detect_roller(roller(), (0.0, 0.0), 0.0)
    assert det is not None
    assert abs(det.top_m - 0.17) < 0.03 and det.length_m >= 0.7 and det.width_m <= 0.2
    assert det.normal[0] > 0.95                                         # across it: away from the camera (+x)
    assert abs(det.centre[0] - 0.72) < 0.08


def test_roller_seen_at_an_angle():
    det = rc.detect_roller(rotate(roller(), math.radians(40)), (0.0, 0.0), 0.0)
    assert det is not None
    assert abs(math.degrees(math.atan2(det.normal[1], det.normal[0])) - 40) < 10


def test_walls_steps_boxes_and_cables_are_not_rollers():
    floor = plane(0.3, 0.75, -0.4, 0.4, 0.0)
    wall = np.vstack([floor, face(0.75, -0.4, 0.4, 0.0, 0.55)])                      # too tall
    step = np.vstack([floor, face(0.75, -0.4, 0.4, 0.0, 0.17), plane(0.75, 1.2, -0.4, 0.4, 0.17)])   # top behind the edge
    box = np.vstack([floor, face(0.75, -0.4, 0.4, 0.0, 0.17), plane(0.75, 0.9, -0.4, 0.4, 0.17),
                     face(0.9, -0.4, 0.4, 0.17, 0.45)])                               # something tall right behind
    cable = np.vstack([floor, face(0.75, -0.4, 0.4, 0.0, 0.04)])                     # too low
    post = np.vstack([floor, face(0.75, -0.05, 0.05, 0.0, 0.17)])                    # too short
    for name, pts in (('wall', wall), ('step', step), ('box', box), ('cable', cable), ('post', post)):
        assert rc.detect_roller(pts, (0.0, 0.0), 0.0) is None, name
    assert rc.detect_roller(np.zeros((0, 3)), (0.0, 0.0), 0.0) is None


def test_floor_height_matters():
    """The same roller on a floor 0.3 m lower in the frame is still a roller once the floor is known."""
    pts = roller()
    pts[:, 2] -= 0.3
    assert rc.detect_roller(pts, (0.0, 0.0), 0.0) is None
    assert rc.detect_roller(pts, (0.0, 0.0), -0.3) is not None


# ------------------------------------------------------------------------------------------ real frames from the bag
def bag_points(d, key, i):
    pts = sc.depth_to_points(d[key][i] * 0.001, tuple(d['intrinsics']), 1) @ d[key + '_R'][i].T
    pts[:, 2] += d[key + '_h'][i]
    return pts


def test_real_roller_from_the_bag():
    """Depth frames of bags/data/roller (every 4th pixel, levelled on the floor in front): the roller is found every time
    the camera faces it, from the front and at an angle; the stepping-block field, wall panels and the corridor are not
    rollers."""
    d = np.load(DATA)
    p = rc.RollerParams(stride=1)                                      # the frames are already subsampled
    for i, t in enumerate(d['roller_t']):
        det = rc.detect_roller(bag_points(d, 'roller', i), (0.0, 0.0), 0.0, p)
        assert det is not None, f'roller not found at t={t:.1f}'
        assert 0.14 < det.top_m < 0.20 and det.length_m >= 0.5 and det.normal[0] > 0.9, (t, det)
    for i, t in enumerate(d['no_roller_t']):
        assert rc.detect_roller(bag_points(d, 'no_roller', i), (0.0, 0.0), 0.0, p) is None, f'roller found at t={t:.1f}'


def test_staircase_from_the_stairs_bag_is_not_a_roller():
    d = np.load(STAIRS)
    to_level, height = d['optical_to_level'], float(d['camera_height_m'])
    for key in ('stairs', 'no_stairs'):
        for frame in d[key]:
            pts = sc.depth_to_points(frame * 0.001, tuple(d['intrinsics']), 1) @ to_level.T
            pts[:, 2] += height
            assert rc.detect_roller(pts, (0.0, 0.0), 0.03, rc.RollerParams(stride=1)) is None, key


def test_real_lidar_sees_only_low_returns_on_the_roller_line():
    """bags/data/roller lidar (livox frame, floor ~0.15 m below it): facing the roller, low-only cells line up along it
    ~0.85 m ahead; looking elsewhere, nothing there."""
    d = np.load(DATA)
    floor = float(d['lidar_floor_z'])

    def line_cells(cloud):
        m = rc.LidarLowMap()
        m.add(cloud, (0.0, 0.0), floor)
        c = m.confirmed()
        return c[(np.abs(c[:, 1]) < 0.5) & (c[:, 0] > 0.6) & (c[:, 0] < 1.0)]
    on = line_cells(d['lidar_roller'])
    assert len(on) >= 8 and on[:, 1].max() - on[:, 1].min() >= 0.3, on
    assert len(line_cells(d['lidar_no_roller'])) == 0


def test_lidar_rejects_walls_and_curtains():
    """Low returns with wall-height ones on top (a wall) or a wall 0.5 m behind (a curtain hem) are not confirmed."""
    rng = np.random.default_rng(1)
    low = face(1.0, -0.3, 0.3, 0.05, 0.2)
    m = rc.LidarLowMap()
    m.add(low + rng.normal(0, 0.002, low.shape), (0.0, 0.0), 0.0)
    assert len(m.confirmed()) > 0
    wall = rc.LidarLowMap()
    wall.add(np.vstack([low, face(1.0, -0.3, 0.3, 0.32, 0.6)]), (0.0, 0.0), 0.0)
    assert len(wall.confirmed()) == 0
    curtain = rc.LidarLowMap()
    curtain.add(np.vstack([low, face(1.5, -0.6, 0.6, 0.32, 0.6)]), (0.0, 0.0), 0.0)
    assert len(curtain.confirmed()) == 0
    near = m.confirmed(near_xy=[[1.0, 0.0]], radius_m=0.1)              # restricted search: only cells near the spot
    assert 0 < len(near) < len(m.confirmed())


def test_roller_map_confirms_spots_seen_in_several_frames_and_counts_lidar():
    det = rc.detect_roller(roller(), (0.0, 0.0), 0.0)
    rm = rc.RollerMap(confirm_hits=3)
    assert not rm.add(det, frame=1) and not rm.add(det, frame=1)          # the same frame counts once
    assert not rm.add(det, frame=2)
    assert rm.add(det, frame=3) and len(rm) > 0
    assert np.all(rm.confirmed()[:, 4] == 0)
    flipped = rc.RollerDetection(det.xy, det.axis, (-det.normal[0], -det.normal[1]), det.top_m, det.length_m,
                                 det.width_m, det.points)
    rm.add(flipped, frame=4)                                            # seen from the other side: one direction kept
    pts = rm.confirmed(lidar_xy=np.array([[0.72, 0.0], [0.72, 0.1]]))
    assert np.all(np.abs(pts[:, 2]) > 0.95) and pts[:, 4].max() >= 1
    assert rc.points_from_json({'points': pts.tolist()}).shape == pts.shape
    assert rc.points_from_json({'points': [[1, 2, 1, 0]]}).shape == (1, 5)   # no lidar column: 0


# ------------------------------------------------------------------------------------------ tiles
GRID = TileGrid(0.0, 0.0, 0.0, 1.0)


def roller_pts(tile=(1, 0), normal=(1.0, 0.0), lidar=1, n=6):
    """Spots of a roller across the middle of `tile` (along its N-S axis when normal is +x)."""
    cx, cy = tile[0] + 0.5, tile[1] + 0.5
    along = (-normal[1], normal[0])
    return np.array([[cx + t * along[0], cy + t * along[1], normal[0], normal[1], lidar]
                     for t in np.linspace(-0.3, 0.3, n)])


def test_roller_tiles_need_spots_mid_tile_and_the_lidar():
    assert set(rc.roller_tiles(GRID, roller_pts())) == {(1, 0)}
    assert rc.roller_tiles(GRID, roller_pts(n=3)) == {}                           # too few spots
    assert rc.roller_tiles(GRID, roller_pts(lidar=0)) == {}                       # the lidar does not confirm it ...
    assert set(rc.roller_tiles(GRID, roller_pts(lidar=0), min_lidar=0)) == {(1, 0)}   # ... unless it is not needed
    long_ridge = roller_pts(lidar=0, n=14)                                        # a long, well-seen ridge: no lidar needed
    assert set(rc.roller_tiles(GRID, long_ridge, depth_only_points=12)) == {(1, 0)}
    assert rc.roller_tiles(GRID, long_ridge) == {} and rc.roller_tiles(GRID, roller_pts(lidar=0, n=8), depth_only_points=12) == {}
    edge = roller_pts()
    edge[:, 0] = 1.05                                                             # along the edge: spill-over
    assert rc.roller_tiles(GRID, edge) == {}


def test_roller_crossing_from_the_tile_beside_it():
    rollers = rc.roller_tiles(GRID, roller_pts())
    c = rc.roller_crossing(GRID, (0, 0), rollers)
    assert c['side'] == 'E' and c['roller'] == (1, 0) and c['beyond'] == (2, 0) and c['from_tile'] == (0, 0)
    assert abs(c['through_m'] - 1.5) < 1e-9 and abs(c['yaw']) < 1e-9 and c['start_xy'] == (0.5, 0.5)
    c = rc.roller_crossing(GRID, (2, 0), rollers)                                 # from the far side: the other way
    assert c['side'] == 'W' and c['beyond'] == (0, 0)
    assert rc.roller_crossing(GRID, (1, 1), rollers) is None                      # beside it: the roller lies along the way
    assert rc.roller_crossing(GRID, (0, 0), rollers, skip=[(1, 0)]) is None


def test_roller_crossing_from_its_own_tile():
    """Nav2 drove into the front half of the roller tile before the roller was confirmed: cross from there."""
    rollers = rc.roller_tiles(GRID, roller_pts())
    c = rc.roller_crossing(GRID, (1, 0), rollers, robot_xy=(1.2, 0.4))
    assert c['side'] == 'E' and c['from_tile'] == (0, 0) and c['beyond'] == (2, 0)
    assert abs(c['start_xy'][0] - 1.15) < 1e-6 and abs(c['start_xy'][1] - 0.5) < 1e-6   # clear_m before the roller
    c = rc.roller_crossing(GRID, (1, 0), rollers, robot_xy=(1.8, 0.5))              # on the far half: cross back
    assert c['side'] == 'W' and c['beyond'] == (0, 0)


# ------------------------------------------------------------------------------------------ the mode
def build(flat=None, lidar=1, yaw=1.0):
    """4 m corridor (tiles 0..3 along x), robot on tile 0,0, a roller across the middle of tile 1,0."""
    cfg = {'modes.exploring.rollers': True, 'modes.exploring.scan_after_reach': False,
           'modes.roller.gait_settle_s': 1.0, 'modes.roller.approach_angle_deg': 0.0}
    cfg.update(flat or {})
    ctx, m = make(cfg, length_m=4.0)
    ctx.pose_[:] = [0.5, 0.5, yaw]
    d = Drive(ctx, np.zeros((0, 4)))
    gait = []
    ctx.hstep = lambda on: gait.append((on, list(ctx.pose_)))
    ctx.tiles.set_roller_points(roller_pts(lidar=lidar))             # before the tile grid exists, like a latched topic
    return ctx, m, d, gait


def run_until(ctx, m, stop, n=200):
    for _ in range(n):
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if stop():
            return True
    return False


def test_roller_beside_the_tile_is_crossed_in_hstep_then_marked_explored():
    ctx, m, d, gait = build()
    assert run_until(ctx, m, lambda: ctx.complete is not None, n=300), modes_seen(m)
    seen = modes_seen(m)
    assert seen.count('roller') == 1 and seen[seen.index('roller') + 1] == 'exploring', seen
    # h-step on squared up at the centre of the tile in front, released at the centre of the tile beyond
    assert [on for on, _ in gait] == [True, False]
    on_pose, off_pose = gait[0][1], gait[1][1]
    assert math.hypot(on_pose[0] - 0.5, on_pose[1] - 0.5) <= 0.12 and abs(on_pose[2]) <= math.radians(4) + 1e-6
    assert off_pose[0] >= 2.5 - 1e-6 and abs(off_pose[1] - 0.5) < 0.05
    assert [c for c in d.cmds if isinstance(c, tuple) and c[0] > 0.15]       # crossed with forward vx
    tiles = ctx.tiles
    assert (1, 0) in tiles.rollers_crossed and (1, 0) in tiles.trail and (2, 0) in tiles.trail
    assert tiles.trail.index((1, 0)) < tiles.trail.index((2, 0))
    assert tiles.tiles[(1, 0)]['status'] == 'visited'                           # explored
    assert all(label != '1,0' for _, label in ctx.drives if _ != 'cmd')        # never a Nav2 target
    assert tiles.known_route((3, 0), (0, 0)) is None                            # the initial tile is closed once left
    tiles.initial_tile = None                                                   # (were it open:)
    assert tiles.known_route((3, 0), (0, 0)) == [(3, 0), (2, 0), (0, 0)]       # back over the roller: one hop ...
    assert tiles.roller_between((2, 0), (0, 0)) == (1, 0)                       # ... that the roller mode crosses
    assert any(k == 'roller_crossed' for k, _ in ctx.events)


def test_roller_tile_is_not_a_target_and_not_unexplored_before_it_is_crossed():
    ctx, m, d, gait = build({'modes.exploring.rollers': False})
    ctx.tiles.establish(ctx.pose_)
    assert ctx.tiles.tiles[(1, 0)]['status'] == 'roller'
    assert '1,0' not in ctx.tiles.unexplored() and ctx.tiles.rollers_left() == ['1,0']
    assert all(t['tile'] != '1,0' for t in ctx.tiles.targets(ctx.pose_[:2]))


def test_no_roller_mode_when_rollers_are_off():
    ctx, m, d, gait = build({'modes.exploring.rollers': False})
    run_until(ctx, m, lambda: ctx.complete is not None, n=150)
    assert 'roller' not in modes_seen(m) and not gait


def test_roller_without_lidar_confirmation_is_not_crossed():
    ctx, m, d, gait = build(lidar=0)
    run_until(ctx, m, lambda: ctx.complete is not None, n=150)
    assert 'roller' not in modes_seen(m) and not ctx.tiles.rollers


def test_stuck_at_the_roller_backs_off_retries_then_gives_up_and_releases_the_gait():
    ctx, m, d, gait = build({'modes.roller.stall_s': 3.0, 'modes.roller.max_attempts': 2})
    d.wall_x = 1.4                                                          # cannot get over it
    assert run_until(ctx, m, lambda: 'roller' in modes_seen(m) and m.current.name != 'roller', n=300)
    seen = modes_seen(m)
    assert seen[seen.index('roller') + 1] == 'recovery', seen               # gave up past the edge: recovery
    assert gait and gait[0][0] is True and gait[-1][0] is False             # h-step released
    assert [k for k, _ in ctx.events].count('roller_retry') == 1
    assert [c for c in d.cmds if isinstance(c, tuple) and c[0] < 0]        # backed off before the 2nd try
    assert (1, 0) not in ctx.tiles.rollers_crossed and (1, 0) in m.rollers_tried
    assert any(k == 'roller_failed' for k, _ in ctx.events)


def test_no_square_up_means_no_hstep_and_back_to_exploring():
    ctx, m, d, gait = build({'modes.roller.align_timeout_s': 2.0})
    d.drive_orig = d.drive

    def no_turn(vx, wz, vy=0.0):
        d.drive_orig(vx, 0.0 if vx == 0.0 else wz, vy)                     # turning in place does nothing
    ctx.drive = no_turn
    assert run_until(ctx, m, lambda: 'roller' in modes_seen(m) and m.current.name != 'roller', n=200)
    seen = modes_seen(m)
    assert seen[seen.index('roller') + 1] == 'exploring' and not gait


def test_leaving_the_roller_mode_mid_crossing_releases_the_gait():
    ctx, m, d, gait = build()
    assert run_until(ctx, m, lambda: m.current.name == 'roller' and m.current.phase == 'cross', n=200)
    ok, _ = m.request('idle', 'operator command')
    assert ok and [on for on, _ in gait] == [True, False]


# ------------------------------------------------------------------------------------------ backtracking over a roller
def roller_tee_world():
    """Corridor tiles 0,0 .. 3,0 with a branch going north from tile 1,0 (tiles 1,1 and 1,2)."""
    from test_mission import RES, GridInfo
    margin = 1.0
    info = GridInfo(int((4.0 + 2 * margin) / RES), int((3.0 + 2 * margin) / RES), RES, -margin, -margin)
    grid = np.full((info.height, info.width), -1, dtype=np.int16)
    rects = [((0.0, 0.0), (4.0, 1.0)), ((1.0, 1.0), (2.0, 3.0))]
    for fill, shrink in ((100, 1), (0, 0)):
        for (x0, y0), (x1, y1) in rects:
            r0, c0 = info.world_to_cell(x0, y0)
            r1, c1 = info.world_to_cell(x1, y1)
            grid[r0 - shrink:r1 + shrink, c0 - shrink:c1 + shrink] = fill
    return grid, info


def build_tee(flat=None):
    """Roller across tile 2,0; tile 1,0 is a split (the branch north, or over the roller to tile 3,0)."""
    cfg = {'modes.exploring.rollers': True, 'modes.roller.gait_settle_s': 1.0,
           'modes.roller.approach_angle_deg': 0.0,
           'modes.roller.min_turn_rad_s': 0.05,       # the fake robot steps 1 s a tick: 0.2 rad/s overshoots the 4 deg tolerance
           'explorations.tiles.neighbours_fallback': False}
    cfg.update(flat or {})
    ctx, m = make(cfg, world=roller_tee_world())
    ctx.pose_[:] = [0.5, 0.5, 0.0]
    d = Drive(ctx, np.zeros((0, 4)))
    gait = []
    ctx.hstep = lambda on: gait.append((on, list(ctx.pose_)))
    ctx.tiles.set_roller_points(roller_pts(tile=(2, 0)))
    return ctx, m, d, gait


def test_backtracking_crosses_back_over_the_roller():
    ctx, m, d, gait = build_tee()
    assert run_until(ctx, m, lambda: ctx.complete is not None, n=800), modes_seen(m)
    seen = modes_seen(m)
    assert seen.count('roller') == 2, seen                                   # over it, and back over it
    i = len(seen) - 1 - seen[::-1].index('roller')
    assert seen[i - 1] == 'backtracking' and seen[i + 1] == 'backtracking', seen
    assert [on for on, _ in gait] == [True, False, True, False]
    assert abs(gait[2][1][2] - math.pi) < math.radians(5) or abs(abs(gait[2][1][2]) - math.pi) < math.radians(5)
    assert any(k == 'split' and dd['tile'] == '1,0' for k, dd in ctx.events)
    trail = ctx.tiles.trail
    assert (3, 0) in trail and (1, 2) in trail and trail.index((3, 0)) < trail.index((1, 2))
    back = [dd for k, dd in ctx.events if k == 'roller_crossed']
    assert back[-1]['back_to'] == 'backtracking' and back[-1]['beyond'] == '1,0'
    assert any(k == 'backtracked' and dd['tile'] == '1,1' for k, dd in ctx.events)


def test_backtracking_looks_for_another_way_when_the_roller_cannot_be_crossed_back():
    ctx, m, d, gait = build_tee({'modes.roller.stall_s': 3.0, 'modes.roller.max_attempts': 1,
                                 'modes.roller.align_timeout_s': 2.0})
    crossed = []
    orig = ctx.tiles.mark_roller_crossed

    def mark(*a, **kw):
        crossed.append(a)
        orig(*a, **kw)
        d.drive_orig = getattr(d, 'drive_orig', d.drive)
        ctx.drive = lambda vx, wz, vy=0.0: d.drive_orig(vx, 0.0 if vx == 0.0 else wz, vy)   # cannot turn any more
    ctx.tiles.mark_roller_crossed = mark
    assert run_until(ctx, m, lambda: len([1 for k, _ in ctx.events if k == 'roller_failed']) > 0, n=800), modes_seen(m)
    failed = next(dd for k, dd in ctx.events if k == 'roller_failed')
    assert failed['back_to'] == 'backtracking'                               # never stepped on it: hand back
    run_until(ctx, m, lambda: m.current.name != 'backtracking', n=20)
    assert any('looking for another way' in msg for _, msg in ctx.logs)
    assert gait[-1][0] is False


def test_recentres_on_the_edge_after_the_roller_tile_with_omni():
    ctx, m, d, gait = build({'modes.roller.min_turn_rad_s': 0.05}, yaw=0.0)
    ctx.pose_[1] += 0.12                                    # off the line through the tile centres
    assert run_until(ctx, m, lambda: ctx.complete is not None or any(k == 'roller_crossed' for k, _ in ctx.events), n=300)
    ev = [dd for k, dd in ctx.events if k == 'roller_recenter']
    assert len(ev) == 1
    assert any(k == 'roller_crossed' for k, _ in ctx.events)


def test_approach_keeps_an_angle_to_the_roller_and_walks_the_straight_line_then_squares_up():
    ctx, m, d, gait = build({'modes.roller.approach_angle_deg': 20.0, 'modes.roller.min_turn_rad_s': 0.05}, yaw=0.0)
    run_until(ctx, m, lambda: any(k == 'roller_crossed' for k, _ in ctx.events), n=300)
    assert any(k == 'roller_crossed' for k, _ in ctx.events)
    on_pose = gait[0][1]                                    # h-step on: still before the roller tile's edge line
    assert abs(math.degrees(on_pose[2]) - 20.0) < 6.0, on_pose
    ys = [p[1] for p in d.path]
    assert max(ys) - min(ys) < 0.15                         # the track stays on the tile line, only the body is turned
    assert abs(math.degrees(ctx.pose_[2])) < 8.0            # squared up after the edge line


def _cross_with_original_heading(flat, original_deg):
    ctx, m, d, gait = build(dict(flat, **{'modes.roller.min_turn_rad_s': 0.05}), yaw=0.0)
    for _ in range(400):
        ctx.t += 1.0
        if m.current.name == 'exploring' or m.current.name == 'roller':
            ctx.initial_heading = math.radians(original_deg)      # the scanning mode saved the start heading before
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if any(k == 'roller_crossed' for k, _ in ctx.events):
            break
    return ctx


def test_heading_goes_back_to_the_original_heading_at_the_midpoint_after_the_roller():
    ctx = _cross_with_original_heading({'modes.roller.approach_angle_deg': 20.0}, -12.0)
    assert any(k == 'roller_recenter' for k, _ in ctx.events)
    assert abs(math.degrees(ctx.pose_[2]) + 12.0) < 6.0, math.degrees(ctx.pose_[2])
    ctx = _cross_with_original_heading({'modes.roller.restore_heading': False}, -12.0)
    assert abs(math.degrees(ctx.pose_[2])) < 6.0                # off: stays square to the roller
