"""Staircases: detection (stairs_core, no ROS), staircase tiles, and the stairs mode climbing over one in AI mode."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lite3_explorer import stairs_core as sc  # noqa: E402
from lite3_explorer.tiles import TileGrid  # noqa: E402
from lite3_explorer import tiles as T  # noqa: E402
from test_curtain import Drive  # noqa: E402
from test_mission import make, modes_seen, rooms  # noqa: E402

import pytest  # noqa: E402

from lite3_explorer.tile_tracker import TileTracker  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), 'data', 'stairs_depth_frames.npz')
AI_ON, AI_OFF = 0x21010528, 0x2101052B


@pytest.fixture(autouse=True)
def corridor_has_no_risers(request, monkeypatch):
    """The fake corridor (test_mission.corridor) is open floor everywhere, staircase tiles too: the real 2D lidar sees
    the risers as a wall, so a real staircase tile is never 'scanned open floor'. The mode tests switch that rule off;
    tests named *open_floor* keep it."""
    if 'open_floor' not in request.node.name:
        monkeypatch.setattr(TileTracker, 'scanned_open_tiles', lambda self: set())


# ------------------------------------------------------------------------------------------ synthetic scenes
def plane(x0, x1, y0, y1, z, step=0.01):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=1)


def riser(x, y0, y1, z0, z1, step=0.01):
    ys, zs = np.meshgrid(np.arange(y0, y1, step), np.arange(z0, z1, step))
    return np.stack([np.full(ys.size, x), ys.ravel(), zs.ravel()], axis=1)


def staircase(steps=3, rise=0.13, run=0.25, start=0.8, floor=True, seed=0):
    """Seen from the origin, going up towards +x: floor, then `steps` treads with risers between them (z up)."""
    parts = [plane(0.3, start, -0.4, 0.4, 0.0)] if floor else []
    for k in range(1, steps + 1):
        x = start + (k - 1) * run
        parts += [riser(x, -0.4, 0.4, (k - 1) * rise, k * rise), plane(x, x + run, -0.4, 0.4, k * rise)]
    pts = np.vstack(parts)
    return pts + np.random.default_rng(seed).normal(0, 0.003, pts.shape)


def rotate(pts, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    out = pts.copy()
    out[:, 0], out[:, 1] = c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]
    return out


def test_staircase_is_detected_with_its_steps_and_the_way_it_goes_up():
    det = sc.detect_stairs(staircase(), (0.0, 0.0))
    assert det is not None and det.steps == 3
    assert abs(det.rise_m - 0.13) < 0.02
    assert det.ascent[0] > 0.95                                        # the steps go up towards +x
    assert det.xy[:, 0].min() >= 0.75 and det.xy[:, 0].max() <= 1.6    # the stair cells are the treads, not the floor


def test_staircase_seen_at_an_angle():
    det = sc.detect_stairs(rotate(staircase(), math.radians(60)), (0.0, 0.0))
    assert det is not None
    assert abs(math.degrees(math.atan2(det.ascent[1], det.ascent[0])) - 60) < 10


def test_walls_boxes_and_ramps_are_not_stairs():
    floor = plane(0.3, 1.0, -0.4, 0.4, 0.0)
    wall = np.vstack([floor, riser(1.0, -0.4, 0.4, 0.0, 1.0)])
    box = np.vstack([floor, riser(1.0, -0.4, 0.4, 0.0, 0.15), plane(1.0, 1.4, -0.4, 0.4, 0.15)])
    xs, ys = np.meshgrid(np.arange(1.0, 1.8, 0.01), np.arange(-0.4, 0.4, 0.01))
    ramp = np.vstack([floor, np.stack([xs.ravel(), ys.ravel(), 0.3 * (xs.ravel() - 1.0)], axis=1)])
    table = np.vstack([floor, plane(1.0, 1.6, -0.4, 0.4, 0.72)])
    for name, pts in (('wall', wall), ('box', box), ('ramp', ramp), ('table', table)):
        assert sc.detect_stairs(pts, (0.0, 0.0), floor_z=0.0) is None, name


def test_two_treads_need_the_floor_height():
    """Close to the stairs the floor in front is out of view: two treads alone are not enough (a box on a box), two
    treads a step above the known floor are."""
    pts = staircase(steps=2, floor=False)
    assert sc.detect_stairs(pts, (0.0, 0.0)) is None
    det = sc.detect_stairs(pts, (0.0, 0.0), floor_z=0.0)
    assert det is not None and det.steps == 2
    assert sc.detect_stairs(pts, (0.0, 0.0), floor_z=-0.4) is None      # 40 cm below: no step up to the first tread


def test_floor_estimate_ignores_frames_where_a_tread_is_the_nearest_level():
    fe = sc.FloorEstimate(window=10)
    floor = sc.find_levels(staircase(steps=1), (0.0, 0.0))
    tread_only = sc.find_levels(staircase(steps=2, floor=False, start=0.3), (0.0, 0.0))
    for k in range(10):
        fe.update(floor if k % 3 else tread_only)                      # every third frame only sees the first tread
    assert abs(fe.value) < 0.02


def test_stair_map_confirms_cells_seen_in_several_frames():
    det = sc.detect_stairs(staircase(), (0.0, 0.0))
    sm = sc.StairMap(confirm_hits=3)
    assert not sm.add(det, frame=1) and not sm.add(det, frame=1)       # the same frame counts once
    assert not sm.add(det, frame=2)
    assert sm.add(det, frame=3) and len(sm) > 0
    pts = sc.StairMap.from_json(sm.to_json())
    assert pts.shape[1] == 4 and np.all(pts[:, 2] > 0.95)


# ------------------------------------------------------------------------------------------ real depth frames
def test_real_metal_staircase_from_the_bag():
    """Depth frames of bags/data/stairs (every 4th pixel): the three approaches to the metal staircase are stairs, the
    walls / floor seen while turning around are not."""
    d = np.load(DATA)
    to_level, height = d['optical_to_level'], float(d['camera_height_m'])
    p = sc.StairParams(stride=1)                                       # the frames are already subsampled

    def detect(depth_mm):
        pts = sc.depth_to_points(depth_mm * 0.001, tuple(d['intrinsics']), 1) @ to_level.T
        pts[:, 2] += height
        return sc.detect_stairs(pts, (0.0, 0.0), p, floor_z=0.03)
    for t, frame in zip(d['stairs_t'], d['stairs']):
        det = detect(frame)
        assert det is not None, f'stairs not found at t={t}'
        assert 0.09 < det.rise_m < 0.17 and det.ascent[0] > 0.7, (t, det.rise_m, det.ascent)
    for t, frame in zip(d['no_stairs_t'], d['no_stairs']):
        assert detect(frame) is None, f'stairs found at t={t}'


# ------------------------------------------------------------------------------------------ tiles
GRID = TileGrid(0.0, 0.0, 0.0, 1.0)


def stair_pts(tile=(1, 0), ascent=(1.0, 0.0)):
    xs = np.linspace(tile[0] + 0.15, tile[0] + 0.85, 4)
    ys = np.linspace(tile[1] + 0.15, tile[1] + 0.85, 3)
    return np.array([[x, y, ascent[0], ascent[1]] for x in xs for y in ys])


def test_stair_tiles_and_the_side_to_climb_from():
    stairs = sc.stair_tiles(GRID, stair_pts())
    assert set(stairs) == {(1, 0)}
    assert sc.stair_tiles(GRID, stair_pts()[:3]) == {}                  # too few stair cells
    c = sc.stair_crossing(GRID, (0, 0), stairs)
    assert c['side'] == 'E' and c['neighbour'] == (1, 0) and c['stairs'] == [(1, 0)] and c['beyond'] == (2, 0)
    assert abs(c['through_m'] - 1.5) < 1e-9 and abs(c['yaw']) < 1e-9
    assert sc.stair_crossing(GRID, (1, -1), stairs) is None              # beside the staircase: its stringer
    assert sc.stair_crossing(GRID, (2, 0), stairs) is None               # at the top: the steps go down from there
    assert sc.stair_crossing(GRID, (0, 0), stairs, skip=[(1, 0)]) is None


def test_stair_cells_inside_a_traversable_tile_belong_to_the_tile_beyond_its_edge():
    # bag curtaintest_25_11_135748: the foot of the steps seen 0-0.26 m inside open tile 0,2 (ascent out of its W edge)
    # made 0,2 the staircase; the steps are in tile -1,2
    foot = np.array([[x, y, -1.0, 0.0] for x in (0.05, 0.12, 0.20, 0.26) for y in (2.3, 2.5, 2.7)])
    assert set(sc.stair_tiles(GRID, foot)) == {(0, 2)}                     # no floor known: where the cells are
    stairs = sc.stair_tiles(GRID, foot, traversable={(0, 2)})
    assert set(stairs) == {(-1, 2)} and stairs[(-1, 2)]['points'] == 12
    assert sc.stair_tiles(GRID, foot, traversable={(0, 2), (-1, 2)}) == {}  # floor on both sides: no staircase
    deep = foot.copy()
    deep[:, 0] += 0.4                                                       # 0.45-0.66 m inside: not the foot of steps
    assert sc.stair_tiles(GRID, deep, traversable={(0, 2)}) == {}
    sideways = foot[foot[:, 1] < 2.6].copy()
    sideways[:, 2:] = (0.0, 1.0)                                            # steps climbing N, 0.5+ m from the N edge
    assert sc.stair_tiles(GRID, sideways, traversable={(0, 2)}) == {}


def test_open_floor_tile_is_never_a_staircase_and_hands_it_to_the_tile_beyond():
    ctx, m = make({}, length_m=4.0)
    tiles = ctx.tiles
    run_until(ctx, m, lambda: tiles.ready, n=20)
    assert (2, 0) in tiles.scanned_open_tiles()                             # the fake corridor: open floor
    foot = np.array([[x, y, 1.0, 0.0] for x in (2.70, 2.78, 2.85, 2.92) for y in (0.3, 0.5, 0.7)])   # E edge of 2,0
    tiles.set_stair_points(foot)
    assert (2, 0) not in tiles.stairs
    tiles.scanned_open_tiles = lambda: {(2, 0)}                             # tile 3,0 holds the risers: not open floor
    tiles.set_stair_points(foot + [0.001, 0, 0, 0])
    assert set(tiles.stairs) == {(3, 0)}
    tiles.update(force=True)
    assert tiles.tiles[(3, 0)]['status'] == 'staircase' and tiles.tiles[(2, 0)]['status'] != 'staircase'


def test_a_staircase_deeper_than_a_tile_is_crossed_whole():
    stairs = sc.stair_tiles(GRID, np.vstack([stair_pts((1, 0)), stair_pts((2, 0))]))
    c = sc.stair_crossing(GRID, (0, 0), stairs)
    assert c['stairs'] == [(1, 0), (2, 0)] and c['beyond'] == (3, 0) and abs(c['through_m'] - 2.5) < 1e-9


# ------------------------------------------------------------------------------------------ the mode
def build(flat=None, ascent=(1.0, 0.0), yaw=0.6):
    """4 m corridor (tiles 0..3 along x), robot on tile 0,0, a staircase on tile 1,0."""
    cfg = {'modes.exploring.stairs': True, 'modes.exploring.scan_after_reach': False,
           'modes.stairs.ai_settle_s': 1.0, 'modes.stairs.stall_s': 3.0}
    cfg.update(flat or {})
    ctx, m = make(cfg, length_m=4.0)
    ctx.pose_[:] = [0.5, 0.5, yaw]
    d = Drive(ctx, np.zeros((0, 4)))
    ai = []

    def ai_mode(on, code):
        ai.append((on, code, list(ctx.pose_)))
        return True
    ctx.ai_mode = ai_mode
    ctx.rebases = []
    ctx.rebase_odom = lambda x, y, yaw: ctx.rebases.append((x, y, yaw))
    ctx.tiles.set_stair_points(stair_pts(ascent=ascent))               # before the tile grid exists, like a latched topic
    return ctx, m, d, ai


def run_until(ctx, m, stop, n=200):
    for _ in range(n):
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if stop():
            return True
    return False


def test_staircase_beside_the_tile_is_climbed_in_ai_mode_then_walled_off():
    ctx, m, d, ai = build()
    assert run_until(ctx, m, lambda: ctx.complete is not None, n=300), modes_seen(m)
    seen = modes_seen(m)
    assert seen.count('stairs') == 1 and seen[seen.index('stairs') + 1:seen.index('stairs') + 3] == ['scanning', 'exploring'], seen
    # AI mode on squared up at the tile in front, off at the centre of the tile beyond
    assert [(on, code) for on, code, _ in ai] == [(True, AI_ON), (False, AI_OFF)]
    on_pose, off_pose = ai[0][2], ai[1][2]
    assert math.hypot(on_pose[0] - 0.5, on_pose[1] - 0.5) <= 0.07 and abs(on_pose[2]) <= math.radians(5) + 1e-6
    assert off_pose[0] >= 2.5 - 1e-6 and abs(off_pose[1] - 0.5) < 0.05
    walks = [c for c in d.cmds if isinstance(c, tuple) and c[0] > 0.15]
    assert walks                                                        # climbed with forward vx
    # the staircase is walled off and never a Nav2 target; the tile beyond was reached
    tiles = ctx.tiles
    assert (1, 0) in tiles.stairs_crossed and tiles.tiles[(1, 0)]['status'] == 'staircase'
    walls = {tuple(map(tuple, s)) for s in tiles.virtual_walls()}
    assert all(tuple(map(tuple, tiles.grid.edge((1, 0), side))) in walls for side in 'NESW')
    assert (2, 0) in tiles.trail and (3, 0) in tiles.trail
    assert '1,0' not in tiles.unexplored() and all(label != '1,0' for _, label in ctx.drives if _ != 'cmd')
    assert tiles.known_route((3, 0), (0, 0)) is None                    # no way back over the staircase
    assert any(k == 'stairs_crossed' for k, _ in ctx.events)


def test_no_stairs_mode_when_stairs_are_off():
    ctx, m, d, ai = build({'modes.exploring.stairs': False})
    run_until(ctx, m, lambda: False, n=40)
    assert 'stairs' not in modes_seen(m) and not ai
    assert ctx.tiles.tiles[(1, 0)]['status'] == 'staircase'             # still never driven onto with Nav2


def test_staircase_seen_from_its_side_is_not_climbed():
    ctx, m, d, ai = build(ascent=(0.0, 1.0))                            # steps go up across the corridor
    run_until(ctx, m, lambda: False, n=40)
    assert 'stairs' not in modes_seen(m) and not ai


def test_stuck_on_the_stairs_turns_ai_mode_off_and_walls_it_off():
    ctx, m, d, ai = build()
    d.wall_x = 1.3                                                      # something on the stairs: never gets over
    assert run_until(ctx, m, lambda: any(k == 'stairs_failed' for k, _ in ctx.events), n=200)
    failed = next(dd for k, dd in ctx.events if k == 'stairs_failed')
    assert failed['ai_was_on'] and failed['back_to'] == 'recovery'
    assert [on for on, _, _ in ai] == [True, False]
    assert (1, 0) in ctx.tiles.stairs_crossed and (1, 0) in m.stairs_tried
    assert modes_seen(m)[-1] == 'recovery'


def test_no_square_up_means_no_ai_mode_and_back_to_exploring():
    ctx, m, d, ai = build({'modes.stairs.align_timeout_s': 3.0})
    moved = d.drive
    d.drive = None
    ctx.drive = lambda vx, wz, vy=0.0: moved(vx, 0.0, vy)               # this robot cannot turn
    assert run_until(ctx, m, lambda: any(k == 'stairs_failed' for k, _ in ctx.events), n=100)
    failed = next(dd for k, dd in ctx.events if k == 'stairs_failed')
    assert not failed['ai_was_on'] and failed['back_to'] == 'exploring'
    assert not ai and (1, 0) not in ctx.tiles.stairs_crossed and (1, 0) in m.stairs_tried


def test_leaving_the_stairs_mode_mid_climb_turns_ai_mode_off():
    ctx, m, d, ai = build()
    assert run_until(ctx, m, lambda: m.current.name == 'stairs' and m.current.phase == 'cross', n=100)
    ok, _ = m.request('idle', 'operator command')
    assert ok and [on for on, _, _ in ai] == [True, False]


def test_hex_string_codes_from_the_yaml():
    ctx, m, d, ai = build({'modes.stairs.ai_on_code': '0x21010528', 'modes.stairs.ai_off_code': '0x2101052B'})
    assert run_until(ctx, m, lambda: len(ai) == 2, n=200)
    assert [code for _, code, _ in ai] == [AI_ON, AI_OFF]


def test_a_tile_the_robot_stood_on_never_becomes_a_staircase():
    ctx, m = make({}, length_m=4.0)
    run_until(ctx, m, lambda: ctx.tiles.ready and len(ctx.tiles.trail) >= 2, n=50)
    walked = ctx.tiles.trail[1]
    assert not ctx.tiles.set_stair_points(stair_pts(walked)) and walked not in ctx.tiles.stairs


# ------------------------------------------------------------------------------------------ IMU crossing (25 Nov fixes)
class FakeImuRobot:
    """Like the real robot: the map pose FREEZES while AI mode is on (odom_tf_bridge), the IMU stays live. The true x
    goes on; counted from where AI mode went on: level for 0.4 m, nose up for 0.5 m, down for 0.5 m, level after. A turn keeps spinning
    `coast` ticks after the last non-zero wz (the robot coasting)."""
    def __init__(self, ctx, d, ai, stuck_at=None, coast=0):
        self.ctx, self.d, self.ai = ctx, d, ai
        self.true_x, self.x0, self.yaw, self.gz, self.last_turn_t = None, None, 0.0, 0.0, -1e9
        self.stuck_at, self.coast, self.yaw_kick = stuck_at, coast, 0.0
        self.nose_at_ai = []
        self.moved = d.drive
        ctx.drive = self.drive
        ctx.imu = self.imu
        self.ai_mode0 = ctx.ai_mode

        def ai_mode(on, code):
            self.nose_at_ai.append(self.nose())
            if on:
                self.true_x = self.x0 = ctx.pose_[0]
            return self.ai_mode0(on, code)
        ctx.ai_mode = ai_mode

    def in_ai(self):
        return bool(self.ai) and self.ai[-1][0]

    def nose(self):
        if self.x0 is None:
            return 0.0
        s = self.true_x - self.x0
        return 15.0 if 0.4 <= s < 0.9 else (-20.0 if 0.9 <= s < 1.4 else 0.0)

    def drive(self, vx, wz, vy=0.0):
        if wz != 0.0:
            self.last_turn_t = self.ctx.t
        self.gz = wz if wz != 0.0 else (0.3 if self.ctx.t - self.last_turn_t < self.coast else 0.0)
        if self.in_ai():
            self.d.cmds.append((vx, wz))
            nx = self.true_x + vx
            self.true_x = nx if self.stuck_at is None else min(nx, self.x0 + self.stuck_at)
            self.yaw += wz
            return                                                     # the map pose does NOT move
        self.moved(vx, wz, vy)
        self.yaw = self.ctx.pose_[2] + self.yaw_kick

    def imu(self):
        return math.radians(-self.nose()), self.yaw, self.gz, 0.0      # ROS pitch: + = nose down (nose_up_sign -1)


def build_imu(flat=None, ascent=(1.0, 0.0), **kw):
    ctx, m, d, ai = build(flat, ascent=ascent)
    return ctx, m, d, ai, FakeImuRobot(ctx, d, ai, **kw)


def test_imu_crossing_gets_over_with_a_frozen_pose_and_turns_ai_off_only_when_level():
    ctx, m, d, ai, r = build_imu()
    assert run_until(ctx, m, lambda: any(k == 'stairs_crossed' for k, _ in ctx.events), n=300), ctx.events[-5:]
    assert [on for on, _, _ in ai] == [True, False]
    assert abs(r.nose_at_ai[-1]) < 4.0                                 # AI mode off on level ground
    assert r.true_x - r.x0 >= 1.4                                      # really over the stairs
    assert ctx.rebases                                                 # odom moved to the tile beyond


def test_enter_ai_waits_until_the_robot_has_stopped_turning():
    ctx, m, d, ai, r = build_imu(coast=4)
    assert run_until(ctx, m, lambda: bool(ai), n=200)
    assert ctx.t - r.last_turn_t >= 4                                  # not while it was still coasting round
    assert abs(r.gz) < 0.05


def test_imu_crossing_never_turns_ai_off_on_the_stairs_before_the_hard_limit():
    ctx, m, d, ai, r = build_imu({'modes.stairs.cross_timeout_s': 20.0}, stuck_at=0.6)   # caught on the first step
    assert run_until(ctx, m, lambda: any(k == 'stairs_failed' for k, _ in ctx.events), n=300)
    assert [on for on, _, _ in ai] == [True, False]
    assert r.nose_at_ai[-1] > 10.0                                    # still tilted: only the hard limit (2x) ended it
    assert any(k == 'stairs_failed' for k, _ in ctx.events)


def test_timeout_after_the_stairs_counts_as_over_them_and_rebases():
    # bag stairs-rebasing_25_11_114524: up, down, level again, then cross_timeout_s ran out during the short walk on
    # ('clear'): it was counted as a failure, odom was never rebased and the robot thought it was on the start tile
    ctx, m, d, ai, r = build_imu({'modes.stairs.cross_timeout_s': 40.0, 'modes.stairs.clear_s': 1000.0})
    assert run_until(ctx, m, lambda: any(k in ('stairs_crossed', 'stairs_failed') for k, _ in ctx.events), n=300)
    assert any(k == 'stairs_crossed' for k, _ in ctx.events), [k for k, _ in ctx.events][-6:]
    assert len(ctx.rebases) == 1 and r.true_x - r.x0 >= 1.4
    assert [on for on, _, _ in ai] == [True, False]


def test_late_imu_messages_never_hand_the_climb_to_the_frozen_map_pose():
    # bag multiterrain_25_11_115442: /imu/data at ~2 Hz; a stale tick fell through to the map-pose check, which (pose
    # frozen in AI mode) saw no progress for stall_s and called the robot stuck after it had crossed: no rebase
    ctx, m, d, ai, r = build_imu()
    live = r.imu

    def gappy():
        p, y, gz, _ = live()
        return p, y, gz, (1.0 if r.in_ai() and int(ctx.t) % 2 else 0.0)     # every other tick older than imu_max_age_s
    ctx.imu = gappy
    assert run_until(ctx, m, lambda: any(k in ('stairs_crossed', 'stairs_failed') for k, _ in ctx.events), n=300)
    assert any(k == 'stairs_crossed' for k, _ in ctx.events), [(k, e.get('why')) for k, e in ctx.events][-4:]
    assert len(ctx.rebases) == 1 and r.true_x - r.x0 >= 1.4


def test_imu_gone_for_good_on_the_stairs_stops_after_imu_lost_s():
    ctx, m, d, ai, r = build_imu({'modes.stairs.imu_lost_s': 3.0})
    live = r.imu
    ctx.imu = lambda: None if r.in_ai() and r.true_x - r.x0 >= 0.6 else live()   # gone half-way up
    assert run_until(ctx, m, lambda: any(k == 'stairs_failed' for k, _ in ctx.events), n=300)
    why = dict(ctx.events)['stairs_failed']['why']
    assert why.startswith('no IMU for') and 'over the staircase' not in why, why
    assert [on for on, _, _ in ai] == [True, False]
    # never rebased to the tile beyond; only released (odom_tf_bridge holds after AI mode) where the pose says it is
    assert ctx.rebases == [tuple(ctx.pose_)], ctx.rebases
    assert not any(k == 'odom_rebase' for k, _ in ctx.events)


# ------------------------------------------------------------------- wall beyond the stairs (bag fullrun_25_11_183742)
def wall_ahead(ctx, r, wall_s, lag=0, floor_when_nose_down=None):
    """ctx.front_gap: a wall wall_s (along the walk, from where AI mode went on) ahead of the nose, seen by a scan `lag`
    ticks old (the robot walked on since). floor_when_nose_down: the gap a nose-down scan shows (the floor below the
    steps is in the scan band), None = it shows the wall."""
    hist = []

    def front_gap(half_width_m=0.2, nose_m=0.305):
        if r.x0 is None:
            return None
        hist.append((ctx.t, r.true_x - r.x0, r.nose()))
        t_cap = ctx.t - lag
        s, nose = next(((s, n) for t, s, n in reversed(hist) if t <= t_cap), hist[0][1:])
        if floor_when_nose_down is not None and nose < -4.0:
            return floor_when_nose_down, float(lag)
        gap = wall_s - s
        return (gap if gap > 0.05 else None), float(lag)          # closer than range_min: the wall drops out (inf)
    ctx.front_gap = front_gap


def test_walk_off_the_stairs_stops_short_of_a_wall_beyond_them():
    # bag fullrun_25_11_183742: level after the stairs, 'clear' walked on 2 s at 0.34 m/s into the wall ~0.5 m past them
    ctx, m, d, ai, r = build_imu({'modes.stairs.clear_s': 5.0})
    wall_s = 1.75                                                      # the nose touches the wall at 1.75 m walked
    wall_ahead(ctx, r, wall_s)
    assert run_until(ctx, m, lambda: any(k in ('stairs_crossed', 'stairs_failed') for k, _ in ctx.events), n=300)
    assert any(k == 'stairs_crossed' for k, _ in ctx.events), [(k, e.get('why')) for k, e in ctx.events][-4:]
    assert r.true_x - r.x0 <= wall_s - 0.05, r.true_x - r.x0          # never at the wall
    assert abs(r.nose_at_ai[-1]) < 4.0 and len(ctx.rebases) == 1


def test_wall_stop_allows_for_the_scan_being_late():
    # the scan is a tick old: the robot walked on since it was taken; the gap is what is left NOW
    ctx, m, d, ai, r = build_imu({'modes.stairs.clear_s': 5.0, 'modes.stairs.speed_mps': 0.3})
    wall_s = 1.9
    wall_ahead(ctx, r, wall_s, lag=1)
    assert run_until(ctx, m, lambda: any(k in ('stairs_crossed', 'stairs_failed') for k, _ in ctx.events), n=300)
    assert any(k == 'stairs_crossed' for k, _ in ctx.events)
    assert r.true_x - r.x0 <= wall_s - 0.05, r.true_x - r.x0


def test_floor_seen_nose_down_never_ends_the_climb_on_the_last_step():
    # nose down, the floor below the steps shows up a few cm ahead in the scan; seen late (lag), the robot is level by
    # the time it arrives: that must slow it, never count as a wall to stop at
    ctx, m, d, ai, r = build_imu()
    wall_ahead(ctx, r, wall_s=50.0, lag=1, floor_when_nose_down=0.05)
    assert run_until(ctx, m, lambda: any(k in ('stairs_crossed', 'stairs_failed') for k, _ in ctx.events), n=300)
    whys = [e.get('why', '') for k, e in ctx.events if k == 'mode']
    assert not any('wall' in w for w in whys), whys
    assert any(k == 'stairs_crossed' for k, _ in ctx.events) and r.true_x - r.x0 >= 1.4


def test_relocation_after_the_stairs_waits_for_slam_to_match_a_scan():
    # bag fullrun_25_11_183742: the scan turn was skipped and the tile was checked 0.17 s after the rebase, before
    # slam_toolbox had seen a single scan from after it: it read the rebase back
    ctx, m, d, ai, r = build_imu({'modes.scanning.relocate_wait_s': 3.0, 'modes.scanning.turn': False})
    assert run_until(ctx, m, lambda: m.current.name == 'scanning' and any(k == 'stairs_crossed' for k, _ in ctx.events),
                     n=300), modes_seen(m)
    assert not any(k == 'stairs_relocated' for k, _ in ctx.events)
    ctx.t += 1.0
    m.tick()
    cx, cy = ctx.tiles.grid.center((3, 0))
    ctx.pose_ = (cx, cy, ctx.pose_[2])                                 # SLAM's correction lands a while after the rebase
    assert run_until(ctx, m, lambda: any(k == 'stairs_relocated' for k, _ in ctx.events), n=10), modes_seen(m)
    assert dict(ctx.events)['stairs_relocated']['actual'] == '3,0'


def test_imu_crossing_holds_the_heading_on_the_imu_yaw():
    ctx, m, d, ai, r = build_imu()
    assert run_until(ctx, m, lambda: m.current.name == 'stairs' and m.current.phase == 'cross', n=200)
    r.yaw += 0.2                                                       # the robot swung left in AI mode
    ctx.t += 1.0
    m.tick()
    assert d.cmds[-1][1] < 0                                           # turns back right; the frozen pose saw nothing


def test_odom_is_rebased_to_the_tile_beyond_after_the_climb():
    ctx, m, d, ai, r = build_imu()
    assert run_until(ctx, m, lambda: any(k == 'stairs_crossed' for k, _ in ctx.events), n=300), modes_seen(m)
    assert len(ctx.rebases) == 1
    x, y, yaw = ctx.rebases[0]
    assert (x, y) == tuple(ctx.tiles.grid.center((2, 0))) and abs(yaw) < 1e-6
    assert any(k == 'odom_rebase' for k, _ in ctx.events)


def test_odom_is_rebased_to_a_tile_beyond_the_staircase_along_the_stairs_not_the_grid():
    a = math.radians(10)                                               # the steps go up 10 deg off the grid's +x
    ctx, m, d, ai, r = build_imu({'modes.stairs.rebase_yaw': 'stairs'}, ascent=(math.cos(a), math.sin(a)))
    assert run_until(ctx, m, lambda: any(k == 'stairs_crossed' for k, _ in ctx.events), n=300), modes_seen(m)
    x, y, yaw = ctx.rebases[0]
    sx, sy = ctx.tiles.grid.center((1, 0))                             # staircase tile centre, one tile up the steps
    assert abs(x - (sx + math.cos(a))) < 1e-9 and abs(y - (sy + math.sin(a))) < 1e-9 and abs(yaw - a) < 1e-9
    ref = ctx.tiles.stair_refs[-1]
    assert ref['used'] and ref['grid_tile'] == (2, 0)
    assert ctx.tiles.as_dict()['stair_refs'][-1]['yaw_deg'] == 10.0
    ev = dict(ctx.events)['odom_rebase']
    assert ev['tile'] == '2,0' and ev['reference'] == 'stairs'


def test_rebase_default_stays_on_the_line_walked_with_the_squared_up_heading():
    a = math.radians(10)
    ctx, m, d, ai, r = build_imu(ascent=(math.cos(a), math.sin(a)))
    assert run_until(ctx, m, lambda: any(k == 'stairs_crossed' for k, _ in ctx.events), n=300), modes_seen(m)
    x, y, yaw = ctx.rebases[0]
    ty = ctx.tiles.grid.center((0, 0))[1]
    assert abs(yaw) < 0.05 and abs(y - ty) < 0.05          # grid edge normal (+x) and the line from the tile centre, not snapped 10 deg off


def test_scan_after_the_stairs_corrects_the_tile_from_slam():
    ctx, m, d, ai, r = build_imu()
    assert run_until(ctx, m, lambda: m.current.name == 'scanning' and any(k == 'stairs_crossed' for k, _ in ctx.events),
                     n=300), modes_seen(m)
    assumed = tuple(m.modes['scanning'].after_stairs)
    assert assumed == (2, 0), assumed
    ctx.pose_ = (ctx.tiles.grid.center((3, 0))[0], ctx.tiles.grid.center((3, 0))[1], ctx.pose_[2])   # SLAM: one tile further
    assert run_until(ctx, m, lambda: any(k == 'stairs_relocated' for k, _ in ctx.events), n=300), modes_seen(m)
    ev = dict(ctx.events)['stairs_relocated']
    assert ev['assumed'] != ev['actual']
    assert (3, 0) in ctx.tiles.trail and assumed not in ctx.tiles.trail


def test_beyond_reference_follows_the_stairs():
    a = math.radians(-20)
    stairs = sc.stair_tiles(GRID, np.vstack([stair_pts((1, 0), (math.cos(a), math.sin(a))),
                                            stair_pts((2, 0), (math.cos(a), math.sin(a)))]))
    ref = sc.beyond_reference(GRID, [(1, 0), (2, 0)], stairs)
    mx, my = 0.5 * (np.add(GRID.center((1, 0)), GRID.center((2, 0))))
    assert math.dist(ref['centre_xy'], (mx + 1.5 * math.cos(a), my + 1.5 * math.sin(a))) < 1e-9
    assert abs(ref['yaw'] - a) < 1e-9
    sw, se, ne, nw = ref['corners']
    assert abs(math.dist(sw, se) - GRID.size) < 1e-9 and abs(math.dist(se, ne) - GRID.size) < 1e-9
    assert abs(math.atan2(se[1] - sw[1], se[0] - sw[0]) - a) < 1e-9   # turned to the stairs
    flat = sc.beyond_reference(GRID, [(1, 0)], {}, fallback_dir=(0.0, 2.0))   # no detector direction: the fallback
    assert abs(flat['yaw'] - math.pi / 2) < 1e-9


def test_no_rebase_when_the_climb_failed():
    ctx, m, d, ai, r = build_imu(stuck_at=0.0)                         # never reaches the stairs: no tilt
    assert run_until(ctx, m, lambda: any(k == 'stairs_failed' for k, _ in ctx.events), n=300)
    assert ctx.rebases == [tuple(ctx.pose_)]                           # only the release of the bridge's hold, in place
    assert not any(k == 'odom_rebase' for k, _ in ctx.events)


# ------------------------------------------------------------------------------------------ opening scan (opening_stairs)
def build_opening(flat=None, **kw):
    """The stair detector only runs for the opening scan (no use_stairs): a staircase beside the start tile is climbed
    straight after the scan."""
    kw.setdefault('yaw', 0.0)                                           # grid square with the corridor
    cfg = {'modes.exploring.stairs': False, 'modes.scanning.opening_stairs': True}
    cfg.update(flat or {})
    ctx, m, d, ai = build(cfg, **kw)
    ctx.detector = []
    ctx.stair_detection = ctx.detector.append
    return ctx, m, d, ai


def test_staircase_seen_in_the_opening_scan_goes_straight_to_the_stairs_mode():
    ctx, m, d, ai = build_opening()
    assert run_until(ctx, m, lambda: m.current.name == 'stairs', n=100), modes_seen(m)
    assert modes_seen(m)[-2:] == ['scanning', 'stairs']               # no exploring decision in between
    assert ctx.detector == [True, False]                               # on for the opening scan, off after it
    assert run_until(ctx, m, lambda: len(ai) == 2, n=200)
    assert [on for on, _, _ in ai] == [True, False]


def test_no_stairs_mode_after_the_opening_scan_when_it_is_off():
    ctx, m, d, ai = build_opening({'modes.scanning.opening_stairs': False})
    run_until(ctx, m, lambda: False, n=40)
    assert 'stairs' not in modes_seen(m) and not ai and ctx.detector == []


def test_opening_scan_staircase_seen_from_its_side_is_not_climbed():
    ctx, m, d, ai = build_opening(ascent=(0.0, 1.0))
    run_until(ctx, m, lambda: False, n=40)
    assert 'stairs' not in modes_seen(m) and not ai and ctx.detector == [True, False]


# ------------------------------------------------------------------------------------------ new map after the stairs
def build_new_map(result='done', **flat):
    cfg = {'modes.stairs.new_map': True}
    cfg.update(flat)
    ctx, m, d, ai = build(cfg)
    ctx.new_map_calls, ctx.new_map_polls = [], 0

    def start_new_map(out_dir=None, timeout_s=60.0):
        ctx.new_map_calls.append((out_dir, timeout_s, tuple(ctx.tiles.stairs_crossed)))
    ctx.start_new_map = start_new_map

    def new_map_state():
        ctx.new_map_polls += 1
        if ctx.new_map_polls < 3:
            return 'working'
        if result == 'done' and ctx.tiles.ready:
            ctx.tiles.reset()                       # what the mission manager does once slam_toolbox runs an empty map
        return result
    ctx.new_map_state = new_map_state
    return ctx, m, d, ai


def test_new_map_after_the_stairs_resets_the_grid_and_scans_from_the_tile_beyond():
    ctx, m, d, ai = build_new_map()
    assert run_until(ctx, m, lambda: ctx.new_map_polls >= 3, n=300), modes_seen(m)
    assert len(ctx.new_map_calls) == 1 and ctx.new_map_calls[0][0] is None and ctx.new_map_calls[0][2] == ((1, 0),)
    assert not ctx.tiles.ready and ctx.tiles.stairs_crossed == set() and ctx.tiles.trail == []
    m.tick()
    assert m.current.name == 'scanning' and m.current.opening and m.current.after_stairs is None   # opening scan: a new grid


def test_failed_new_map_keeps_the_old_map_and_scans_as_before():
    ctx, m, d, ai = build_new_map('failed')
    assert run_until(ctx, m, lambda: ctx.new_map_polls >= 3, n=300), modes_seen(m)
    m.tick()
    assert m.current.name == 'scanning' and m.current.after_stairs == (2, 0)
    assert ctx.tiles.ready and (1, 0) in ctx.tiles.stairs_crossed


def test_new_map_is_off_by_default():
    ctx, m, d, ai = build()
    ctx.start_new_map = lambda *a, **k: (_ for _ in ()).throw(AssertionError('new map asked for'))
    assert run_until(ctx, m, lambda: ctx.complete is not None, n=300), modes_seen(m)


def test_tile_tracker_reset_forgets_the_map_but_keeps_the_settings():
    from lite3_explorer.tile_tracker import TileTracker
    t = TileTracker(size_m=1.0)
    t.align_on_entry, t.skip_scene, t.stair_min_points = False, True, 7
    t.grid, t.trail, t.stairs_crossed, t.start_wall = object(), [(0, 0)], {(1, 0)}, ((0, 0), (1, 0))
    t.set_map(np.zeros((4, 4)), None)
    v = t.version
    t.reset()
    assert not t.ready and t.map is None and t.trail == [] and t.stairs_crossed == set() and t.start_wall is None
    assert t.align_on_entry is False and t.skip_scene is True and t.stair_min_points == 7 and t.version > v


def test_new_map_keeps_the_old_tile_numbering_and_the_staircase_wall():
    from lite3_explorer import tiles as TT
    from lite3_explorer.tile_tracker import TileTracker
    t = TileTracker()
    old = TT.TileGrid(0.0, 0.0, 0.0, 1.0)                    # old frame: robot crossed stairs on tiles 0,1 + 0,2 into 0,3
    a, b = (0.5, 3.5, math.pi / 2), (0.2, -0.1, 0.0)         # robot pose in the old frame / in the new map's frame
    d = b[2] - a[2]
    c, s = math.cos(d), math.sin(d)
    ox, oy = old.origin_x - a[0], old.origin_y - a[1]
    t.carry = {'grid': TT.TileGrid(b[0] + c * ox - s * oy, b[1] + s * ox + c * oy, TT.wrap(old.yaw + d), 1.0),
               'stairs_crossed': {(0, 1), (0, 2)}}
    start = t.establish([b[0], b[1], 0.0], 'test')
    assert tuple(start) == (0, 3) and t.stairs_crossed == {(0, 1), (0, 2)}
    assert t.init_info['how'] == 'inherited from the previous map'
    # the stairs lie behind the robot, i.e. on the start tile's back edge: that edge is the virtual wall
    assert t.carry is None and t.start_wall_side in 'NESW'
    (x0, y0), (x1, y1) = t.start_wall
    assert abs((x0 + x1) / 2 - (b[0] - 0.5)) < 1e-6 and abs((y0 + y1) / 2 - b[1]) < 1e-6   # 0.5 m behind the robot (heading +x)


# ------------------------------------------------------------------------------- one staircase per run: the tile is locked
def test_the_first_staircase_tile_is_locked_for_the_rest_of_the_run():
    ctx, m = make({}, length_m=4.0)
    tiles = ctx.tiles
    run_until(ctx, m, lambda: tiles.ready, n=20)
    tiles.scanned_open_tiles = lambda: set()
    assert tiles.set_stair_points(stair_pts((2, 0)))                         # found: tile 2,0
    assert set(tiles.stairs) == {(2, 0)}
    # more cells, now mostly in another tile: still 2,0 (only one staircase in the maze)
    more = np.vstack([stair_pts((2, 0)), stair_pts((3, 0)), stair_pts((3, 0))])
    assert not tiles.set_stair_points(more) and set(tiles.stairs) == {(2, 0)}
    # the stair cells vanish, or a later scan makes the tile look like open floor: still the staircase
    tiles.scanned_open_tiles = lambda: {(2, 0)}
    assert not tiles.set_stair_points(np.zeros((0, 4))) and set(tiles.stairs) == {(2, 0)}
    tiles.update(force=True)
    assert tiles.tiles[(2, 0)]['status'] == 'staircase'


def test_several_tiles_at_first_sight_lock_the_one_with_the_most_stair_cells():
    ctx, m = make({}, length_m=4.0)
    tiles = ctx.tiles
    run_until(ctx, m, lambda: tiles.ready, n=20)
    tiles.scanned_open_tiles = lambda: set()
    assert tiles.set_stair_points(np.vstack([stair_pts((2, 0))[:5], stair_pts((3, 0))]))
    assert set(tiles.stairs) == {(3, 0)}


def test_no_new_staircase_once_the_staircase_was_climbed():
    ctx, m = make({}, length_m=4.0)
    tiles = ctx.tiles
    run_until(ctx, m, lambda: tiles.ready, n=20)
    tiles.scanned_open_tiles = lambda: set()
    tiles.stairs_crossed = {(9, 9)}                                          # climbed (e.g. carried over to a new map)
    assert not tiles.set_stair_points(stair_pts((2, 0))) and not tiles.stairs


# ------------------------------- the staircase closes one side along i (relative to the robot's original heading)
# An open 4 m x 4 m room (x, y in 0..4). The tile grid is anchored at the robot's start tile, so tiles are named through
# tile_of(world x, y) here.
def _room(start_xy, heading, stair_xy, ascent):
    from test_mission import corridor
    tiles = TileTracker()
    tiles.align_on_entry = False
    tiles.set_map(*corridor(4.0, 4.0))
    tiles.establish([start_xy[0], start_xy[1], heading])
    tiles.scanned_open_tiles = lambda: set()
    tiles.stairs = {tuple(tiles.tile_of(*stair_xy)): {'points': 12, 'ascent': ascent}}
    tiles.update(force=True)
    return tiles


def _room_tiles(tiles):
    return {tuple(tiles.tile_of(x + 0.5, y + 0.5)) for x in range(4) for y in range(4)}


def _column(tiles, x):
    """Room tiles whose centre has this world x (+0.5)."""
    return {tuple(tiles.tile_of(x + 0.5, y + 0.5)) for y in range(4)}


def _closed(tiles):
    """Closed tiles inside the room (the wall / unknown tiles around it close with their column too)."""
    return {t for t, d in tiles.tiles.items() if d['status'] == 'closed'} & _room_tiles(tiles)


def test_before_the_climb_tiles_beyond_the_staircase_are_closed():
    tiles = _room((0.5, 1.5), 0.0, (2.5, 1.5), (1.0, 0.0))                   # facing +x, staircase in column x=2
    assert tiles.forward == (0, 1)
    assert _closed(tiles) == _column(tiles, 3)                                # beyond it
    beyond = {T.tile_label(t) for t in _column(tiles, 3)}
    assert not beyond & {t['tile'] for t in tiles.targets([0.5, 1.5])}
    stair = tiles.staircase_tile
    walls = tiles.virtual_walls(start_wall=False)
    assert all(tiles.grid.edge((stair[0], j), 'E') in walls for _, j in _column(tiles, 2))   # Nav2 kept out of it


def test_after_the_climb_tiles_behind_the_staircase_are_closed_and_there_is_no_way_back():
    tiles = _room((0.5, 1.5), 0.0, (2.5, 1.5), (1.0, 0.0))
    stair = tiles.staircase_tile
    tiles.track([1.5, 1.5, 0.0])                                              # walked to the tile in front of the stairs ...
    front = tuple(tiles.current)
    tiles.mark_split(front, [T.tile_label(tiles.tile_of(1.5, 0.5)), T.tile_label(tiles.tile_of(1.5, 2.5))])
    tiles.track([3.5, 1.5, 0.0])                                              # ... climbed over them onto the tile beyond
    tiles.mark_stairs_crossed([stair])
    assert _closed(tiles) == _column(tiles, 0) | _column(tiles, 1)            # behind it
    assert not _closed(tiles) & (_column(tiles, 2) | _column(tiles, 3))
    assert tiles.known_route(tiles.current, tiles.tile_of(0.5, 1.5)) is None  # no backtracking through the stairs
    assert tiles.last_split() is None                                         # the split behind is dropped
    rec = tiles.recovery_targets([3.5, 1.5], min_distance_m=0.0, limit=50)
    assert rec and not {r['tile'] for r in rec} & {T.tile_label(t) for t in _closed(tiles)}
    walls = tiles.virtual_walls(start_wall=False)
    assert all(tiles.grid.edge((stair[0], j), 'W') in walls for _, j in _column(tiles, 2))


def test_facing_minus_x_the_sides_flip():
    tiles = _room((3.5, 1.5), math.pi, (1.5, 1.5), (-1.0, 0.0))              # facing -x, staircase in column x=1
    assert tiles.forward == (0, -1)
    assert _closed(tiles) == _column(tiles, 0)                                # beyond = further along -x
    tiles.mark_stairs_crossed([tiles.staircase_tile])
    assert _closed(tiles) == _column(tiles, 2) | _column(tiles, 3)            # behind = the start side


def _row(tiles, y):
    """Room tiles whose centre has this world y (+0.5)."""
    return {tuple(tiles.tile_of(x + 0.5, y + 0.5)) for x in range(4)}


def test_facing_along_j_the_tiles_beyond_are_closed_along_j():
    tiles = _room((1.5, 0.5), math.pi / 2, (1.5, 2.5), (0.0, 1.0))           # facing +y, staircase in row y=2
    assert tiles.forward == (1, 1)
    assert _closed(tiles) == _row(tiles, 3)                                   # beyond = further along +j
    stair = tiles.staircase_tile
    walls = tiles.virtual_walls(start_wall=False)
    assert all(tiles.grid.edge((i, stair[1]), 'N') in walls for i, _ in _row(tiles, 2))
    tiles.mark_stairs_crossed([stair])
    assert _closed(tiles) == _row(tiles, 0) | _row(tiles, 1)                  # behind = the start side
    walls = tiles.virtual_walls(start_wall=False)
    assert all(tiles.grid.edge((i, stair[1]), 'S') in walls for i, _ in _row(tiles, 2))


def test_facing_minus_j():
    tiles = _room((1.5, 3.5), -math.pi / 2, (1.5, 1.5), (0.0, -1.0))         # facing -y, staircase in row y=1
    assert tiles.forward == (1, -1)
    assert _closed(tiles) == _row(tiles, 0)


def test_forward_axis_picks_the_closest_grid_axis():
    from lite3_explorer.tile_tracker import forward_axis
    assert forward_axis(0.3) == (0, 1)
    assert forward_axis(math.pi - 0.3) == (0, -1)
    assert forward_axis(math.pi / 2 + 0.6) == (1, 1)
    assert forward_axis(-math.pi / 2 - 0.6) == (1, -1)


def test_the_initial_tile_is_closed_once_the_robot_leaves_it():
    tiles = _room((0.5, 1.5), 0.0, (2.5, 1.5), (1.0, 0.0))
    start = tuple(tiles.start_tile)
    assert not tiles.is_closed(start) and tiles.tiles[start]['status'] == 'visited'
    walls = tiles.virtual_walls(start_wall=False)
    assert tiles.grid.edge(start, 'E') not in walls                           # the robot is still on it
    tiles.track([1.5, 1.5, 0.0])
    tiles.update(force=True)
    assert tiles.is_closed(start) and tiles.tiles[start]['status'] == 'closed'
    walls = tiles.virtual_walls(start_wall=False)
    assert all(tiles.grid.edge(start, side) in walls for side in 'NESW')
    assert T.tile_label(start) not in {r['tile'] for r in tiles.recovery_targets([1.5, 1.5], min_distance_m=0.0)}
    assert tiles.known_route(tiles.current, start) is None


def test_nothing_is_closed_before_a_staircase_is_found():
    tiles = _room((0.5, 1.5), 0.0, (2.5, 1.5), (1.0, 0.0))
    tiles.stairs = {}
    tiles.update(force=True)
    assert not _closed(tiles)


def test_stairs_wait_until_this_side_is_explored():
    tiles = _room((0.5, 1.5), 0.0, (2.5, 1.5), (1.0, 0.0))
    assert tiles.unexplored() and not tiles.stairs_ready()
    for t, d in tiles.tiles.items():
        if d['status'] == 'to_visit':
            tiles.trail.append(t)
    tiles.update(force=True)
    assert tiles.stairs_ready()


def test_the_robot_explores_behind_it_before_climbing_the_stairs_beside_the_start():
    # corridor x 0..4; the robot starts at x=1.5 facing +x with the staircase right ahead at x=2.5: the tile behind it
    # (x=0.5) must be explored first, then it comes back in front of the stairs and climbs
    ctx, m, d, ai = build(yaw=0.0)
    ctx.pose_[:] = [1.5, 0.5, 0.0]
    ctx.tiles.stair_points = np.zeros((0, 4))
    ctx.tiles.set_stair_points(stair_pts((2, 0)))
    assert run_until(ctx, m, lambda: m.current.name == 'stairs', n=400), modes_seen(m)
    tiles = ctx.tiles
    assert tuple(tiles.tile_of(0.5, 0.5)) in tiles.trail                     # explored behind it first
    assert tuple(tiles.current) == tuple(tiles.tile_of(1.5, 0.5))             # climbs from in front of the stairs
    assert any('not yet' in str(msg) for _, msg in ctx.logs)


def _stuck_beside_an_unreachable_room(when):
    """2 x 4 hall (tiles 0..3, 0..1), staircase on 2,0 going up +x, and a separate mapped room (tiles 0,3 and 1,3) that
    touches no visited tile: it stays 'to_visit' but is never a candidate (beyond max_adjacency), so it would hold the
    stairs back (stairs_ready) for ever -- bag fullrun_25_11_232027 looped recovery / retry like this."""
    cfg = {'modes.exploring.stairs': True, 'modes.exploring.scan_after_reach': False,
           'modes.stairs.ai_settle_s': 1.0, 'modes.stairs.stall_s': 3.0, 'modes.exploring.retry_period_s': 3.0,
           'modes.exploring.stairs_when_stuck': when}
    ctx, m = make(cfg, world=rooms((0.0, 0.0, 4.0, 2.0), (0.0, 3.0, 2.0, 4.0)))
    ctx.pose_[:] = [0.5, 0.5, 0.0]
    Drive(ctx, np.zeros((0, 4)))
    ctx.ai_mode = lambda on, code: True
    ctx.rebase_odom = lambda x, y, yaw: None
    ctx.tiles.set_stair_points(stair_pts((2, 0)))
    return ctx, m


def test_nothing_reachable_left_on_this_side_goes_for_the_stairs():
    ctx, m = _stuck_beside_an_unreachable_room(True)
    assert run_until(ctx, m, lambda: m.current.name == 'stairs', n=600), modes_seen(m)
    forced = [d for k, d in ctx.events if k == 'stairs_forced']
    assert len(forced) == 1 and forced[0]['stairs'] == '2,0' and {'0,3', '1,3'} <= set(forced[0]['given_up'])
    assert any(k == 'retry' for k, _ in ctx.events)                          # one retry round first
    assert tuple(ctx.tiles.current) == (1, 0)                                # climbs from in front of the stairs


def test_forced_stairs_backtrack_to_the_tile_in_front_without_recovery():
    ctx, m = _stuck_beside_an_unreachable_room(True)
    assert run_until(ctx, m, lambda: any(k == 'stairs_forced' for k, _ in ctx.events), n=600), modes_seen(m)
    ctx.no_path |= {'1,0', '0,1', '1,1', '2,1'}                              # Nav2 finds no path to anything now
    ctx.pose_[:] = [2.5, 1.5, 0.0]                                           # away from the tile in front of the stairs
    ctx.tiles.track(ctx.pose_)
    n = len(m.transitions)
    assert run_until(ctx, m, lambda: m.current.name == 'stairs', n=300), modes_seen(m)
    after = [t[2] for t in m.transitions[n:]]
    assert 'recovery' not in after and after[-2:] == ['backtracking', 'stairs'], after
    assert any(k == 'stairs_backtrack' and d['split'] == '1,0' for k, d in ctx.events)
    assert any(k == 'backtracked' and d.get('stairs') for k, d in ctx.events)
    assert not any(s['tile'] == (1, 0) for s in ctx.tiles.splits)            # the stairs split is dropped
    assert tuple(ctx.tiles.current) == (1, 0)


def test_without_stairs_when_stuck_the_unreachable_room_holds_the_stairs_back():
    ctx, m = _stuck_beside_an_unreachable_room(False)
    assert not run_until(ctx, m, lambda: m.current.name == 'stairs', n=300)
    assert not any(k == 'stairs_forced' for k, _ in ctx.events)


def test_tile_beyond_the_stairs_is_predicted_only_after_squaring_up():
    ctx, m, d, ai = build()
    seen = []

    def watch():
        if m.current.name == 'stairs':
            seen.append((m.current.phase, len(ctx.tiles.stair_refs)))
        return bool(ai)
    assert run_until(ctx, m, watch, n=300), modes_seen(m)
    assert seen and all(n == 0 for phase, n in seen if phase in ('center', 'align'))   # nothing predicted before
    assert len(ctx.tiles.stair_refs) == 1                                             # one prediction, then AI on
    ev = [e for k, e in ctx.events if k == 'stairs_reference']
    assert len(ev) == 1 and ev[0]['pose'] is not None and abs(ev[0]['pose'][2]) <= math.radians(5) + 1e-6
    assert 'reference' not in next(e for k, e in ctx.events if k == 'stairs')


def test_stuck_on_the_way_to_forced_stairs_backtracks_again_instead_of_recovering():
    ctx, m = _stuck_beside_an_unreachable_room(True)
    assert run_until(ctx, m, lambda: any(k == 'stairs_forced' for k, _ in ctx.events), n=600), modes_seen(m)
    ctx.pose_[:] = [2.5, 1.5, 0.0]
    ctx.tiles.track(ctx.pose_)
    ctx.stuck_tiles = {'1,0'}                                                # never gets to the tile in front of them
    n = len(m.transitions)

    def tries():
        return sum(1 for k, _ in ctx.events if k == 'stairs_backtrack')
    assert run_until(ctx, m, lambda: tries() >= 2, n=300), modes_seen(m)
    assert any(k == 'stuck' for k, _ in ctx.events)
    ctx.stuck_tiles = set()
    assert run_until(ctx, m, lambda: m.current.name == 'stairs', n=300), modes_seen(m)
    assert 'recovery' not in [t[2] for t in m.transitions[n:]], modes_seen(m)
