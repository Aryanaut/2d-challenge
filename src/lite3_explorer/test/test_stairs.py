"""Staircases: detection (stairs_core, no ROS), staircase tiles, and the stairs mode climbing over one in AI mode."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lite3_explorer import stairs_core as sc  # noqa: E402
from lite3_explorer.tiles import TileGrid  # noqa: E402
from test_curtain import Drive  # noqa: E402
from test_mission import make, modes_seen  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), 'data', 'stairs_depth_frames.npz')
AI_ON, AI_OFF = 0x21010528, 0x2101052B


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


def test_a_staircase_deeper_than_a_tile_is_crossed_whole():
    stairs = sc.stair_tiles(GRID, np.vstack([stair_pts((1, 0)), stair_pts((2, 0))]))
    c = sc.stair_crossing(GRID, (0, 0), stairs)
    assert c['stairs'] == [(1, 0), (2, 0)] and c['beyond'] == (3, 0) and abs(c['through_m'] - 2.5) < 1e-9


# ------------------------------------------------------------------------------------------ the mode
def build(flat=None, ascent=(1.0, 0.0), yaw=1.0):
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
    assert seen.count('stairs') == 1 and seen[seen.index('stairs') + 1] == 'exploring', seen
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


# ------------------------------------------------------------------------------------------ climber: true
class FakeClimber:
    """A stair_climber node: after enable, walks through `script` (state, ticks), one state per tick-count."""
    def __init__(self, ctx, script, cross_to_x=2.5):
        self.ctx, self.script, self.cross_to_x = ctx, list(script), cross_to_x
        self.enables, self.flags, self.state, self.left = [], [], None, 0
        self.drives_while_on = 0
        ctx.stair_climber_enable = self.enable
        ctx.ai_flag = self.flags.append
        ctx.stair_climber_state = self.get
        self.on = False

    def enable(self, on):
        self.enables.append(on)
        self.on = on
        if on:
            self.queue, self.state, self.left = list(self.script), None, 0
        else:
            self.state = 'IDLE' if self.state is not None else None

    def get(self):
        if self.on:
            if self.left <= 0 and self.queue:
                self.state, self.left = self.queue.pop(0)
                if self.state == 'DONE':
                    self.ctx.pose_[0] = self.cross_to_x            # it walked over the stairs
            self.left -= 1
        return (self.state, 0.05) if self.state is not None else (None, float('inf'))


CROSS = [('SEARCH', 2), ('ALIGN', 3), ('WALK_TO_AI_ZONE', 1), ('ENTER_AI', 2), ('CROSS_AI', 5), ('EXIT_AI', 2),
         ('SWITCH_TO_FLAT_GAIT', 2), ('DONE', 100)]


def build_climber(script, flat=None):
    cfg = {'modes.stairs.climber': True}
    cfg.update(flat or {})
    ctx, m, d, ai = build(cfg)
    return ctx, m, d, ai, FakeClimber(ctx, script)


def test_climber_does_the_climb_and_the_staircase_is_walled_off():
    ctx, m, d, ai, cl = build_climber(CROSS)
    assert run_until(ctx, m, lambda: any(k == 'stairs_crossed' for k, _ in ctx.events), n=200), modes_seen(m)
    assert ai == []                                      # the mission sent no AI codes: stair_climber did
    assert cl.enables == [True, False] and cl.flags == [True, False]
    in_climber = [i for i, (mode, ph) in enumerate(ctx.phases) if mode == 'stairs' and ph == 'climber']
    assert in_climber
    tiles = ctx.tiles
    assert (1, 0) in tiles.stairs_crossed and (2, 0) in tiles.trail
    assert modes_seen(m)[-1] == 'exploring'


def test_climber_sends_no_cmd_vel_while_it_climbs():
    ctx, m, d, ai, cl = build_climber(CROSS)
    assert run_until(ctx, m, lambda: m.current.name == 'stairs' and m.current.phase == 'climber', n=100)
    k = len(d.cmds)
    for _ in range(8):
        ctx.t += 1.0
        m.tick()
    assert all(c == 'stop' for c in d.cmds[k:]), d.cmds[k:]


def test_climber_abort_in_ai_mode_turns_ai_off_and_walls_it_off():
    ctx, m, d, ai, cl = build_climber([('SEARCH', 1), ('ALIGN', 2), ('ENTER_AI', 2), ('CROSS_AI', 2), ('ABORT', 100)])
    assert run_until(ctx, m, lambda: any(k == 'stairs_failed' for k, _ in ctx.events), n=200)
    failed = next(dd for k, dd in ctx.events if k == 'stairs_failed')
    assert failed['ai_was_on'] and failed['back_to'] == 'recovery'
    assert [(on, code) for on, code, _ in ai] == [(False, AI_OFF)]
    assert cl.enables == [True, False] and (1, 0) in ctx.tiles.stairs_crossed


def test_climber_that_never_sees_the_stairs_gives_up_without_climbing():
    ctx, m, d, ai, cl = build_climber([('SEARCH', 1000)], {'modes.stairs.climber_search_timeout_s': 6.0})
    assert run_until(ctx, m, lambda: any(k == 'stairs_failed' for k, _ in ctx.events), n=200)
    failed = next(dd for k, dd in ctx.events if k == 'stairs_failed')
    assert not failed['ai_was_on'] and failed['back_to'] == 'exploring' and 'did not see' in failed['why']
    assert ai == [] and (1, 0) not in ctx.tiles.stairs_crossed and (1, 0) in m.stairs_tried


def test_climber_not_running_gives_up():
    ctx, m, d, ai, cl = build_climber([])
    assert run_until(ctx, m, lambda: any(k == 'stairs_failed' for k, _ in ctx.events), n=200)
    failed = next(dd for k, dd in ctx.events if k == 'stairs_failed')
    assert 'no state' in failed['why'] and not failed['ai_was_on'] and ai == []


def test_leaving_mid_climber_disables_it_and_turns_ai_off():
    ctx, m, d, ai, cl = build_climber([('ENTER_AI', 1), ('CROSS_AI', 1000)])
    assert run_until(ctx, m, lambda: cl.state == 'CROSS_AI', n=100)
    ok, _ = m.request('idle', 'operator command')
    assert ok and cl.enables[-1] is False and cl.flags[-1] is False
    assert [(on, code) for on, code, _ in ai] == [(False, AI_OFF)]
