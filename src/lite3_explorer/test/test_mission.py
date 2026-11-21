"""Mission manager logic in a simulated corridor (no ROS): modes, stuck recovery and end conditions."""
import math

import numpy as np
import pytest

from lite3_explorer.map_grid import GridInfo
from lite3_explorer.mission.config import DEFAULTS, merge_flat
from lite3_explorer.mission.core import ModeMachine
from lite3_explorer.mission.end_conditions import MissionSnapshot, build_end_condition
from lite3_explorer.mission.stuck import StuckDetector
from lite3_explorer.mission.traversal import FAILED, RUNNING, SUCCEEDED, Traversal, TraversalResult
from lite3_explorer.tile_tracker import TileTracker

RES = 0.05


def room(w_m, h_m):
    return corridor(w_m, h_m)


def corridor(length_m=4.0, width_m=1.0):
    """A fully mapped 1 m wide corridor from x=0 to x=length_m, walls one cell thick around it."""
    margin = 1.0
    info = GridInfo(int((length_m + 2 * margin) / RES), int((width_m + 2 * margin) / RES), RES, -margin, -margin)
    grid = np.full((info.height, info.width), -1, dtype=np.int16)
    r0, c0 = info.world_to_cell(0.0, 0.0)
    r1, c1 = info.world_to_cell(length_m, width_m)
    grid[r0 - 1:r1 + 1, c0 - 1:c1 + 1] = 100
    grid[r0:r1, c0:c1] = 0
    return grid, info


class FakeScanner:
    ready = True

    def __init__(self):
        self.turns = []

    def __init__(self, ctx=None):
        self.turns, self.ctx = [], ctx

    def start(self, deg, allowance):
        self.turns.append(deg)
        if self.ctx is not None:
            self.ctx.reveal()

    def poll(self):
        return SUCCEEDED

    def cancel(self):
        pass

    def has_room(self):
        return True


class FakePaths:
    def __init__(self, ctx):
        self.ctx, self.goals = ctx, None

    def start(self, goals):
        self.goals = goals
        return True

    def poll(self):
        out = []
        for g in self.goals:
            label = '%d,%d' % self.ctx.tiles.tile_of(*g)
            out.append(None if label in self.ctx.no_path else [tuple(self.ctx.pose_[:2]), tuple(g)])
        return out

    def cancel(self):
        pass


class FakeTraversal(Traversal):
    uses_route = True

    def available(self):
        return self.ctx.nav_up

    def start(self, goal_xy, yaw, route=None):
        self.goal, self.started, self.cancelled = goal_xy, self.ctx.t, False
        self.ctx.drives.append((self.name, '%d,%d' % self.ctx.tiles.tile_of(*goal_xy)))
        return True

    def poll(self):
        label = '%d,%d' % self.ctx.tiles.tile_of(*self.goal)
        if label in self.ctx.stuck_tiles:
            return TraversalResult(RUNNING)          # never arrives, never moves
        if label in self.ctx.fail_tiles:
            return TraversalResult(FAILED, 'aborted by Nav2')
        if getattr(self.ctx, 'edge_walk', False) and self.ctx.t - self.started == 1:
            g = self.ctx.tiles.grid                  # the robot passes 0.05 m inside the goal tile's west edge
            i, j = g.tile_of(*self.goal)
            x, y = g.to_world(i * g.size + 0.05, (j + 0.5) * g.size)
            self.ctx.pose_ = [x, y, 0.0]
            return TraversalResult(RUNNING)
        if self.ctx.t - self.started >= getattr(self.ctx, 'drive_ticks', 2):
            self.ctx.pose_ = [self.goal[0], self.goal[1], 0.0]
            return TraversalResult(SUCCEEDED)
        return TraversalResult(RUNNING)

    def cancel(self):
        self.cancelled = True


class FakeCtx:
    def __init__(self, cfg, pose=(0.5, 0.5, 0.0), length_m=4.0, world=None, reveal_m=None):
        self.cfg = cfg
        self.t = 0.0
        self.pose_ = list(pose)
        self.truth, self.info = world if world is not None else corridor(length_m)
        self.reveal_m = reveal_m              # None: the whole map is known from the start
        self.tiles = TileTracker(max_failures=cfg['tiles']['max_failures'])
        if reveal_m is None:
            self.tiles.set_map(self.truth, self.info)
        else:
            self.known = np.full_like(self.truth, -1)
            self.reveal()
        self.scanner, self.paths = FakeScanner(self), FakePaths(self)
        self.events, self.logs, self.drives = [], [], []
        self.complete, self.cost, self.cleared = None, 0, 0
        self.stuck_tiles, self.fail_tiles, self.no_path = set(), set(), set()
        self.nav_up = True

    def now(self):
        return self.t

    def reveal(self):
        """Fake lidar: copy the true map within reveal_m of the robot into the known map."""
        if self.reveal_m is None:
            return
        info = self.info
        rows, cols = np.mgrid[0:info.height, 0:info.width]
        xs = info.origin_x + (cols + 0.5) * info.resolution
        ys = info.origin_y + (rows + 0.5) * info.resolution
        near = np.hypot(xs - self.pose_[0], ys - self.pose_[1]) <= self.reveal_m
        self.known[near] = self.truth[near]
        self.tiles.set_map(self.known.copy(), info)

    def log(self, level, msg):
        self.logs.append((level, msg))

    def pose(self):
        return list(self.pose_)

    def local_map_ready(self, pose):
        return True

    def ready_for_start(self):
        return True, []

    def gait_hold(self, kind):
        pass

    # terrain classifier: asked for a tile; tiles in `unseen_tiles` give no answer
    def classify_tile(self, label):
        self.classified = getattr(self, 'classified', []) + [(label, list(self.pose_))]

    def terrain_status(self, label):
        return 'unseen' if label in getattr(self, 'unseen_tiles', set()) else 'seen'

    # robot animations: a fake animation_player that finishes 2 ticks after it starts
    def animation_play(self, name):
        self.played = getattr(self, 'played', []) + [name]
        self._anim_left = 2
        return True

    def animation_state(self):
        return 'running' if getattr(self, '_anim_left', 0) > 0 and not self._tick_anim() else 'done'

    def _tick_anim(self):
        self._anim_left -= 1
        return self._anim_left <= 0

    def animation_cancel(self):
        self._anim_left = 0

    def traversal(self, name):
        if name not in self.cfg['traversals']:
            raise KeyError(name)
        return FakeTraversal(self, name, self.cfg['traversals'][name])

    def costmap_cost_at_robot(self):
        return self.cost

    def clear_costmaps(self):
        self.cleared += 1

    def set_target(self, label, goal_xy, route):
        pass

    def event(self, kind, **data):
        self.events.append((kind, data))

    def on_mode_change(self, mode, phase):
        pass

    # scene mode: the robot turns in place (drive) and a fake AI answers describe_scene
    def drive(self, vx, wz, vy=0.0):
        self.pose_[2] += wz * 0.2                # one tick = 0.2 s of turning in the fake
        self.drives.append(('cmd', wz))

    def stop(self):
        pass

    def on_complete(self, reason):
        self.complete = reason

    def describe_scene(self, label, pose):
        self.asked = getattr(self, 'asked', []) + [(label, list(pose))]
        return FakeCall(ok=label not in getattr(self, 'ai_fails', set()))

    def record_scene(self, label, pose, call):
        self.recorded = getattr(self, 'recorded', []) + [(label, call.ok)]


class FakeCall:
    done = True

    def __init__(self, ok=True):
        self.ok, self.result = ok, {'scene_description': 'a table with a vase'}

    def failure(self):
        return 'no reply'


def run(ctx, machine, steps=400, until_complete=True):
    for _ in range(steps):
        ctx.t += 1.0
        machine.tick()
        ctx.tiles.track(ctx.pose_)
        if until_complete and ctx.complete:
            break


def make(flat=None, **ctx_kw):
    cfg = merge_flat(DEFAULTS, flat or {})
    ctx = FakeCtx(cfg, **ctx_kw)
    m = ModeMachine(ctx, cfg)
    m.start()
    return ctx, m


def modes_seen(m):
    return [t[2] for t in m.transitions]


# ------------------------------------------------------------------------------------------------ tests
def test_explores_every_tile_then_completes():
    ctx, m = make()
    run(ctx, m)
    assert ctx.complete and ctx.complete.startswith('all_tiles_explored')
    assert ctx.tiles.unexplored() == []
    assert ctx.tiles.visited_count() == 4
    seen = modes_seen(m)
    assert seen[:3] == ['idle', 'scanning', 'exploring'] and seen[-1] == 'complete'
    assert 'recovery' not in seen
    assert ctx.scanner.turns[:2] == [-90.0, 180.0]            # opening sweep first, then full turns
    assert 360.0 in ctx.scanner.turns
    assert all(name == 'route' for name, _ in ctx.drives)


def test_stuck_without_motion_goes_to_nearest_tile_centre_then_gives_up_on_the_tile():
    ctx, m = make({'tiles.max_failures': 2, 'stuck.window_s': 5.0, 'stuck.cooldown_s': 0.0})
    ctx.stuck_tiles = {'2,0'}
    run(ctx, m, steps=600)
    seen = modes_seen(m)
    assert 'recovery' in seen
    stuck = [d for k, d in ctx.events if k == 'stuck']
    assert stuck and stuck[0]['target'] == '2,0' and 'moved only' in stuck[0]['why']
    # recovery drives to a tile centre with the 'direct' profile, never the tile it got stuck on
    rec = [tile for name, tile in ctx.drives if name == 'direct']
    assert rec and '2,0' not in rec
    assert ctx.cleared >= 1
    assert '2,0' in ctx.tiles.unreachable
    assert ctx.complete and 'unreachable skipped' in ctx.complete


def test_costmap_trigger():
    ctx, m = make({'stuck.triggers': ['costmap'], 'stuck.costmap_hold_s': 2.0, 'stuck.cooldown_s': 0.0})
    run(ctx, m, steps=6, until_complete=False)
    assert m.current.name == 'exploring'
    ctx.cost = 100
    run(ctx, m, steps=4, until_complete=False)
    assert 'recovery' in modes_seen(m)
    assert any(k == 'stuck' and 'costmap' in d['why'] for k, d in ctx.events)


def test_no_path_to_any_tile_recovers_and_finally_gives_up():
    ctx, m = make({'modes.recovery.give_up_after': 2})
    ctx.no_path = {'1,0', '2,0', '3,0'}
    run(ctx, m, steps=300)
    assert modes_seen(m).count('recovery') >= 2
    assert ctx.tiles.unexplored() == []
    assert ctx.tiles.unreachable                          # given up on after the recoveries
    assert ctx.complete


def test_failed_goal_chooses_another_tile_and_marks_unreachable():
    ctx, m = make({'tiles.max_failures': 1})
    ctx.fail_tiles = {'1,0'}
    run(ctx, m, steps=200)
    assert '1,0' in ctx.tiles.unreachable
    assert ctx.complete


def test_time_limit_end_condition():
    ctx, m = make({'end_condition.conditions': ['all_tiles_explored', 'time_limit'],
                   'end_condition.time_limit.seconds': 5.0})
    ctx.stuck_tiles = {'1,0'}
    run(ctx, m, steps=50)
    assert ctx.complete and ctx.complete.startswith('time_limit')


def test_combine_all_needs_every_condition():
    cond = build_end_condition({'combine': 'all', 'conditions': ['all_tiles_explored', 'max_tiles_visited'],
                                'all_tiles_explored': {'min_visited': 1}, 'max_tiles_visited': {'count': 3}})
    snap = MissionSnapshot({}, visited=2, unexplored=[])
    assert cond.check(snap, checkpoint=True) is None
    snap.visited = 3
    assert cond.check(snap, checkpoint=True)


def test_checkpoint_conditions_only_change_at_checkpoints():
    cond = build_end_condition({'conditions': ['all_tiles_explored']})
    assert cond.check(MissionSnapshot({}, visited=3, unexplored=[]), checkpoint=False) is None
    assert cond.check(MissionSnapshot({}, visited=3, unexplored=[]), checkpoint=True)


def test_unknown_end_condition_and_mode_are_rejected():
    with pytest.raises(ValueError):
        build_end_condition({'conditions': ['nope']})
    cfg = merge_flat(DEFAULTS, {'modes.exploring.scan_mode': 'look_around'})
    with pytest.raises(ValueError, match='look_around'):
        ModeMachine(FakeCtx(cfg), cfg)


def test_custom_mode_instance_and_traversal_profile_from_config():
    ctx, m = make({'mission.modes': ['idle', 'scanning', 'explore_direct', 'recovery', 'complete'],
                   'modes.scanning.next_mode': 'explore_direct',
                   'modes.explore_direct.type': 'exploring',
                   'modes.explore_direct.traversal': 'careful',
                   'modes.explore_direct.tile_traversals': ['3,0:direct'],
                   'traversals.careful.type': 'nav2_pose'})
    run(ctx, m)
    assert ctx.complete
    names = {tile: name for name, tile in ctx.drives}
    assert names['1,0'] == 'careful' and names['3,0'] == 'direct'


def test_pause_and_resume():
    ctx, m = make()
    run(ctx, m, steps=6, until_complete=False)
    assert m.activate(False) == 'idle'
    run(ctx, m, steps=10, until_complete=False)
    assert m.current.name == 'idle' and m.current.phase == 'paused'
    m.activate(True)
    assert m.current.name == 'scanning'
    run(ctx, m)
    assert ctx.complete


def test_stuck_detector_no_motion():
    d = StuckDetector(['no_motion'], window_s=5.0, min_motion_m=0.2)
    for t in range(5):
        assert d.update(float(t), (0.0, 0.0), driving=True) is None
    assert d.update(5.0, (0.05, 0.0), driving=True)
    d.reset()
    for t in range(10):
        assert d.update(float(t), (0.1 * t, 0.0), driving=True) is None     # moving: never stuck
    assert d.update(11.0, (0.0, 0.0), driving=False) is None               # not driving: never stuck


def test_stuck_detector_costmap():
    d = StuckDetector(['costmap'], costmap_cost=99, costmap_hold_s=2.0)
    assert d.update(0.0, None, False, 100) is None
    assert d.update(1.0, None, False, 50) is None          # left the zone: timer restarts
    assert d.update(2.0, None, False, 99) is None
    assert d.update(4.0, None, False, 99)
    with pytest.raises(ValueError):
        StuckDetector(['bogus'])


def test_recovery_targets_are_tile_centres_nearest_first():
    ctx, _ = make()
    ctx.tiles.establish([1.5, 0.5, 0.0])
    label = lambda x: '%d,%d' % ctx.tiles.tile_of(x, 0.5)  # noqa: E731
    near = ctx.tiles.recovery_targets([1.4, 0.5], min_distance_m=0.0)
    assert [t['tile'] for t in near][:3] == [label(1.5), label(0.5), label(2.5)]
    first = ctx.tiles.recovery_targets([1.4, 0.5], min_distance_m=0.25)[0]
    assert first['tile'] == label(0.5)                                   # the robot's own centre is skipped
    assert math.dist(first['goal_xy'], first['centre_xy']) < 0.1          # the centre itself in a clear corridor
    assert ctx.tiles.recovery_targets([1.4, 0.5], exclude=[label(0.5)])[0]['tile'] == label(2.5)


def test_merge_flat_drops_empty_string_lists():
    cfg = merge_flat(DEFAULTS, {'modes.exploring.tile_traversals': [''], 'stuck.window_s': 9.0})
    assert cfg['modes']['exploring']['tile_traversals'] == []
    assert cfg['stuck']['window_s'] == 9.0
    assert DEFAULTS['stuck']['window_s'] == 60.0


def test_waits_for_nav2_instead_of_giving_up_on_tiles():
    ctx, m = make()
    ctx.nav_up = False
    run(ctx, m, steps=60, until_complete=False)
    assert m.current.name == 'exploring' and m.current.phase in ('wait_traversal', 'decide', 'checking_paths')
    assert not ctx.tiles.unreachable and not ctx.complete
    ctx.nav_up = True
    run(ctx, m)
    assert ctx.complete and not ctx.tiles.unreachable



# ------------------------------------------------------------------------- exploration strategies
def targets_chosen(ctx):
    return [d for k, d in ctx.events if k == 'target']


def test_next_tile_shares_a_side_with_an_explored_tile():
    # 3 m x 2 m open room: from tile 0,0 the diagonal 1,1 is never chosen before one of its side neighbours
    ctx, m = make(world=room(3.0, 2.0))
    run(ctx, m)
    assert ctx.complete and ctx.tiles.unexplored() == []
    chosen = targets_chosen(ctx)
    assert len(chosen) == 5 and all(t['adjacency'] in (0, 1, 2) for t in chosen)
    assert chosen[0]['adjacency'] == 0                    # first pick: a neighbour of the current tile


def test_adjacency_levels_rank_the_current_tiles_neighbours_first():
    ctx, m = make(world=room(3.0, 2.0))
    run(ctx, m)
    # with prefer_current every pick is a neighbour of the tile the robot stands on (the room is open)
    assert all(t['adjacency'] == 0 for t in targets_chosen(ctx))


def test_unreachable_tiles_are_eliminated_after_strikes():
    ctx, m = make({'explorations.tiles.eliminate_after': 2, 'explorations.tiles.max_adjacency': 3},
                  world=room(3.0, 2.0))
    ctx.no_path = {'2,1'}
    run(ctx, m)
    assert '2,1' in ctx.tiles.unreachable and 'no Nav2 path' in ctx.tiles.unreachable['2,1']
    assert any(k == 'eliminated' and d['tiles'] == ['2,1'] for k, d in ctx.events)
    assert 'recovery' not in modes_seen(m)                # other tiles stayed reachable: no recovery needed
    assert ctx.complete and 'unreachable skipped' in ctx.complete


def test_all_paths_failing_eliminates_nothing_and_recovers():
    ctx, m = make({'modes.recovery.give_up_after': 0, 'modes.exploring.no_path_retries': 0})
    run(ctx, m, steps=6, until_complete=False)
    ctx.no_path = {'1,0', '2,0', '3,0'}
    ctx.stuck_tiles = {'1,0', '2,0', '3,0'}               # and recovery can't move into them either
    run(ctx, m, steps=40, until_complete=False)
    assert 'recovery' in modes_seen(m)
    assert not any(k == 'eliminated' for k, _ in ctx.events)


def test_tiles_only_finishes_when_identified_tiles_are_done():
    # the fake lidar sees 1.2 m: the next tile is never 70 % mapped, so it is never identified
    ctx, m = make(world=corridor(6.0), reveal_m=1.2)
    run(ctx, m)
    assert ctx.complete and ctx.tiles.visited_count() < 6
    assert all(t['target_kind'] == 'tile' for t in targets_chosen(ctx))


def test_optional_frontier_fallback_opens_up_the_map():
    ctx, m = make({'modes.exploring.exploration': ['tiles', 'frontier'],
                   'end_condition.combine': 'all',
                   'end_condition.conditions': ['all_tiles_explored', 'no_frontiers_left']},
                  world=corridor(6.0), reveal_m=1.2)
    run(ctx, m, steps=800)
    kinds = [t['target_kind'] for t in targets_chosen(ctx)]
    assert 'frontier' in kinds and 'tile' in kinds
    assert ctx.complete and 'no_frontiers_left' in ctx.complete
    assert ctx.tiles.visited_count() == 6


def test_kind_traversal_override_and_bad_exploration_profile():
    ctx, m = make({'modes.exploring.exploration': ['tiles', 'frontier'],
                   'modes.exploring.kind_traversals': ['frontier:direct'],
                   'end_condition.combine': 'all',
                   'end_condition.conditions': ['all_tiles_explored', 'no_frontiers_left']},
                  world=corridor(6.0), reveal_m=1.2)
    run(ctx, m, steps=800)
    by_kind = {t['target_kind']: t['traversal'] for t in targets_chosen(ctx)}
    assert by_kind == {'tile': 'route', 'frontier': 'direct'}
    cfg = merge_flat(DEFAULTS, {'modes.exploring.exploration': ['nope']})
    with pytest.raises(KeyError, match='nope'):
        ModeMachine(FakeCtx(cfg), cfg)


def test_adjacency_fallback_when_the_link_tile_is_eliminated():
    ctx, m = make({'tiles.max_failures': 1})
    ctx.fail_tiles = {'1,0'}                              # tiles 2,0 and 3,0 only touch 1,0
    run(ctx, m)
    assert '1,0' in ctx.tiles.unreachable
    assert ctx.tiles.unexplored() == [] and ctx.complete
    assert any(t['adjacency'] == 3 for t in targets_chosen(ctx))


def test_nav2_gets_three_waypoints_then_the_exact_tile_centre():
    ctx, m = make()
    run(ctx, m)
    chosen = targets_chosen(ctx)
    assert chosen
    for t in chosen:
        route = t['route']
        assert len(route) == 4                                       # 3 intermediate waypoints + the goal
        tile = tuple(int(v) for v in t['tile'].split(','))
        cx, cy = ctx.tiles.grid.center(tile)
        assert math.dist((route[-1]['x'], route[-1]['y']), (cx, cy)) < 1e-3   # the goal IS the tile centre
    ctx, m = make({'traversals.route.waypoints': 5})
    run(ctx, m)
    assert all(len(t['route']) == 6 for t in targets_chosen(ctx))


def test_failed_goal_is_followed_by_a_different_tile():
    ctx, m = make({'tiles.max_failures': 3, 'explorations.tiles.fallback_any': True, 'modes.exploring.scan_after_reach': False}, length_m=4.0)
    ctx.fail_tiles = {'1,0'}                               # the nearest tile always aborts
    run(ctx, m, steps=40, until_complete=False)
    tiles = [t for _, t in ctx.drives]
    assert tiles[0] == '1,0'
    assert tiles[1] != '1,0', tiles                        # re-decided on another tile, not a retry of the failed one


def test_goal_yaw_faces_away_from_the_most_walls():
    import math
    from lite3_explorer.approach import yaw_away_from_walls as f
    deg = lambda y: round(math.degrees(y))
    assert deg(f({'N': 'wall', 'E': 'wall', 'W': 'wall', 'S': 'open'}, 0.0, 0.0)) == -90      # dead end: face S
    assert deg(f({'N': 'wall', 'S': 'open', 'E': 'open', 'W': 'open'}, 0.0, 0.0)) == -90      # wall behind, never ahead
    assert deg(f({}, math.radians(10), math.radians(80))) == 100                               # no walls: nearest axis
    assert deg(f({'E': 'wall', 'W': 'wall'}, 0.0, math.radians(85))) == 90                     # corridor: keep travel axis


def test_direct_path_through_shared_side():
    from lite3_explorer import tiles as T
    tg = T.TileGrid(0.0, 0.0, 0.0, 1.0)
    f = T.direct_path_through_shared_side
    assert f(tg, (0, 0), (1, 0), [(0.5, 0.5), (1.0, 0.5), (1.5, 0.5)])
    assert not f(tg, (0, 0), (1, 0), [(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)])      # detours through tile 0,1 / 1,1
    assert not f(tg, (0, 0), (2, 0), [(0.5, 0.5), (2.5, 0.5)])                              # not a neighbour
    assert not f(tg, (0, 0), (1, 1), [(0.5, 0.5), (1.5, 1.5)])                              # diagonal
    assert f(tg, (1, 1), (1, 0), [(1.5, 1.5), (1.5, 0.5)])                                  # towards the south


def test_next_tile_is_a_neighbour_through_a_direct_path():
    ctx, m = make({'modes.exploring.scan_after_reach': False}, length_m=4.0)
    run(ctx, m, steps=200)
    order = [t for _, t in ctx.drives]
    assert order[:3] == ['1,0', '2,0', '3,0']
    assert ctx.complete


def test_quickscan_on_tile_edge_runs_once_per_tile_and_resumes_exploring():
    ctx, m = make({'modes.exploring.quickscan_on_edge': True})
    ctx.edge_walk = True
    run(ctx, m)
    assert ctx.complete
    assert 'quickscan' in modes_seen(m)
    assert len(ctx.played) == len(m.quickscanned) and set(ctx.played) == {'quickscan'}
    seen = modes_seen(m)
    i = seen.index('quickscan')
    assert seen[i + 1] == 'exploring'


def test_no_quickscan_by_default():
    ctx, m = make()
    run(ctx, m)
    assert 'quickscan' not in modes_seen(m) and not getattr(ctx, 'played', [])


def test_no_path_retries_before_recovery():
    ctx, m = make({'modes.recovery.give_up_after': 0, 'modes.exploring.no_path_retries': 1,
                   'modes.exploring.retry_period_s': 1.0})
    run(ctx, m, steps=6, until_complete=False)
    ctx.no_path = {'1,0', '2,0', '3,0'}
    ctx.stuck_tiles = {'1,0', '2,0', '3,0'}
    run(ctx, m, steps=40, until_complete=False)
    assert any(k == 'retry' for k, _ in ctx.events)       # waited and retried first ...
    assert 'recovery' in modes_seen(m)                     # ... then recovered


def test_target_tile_is_classified_once_the_robot_points_at_it():
    ctx, m = make({'modes.exploring.scan_after_reach': False}, length_m=4.0)
    run(ctx, m, steps=40, until_complete=False)
    asked = [lab for lab, _ in ctx.classified]
    assert asked and asked[0] == ctx.drives[0][1]                # the tile chosen first is the one asked about
    import math
    lab, pose = ctx.classified[0]
    i, j = (int(v) for v in lab.split(','))
    gx, gy = ctx.tiles.grid.center((i, j))
    off = abs(math.atan2(math.sin(math.atan2(gy - pose[1], gx - pose[0]) - pose[2]),
                         math.cos(math.atan2(gy - pose[1], gx - pose[0]) - pose[2])))
    assert off <= math.radians(20.0) + 1e-6


def test_unseen_target_tile_is_dropped_for_another():
    ctx, m = make({'modes.exploring.scan_after_reach': False}, length_m=4.0)
    ctx.unseen_tiles = {'1,0'}                                   # the nearest tile's floor can't be seen
    ctx.drive_ticks = 8
    run(ctx, m, steps=60, until_complete=False)
    assert any(k == 'target_unseen' for k, _ in ctx.events)
    asked = [lab for lab, _ in ctx.classified]
    assert asked.count('1,0') == 1                               # asked once, never in a loop
    assert len(ctx.drives) > 1                                   # and the mission went on to other tiles


def test_only_tiles_sharing_a_side_with_the_current_tile_are_candidates():
    ctx, m = make(length_m=4.0)
    ctx.nav_up = False                                       # scan and place the grid, but never drive
    run(ctx, m, steps=30, until_complete=False)
    assert ctx.tiles.current == (0, 0)
    strat = m.exploration('tiles')
    cands = strat.candidates(ctx.pose())
    assert [c['tile'] for c in cands] == ['1,0']             # 2,0 and 3,0 are not neighbours of 0,0
    assert ctx.tiles.tiles[(1, 0)]['relevant'] is True and ctx.tiles.tiles[(2, 0)]['relevant'] is False
    strat.cfg.update(neighbours_only=False, max_adjacency=3)
    assert len(strat.candidates(ctx.pose())) > 1


def scene_world():
    """A 3 m corridor whose last tile (2,0) is a scene tile: fully scanned, half of it occupied."""
    grid, info = corridor(3.0)
    r0, c0 = info.world_to_cell(2.0, 0.0)
    r1, c1 = info.world_to_cell(3.0, 0.5)
    grid[r0:r1, c0:c1] = 100
    return grid, info


def test_scene_tile_beside_the_current_tile_is_looked_at_never_entered():
    ctx, m = make(world=scene_world())
    run(ctx, m, 300)
    assert ctx.tiles.tiles[(2, 0)]['scene'] is True
    assert [a[0] for a in ctx.asked] == ['2,0']
    assert ctx.recorded == [('2,0', True)]
    assert (2, 0) not in ctx.tiles.trail and (2, 0) in ctx.tiles.described      # explored, not entered
    assert ctx.tiles.tiles[(2, 0)]['status'] == 'visited'
    assert all(lab != '2,0' for name, lab in ctx.drives if name != 'cmd')       # never driven to
    assert 'scene' in modes_seen(m) and ctx.complete


def test_scene_tile_is_marked_explored_even_when_the_ai_fails():
    ctx, m = make(world=scene_world())
    ctx.ai_fails = {'2,0'}
    run(ctx, m, 300)
    assert len(ctx.asked) == 2 and (2, 0) in ctx.tiles.described and ctx.complete


def test_no_scene_mode_when_scenes_are_off():
    ctx, m = make(flat={'modes.exploring.scenes': False}, world=scene_world())
    run(ctx, m, 300)
    assert not getattr(ctx, 'asked', [])


def test_a_flat_answer_never_replaces_an_uneven_one_within_one_look():
    t = TileTracker()
    t.begin_terrain_look('-1,1')
    assert t.set_terrain('-1,1', {'kind': 'uneven', 'source': 'terrain_type'}) is True
    assert t.set_terrain('-1,1', {'kind': 'flat', 'n_cells': 9}) is False       # 0.4 s later, from 9 cells: ignored
    assert t.terrain['-1,1']['kind'] == 'uneven'
    assert t.set_terrain('-1,1', {'kind': 'slats'}) is True                     # a more specific non-flat answer is taken
    t.begin_terrain_look('-1,1')                                                # asked again: a fresh look
    assert t.set_terrain('-1,1', {'kind': 'flat'}) is True and t.terrain['-1,1']['kind'] == 'flat'


def test_scene_turn_does_not_flip_direction_when_the_scene_tile_is_behind():
    ctx, m = make(world=scene_world(), pose=(1.5, 0.5, math.pi - 0.05))      # tile 2,0 (east) is straight behind
    ctx.nav_up = False
    run(ctx, m, steps=20, until_complete=False)                              # the opening scan places the tile grid
    ctx.pose_[:] = [1.5, 0.5, math.pi - 0.05]
    orig = ctx.drive
    state = {'n': 0}

    def kicked(vx, wz, vy=0.0):                                              # the robot coasts the wrong way once
        orig(vx, wz, vy)
        state['n'] += 1
        if state['n'] == 2:
            ctx.pose_[2] += 0.4
    ctx.drive = kicked
    m.request('scene', tile='2,0')
    run(ctx, m, steps=80, until_complete=False)
    turns = [w for name, w in ctx.drives if name == 'cmd' and abs(w) > 1e-6]
    assert turns and len({w > 0 for w in turns}) == 1                         # one direction all the way round


def test_scene_tiles_only_count_beside_the_current_tile_and_are_checked_once_per_arrival():
    ctx, m = make(world=scene_world())
    ctx.nav_up = False
    run(ctx, m, steps=30, until_complete=False)               # grid placed, robot on tile 0,0; the scene tile 2,0 is two tiles away
    assert ctx.tiles.tiles[(2, 0)]['scene'] is True and ctx.tiles.current == (0, 0)
    assert ctx.tiles.scene_tiles() == [] and not ctx.tiles.is_scene((2, 0))     # not beside the current tile: ordinary tile
    assert ctx.tiles._drivable('2,0') or not ctx.tiles.skip_scene
    ctx.tiles.trail.append((1, 0))                            # now it is next to 1,0
    assert ctx.tiles.scene_tiles() == [(2, 0)] and ctx.tiles.is_scene((2, 0))
    ctx.tiles.trail.append((0, 0))                            # and away again
    assert ctx.tiles.scene_tiles() == []
