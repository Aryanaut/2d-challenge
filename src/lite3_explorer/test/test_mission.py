"""Mission manager logic in a simulated corridor (no ROS): modes, stuck recovery and end conditions."""
import math

import numpy as np
import pytest

from lite3_explorer import tiles as T
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
            if label in self.ctx.no_path:
                out.append(None)
            elif getattr(self.ctx, 'grid_paths', False):
                out.append(self._bfs(tuple(self.ctx.pose_[:2]), tuple(g)))
            else:
                out.append([tuple(self.ctx.pose_[:2]), tuple(g)])
        return out

    def _bfs(self, a, b):
        """A path through the free cells of the true map (cell centres), like Nav2's: it follows the corridors, so
        tiles_crossed sees the tiles it runs through. None when there is none."""
        from collections import deque
        grid, info = self.ctx.truth, self.ctx.info
        start, goal = info.world_to_cell(*a), info.world_to_cell(*b)
        prev, q = {start: None}, deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dr, c[1] + dc)
                if n not in prev and 0 <= n[0] < info.height and 0 <= n[1] < info.width and grid[n] == 0:
                    prev[n] = c
                    q.append(n)
        if goal not in prev:
            return None
        cells, c = [], goal
        while c is not None:
            cells.append(c)
            c = prev[c]
        return [(info.origin_x + (col + 0.5) * info.resolution, info.origin_y + (row + 0.5) * info.resolution)
                for row, col in reversed(cells)]

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
            dx, dy = getattr(self.ctx, 'arrive_off', {}).get(self.name, (0.0, 0.0))   # e.g. Nav2's goal tolerance
            self.ctx.pose_ = [self.goal[0] + dx, self.goal[1] + dy, 0.0]
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
        self.tiles.align_on_entry = False                 # these tests assert exact tile centres (alignment: test_tiles.py)
        self.tiles.skip_scene = bool(cfg['modes']['exploring'].get('scenes', False))   # as mission_manager does
        if reveal_m is None:
            self.tiles.set_map(self.truth, self.info)
        else:
            self.known = np.full_like(self.truth, -1)
            self.reveal()
        self.scanner, self.paths = FakeScanner(self), FakePaths(self)
        self.events, self.logs, self.drives = [], [], []
        self.phases = []
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

    def animation_markers(self):                 # a scenescan reports its three stops once it is under way
        if getattr(self, 'played', [])[-1:] == ['scenescan'] and getattr(self, '_anim_left', 2) < 2 \
                and not getattr(self, '_views_sent', False):
            self._views_sent = True
            return ['left', 'centre', 'right']
        return []

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
        self.phases.append((mode, phase))

    # scene mode: the robot turns in place (drive) and a fake AI answers describe_scene
    def drive(self, vx, wz, vy=0.0):
        self.pose_[2] += wz * 0.2                # one tick = 0.2 s of turning in the fake
        self.drives.append(('cmd', wz))

    def stop(self):
        pass

    def on_complete(self, reason):
        self.complete = reason

    def describe_scene(self, label, pose, view=None):
        self.asked = getattr(self, 'asked', []) + [(label, list(pose))]
        self.views = getattr(self, 'views', []) + [view]
        call = FakeCall(ok=label not in getattr(self, 'ai_fails', set()))
        if getattr(self, 'ai_slow', False):          # the answer only comes when the test says so
            call.done = False
            self.slow_calls = getattr(self, 'slow_calls', []) + [call]
        return call

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
    assert ctx.scanner.turns[:2] == [80.0, -80.0]             # opening sweep first (front only), then full turns
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
    # recovery drives to a tile centre with the 'recovery_mppi' profile, never the tile it got stuck on
    rec = [tile for name, tile in ctx.drives if name == 'recovery_mppi']
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


def test_recovery_targets_are_explored_tile_centres_nearest_first():
    ctx, _ = make()
    ctx.tiles.establish([1.5, 0.5, 0.0])
    ctx.tiles.initial_tile = None              # the start tile stays open here (closing it: test_stairs)
    label = lambda x: '%d,%d' % ctx.tiles.tile_of(x, 0.5)  # noqa: E731
    # only the tile the robot stands on is explored: the seen-but-unvisited neighbours are never setpoints
    assert [t['tile'] for t in ctx.tiles.recovery_targets([1.4, 0.5], min_distance_m=0.0)] == [label(1.5)]
    assert ctx.tiles.recovery_targets([1.4, 0.5], min_distance_m=0.25) == []
    for x in (0.5, 2.5, 3.5):                                            # walk the corridor
        ctx.tiles.track([x, 0.5, 0.0])
    ctx.tiles.update(force=True)
    near = ctx.tiles.recovery_targets([1.4, 0.5], min_distance_m=0.0)
    assert [t['tile'] for t in near][:3] == [label(1.5), label(0.5), label(2.5)]
    first = ctx.tiles.recovery_targets([1.4, 0.5], min_distance_m=0.25)[0]
    assert first['tile'] == label(0.5)                                   # the robot's own centre is skipped
    assert math.dist(first['goal_xy'], first['centre_xy']) < 0.1          # the centre itself in a clear corridor
    assert ctx.tiles.recovery_targets([1.4, 0.5], exclude=[label(0.5)])[0]['tile'] == label(2.5)
    # adjacent_only: the robot's own explored tile is kept first ...
    own = ctx.tiles.recovery_targets([1.4, 0.5], exclude=[label(0.5), label(2.5)], adjacent_only=True)
    assert [t['tile'] for t in own] == [label(1.5)]
    # ... and with no explored tile there or beside it left -> the nearest explored tile further away
    far = ctx.tiles.recovery_targets([1.4, 0.5], exclude=[label(0.5), label(1.5), label(2.5)], adjacent_only=True)
    assert [t['tile'] for t in far] == [label(3.5)]


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
    assert {a[0] for a in ctx.asked} == {'2,0'}
    assert ctx.recorded == [('2,0', True)]
    assert (2, 0) not in ctx.tiles.trail and (2, 0) in ctx.tiles.described      # explored, not entered
    assert ctx.tiles.tiles[(2, 0)]['status'] == 'visited'
    assert all(lab != '2,0' for name, lab in ctx.drives if name != 'cmd')       # never driven to
    assert 'scene' in modes_seen(m) and ctx.complete


def test_scene_tile_takes_three_views_one_per_angle():
    ctx, m = make(world=scene_world())
    run(ctx, m, 300)
    assert ctx.played.count('scenescan') == 1
    assert ctx.views == ['left', 'centre', 'right'] and [a[0] for a in ctx.asked] == ['2,0'] * 3
    assert ctx.recorded == [('2,0', True)] and (2, 0) in ctx.tiles.described


def test_scene_tile_the_ai_calls_traversable_is_an_ordinary_tile_again():
    ctx, m = make(world=scene_world())
    run(ctx, m, 300)
    assert (2, 0) in ctx.tiles.described
    ctx.tiles.mark_traversable((2, 0))
    assert not ctx.tiles.is_scene((2, 0)) and (2, 0) not in ctx.tiles.described
    assert ctx.tiles.tiles[(2, 0)]['status'] in ('to_visit', 'visited') and (2, 0) not in ctx.tiles.scene_tiles()


def test_scene_tile_is_marked_explored_even_when_the_ai_fails():
    ctx, m = make(world=scene_world())
    ctx.ai_fails = {'2,0'}
    run(ctx, m, 300)
    assert len(ctx.asked) == 4 and (2, 0) in ctx.tiles.described and ctx.complete   # 3 views, then 1 single try


def test_scene_described_in_the_background_lets_the_robot_explore_on():
    ctx, m = make(flat={'modes.scene.background': True}, world=scene_world())
    ctx.ai_slow = True
    run(ctx, m, 300)
    assert ctx.views == ['left', 'centre', 'right']
    assert ctx.complete and (2, 0) in ctx.tiles.described          # explored on without waiting for the AI
    assert not getattr(ctx, 'recorded', []) and '2,0' in m.scene_calls
    assert [mode for mode in modes_seen(m)[modes_seen(m).index('scene') + 1:]][0] == 'exploring'
    for call in ctx.slow_calls:
        call.done = True
    m.tick()
    assert ctx.recorded == [('2,0', True)] and not m.scene_calls


def test_scene_background_failure_is_recorded_without_a_retry():
    ctx, m = make(flat={'modes.scene.background': True}, world=scene_world())
    ctx.ai_slow, ctx.ai_fails = True, {'2,0'}
    run(ctx, m, 300)
    for call in ctx.slow_calls:
        call.done = True
    m.tick()
    assert len(ctx.asked) == 3 and ctx.recorded == [('2,0', False)] and (2, 0) in ctx.tiles.described


def test_scene_waits_for_the_ai_when_not_in_the_background():
    ctx, m = make(flat={'modes.scene.background': False}, world=scene_world())
    ctx.ai_slow = True
    run(ctx, m, 60, until_complete=False)
    assert m.current.name == 'scene' and m.current.phase == 'describe' and not m.scene_calls


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


def test_scene_tiles_are_detected_anywhere_but_looked_at_only_from_a_neighbour():
    ctx, m = make(world=scene_world())
    ctx.nav_up = False
    run(ctx, m, steps=30, until_complete=False)               # grid placed, robot on tile 0,0; the scene tile 2,0 is two tiles away
    tiles = ctx.tiles
    assert tiles.tiles[(2, 0)]['scene'] is True and tiles.current == (0, 0)
    assert tiles.scene_tiles() == [(2, 0)] and tiles.is_scene((2, 0))         # detected from afar
    assert tiles.skip_scene and not tiles._drivable('2,0')                    # so never a target
    assert '2,0' not in tiles.unreachable                                     # and not eliminated (no clear goal point)
    assert tiles.scene_neighbours((0, 0)) == []                               # not beside 0,0: no scene mode here
    assert 'scene' not in modes_seen(m)
    assert tiles.scene_neighbours((1, 0)) == [(2, 0)]                         # from 1,0 the scene mode takes it
    assert tiles.scene_viewpoints((2, 0)) == [(1, 0)]


def test_no_scene_mode_across_a_wall_between_the_current_tile_and_the_scene_tile():
    grid, info = scene_world()
    r0, c = info.world_to_cell(2.12, 0.0)          # a wall across the whole edge, 12 cm into the scene tile (grid a bit off)
    r1, _ = info.world_to_cell(2.12, 1.0)
    grid[r0:r1, c:c + 1] = 100
    ctx, m = make(world=(grid, info))
    ctx.nav_up = False
    run(ctx, m, steps=30, until_complete=False)
    tiles = ctx.tiles
    assert tiles.is_scene((2, 0))
    assert tiles.wall_between((1, 0), (2, 0))
    assert tiles.scene_neighbours((1, 0)) == [] and tiles.scene_viewpoints((2, 0)) == []
    tiles._visit((1, 0), list(tiles.grid.center((1, 0))) + [0.0], 'test')
    ctx.pose_[:] = list(tiles.grid.center((1, 0))) + [0.0]
    m.request('scene', tile='2,0')                 # asked anyway (e.g. after a recovery): never looked at through the wall
    run(ctx, m, steps=20, until_complete=False)
    assert not getattr(ctx, 'asked', None) and '2,0' not in m.scenes_done and (2, 0) not in tiles.described


def test_clutter_against_the_edge_of_a_scene_tile_is_no_wall():
    ctx, m = make(world=scene_world())             # the scene's clutter covers half the shared edge
    ctx.nav_up = False
    run(ctx, m, steps=30, until_complete=False)
    assert not ctx.tiles.wall_between((1, 0), (2, 0))


# ------------------------------------------------------------------------------------------------ backtracking
def tee_world():
    """Corridor tiles 0,0 .. 2,0 with a branch going north from tile 1,0 (tiles 1,1 and 1,2): tile 1,0 is a split."""
    margin = 1.0
    info = GridInfo(int((3.0 + 2 * margin) / RES), int((3.0 + 2 * margin) / RES), RES, -margin, -margin)
    grid = np.full((info.height, info.width), -1, dtype=np.int16)
    rects = [((0.0, 0.0), (3.0, 1.0)), ((1.0, 1.0), (2.0, 3.0))]
    for (x0, y0), (x1, y1) in rects:
        r0, c0 = info.world_to_cell(x0, y0)
        r1, c1 = info.world_to_cell(x1, y1)
        grid[r0 - 1:r1 + 1, c0 - 1:c1 + 1] = 100
    for (x0, y0), (x1, y1) in rects:
        r0, c0 = info.world_to_cell(x0, y0)
        r1, c1 = info.world_to_cell(x1, y1)
        grid[r0:r1, c0:c1] = 0
    return grid, info


def test_split_tile_is_marked_and_backtracking_takes_the_untravelled_way():
    # neighbours_fallback off: at the end of a branch no exploration target is left -> backtracking (not retry_wait)
    ctx, m = make({'explorations.tiles.neighbours_fallback': False}, world=tee_world())
    run(ctx, m, steps=400)
    splits = [d for k, d in ctx.events if k == 'split']
    assert splits and splits[0]['tile'] == '1,0' and splits[0]['options'] == ['1,1', '2,0']
    seen = modes_seen(m)
    assert 'backtracking' in seen and 'recovery' not in seen
    bt = [d for k, d in ctx.events if k == 'backtracked']
    assert bt and bt[0]['split'] == '1,0'
    # the robot went back to the split tile, then on to the option it had not travelled to
    i = seen.index('backtracking')
    assert seen[i + 1] == 'scanning' and seen[i + 2] == 'exploring'
    assert ctx.complete and ctx.tiles.unexplored() == []
    trail = [T.tile_label(t) for t in ctx.tiles.trail]
    first_branch = trail[trail.index('1,0') + 1]
    other = '1,1' if first_branch == '2,0' else '2,0'
    back = trail.index('1,0', trail.index(first_branch))       # returned to the split tile ...
    assert trail[back + 1] == other                            # ... and took the untravelled way


def test_path_only_through_other_tiles_backtracks_instead_of_recovering():
    # neighbours_fallback on (the default): at the end of a branch the split's other option IS offered, and Nav2 has a
    # path to it, but through the split tile (reject_via_other_tiles). That is not a missing path: backtrack to the
    # split (fullrun_25_11_210402: recovered on the spot 3 times, then gave the split's last option up as unreachable).
    ctx, m = make({}, world=tee_world())
    ctx.grid_paths = True                      # paths follow the corridors (through the split tile)
    run(ctx, m, steps=400)
    assert any(k == 'via_other_tiles' for k, _ in ctx.events)
    seen = modes_seen(m)
    assert 'backtracking' in seen and 'recovery' not in seen
    assert not any(k == 'no_path' for k, _ in ctx.events)
    assert not ctx.tiles.unreachable
    assert ctx.complete and ctx.tiles.unexplored() == []


def test_no_path_with_a_split_left_backtracks_instead_of_recovering():
    # a genuine no-path (after the retries) with a split behind: back to the split, not recovery
    ctx, m = make({'explorations.tiles.neighbours_fallback': False, 'modes.exploring.no_path_retries': 0},
                  world=tee_world())
    for _ in range(200):                       # until the robot has left the split tile for one of its options
        run(ctx, m, steps=1, until_complete=False)
        trail = [T.tile_label(t) for t in ctx.tiles.trail]
        if '1,0' in trail and trail[-1] in ('1,1', '2,0'):
            break
    ahead = '1,2' if trail[-1] == '1,1' else None
    if ahead is None:
        pytest.skip('took the dead-end branch first: nothing ahead to have no path to')
    ctx.no_path = {ahead}
    run(ctx, m, steps=60, until_complete=False)
    seen = modes_seen(m)
    assert 'backtracking' in seen
    assert seen.index('backtracking') < (seen.index('recovery') if 'recovery' in seen else len(seen))


def test_no_backtracking_when_turned_off():
    ctx, m = make({'modes.exploring.backtracking': False, 'explorations.tiles.neighbours_fallback': False},
                  world=tee_world())
    run(ctx, m, steps=60, until_complete=False)
    assert 'backtracking' not in modes_seen(m)
    assert not [d for k, d in ctx.events if k == 'split']
    assert any('retrying for Nav2 paths' in msg for _, msg in ctx.logs)      # the old behaviour: wait and retry


def test_unreachable_split_option_is_retried_by_backtracking():
    ctx, m = make({'explorations.tiles.neighbours_fallback': False}, world=tee_world())
    for _ in range(200):                       # until the robot has left the split tile for one of its options
        run(ctx, m, steps=1, until_complete=False)
        trail = [T.tile_label(t) for t in ctx.tiles.trail]
        if '1,0' in trail and trail[-1] in ('1,1', '2,0'):
            break
    other = '1,1' if trail[-1] == '2,0' else '2,0'
    ctx.tiles.mark_unreachable(other, 'test: given up on')
    run(ctx, m, steps=400)
    assert 'backtracking' in modes_seen(m)
    assert any(k == 'backtracked' and d['tile'] == other for k, d in ctx.events)
    assert ctx.complete and ctx.tiles.unexplored() == []


def test_backtracking_drives_through_tile_centres_with_three_waypoints_per_hop():
    # the fake turns 0.2 s worth per 1 s tick: give the in-place turns time to finish
    ctx, m = make({'explorations.tiles.neighbours_fallback': False, 'modes.backtracking.align_timeout_s': 120.0},
                  world=tee_world())
    run(ctx, m, steps=400)
    routes = [d for k, d in ctx.events if k == 'backtracking_route']
    assert routes and all(r['route'] for r in routes)
    g = ctx.tiles.grid
    for r in routes:
        via = [tuple(int(v) for v in lab.split(',')) for lab in r['via']]
        poses = [(p['x'], p['y']) for p in r['route']]
        assert len(poses) == 4 * (len(via) - 1)                   # per hop: 3 waypoints, then the next centre
        for h, (a, b) in enumerate(zip(via, via[1:])):
            (ax, ay), (bx, by) = g.center(a), g.center(b)
            for i in range(4):                                   # 1/4, 1/2, 3/4 and the centre: on the straight line
                x, y = poses[4 * h + i]
                f = (i + 1) / 4
                assert math.hypot(x - (ax + f * (bx - ax)), y - (ay + f * (by - ay))) < 0.01
            hop_yaw = math.degrees(math.atan2(by - ay, bx - ax))
            yaws = [p['yaw_deg'] for p in r['route'][4 * h:4 * h + 4]]
            assert all(abs(y - hop_yaw) < 0.2 for y in yaws[:3])     # the waypoints face the tile the hop goes to
            if h + 2 < len(via):                                   # a tile centre on the way faces the NEXT tile
                (cx, cy) = g.center(via[h + 2])
                assert abs(yaws[3] - math.degrees(math.atan2(cy - by, cx - bx))) < 0.2
    assert any(r['phase'] == 'to_split' for r in routes) and any(r['phase'] == 'to_option' for r in routes)
    phases = [ph for md, ph in ctx.phases if md == 'backtracking']
    assert 'align' in phases                                      # turned in place at a tile centre before a hop
    assert not any('could not turn' in msg for _, msg in ctx.logs)  # ... until it faced the next tile
    assert ctx.complete


# ------------------------------------------------------------------------------------------------ recovery: mppi + omni
def test_recovery_uses_mppi_then_falls_back_to_omni_on_the_same_tile_centre():
    ctx, m = make()
    run(ctx, m, steps=8, until_complete=False)
    calls = []

    class Flaky(FakeTraversal):
        def poll(self):
            if self.name == 'recovery_mppi':
                return TraversalResult(FAILED, 'controller failed')
            return super().poll()

    ctx.traversal = lambda name: (calls.append(name), Flaky(ctx, name, ctx.cfg['traversals'][name]))[1]
    m.request('recovery', 'test')
    run(ctx, m, steps=12, until_complete=False)
    assert calls[:2] == ['recovery_mppi', 'omni']
    fb = [d for k, d in ctx.events if k == 'recovery_fallback']
    assert fb and fb[0]['failed'] == 'recovery_mppi' and fb[0]['traversal'] == 'omni'
    rec = [d for k, d in ctx.events if k == 'recovered']
    assert rec and rec[0]['traversal'] == 'omni' and rec[0]['tile'] == fb[0]['tile']   # same tile centre


def test_omni_traversal_slides_with_x_and_y_and_holds_the_heading():
    from lite3_explorer.mission.traversal import OmniTraversal

    class Robot:
        def __init__(self):
            self.p, self.t, self.cmds = [0.0, 0.0, 0.5], 0.0, []

        def pose(self):
            return list(self.p)

        def now(self):
            return self.t

        def drive(self, vx, wz, vy=0.0):
            self.cmds.append((vx, vy, wz))
            c, s = math.cos(self.p[2]), math.sin(self.p[2])
            self.p[0] += vx * c - vy * s          # 1 s steps, robot frame -> map
            self.p[1] += vx * s + vy * c
            self.p[2] += wz

        def stop(self):
            self.cmds.append('stop')

    r = Robot()
    tr = OmniTraversal(r, 'omni', DEFAULTS['traversals']['omni'])
    assert tr.start([0.0, 1.0], 0.0)                    # straight "left" of the map, robot turned 0.5 rad
    for _ in range(60):
        r.t += 1.0
        if tr.poll().done:
            break
    assert tr.poll().state == SUCCEEDED
    assert math.dist(r.p[:2], (0.0, 1.0)) <= 0.10
    assert abs(r.p[2] - 0.5) < 1e-6                     # heading held: no turning
    moves = [c for c in r.cmds if c != 'stop']
    assert any(abs(vy) > 0.01 for _, vy, _ in moves) and any(abs(vx) > 0.01 for vx, _, _ in moves)
    assert r.cmds[-1] == 'stop'


def test_omni_traversal_stops_at_a_wall_in_the_costmap():
    from lite3_explorer.mission.traversal import OmniTraversal
    grid, info = corridor(4.0)                          # wall cells (100) just outside y = 0..1

    class Robot:
        p, t = [0.5, 0.5, 0.0], 0.0

        def pose(self):
            return list(self.p)

        def now(self):
            return self.t

        def drive(self, vx, wz, vy=0.0):
            self.p[0] += vx * 0.1
            self.p[1] += vy * 0.1

        def stop(self):
            pass

        def costmap_snapshot(self):
            return grid, info

    r = Robot()
    tr = OmniTraversal(r, 'omni', DEFAULTS['traversals']['omni'])
    tr.start([0.5, 2.0], 0.0)                           # through the north wall
    for _ in range(100):
        r.t += 0.1
        if tr.poll().done:
            break
    res = tr.poll()
    assert res.state == FAILED and 'wall' in res.reason
    assert r.p[1] < 1.0                                 # never drove into it


def test_scene_tile_is_looked_at_from_a_visited_neighbour():
    ctx, m = make()                                     # corridor tiles 0,0 .. 3,0
    ctx.nav_up = False                                  # scan and place the grid, but never drive
    run(ctx, m, steps=30, until_complete=False)
    tiles = ctx.tiles
    for t in [(1, 0), (2, 0), (1, 0)]:                   # been to 2,0, now back on 1,0
        tiles._visit(t, list(tiles.grid.center(t)) + [0.0], 'test')
    tiles.update(force=True)
    strat = m.exploration('tiles')
    target = {'tile': '3,0', 'label': '3,0', 'kind': 'tile'}
    assert strat.still_valid(target)
    tiles.tiles[(3, 0)]['scene'] = True                  # 3,0 turned out to be a scene tile after the robot left 2,0
    assert tiles.is_scene((3, 0)) and not tiles._drivable('3,0')
    assert not strat.still_valid(target)                 # a drive towards it would stop
    assert '3,0' not in [t['tile'] for t in tiles.targets(ctx.pose_[:2])]           # never driven into
    assert tiles.scene_neighbours((1, 0)) == []          # not beside 1,0: no scene mode from here
    cands = strat.candidates(list(tiles.grid.center((1, 0))) + [0.0])
    assert [c['tile'] for c in cands if c.get('scene') == '3,0'] == ['2,0']   # go back to the neighbour it shares a side with
    assert '3,0' not in tiles.unreachable
    tiles._visit((2, 0), list(tiles.grid.center((2, 0))) + [0.0], 'test')
    assert tiles.scene_neighbours((2, 0)) == [(3, 0)]    # standing there, the scene mode takes it
    tiles.mark_described((3, 0))
    assert not tiles.pending_scenes() and not [c for c in strat.candidates(ctx.pose_) if c.get('scene')]


def test_scene_mode_reaches_the_tile_centre_before_turning_to_the_scene():
    ctx, m = make(world=scene_world())
    cmds = []

    def drive(vx, wz, vy=0.0):                   # a robot that slides with vx / vy and turns with wz (1 tick = 1 s)
        cmds.append((vx, vy, wz))
        p = ctx.pose_
        c, s = math.cos(p[2]), math.sin(p[2])
        p[0], p[1], p[2] = p[0] + vx * c - vy * s, p[1] + vx * s + vy * c, p[2] + wz
    ctx.drive = drive
    at_align = []
    spy = ctx.on_mode_change

    def phases(mode, phase):
        if mode == 'scene' and phase == 'rest':
            ctx.pose_[:] = [1.32, 0.31, math.pi / 2]           # nudged off the centre of tile 1,0, facing away
        if mode == 'scene' and phase == 'align':
            at_align.append((list(ctx.pose_), len(cmds)))
        return spy(mode, phase)
    ctx.on_mode_change = phases
    run(ctx, m, 300)
    assert ctx.asked and ctx.asked[0][0] == '2,0'
    pose, n = at_align[0]
    assert math.hypot(pose[0] - 1.5, pose[1] - 0.5) <= 0.06 + 1e-6            # centred first ...
    centring, turning = cmds[:n], cmds[n:]
    assert centring and all(abs(vx) + abs(vy) > 0 for vx, vy, _ in centring)  # ... sliding, heading held ...
    assert abs(pose[2] - math.pi / 2) < 0.05
    assert turning and all(vx == 0 and vy == 0 for vx, vy, _ in turning)      # ... then only turned in place
    looked = ctx.asked[0][1]
    assert abs(math.atan2(math.sin(looked[2]), math.cos(looked[2]))) <= math.radians(6) + 1e-6   # facing east, at it


def test_scene_found_while_driving_recovers_to_the_previous_tile_then_looks_at_it():
    ctx, m = make(length_m=4.0)
    ctx.drive_ticks = 8
    for _ in range(300):                                          # until driving to a tile
        ctx.t += 1.0
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if m.current.name == 'exploring' and m.current.phase == 'driving':
            break
    target = m.current.target['tile']
    here = ctx.tiles.current
    n_before = len(m.transitions)
    real = ctx.tiles._scene_flag
    ctx.tiles._scene_flag = lambda t: tuple(t) == tuple(int(v) for v in target.split(',')) or real(t)   # detected on the way
    run(ctx, m, steps=60, until_complete=False)
    seen = [t[2] for t in m.transitions[n_before:]]
    assert seen[:2] == ['recovery', 'scene']
    assert any(k == 'scene_while_driving' for k, _ in ctx.events)
    assert target in m.scenes_done
    assert ctx.tiles.current == here and tuple(int(v) for v in target.split(',')) not in ctx.tiles.trail


def test_recovery_never_drives_into_a_tile_that_turns_into_a_scene_on_the_way():
    # bag reporter3_25_11_180333: recovery picked 1,-1 (not yet a scene), the scan flagged it while the omni fallback
    # was driving there, and the robot ended on it as an ordinary visited tile
    ctx, m = make(length_m=4.0)
    ctx.tiles.skip_scene = True
    run(ctx, m, steps=8, until_complete=False)
    ctx.drive_ticks = 8
    m.request('recovery', 'test')
    for _ in range(20):
        ctx.t += 1.0
        m.tick()
        if m.current.name == 'recovery' and m.current.phase == 'driving':
            break
    assert m.current.name == 'recovery' and m.current.phase == 'driving'
    target = m.current.target['tile']
    tile = tuple(int(v) for v in target.split(','))
    real = ctx.tiles._scene_flag
    ctx.tiles._scene_flag = lambda t: tuple(t) == tile or real(t)      # flagged on the way
    entries = ctx.tiles.trail.count(tile)
    run(ctx, m, steps=30, until_complete=False)
    assert ('target_dropped', {'label': target, 'why': 'scene tile', 'mode': 'recovery'}) in ctx.events
    assert not any(k == 'recovered' and d['tile'] == target for k, d in ctx.events)
    assert ctx.tiles.trail.count(tile) == entries                       # never entered


class BlockedScanner(FakeScanner):
    """Nav2 Spin aborting with 'Collision Ahead': every turn of up to 180 deg stops after 10 % of it; only the long way
    round (> 180 deg) gets through."""

    def start(self, deg, allowance):
        self.turns.append(deg)
        self.ok = abs(deg) > 180.0
        self.ctx.pose_[2] += math.radians(deg if self.ok else 0.1 * deg)

    def poll(self):
        return SUCCEEDED if self.ok else FAILED


class RotatingScanner(FakeScanner):
    """Every turn completes and really rotates the robot; records the heading swept, relative to the start."""

    def start(self, deg, allowance):
        self.turns.append(deg)
        rel0 = math.degrees(self.ctx.pose_[2])
        self.swept.append((rel0, rel0 + deg))
        self.ctx.pose_[2] += math.radians(deg)
        self.ctx.reveal()


def test_opening_spin_sweeps_the_front_never_through_the_wall_behind():
    for setpoints in ([0, 80, -80], [0, 80, -160, -80], [0, -120, 120]):
        ctx, m = make({'modes.scanning.opening_setpoints_deg': setpoints})
        ctx.scanner = RotatingScanner(ctx)
        ctx.scanner.swept = []
        start = ctx.pose_[2]
        for _ in range(60):
            ctx.t += 1.0
            m.tick()
            if m.current is not None and m.current.name == 'exploring':
                break
        assert ctx.scanner.swept, setpoints
        # every opening turn starts and ends within +-180 of the start heading: none swings round through the back
        assert all(-180.0 <= a <= 180.0 and -180.0 <= b <= 180.0 for a, b in ctx.scanner.swept), (setpoints, ctx.scanner.swept)
        assert abs(math.degrees(ctx.pose_[2] - start)) < 3.0, setpoints          # back to the start heading
    ctx, m = make()
    ctx.scanner = RotatingScanner(ctx)
    ctx.scanner.swept = []
    for _ in range(60):
        ctx.t += 1.0
        m.tick()
        if m.current is not None and m.current.name == 'exploring':
            break
    assert max(abs(v) for ab in ctx.scanner.swept for v in ab) <= 90.0           # default: a front sweep only


def test_opening_spin_still_turns_back_to_the_start_heading_when_the_sweep_keeps_failing():
    ctx, m = make()
    ctx.scanner = BlockedScanner(ctx)
    for _ in range(60):                                  # until the opening scan hands over (driving resets the yaw)
        ctx.t += 1.0
        m.tick()
        if m.current is not None and m.current.name == 'exploring':
            break
    yaw = ctx.pose_[2]
    assert abs(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))) < 3.0    # facing the start heading again
    assert abs(ctx.scanner.turns[-1]) > 180.0                                    # short way blocked: the long way round
    assert 'exploring' in modes_seen(m)


def test_omni_traversal_checks_the_body_not_just_the_centre():
    from lite3_explorer.mission.traversal import OmniTraversal
    import numpy as np
    from lite3_explorer.map_grid import GridInfo
    info = GridInfo(width=80, height=80, resolution=0.05, origin_x=-2.0, origin_y=-2.0)
    grid = np.zeros((80, 80), dtype=np.int16)
    r0, c0 = info.world_to_cell(0.0, 0.27)               # a wall 0.27 m beside the path: the centre line misses it,
    grid[r0, c0 - 20:c0 + 20] = 100                      # the 0.37 m wide body (half 0.185 + pad) does not

    class Robot:
        p, t = [0.0, 0.0, 0.0], 0.0

        def pose(self):
            return list(self.p)

        def now(self):
            return self.t

        def drive(self, vx, wz, vy=0.0):
            self.p[0] += vx * 0.1

        def stop(self):
            pass

        def costmap_snapshot(self):
            return grid, info

    grid[:, :] = 0
    grid[info.world_to_cell(0.0, 0.27)[0], c0 - 20:c0 + 20] = 100     # 0.27 m beside the centre line: clear of the body
    r = Robot()
    tr = OmniTraversal(r, 'omni', DEFAULTS['traversals']['omni'])
    tr.start([1.0, 0.0], 0.0)
    r.t += 0.1
    assert tr.poll().state == RUNNING
    grid[info.world_to_cell(0.0, 0.17)[0], c0 - 20:c0 + 20] = 100   # inside the body half-width: blocks
    r2 = Robot()
    tr2 = OmniTraversal(r2, 'omni', DEFAULTS['traversals']['omni'])
    tr2.start([1.0, 0.0], 0.0)
    r2.t += 0.1
    res = tr2.poll()
    assert res.state == FAILED and 'wall' in res.reason


# ------------------------------------------------------------------------------------------- scene views: majority vote
def test_scene_views_vote_by_majority_not_any_single_view():
    from lite3_explorer.mission.modes import _ViewsCall
    vote = _ViewsCall.vote
    # bags 104357 / 175813: one view catching the edge of a scene no longer makes the tile a scene
    assert vote([('traversable', 0.9), ('kitchen', 0.8), ('traversable', 0.9)]) == 'traversable'
    assert vote([('traversable', 0.9), ('traversable', 0.9), ('workshop', 0.99)]) == 'traversable'
    assert vote([('traversable', 0.9)] * 3) == 'traversable'
    # a real scene seen by most views stays one (bag 174513: cleaning_room / traversable / cleaning_room)
    assert vote([('cleaning_room', 0.9), ('traversable', 0.9), ('cleaning_room', 0.8)]) == 'cleaning_room'
    assert vote([('market', 0.8), ('market', 0.9), ('market', 0.95)]) == 'market'
    # one view failed: a tie goes by summed confidence, traversable when that ties too
    assert vote([('kitchen', 0.95), ('traversable', 0.6)]) == 'kitchen'
    assert vote([('kitchen', 0.6), ('traversable', 0.95)]) == 'traversable'
    assert vote([('kitchen', 0.8), ('traversable', 0.8)]) == 'traversable'
    # two scene types named: the commonest, then the more confident
    assert vote([('kitchen', 0.7), ('market', 0.9), ('kitchen', 0.6)]) == 'kitchen'
    assert vote([('kitchen', 0.7), ('market', 0.9)]) == 'market'


class _Call:
    def __init__(self, result):
        self.result, self.done, self.ok = result, True, True


def test_scene_views_result_carries_the_vote():
    from lite3_explorer.mission.modes import _ViewsCall
    views = {'left': _Call({'scene_type': 'traversable', 'confidence': '0.9', 'scene_description': 'traversable: wall'}),
             'centre': _Call({'scene_type': 'kitchen', 'confidence': '0.8', 'scene_description': 'kitchen: oven edge'}),
             'right': _Call({'scene_type': 'traversable', 'confidence': '0.9', 'scene_description': 'traversable: steps'})}
    out = _ViewsCall(views).result
    assert out['scene_type'] == 'traversable'
    assert out['scene_votes'] == {'centre': 'kitchen', 'left': 'traversable', 'right': 'traversable'}


# ------------------------------------------------------------------------------------- ending_tile end condition
def rooms(*rects, bbox=None):
    """Free rectangles (x0, y0, x1, y1) in metres; everything else inside their bounding box (+ one cell) is wall."""
    margin = 1.0
    xs = [r[0] for r in rects] + [r[2] for r in rects]
    ys = [r[1] for r in rects] + [r[3] for r in rects]
    x0, y0 = min(xs) - margin, min(ys) - margin
    info = GridInfo(int((max(xs) - x0 + margin) / RES), int((max(ys) - y0 + margin) / RES), RES, x0, y0)
    grid = np.full((info.height, info.width), -1, dtype=np.int16)
    ra, ca = info.world_to_cell(min(xs), min(ys))
    rb, cb = info.world_to_cell(max(xs), max(ys))
    grid[ra - 1:rb + 1, ca - 1:cb + 1] = 100
    for rx0, ry0, rx1, ry1 in rects:
        r0, c0 = info.world_to_cell(rx0, ry0)
        r1, c1 = info.world_to_cell(rx1, ry1)
        grid[r0:r1, c0:c1] = 0
    return grid, info


ENDING = {'end_condition.conditions': ['ending_tile', 'all_tiles_explored']}


def l_then_hallway():
    """Start tile 0,0 (3 walls), corner 0,1 (2 walls), then a hallway 1,1..4,1 (4 tiles) heading east."""
    return rooms((0.0, 0.0, 1.0, 2.0), (0.0, 1.0, 5.0, 2.0))


def corridor_into_room():
    """Corridor 0,0 -> 1,0 opening into a 3 x 3 room (tiles 2..4, -1..1): 1,0 has 2 walls, 2,0 and 3,0 none."""
    return rooms((0.0, 0.0, 2.0, 1.0), (2.0, -1.0, 5.0, 2.0))


def classified(world, trail):
    grid, info = world
    tg = T.TileGrid(0.0, 0.0, 0.0)
    tiles = T.classify_tiles(grid, info, tg, trail)
    for t, d in tiles.items():
        if d['status'] != 'unknown':
            d['sides'] = T.tile_sides(grid, info, tg, t)
    return tiles


def test_find_ending_tile_before_a_hallway():
    tiles = classified(l_then_hallway(), [(0, 0)])
    assert T.hallways(tiles, 4) == [[(1, 1), (2, 1), (3, 1), (4, 1)]]
    assert T.hallways(tiles, 5) == []
    found = T.find_ending_tile(tiles, [(0, 0)])
    assert found['tile'] == (0, 1) and found['trigger_tile'] == (1, 1) and 'hallway of 4' in found['trigger']
    assert T.find_ending_tile(tiles, [(0, 0)], min_hallway_tiles=5) is None
    assert T.find_ending_tile(tiles, [(0, 0)], triggers=['open_tile']) is None


def test_find_ending_tile_entered_only_needs_the_open_tile_on_the_trail():
    tiles = classified(corridor_into_room(), [(0, 0), (1, 0)])
    assert T.find_ending_tile(tiles, [(0, 0), (1, 0)], triggers=['open_tile'])['tile'] == (1, 0)   # seen from afar
    assert T.find_ending_tile(tiles, [(0, 0), (1, 0)], triggers=['open_tile'], entered_only=True) is None
    tiles = classified(corridor_into_room(), [(0, 0), (1, 0), (2, 0)])
    found = T.find_ending_tile(tiles, [(0, 0), (1, 0), (2, 0)], triggers=['open_tile'], entered_only=True)
    assert found['tile'] == (1, 0) and found['trigger_tile'] == (2, 0)
    found = T.find_ending_tile(tiles, [(0, 0), (1, 0), (2, 0)], triggers=['open_tile'], entered_only=True,
                               preceding=False)
    assert found['tile'] == (2, 0)


OPEN_TILE_ONLY = dict(ENDING, **{'end_condition.ending_tile.triggers': ['open_tile']})


def test_find_ending_tile_next_tile_seen_from_the_current_tile():
    world = corridor_into_room()
    grid, info = world
    tg = T.TileGrid(0.0, 0.0, 0.0)
    tiles = classified(world, [(0, 0), (1, 0)])
    for t, d in tiles.items():
        if d['status'] != 'unknown':
            sh = T.side_shares(grid, info, tg, t)
            d['wall_share'], d['free_share'] = ({s: v[k] for s, v in sh.items()} for k in (0, 1))
    found = T.find_ending_tile(tiles, [(0, 0), (1, 0)], triggers=['next_tile'], solid_wall_share=0.75)
    assert found == {'tile': (1, 0), 'trigger_tile': (2, 0), 'trigger': 'the scan shows next tile 2,0 with no walls'}
    assert T.find_ending_tile(tiles, [(0, 0)], triggers=['next_tile'], solid_wall_share=0.75) is None   # 1,0 has walls
    assert T.find_ending_tile(tiles, [(0, 0), (1, 0), (2, 0)], triggers=['next_tile'],
                              solid_wall_share=0.75) is None        # already outside: next_tile does not apply
    tiles[(2, 0)]['free_share'] = {s: 0.0 for s in T.SIDES}         # edges not mapped yet
    tiles[(2, 0)]['wall_share'] = {s: 0.2 for s in T.SIDES}
    assert T.find_ending_tile(tiles, [(0, 0), (1, 0)], triggers=['next_tile'], solid_wall_share=0.75) is None


def test_ending_tile_when_the_scan_shows_the_next_tile_open_never_leaves_the_maze():
    ctx, m = make(ENDING, world=corridor_into_room(), reveal_m=1.7)
    run(ctx, m)
    assert ctx.complete and 'ending tile 1,0' in ctx.complete and 'next tile 2,0 with no walls' in ctx.complete
    assert (2, 0) not in ctx.tiles.trail and not [d for d in ctx.drives if d[1] != '1,0']     # never drove outside
    assert ctx.tiles.trail[-1] == (1, 0) and ('direct', '1,0') in ctx.drives                  # centred on 1,0
    assert math.hypot(ctx.pose_[0] - 1.5, ctx.pose_[1] - 0.5) < 0.05
    assert any(k == 'ending_tile' and d['tile'] == '1,0' for k, d in ctx.events)


def test_ending_centres_with_omni_when_nav2_stops_within_its_tolerance():
    ctx, m = make(ENDING, world=corridor_into_room(), reveal_m=1.7)
    # 0.25 m off: inside reach_radius_m (0.30). (Stopping 0.2 m short of the opening instead puts 2,0's far edge past
    # the 1.7 m lidar: next_tile cannot see it open, the robot steps out and the open_tile safety net brings it back.)
    ctx.arrive_off = {'route': (0.2, 0.15), 'direct': (0.2, 0.15)}
    run(ctx, m)
    assert ctx.complete and 'ending tile 1,0' in ctx.complete and (2, 0) not in ctx.tiles.trail
    assert ctx.drives[-2:] == [('direct', '1,0'), ('omni', '1,0')]
    assert math.hypot(ctx.pose_[0] - 1.5, ctx.pose_[1] - 0.5) < 0.05


def test_ending_tile_after_entering_a_tile_with_no_walls():
    # safety net: the open_tile trigger alone (or the outside not seen from the last maze tile)
    ctx, m = make(OPEN_TILE_ONLY, world=corridor_into_room(), reveal_m=1.7)
    run(ctx, m)
    assert ctx.complete and 'ending tile 1,0' in ctx.complete and 'tile 2,0 has no walls' in ctx.complete
    trail = ctx.tiles.trail
    assert (2, 0) in trail and trail.index((2, 0)) < len(trail) - 1      # entered the open tile, then went back
    assert trail[-1] == (1, 0) and ('direct', '1,0') in ctx.drives
    assert math.hypot(ctx.pose_[0] - 1.5, ctx.pose_[1] - 0.5) < 0.3


def test_ending_tile_not_before_the_open_tile_is_entered():
    # open_tile only, the whole map known from the start: nothing happens until 2,0 is entered
    ctx, m = make(OPEN_TILE_ONLY, world=corridor_into_room())
    run(ctx, m)
    assert 'ending tile 1,0' in ctx.complete
    assert ctx.drives.index(('route', '2,0')) < ctx.drives.index(('direct', '1,0'))


def test_ending_tile_preceding_false_ends_on_the_open_tile():
    ctx, m = make(dict(OPEN_TILE_ONLY, **{'end_condition.ending_tile.preceding': False}), world=corridor_into_room(),
                  reveal_m=1.7)
    run(ctx, m)
    assert ctx.complete and 'ending tile 2,0' in ctx.complete
    assert ctx.tiles.trail[-1] == (2, 0)


def test_ending_tile_hallway_trigger_once_inside_the_hallway():
    ctx, m = make(dict(ENDING, **{'end_condition.ending_tile.triggers': ['hallway']}), world=l_then_hallway(),
                  reveal_m=1.7)
    run(ctx, m)
    assert ctx.complete and 'ending tile 0,1' in ctx.complete and 'hallway of 4' in ctx.complete
    assert math.hypot(ctx.pose_[0] - 0.5, ctx.pose_[1] - 1.5) < 0.3


def test_ending_tile_not_found_in_a_plain_corridor_falls_back_to_all_tiles():
    ctx, m = make(ENDING)
    run(ctx, m)
    assert ctx.complete and ctx.complete.startswith('all_tiles_explored')
    assert 'ending' not in modes_seen(m)


def test_ending_falls_back_then_rescans_when_the_centre_is_not_reached():
    ctx, m = make(ENDING, world=corridor_into_room(), reveal_m=1.7)
    for _ in range(400):                      # fail the ending drive the first time only
        ctx.t += 1.0
        if m.current.name == 'ending' and not any(k == 'ending_failed' for k, _ in ctx.events):
            ctx.fail_tiles = {'1,0'}
        elif any(k == 'ending_failed' for k, _ in ctx.events):
            ctx.fail_tiles = set()
        m.tick()
        ctx.tiles.track(ctx.pose_)
        if ctx.complete:
            break
    assert ('direct', '1,0') in ctx.drives and ('omni', '1,0') in ctx.drives
    assert any(k == 'ending_failed' for k, _ in ctx.events)
    assert ctx.complete and 'ending tile 1,0' in ctx.complete


def _bag_tracker(name):
    """TileTracker on the bag endconditiontest_25_11_201356 map: 'before' (t=50 s, robot in 1,0, the last maze tile) or
    'entered' (t=66 s, robot in 2,0: outside the maze, open floor with furniture partly on its east edge)."""
    import os
    z = np.load(os.path.join(os.path.dirname(__file__), 'data', 'endcondition_outside.npz'))
    w, h, res, ox, oy = z[name + '_info']
    ox_t, oy_t, yaw = z[name + '_tg']
    tr = TileTracker()
    tr.grid = T.TileGrid(float(ox_t), float(oy_t), math.radians(float(yaw)))
    tr.trail = [tuple(int(v) for v in t) for t in z[name + '_trail']]
    tr.set_map(z[name + '_grid'].astype(np.int16), GridInfo(int(w), int(h), float(res), float(ox), float(oy)))
    tr.update()
    return tr


def test_bag_furniture_on_an_edge_is_a_wall():
    tr = _bag_tracker('entered')
    out = tr.tiles[(2, 0)]
    assert out['sides']['E'] == 'wall' and 0.5 < out['wall_share']['E'] < 0.75      # furniture outside the maze
    assert tr.tiles[(1, 0)]['wall_share']['N'] >= 0.95 and tr.tiles[(1, 0)]['wall_share']['S'] >= 0.95
    assert (2, 0) not in T.open_tiles(tr.tiles, 0.4)       # any wall detected on an edge rules the tile out
    assert (2, 0) in T.open_tiles(tr.tiles, 0.75)          # (only a looser solid_wall_share would let it through)


def _bag_snap(tr, pose):
    return MissionSnapshot(tr.counts(), tr.visited_count(), tiles=tr.tiles, trail=list(tr.trail), pose=pose,
                           grid=tr.grid)


def test_bag_no_ending_while_a_wall_is_detected_on_the_next_tiles_edge():
    # t=50 s, robot in 1,0: the scan shows 2,0 open on three sides, but furniture along its east edge (0.65): no ending
    cond = build_end_condition({'conditions': ['ending_tile'], 'ending_tile': DEFAULTS['end_condition']['ending_tile']})
    before = _bag_tracker('before')
    assert not T.looks_open(before.tiles[(2, 0)])
    assert T.looks_open(before.tiles[(2, 0)], solid_wall_share=0.75)    # seen well enough: only the wall rules it out
    assert not [t for t in [(-1, 0), (0, 0), (1, 0)] if T.looks_open(before.tiles[t])]      # the maze tiles
    assert cond.check(_bag_snap(before, [3.058, 4.99]), checkpoint=True) is None and cond.goal() is None
    # KNOWN LIMIT: once the robot stands in 2,0 (outside), 2,0 counts as walled (the furniture) and the scan shows 2,1
    # with no walls at all: the ending tile becomes 2,0, outside the maze
    entered = _bag_tracker('entered')
    cond.check(_bag_snap(entered, list(entered.grid.center((2, 0)))), checkpoint=True)
    assert cond.goal()['tile'] == (2, 0) and 'next tile 2,1 with no walls' in cond.goal()['reason']


def test_bag_looser_wall_share_finds_the_last_maze_tile():
    cfg = dict(DEFAULTS['end_condition']['ending_tile'], solid_wall_share=0.75)
    cond = build_end_condition({'conditions': ['ending_tile'], 'ending_tile': cfg})
    before = _bag_tracker('before')
    assert cond.check(_bag_snap(before, [3.058, 4.99]), checkpoint=True)      # within reach_radius_m ...
    assert cond.goal()['tile'] == (1, 0) and 'next tile 2,0 with no walls' in cond.goal()['reason']
    assert math.dist([3.058, 4.99], before.grid.center((1, 0))) > 0.2       # ... but the ending mode still centres


def test_no_ending_tile_when_clutter_stands_on_the_open_tiles_edge():
    grid, info = corridor_into_room()
    r0, c = info.world_to_cell(3.0, 0.2)                 # a box along 60 % of 2,0's east edge
    r1, _ = info.world_to_cell(3.0, 0.8)
    grid[r0:r1, c:c + 2] = 100
    ctx, m = make(ENDING, world=(grid, info), reveal_m=1.7)
    run(ctx, m)
    # every tile of the small room touches a wall or the box: none is open, the mission ends on all tiles explored
    assert ctx.complete and ctx.complete.startswith('all_tiles_explored') and 'ending' not in modes_seen(m)


def start_beside_a_room():
    """Start tile 0,0 (walls N, S, W) opening straight into a 3 x 3 room (tiles 1..3, -1..1): 2,0 has no walls."""
    return rooms((0.0, 0.0, 1.0, 1.0), (1.0, -1.0, 4.0, 2.0))


def test_find_ending_tile_never_the_initial_tile():
    tiles = classified(corridor_into_room(), [(1, 0)])
    for d in tiles.values():
        if 'sides' in d:
            d['wall_share'] = {s: (1.0 if v == 'wall' else 0.0) for s, v in d['sides'].items()}
            d['free_share'] = {s: (0.0 if v == 'wall' else 1.0) for s, v in d['sides'].items()}
    assert T.find_ending_tile(tiles, [(1, 0)], triggers=['next_tile']) is None             # 1,0 is the initial tile
    assert T.find_ending_tile(tiles, [(0, 0), (1, 0)], triggers=['next_tile'])['tile'] == (1, 0)
    assert T.find_ending_tile(tiles, [(1, 0), (2, 0)], triggers=['open_tile'], entered_only=True) is None
    assert T.find_ending_tile(tiles, [(0, 0), (1, 0), (2, 0)], triggers=['open_tile'], entered_only=True)['tile'] == (1, 0)


def test_no_ending_on_the_initial_tile_even_beside_an_open_tile():
    ctx, m = make(ENDING, world=start_beside_a_room(), reveal_m=1.7)
    run(ctx, m)
    assert ctx.complete and 'ending tile 0,0' not in ctx.complete
    assert not any(k == 'ending_tile' and d['tile'] == '0,0' for k, d in ctx.events)


def test_ending_tile_solid_wall_share_zero_needs_every_side_open():
    grid, info = corridor_into_room()
    r0, c = info.world_to_cell(3.0, 0.2)
    r1, _ = info.world_to_cell(3.0, 0.8)
    grid[r0:r1, c:c + 2] = 100
    ctx, m = make(dict(ENDING, **{'end_condition.ending_tile.solid_wall_share': 0.0}), world=(grid, info), reveal_m=1.7)
    run(ctx, m)
    assert ctx.complete and '2,0 with no walls' not in ctx.complete and '2,0 has no walls' not in ctx.complete


def test_ending_tile_needs_the_ending_mode():
    cfg = merge_flat(DEFAULTS, dict(ENDING, **{'mission.modes': ['idle', 'scanning', 'exploring', 'recovery',
                                                                 'complete']}))
    with pytest.raises(ValueError, match='ending'):
        ModeMachine(FakeCtx(cfg), cfg)
