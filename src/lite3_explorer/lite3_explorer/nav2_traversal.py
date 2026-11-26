"""Nav2 traversal types (ROS): nav2_pose (NavigateToPose) and nav2_route (NavigateThroughPoses through a route).

Registered on import into mission.traversal.TRAVERSAL_TYPES. Profile options (traversals.<name>.*):
  type            nav2_pose | nav2_route
  action          action name (default navigate_to_pose / navigate_through_poses, or follow_waypoints with backend)
  backend         nav2_route only: through_poses (default: one continuous path through all the waypoints, the robot
                  only stops at the last one) | follow_waypoints (one NavigateToPose per waypoint: stops at each)
  behavior_tree   nav2_pose only: a behaviour tree XML for this profile ('' = Nav2's default); a bare file name is
                  taken from this package's share/behavior_trees (e.g. recovery_mppi.xml)
  waypoints, skip_start_m, drop_backward_waypoints
                  nav2_route only: how the exploring mode builds the route (approach.default_route)
"""
import math
import os

from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints, NavigateThroughPoses, NavigateToPose

from lite3_explorer.mission.traversal import (FAILED, RUNNING, SUCCEEDED, Traversal, TraversalResult,
                                              register_traversal)


def resolve_behavior_tree(path):
    """A bare file name (e.g. recovery_mppi.xml) is looked up in this package's share/behavior_trees."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('lite3_explorer'), 'behavior_trees', path)


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class _Nav2Traversal(Traversal):
    action_type = None
    default_action = ''

    def __init__(self, ctx, name, cfg):
        super().__init__(ctx, name, cfg)
        self._configure(cfg)
        self.node = ctx
        self.action_name = cfg.get('action') or self.default_action
        self.client = ctx.action_client(self.action_type, self.action_name)
        self.seq = 0
        self.handle = None
        self.result = TraversalResult(FAILED, 'not started')

    def _configure(self, cfg):
        """Hook for subclasses to pick action_type / default_action from the profile before the client is made."""

    def available(self):
        return self.client.server_is_ready()

    def _pose(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = self.node.global_frame
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        o = ps.pose.orientation
        o.x, o.y, o.z, o.w = yaw_to_quaternion(yaw)
        return ps

    def _send(self, goal):
        if not self.client.wait_for_server(timeout_sec=0.5):
            self.result = TraversalResult(FAILED, f'Nav2 {self.action_name} server not available')
            return False
        self.seq += 1
        seq = self.seq
        self.handle = None
        self.result = TraversalResult(RUNNING)
        self.client.send_goal_async(goal).add_done_callback(lambda f: self._on_response(f, seq))
        return True

    def _on_response(self, future, seq):
        handle = future.result()
        if seq != self.seq:
            if handle is not None and handle.accepted:
                handle.cancel_goal_async()
            return
        if handle is None or not handle.accepted:
            self.result = TraversalResult(FAILED, 'rejected by Nav2')
            return
        self.handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, seq))

    def _on_result(self, future, seq):
        if seq != self.seq:
            return
        wrapped = future.result()
        self.handle = None
        self.result = self._interpret(wrapped)

    def _interpret(self, wrapped):
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            return TraversalResult(SUCCEEDED)
        return TraversalResult(FAILED, f'Nav2 status {wrapped.status}')

    def poll(self):
        return self.result

    def cancel(self):
        self.seq += 1
        if self.handle is not None:
            self.handle.cancel_goal_async()
        self.handle = None
        if self.result.state == RUNNING:
            self.result = TraversalResult(FAILED, 'cancelled')


@register_traversal('nav2_pose')
class Nav2PoseTraversal(_Nav2Traversal):
    action_type = NavigateToPose
    default_action = 'navigate_to_pose'

    def start(self, goal_xy, yaw, route=None):
        goal = NavigateToPose.Goal()
        goal.pose = self._pose(goal_xy[0], goal_xy[1], yaw)
        if self.cfg.get('behavior_tree'):
            goal.behavior_tree = resolve_behavior_tree(str(self.cfg['behavior_tree']))
        self.node.publish_goal(goal.pose, self.name)
        return self._send(goal)


@register_traversal('nav2_route')
class Nav2RouteTraversal(_Nav2Traversal):
    """Drive the whole route in one go. Default backend NavigateThroughPoses: Nav2 plans one path through all the
    waypoints and the controller only slows to a stop at the last one (the tile centre); passed waypoints are
    dropped by the BT's RemovePassedGoals. backend: follow_waypoints uses FollowWaypoints instead, which sends one
    NavigateToPose per waypoint and so stops at each. Without a route it drives straight to the goal."""
    action_type = NavigateThroughPoses
    default_action = 'navigate_through_poses'
    uses_route = True

    def _configure(self, cfg):
        self.backend = str(cfg.get('backend', 'through_poses'))
        if self.backend == 'follow_waypoints':
            self.action_type = FollowWaypoints
            self.default_action = 'follow_waypoints'
        elif self.backend != 'through_poses':
            raise ValueError(f"traversal {self.name}: unknown nav2_route backend "
                             f"'{self.backend}' (through_poses | follow_waypoints)")

    def start(self, goal_xy, yaw, route=None):
        poses = list(route or [])[:-1] + [(goal_xy[0], goal_xy[1], yaw)]
        goal = self.action_type.Goal()
        goal.poses = [self._pose(x, y, w) for x, y, w in poses]
        if self.backend == 'through_poses' and self.cfg.get('behavior_tree'):
            goal.behavior_tree = resolve_behavior_tree(str(self.cfg['behavior_tree']))
        self.n_poses = len(goal.poses)
        self.node.publish_goal(goal.poses[-1], self.name)
        return self._send(goal)

    def _interpret(self, wrapped):
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            if self.backend == 'follow_waypoints' and \
                    (self.n_poses - 1) in list(wrapped.result.missed_waypoints):
                return TraversalResult(FAILED, 'the final waypoint was missed')
            return TraversalResult(SUCCEEDED)
        return TraversalResult(FAILED, f'Nav2 status {wrapped.status}')
