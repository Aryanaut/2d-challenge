"""Frontier exploration node for the DEEP Robotics Lite3 (ROS 2 Humble + Nav2).

Combines the three roles of shimmer0909/Frontier-Exploration in one node:
map listener (/map), frontier detector (publishes /frontier_list) and frontier
selector (sends the best cluster to Nav2's NavigateToPose action and publishes
/frontier_reached on success).

Interfaces
  sub  /map                         nav_msgs/OccupancyGrid
  pub  /frontier_list               geometry_msgs/PoseArray   (all cluster goals)
  pub  ~/markers                    visualization_msgs/MarkerArray
  pub  ~/current_goal               geometry_msgs/PoseStamped (latched) goal sent to Nav2 (show in RViz)
  pub  /frontier_reached            std_msgs/Bool
  pub  /exploration_complete        std_msgs/Bool (latched)
  pub  ~/status                     std_msgs/String
  pub  /frontier_candidates         std_msgs/String  JSON: numbered usable frontiers + wall clearance
                                    (published while not driving, for the AI to choose from)
  sub  ~/preferred_goal             geometry_msgs/PointStamped  goal chosen by the AI: a frontier,
                                    or a tile to visit (any free point). Used for the next goal if
                                    it has enough wall clearance, otherwise the cost ranking decides
  srv  ~/set_active                 std_srvs/SetBool  (False cancels the current goal)
  act  navigate_to_pose             nav2_msgs/NavigateToPose (client)

With hold_after_reach=true the node waits after each reached frontier until
~/set_active(True) is called (the environment reporter uses that pause to look
around with the camera) or hold_timeout_s expires.
"""
import json
import math
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, Pose, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_kitchen_explorer import frontier_core as fc


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class FrontierExplorer(Node):
    IDLE, NAVIGATING, HOLD, PAUSED, COMPLETE = 'IDLE', 'NAVIGATING', 'HOLD', 'PAUSED', 'COMPLETE'

    def __init__(self):
        super().__init__('frontier_explorer')
        p = self.declare_parameter
        self.global_frame = p('global_frame', 'map').value
        self.robot_frame = p('robot_frame', 'base_link').value
        self.map_topic = p('map_topic', '/map').value
        self.start_active = p('start_active', True).value
        self.free_threshold = p('free_threshold', 25).value
        self.occupied_threshold = p('occupied_threshold', 65).value
        self.min_cluster_cells = p('min_cluster_cells', 8).value
        self.goal_clearance_m = p('goal_clearance_m', 0.30).value
        self.distance_weight = p('distance_weight', 1.0).value
        self.size_weight = p('size_weight', 0.5).value
        self.min_goal_distance_m = p('min_goal_distance_m', 0.6).value
        self.blacklist_radius_m = p('blacklist_radius_m', 0.6).value
        self.goal_timeout_s = p('goal_timeout_s', 90.0).value
        self.progress_timeout_s = p('progress_timeout_s', 25.0).value
        self.progress_min_m = p('progress_min_m', 0.25).value
        self.replan_when_goal_explored = p('replan_when_goal_explored', True).value
        self.goal_frontier_radius_m = p('goal_frontier_radius_m', 0.5).value
        self.face_frontier = p('face_frontier', True).value
        self.hold_after_reach = p('hold_after_reach', True).value
        self.hold_timeout_s = p('hold_timeout_s', 90.0).value
        self.empty_plans_before_complete = p('empty_plans_before_complete', 5).value
        self.plan_period_s = p('plan_period_s', 1.0).value
        self.nav_action = p('navigate_action', 'navigate_to_pose').value
        self.candidate_limit = p('candidate_limit', 8).value
        self.preferred_timeout_s = p('preferred_timeout_s', 60.0).value
        self.preferred_match_radius_m = p('preferred_match_radius_m', 0.6).value
        self.preferred_min_clearance_m = p('preferred_min_clearance_m', 0.30).value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action)

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, map_qos)
        self.frontier_pub = self.create_publisher(PoseArray, '/frontier_list', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', 10)
        self.reached_pub = self.create_publisher(Bool, '/frontier_reached', 10)
        self.complete_pub = self.create_publisher(Bool, '/exploration_complete', latched)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.candidates_pub = self.create_publisher(String, '/frontier_candidates', latched)
        self.goal_pub = self.create_publisher(PoseStamped, '~/current_goal', latched)
        self.create_subscription(PointStamped, '~/preferred_goal', self._on_preferred, 10)
        self.preferred = None        # (x, y, received monotonic time)
        self.create_service(SetBool, '~/set_active', self._on_set_active)

        self.grid = None
        self.info = None
        self.state = self.IDLE if self.start_active else self.PAUSED
        self.goal_handle = None
        self.goal_seq = 0
        self.goal_xy = None
        self.goal_direct = False
        self.goal_sent_at = 0.0
        self.best_dist = math.inf
        self.last_progress_at = 0.0
        self.hold_since = 0.0
        self.empty_plans = 0
        self.blacklist = []

        self.create_timer(self.plan_period_s, self._tick)
        self.get_logger().info(f'Frontier explorer ready (state={self.state}).')

    # ------------------------------------------------------------------ inputs
    def _on_map(self, msg: OccupancyGrid):
        info = msg.info
        self.info = fc.GridInfo(info.width, info.height, info.resolution,
                                info.origin.position.x, info.origin.position.y)
        self.grid = fc.grid_from_msg_data(msg.data, info.width, info.height)

    def _on_preferred(self, msg: PointStamped):
        self.preferred = (msg.point.x, msg.point.y, time.monotonic())
        self.get_logger().info(f'AI prefers frontier at ({msg.point.x:.2f}, {msg.point.y:.2f}).')

    def _on_set_active(self, request, response):
        if request.data:
            if self.state in (self.PAUSED, self.HOLD, self.COMPLETE):
                if self.state == self.COMPLETE:
                    self.empty_plans = 0
                self._set_state(self.IDLE)
            response.message = f'active ({self.state})'
        else:
            self._cancel_goal()
            self._set_state(self.PAUSED)
            response.message = 'paused'
        response.success = True
        return response

    def _robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, Time(),
                                                timeout=Duration(seconds=0.2))
        except TransformException as exc:
            self.get_logger().warn(f'No {self.global_frame}->{self.robot_frame} TF: {exc}',
                                   throttle_duration_sec=5.0)
            return None
        tr = t.transform.translation
        return tr.x, tr.y

    # -------------------------------------------------------------------- loop
    def _tick(self):
        now = time.monotonic()
        if self.state == self.HOLD and now - self.hold_since > self.hold_timeout_s:
            self.get_logger().warn('Hold timed out; resuming exploration.')
            self._set_state(self.IDLE)
        if self.grid is None:
            self._status('waiting for /map')
            return
        robot = self._robot_pose()
        if robot is None:
            return
        if self.state == self.NAVIGATING:
            self._monitor_goal(robot, now)
        elif self.state == self.IDLE:
            self._plan(robot)
        else:
            self._score(robot)      # keep /frontier_candidates fresh while paused / holding

    def _score(self, robot_xy):
        clusters = fc.build_clusters(self.grid, self.info,
                                     free_threshold=self.free_threshold,
                                     occupied_threshold=self.occupied_threshold,
                                     min_cluster_cells=self.min_cluster_cells,
                                     clearance_m=self.goal_clearance_m)
        best = fc.select_frontier(clusters, robot_xy,
                                  distance_weight=self.distance_weight,
                                  size_weight=self.size_weight,
                                  min_goal_distance_m=self.min_goal_distance_m,
                                  blacklist=self.blacklist,
                                  blacklist_radius_m=self.blacklist_radius_m)
        candidates = fc.rank_candidates(clusters, self.candidate_limit)
        self.candidates_pub.publish(String(data=json.dumps({
            'stamp': self.get_clock().now().nanoseconds * 1e-9,
            'frame': self.global_frame, 'robot_xy': [round(v, 3) for v in robot_xy],
            'candidates': candidates})))
        return clusters, best

    def _apply_preference(self, clusters, best):
        if self.preferred is None:
            return best
        px, py, received = self.preferred
        self.preferred = None
        if time.monotonic() - received > self.preferred_timeout_s:
            self.get_logger().warn('AI frontier choice is stale; using cost ranking.')
            return best
        usable = [c for c in clusters if c.extra.get('usable')]
        match = min(usable, key=lambda c: math.hypot(c.goal_xy[0] - px, c.goal_xy[1] - py), default=None)
        if match is None or math.hypot(match.goal_xy[0] - px, match.goal_xy[1] - py) > self.preferred_match_radius_m:
            return self._direct_goal(px, py, best)
        if match.extra.get('clearance_m', 0.0) < self.preferred_min_clearance_m:
            self.get_logger().warn(f'AI-chosen frontier is only {match.extra["clearance_m"]:.2f} m from a wall '
                                   f'(< {self.preferred_min_clearance_m:.2f} m); using cost ranking.')
            return best
        self.get_logger().info('Following the AI-chosen frontier.')
        return match

    def _direct_goal(self, px, py, best):
        """AI picked a point that is not a frontier (e.g. the next tile to visit): accept it if free and clear."""
        r, c = self.info.world_to_cell(px, py)
        inside = 0 <= r < self.info.height and 0 <= c < self.info.width
        if not inside or not (0 <= self.grid[r, c] <= self.free_threshold):
            self.get_logger().warn('AI-chosen goal is not on mapped free floor; using cost ranking.')
            return best
        clearance = fc.wall_clearance_m(self.grid, self.info, (px, py), self.occupied_threshold)
        if clearance < self.preferred_min_clearance_m:
            self.get_logger().warn(f'AI-chosen goal is only {clearance:.2f} m from a wall '
                                   f'(< {self.preferred_min_clearance_m:.2f} m); using cost ranking.')
            return best
        robot = self._robot_pose()
        dist = math.hypot(px - robot[0], py - robot[1]) if robot else 0.0
        if fc.is_blacklisted((px, py), self.blacklist, self.blacklist_radius_m):
            self.get_logger().warn('AI-chosen goal failed before; using cost ranking.')
            return best
        self.get_logger().info(f'Following the AI-chosen tile goal ({px:.2f}, {py:.2f}), '
                               f'{clearance:.2f} m from walls.')
        return fc.FrontierCluster(cells=[], centroid_xy=(px, py), goal_xy=(px, py), size_m=0.0,
                                  distance_m=dist, extra={'direct': True, 'clearance_m': clearance})

    def _plan(self, robot_xy):
        clusters, best = self._score(robot_xy)
        best = self._apply_preference(clusters, best)   # an AI tile goal is valid even with no frontiers
        self._publish_frontiers(clusters, best)
        if best is None:
            self.empty_plans += 1
            self._status(f'no reachable frontier ({self.empty_plans}/{self.empty_plans_before_complete})')
            if self.empty_plans >= self.empty_plans_before_complete:
                self.get_logger().info('Exploration complete: no frontiers left.')
                self.complete_pub.publish(Bool(data=True))
                self._set_state(self.COMPLETE)
            return
        self.empty_plans = 0
        self._send_goal(best, robot_xy)

    def _send_goal(self, cluster, robot_xy):
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self._status('waiting for Nav2 navigate_to_pose server')
            return
        gx, gy = cluster.goal_xy
        if self.face_frontier:
            yaw = math.atan2(cluster.centroid_xy[1] - robot_xy[1], cluster.centroid_xy[0] - robot_xy[0])
        else:
            yaw = math.atan2(gy - robot_xy[1], gx - robot_xy[0])
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = gx, gy
        q = yaw_to_quaternion(yaw)
        o = goal.pose.pose.orientation
        o.x, o.y, o.z, o.w = q

        self.goal_pub.publish(goal.pose)
        self.goal_direct = bool(cluster.extra.get('direct'))
        self.goal_seq += 1
        seq = self.goal_seq
        self.goal_xy = (gx, gy)
        self.goal_sent_at = self.last_progress_at = time.monotonic()
        self.best_dist = cluster.distance_m
        self._set_state(self.NAVIGATING)
        self.get_logger().info(f'Goal #{seq}: ({gx:.2f}, {gy:.2f}) dist={cluster.distance_m:.2f} m '
                               f'size={cluster.size_m:.2f} m cost={cluster.cost:.2f}')
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_goal_response(f, seq))

    def _on_goal_response(self, future, seq):
        handle = future.result()
        if seq != self.goal_seq:
            # Paused or replanned while Nav2 was still accepting this goal: stop it.
            if handle is not None and handle.accepted:
                handle.cancel_goal_async()
            return
        if handle is None or not handle.accepted:
            self.get_logger().warn('Nav2 rejected the frontier goal; blacklisting it.')
            self._fail_goal()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, seq))

    def _on_result(self, future, seq):
        if seq != self.goal_seq or self.state != self.NAVIGATING:
            return  # stale goal, or cancelled on purpose (pause / replan)
        status = future.result().status
        self.goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Frontier #{seq} reached.')
            self.reached_pub.publish(Bool(data=True))
            if self.hold_after_reach:
                self.hold_since = time.monotonic()
                self._set_state(self.HOLD)
            else:
                self._set_state(self.IDLE)
        else:
            self.get_logger().warn(f'Frontier #{seq} failed (status {status}); blacklisting.')
            self._fail_goal()

    def _monitor_goal(self, robot_xy, now):
        if self.goal_xy is None:
            return
        dist = math.hypot(self.goal_xy[0] - robot_xy[0], self.goal_xy[1] - robot_xy[1])
        if dist < self.best_dist - self.progress_min_m:
            self.best_dist = dist
            self.last_progress_at = now
        if now - self.goal_sent_at > self.goal_timeout_s:
            self.get_logger().warn('Frontier goal timed out.')
            self._cancel_goal()
            self._fail_goal()
        elif now - self.last_progress_at > self.progress_timeout_s:
            self.get_logger().warn('No progress toward frontier (stuck?).')
            self._cancel_goal()
            self._fail_goal()
        elif self.replan_when_goal_explored and not self.goal_direct and not fc.frontier_near(
                self.grid, self.info, self.goal_xy, self.goal_frontier_radius_m, self.free_threshold):
            self.get_logger().info('Goal area already explored; replanning.')
            self._cancel_goal()
            self._set_state(self.IDLE)

    def _fail_goal(self):
        if self.goal_xy is not None:
            self.blacklist.append(self.goal_xy)
        self.goal_xy = None
        self.goal_handle = None
        if self.state == self.NAVIGATING:
            self._set_state(self.IDLE)

    def _cancel_goal(self):
        self.goal_seq += 1  # invalidates callbacks of the old goal
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.goal_handle = None

    # ----------------------------------------------------------------- output
    def _set_state(self, state):
        if state != self.state:
            self.get_logger().info(f'{self.state} -> {state}')
        self.state = state
        self._status('')

    def _status(self, detail):
        self.status_pub.publish(String(data=f'{self.state}: {detail}' if detail else self.state))

    def _publish_frontiers(self, clusters, best):
        stamp = self.get_clock().now().to_msg()
        poses = PoseArray()
        poses.header.frame_id = self.global_frame
        poses.header.stamp = stamp
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for i, c in enumerate(clusters):
            pose = Pose()
            pose.position.x, pose.position.y = c.goal_xy
            pose.orientation.w = 1.0
            poses.poses.append(pose)
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = 'frontiers', i, Marker.SPHERE, Marker.ADD
            m.pose = pose
            size = 0.35 if c is best else 0.2
            m.scale.x = m.scale.y = m.scale.z = size
            if c is best:
                m.color.r, m.color.g, m.color.b = 0.1, 0.9, 0.1
            elif c.blacklisted:
                m.color.r, m.color.g, m.color.b = 0.9, 0.1, 0.1
            else:
                m.color.r, m.color.g, m.color.b = 0.1, 0.4, 1.0
            m.color.a = 0.9
            markers.markers.append(m)
        self.frontier_pub.publish(poses)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cancel_goal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
