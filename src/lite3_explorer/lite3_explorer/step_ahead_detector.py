"""Step-ahead detector: a step up in front of the robot, seen in the depth image (fallback trigger for the h-step gait).

  sub  /camera/depth/image_raw      depth image (16UC1 millimetres or 32FC1 metres)
  sub  /camera/depth/camera_info    intrinsics (else depth_hfov_deg is used)
  sub  /imu/data                    body roll / pitch (the odom -> base_link TF has yaw only)
  sub  /ai_mode                     std_msgs/Bool   true while the stairs mode climbs in AI mode: detection pauses
  pub  /step_ahead                  std_msgs/String JSON, one per frame processed:
                                    {"step": true, "dist_m": .., "rise_m": .., ...} once a step was seen in confirm_frames
                                    frames in a row, else {"step": false}. Nothing while the body is tilted (on the step
                                    already) or the IMU is stale.

The mission manager (step_up_depth in mission.yaml) turns a step seen while it drives to a target into /step_up for that
target, alongside the tile elevation rule, which misses steps whose tile elevation is wrong, late or never measured
(fullrun_26_11_210206: backtracking to the +0.14 m tile 1,1 twice with /step_up off). Detection: step_ahead.py.
"""
import json
import math
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from lite3_explorer import step_ahead as sa
from lite3_explorer.curtain_detector import quat_to_matrix


class StepAheadDetector(Node):
    def __init__(self):
        super().__init__('step_ahead_detector')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.depth_topic = p('depth_topic', '/camera/depth/image_raw')
        self.hfov = math.radians(float(p('depth_hfov_deg', 81.4)))         # used until camera_info arrives
        self.depth_scale = float(p('depth_scale', 0.001))                  # 16UC1 -> metres
        self.base_frame = p('base_frame', 'base_link')
        self.rate_hz = float(p('process_hz', 5.0))
        self.imu_max_age_s = float(p('imu_max_age_s', 0.3))
        d = sa.StepAheadParams()
        self.params = sa.StepAheadParams(**{k: type(getattr(d, k))(p(k, getattr(d, k))) for k in d.__dataclass_fields__})
        self.confirm = sa.StepConfirm(int(p('confirm_frames', 2)))
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.cam = None                                   # (R, t) base_frame <- depth optical frame (static)
        self.intrinsics = None
        self.imu = None                                   # (roll, pitch, monotonic time)
        self.paused = False
        self.last = 0.0
        self.was_step = False
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraInfo, p('camera_info_topic', '/camera/depth/camera_info'), self._on_info, 1)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Imu, p('imu_topic', '/imu/data'), self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Bool, p('pause_topic', '/ai_mode'), self._on_pause, latched)
        self.pub = self.create_publisher(String, p('step_topic', '/step_ahead'), 10)
        q = self.params
        self.get_logger().info(
            f'step-ahead detector on {self.depth_topic} at {self.rate_hz:g} Hz: a flat top {q.min_rise_m * 100:.0f}..'
            f'{q.max_rise_m * 100:.0f} cm above the floor, starting {q.near_m:.2f}..{q.far_m:.2f} m ahead in a '
            f'{2 * q.half_width_m:.2f} m corridor, {self.confirm.confirm_frames} frames in a row')

    def _on_info(self, msg):
        if msg.k[0] > 0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_imu(self, msg):
        o = msg.orientation
        roll, pitch, _ = sa.rpy_from_quat(o.x, o.y, o.z, o.w)
        self.imu = (roll, pitch, time.monotonic())

    def _on_pause(self, msg):
        if bool(msg.data) != self.paused:
            self.paused = bool(msg.data)
            self.confirm.hits = 0
            self.get_logger().info('AI mode on: detection paused.' if self.paused else 'AI mode off: detecting again.')

    def _camera(self, frame):
        if self.cam is None:
            try:
                t = self.tf.lookup_transform(self.base_frame, frame, Time(), Duration(seconds=0.1))
            except Exception as e:  # noqa: BLE001 - static TF not in yet
                self.get_logger().warn(f'no TF {self.base_frame} <- {frame}: {e}', throttle_duration_sec=5.0)
                return None
            tr = t.transform.translation
            self.cam = (quat_to_matrix(t.transform.rotation), np.array([tr.x, tr.y, tr.z]))
        return self.cam

    def _on_depth(self, msg):
        now = time.monotonic()
        if self.paused or now - self.last < 1.0 / self.rate_hz:
            return
        if self.imu is None or now - self.imu[2] > self.imu_max_age_s:
            self.get_logger().warn('no fresh IMU: no step detection', throttle_duration_sec=5.0)
            return
        roll, pitch, _ = self.imu
        if sa.tilted(roll, pitch, self.params):
            return                                         # climbing already / tilted: the profile would be skewed
        if msg.encoding == '16UC1':
            depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width) * self.depth_scale
        elif msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        else:
            self.get_logger().warn(f'depth encoding {msg.encoding} not handled', throttle_duration_sec=30.0)
            return
        cam = self._camera(msg.header.frame_id)
        if cam is None:
            return
        self.last = now
        fx = (msg.width / 2) / math.tan(self.hfov / 2)
        k = self.intrinsics or (fx, fx, msg.width / 2, msg.height / 2)
        pts = sa.to_level_base(sa.depth_to_points(depth, k, self.params), cam[0], cam[1], roll, pitch)
        det = sa.detect(pts, self.params)
        step = self.confirm.update(det)
        out = {'step': step, **(det.to_dict() if step else {})}
        self.pub.publish(String(data=json.dumps(out)))
        if step != self.was_step:
            self.was_step = step
            self.get_logger().info(f'step up ahead: {det.rise_m * 100:+.0f} cm, {det.dist_m:.2f} m ahead of '
                                   f'{self.base_frame}' + (' (at the edge already)' if det.at_edge else '')
                                   if step else 'no step ahead any more.')


def main(args=None):
    rclpy.init(args=args)
    node = StepAheadDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
