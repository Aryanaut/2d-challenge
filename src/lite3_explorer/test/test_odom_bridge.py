"""odom_tf_bridge around AI mode: hold until the rebase (bag fullrun_25_11_183742), no interpolated half-way pose."""
import math

import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import PoseStamped, TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402

from lite3_explorer.odom_tf_bridge import OdomTfBridge  # noqa: E402


@pytest.fixture(scope='module', autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def odom(x, y=0.0, yaw=0.0):
    m = Odometry()
    m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return m


def rebase(x, y, yaw=0.0, frame='odom'):
    p = PoseStamped()
    p.header.frame_id = frame
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


@pytest.fixture
def bridge():
    b = OdomTfBridge()
    b.broadcast_tf = True
    b.sent = []
    b.broadcaster.sendTransform = lambda t: b.sent.append(t)
    yield b
    b.destroy_node()


def xy(t: TransformStamped):
    return round(t.transform.translation.x, 3), round(t.transform.translation.y, 3)


def ns(t: TransformStamped):
    return t.header.stamp.sec * 1_000_000_000 + t.header.stamp.nanosec


def test_after_ai_mode_the_pose_stays_held_until_the_rebase(bridge):
    b = bridge
    b._on_odom(odom(1.0))
    b._on_ai_mode(Bool(data=True))
    b._on_odom(odom(3.0))                      # AI gait: leg odometry runs off
    b._on_ai_mode(Bool(data=False))
    b._on_odom(odom(3.02))                     # the robot shifts a little while AI mode winds down
    assert xy(b.sent[-1]) == (1.0, 0.0)        # still held: slam_toolbox sees no motion, integrates nothing
    b._on_rebase(rebase(2.5, 0.1))
    assert xy(b.sent[-1]) == (2.5, 0.1)
    b._on_odom(odom(3.12))
    assert xy(b.sent[-1]) == (2.6, 0.1)        # tracks the leg odometry from the rebased pose


def test_the_rebase_republishes_the_old_pose_just_before_the_jump(bridge):
    b = bridge
    b._on_odom(odom(1.0))
    b._on_ai_mode(Bool(data=True))
    b._on_ai_mode(Bool(data=False))
    n = len(b.sent)
    b._on_rebase(rebase(2.5, 0.0))
    old, new = b.sent[n:n + 2]
    assert xy(old) == (1.0, 0.0) and xy(new) == (2.5, 0.0)
    assert 0 < ns(new) - ns(old) <= 1_000_000  # a scan stamped between them can only fall in this 1 ms


def test_no_rebase_after_ai_mode_releases_after_the_timeout(bridge):
    b = bridge
    b.hold_until_rebase_s = 5.0
    b._on_odom(odom(1.0))
    b._on_ai_mode(Bool(data=True))
    b._on_odom(odom(3.0))
    b._on_ai_mode(Bool(data=False))
    b._on_odom(odom(3.1))
    assert xy(b.sent[-1]) == (1.0, 0.0)
    b.release_at = b._now_s() - 0.01           # time is up
    b._on_odom(odom(3.2))
    assert xy(b.sent[-1]) == (1.0, 0.0) and not b.frozen     # released: no jump ...
    b._on_odom(odom(3.3))
    assert xy(b.sent[-1]) == (1.1, 0.0)                      # ... and carries on from the held pose


def test_a_rebase_during_ai_mode_needs_no_second_one(bridge):
    b = bridge
    b._on_odom(odom(1.0))
    b._on_ai_mode(Bool(data=True))
    b._on_rebase(rebase(2.0, 0.0))             # arrived before /ai_mode false
    b._on_odom(odom(4.0))
    b._on_ai_mode(Bool(data=False))
    b._on_odom(odom(4.1))
    assert xy(b.sent[-1]) == (2.1, 0.0)


def test_hold_until_rebase_off_releases_at_ai_off(bridge):
    b = bridge
    b.hold_until_rebase_s = 0.0
    b._on_odom(odom(1.0))
    b._on_ai_mode(Bool(data=True))
    b._on_odom(odom(3.0))
    b._on_ai_mode(Bool(data=False))
    b._on_odom(odom(3.1))
    assert xy(b.sent[-1]) == (1.1, 0.0)        # the old behaviour
