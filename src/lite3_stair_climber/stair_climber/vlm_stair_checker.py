#!/usr/bin/env python3
"""Optional camera confirmation: "is there a staircase ahead?"

When the state machine sees stairs with the lidar (and use_vlm is true) it
publishes /stair_climber/vlm_request. This node grabs the latest camera
frame, shrinks it to fit 1024x768 (the server's limit from Lab 6), sends it
to the scene-understanding server and publishes True/False on
/stair_climber/vlm_result.

>>> YOU MUST FILL IN `query_scene_server()` <<<
Copy the HTTP request code from the Lab 6 `client.py` (same endpoint,
same form fields, same group id). If the call fails or times out, this node
publishes nothing and the state machine falls back after `vlm_timeout`.
"""

import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Empty

PROMPT = (
    'You are the camera of a legged robot. Look at the image. '
    'Is there a staircase (steps going up) directly in front of the robot? '
    'Answer with exactly one word: YES or NO.'
)


def query_scene_server(server: str, group: str, jpeg_bytes: bytes, prompt: str,
                       timeout: float) -> str:
    """Send one image + prompt, return the model's text answer ("result" field).

    TODO: replace this body with the request from your Lab 6 client.py.
    The server's JSON reply looks like:
        {"success": true, "result": "...", "processing_time_s": 3.7, ...}
    """
    raise NotImplementedError('copy the request code from Lab 6 client.py here')


class VlmStairChecker(Node):

    def __init__(self):
        super().__init__('vlm_stair_checker')
        self.declare_parameter('image_topic', '/image_raw/compressed')
        self.declare_parameter('server', 'http://192.168.10.141:8000')
        self.declare_parameter('group', 'groupXX')
        self.declare_parameter('timeout', 8.0)
        self.server = self.get_parameter('server').value
        self.group = self.get_parameter('group').value
        self.timeout = self.get_parameter('timeout').value

        self.latest = None
        self.busy = False
        self.create_subscription(CompressedImage, self.get_parameter('image_topic').value,
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(Empty, '/stair_climber/vlm_request', self.on_request, 10)
        self.pub = self.create_publisher(Bool, '/stair_climber/vlm_result', 10)

    def on_image(self, msg):
        self.latest = msg

    def on_request(self, _):
        if self.busy:
            return
        if self.latest is None:
            self.get_logger().warn('VLM requested but no camera image yet')
            return
        self.busy = True
        threading.Thread(target=self._worker, args=(self.latest,), daemon=True).start()

    def _worker(self, msg):
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            h, w = frame.shape[:2]
            scale = min(1024 / w, 768 / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            answer = query_scene_server(self.server, self.group, jpg.tobytes(),
                                        PROMPT, self.timeout)
            verdict = answer.strip().upper().startswith('YES')
            self.get_logger().info(f'VLM answer: {answer!r} -> stairs={verdict}')
            self.pub.publish(Bool(data=verdict))
        except Exception as e:  # noqa: BLE001  never crash the robot over the VLM
            self.get_logger().error(f'VLM query failed: {e}')
        finally:
            self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = VlmStairChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
