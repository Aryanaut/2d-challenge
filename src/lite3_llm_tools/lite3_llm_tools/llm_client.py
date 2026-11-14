"""llm_client: the AI server client node.

It owns everything that talks to the AI server and builds the AI's context from live sensors:
  - camera frames (kept on request, one per view),
  - the lidar: the SLAM /map (drawn as a top-down image) and the /scan ranges (as a line of text),
and answers JSON requests from any explorer. The explorer needs no HTTP, image or prompt code of its own.

Interfaces (JSON strings in std_msgs/String, so they can be recorded with `ros2 bag`)
  sub  /llm/request     {"id": "...", "task": "describe_views" | "place_initial_tile" | "choose_target" |
                         "describe_tile" | "render_map" | "capture_view" | "ping", ...task fields}   (see tools.py)
  pub  /llm/response    {"id": "...", "task": "...", "ok": true, "result": {...}, "elapsed_s": 3.1}
                        or {"id": "...", "ok": false, "error": "..."}
  pub  /llm/status      latched, 1 Hz: {"ready": true, "server": "...", "queries": 12, "last_error": "", ...}
  pub  /llm/last_result every answer from the AI (JSON), for a status pane
  sub  camera image     sensor_msgs/CompressedImage or Image (image_topic)
  sub  /map             nav_msgs/OccupancyGrid
  sub  /scan            sensor_msgs/LaserScan
"""
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import time
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import String

from lite3_llm_tools import gridmap
from lite3_llm_tools import lidar_context
from lite3_llm_tools import prompts as P
from lite3_llm_tools.image_utils import encode_jpeg, image_msg_to_bgr
from lite3_llm_tools.perception_client import PerceptionClient, read_prompt
from lite3_llm_tools.tools import LlmTools, Settings

REQUEST_TOPIC, RESPONSE_TOPIC, STATUS_TOPIC, LAST_RESULT_TOPIC = (
    '/llm/request', '/llm/response', '/llm/status', '/llm/last_result')


class LlmClient(Node):
    def __init__(self):
        super().__init__('llm_client')
        p = self.declare_parameter
        server_url = p('server_url', 'http://127.0.0.1:8000').value
        group_id = str(p('group_id', '').value)
        self.image_topic = p('image_topic', '/camera/color/image_raw/compressed').value
        self.image_compressed = p('image_compressed', True).value
        self.scan_topic = p('scan_topic', '/scan').value
        self.map_topic = p('map_topic', '/map').value
        self.max_image_width = p('max_image_width', 1024).value
        self.max_image_height = p('max_image_height', 768).value
        self.jpeg_quality = p('jpeg_quality', 85).value
        self.image_wait_s = p('image_wait_s', 5.0).value          # how long capture_view waits for a new frame
        self.scan_max_age_s = p('scan_max_age_s', 2.0).value
        self.scan_forward_offset_deg = p('scan_forward_offset_deg', 0.0).value
        self.workers = max(1, int(p('request_workers', 4).value))
        settings = Settings(
            output_dir=p('output_dir', '~/lite3_survey_runs').value,
            query_retries=int(p('query_retries', 1).value),
            error_backoff_s=float(p('error_backoff_s', 5.0).value),
            view_query_workers=int(p('view_query_workers', 3).value),
            wall_band_m=float(p('min_wall_clearance_m', 0.35).value),
            min_clearance_m=float(p('target_min_clearance_m', 0.30).value),
            max_upload_w=int(self.max_image_width), max_upload_h=int(self.max_image_height),
            lidar_context=bool(p('lidar_context', True).value))
        overrides = {task: p(f'{task}_prompt_file', '').value for task in P.PROMPT_FILES}
        overrides['describe_view'] = overrides['describe_view'] or p('prompt_file', '').value
        client = PerceptionClient(
            server_url, group_id or 'unused', read_prompt(P.prompt_path('describe_view', overrides['describe_view'])),
            read_timeout_s=p('read_timeout_s', 75.0).value, connect_timeout_s=p('connect_timeout_s', 5.0).value,
            connect_retries=int(p('connect_retries', 3).value), connect_backoff_s=float(p('connect_backoff_s', 2.0).value))
        self.group_set = bool(group_id) and group_id != 'CHANGE_ME'
        self.tools = LlmTools(client, settings, overrides, get_map=self._map_snapshot, get_scan_text=self._scan_text)

        self.map = None
        self.scan = None
        self.scan_at = 0.0
        self.latest_image = None
        self.waiting = []                  # capture_view requests waiting for the next frame
        self.responses: 'queue.Queue[dict]' = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        self.started = time.monotonic()

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        reliable = QoSProfile(depth=50)
        self.create_subscription(CompressedImage if self.image_compressed else Image, self.image_topic,
                                 self._on_image, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, latched)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(String, REQUEST_TOPIC, self._on_request, reliable)
        self.response_pub = self.create_publisher(String, RESPONSE_TOPIC, reliable)
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, latched)
        self.last_result_pub = self.create_publisher(String, LAST_RESULT_TOPIC, 10)
        self.create_timer(0.05, self._flush)
        self.create_timer(1.0, self._publish_status)
        self._publish_status()
        self.get_logger().info(f'AI server {client.url} (group set: {self.group_set}); camera {self.image_topic}, '
                               f'lidar {self.scan_topic} + {self.map_topic}; saving to {self.tools.default_dir}. '
                               f'Listening on {REQUEST_TOPIC}.')
        if server_url.rstrip('/') in ('http://127.0.0.1:8000', 'http://localhost:8000'):
            self.get_logger().error(
                f'server_url is the built-in default ({server_url}): config/llm_client.yaml was not loaded, so every '
                'request will be refused. Start the node with "ros2 launch lite3_llm_tools llm_client.launch.py" '
                '(or pass --ros-args --params-file), not "ros2 run".')
        if not self.group_set:
            self.get_logger().warn('group_id is empty or CHANGE_ME: the server will reject every request.')

    # ---------------------------------------------------------------- sensors
    def _on_map(self, msg: OccupancyGrid):
        i = msg.info
        self.map = (gridmap.grid_from_msg_data(msg.data, i.width, i.height),
                    gridmap.GridInfo(i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y))

    def _map_snapshot(self):
        return self.map

    def _on_scan(self, msg: LaserScan):
        self.scan = msg
        self.scan_at = time.monotonic()

    def _scan_text(self) -> Optional[str]:
        if self.scan is None or time.monotonic() - self.scan_at > self.scan_max_age_s:
            return None
        return lidar_context.describe(lidar_context.summarize_scan(self.scan, self.scan_forward_offset_deg),
                                      self.scan.range_max)

    def _on_image(self, msg):
        self.latest_image = msg
        if not self.waiting:
            return
        try:
            jpeg = encode_jpeg(image_msg_to_bgr(msg), self.max_image_width, self.jpeg_quality, self.max_image_height)
        except ValueError as exc:
            self.get_logger().error(f'Image conversion failed: {exc}', throttle_duration_sec=5.0)
            return
        scan_text = self._scan_text()
        waiting, self.waiting = self.waiting, []
        for req, asked_at in waiting:    # every capture requested before this frame gets it
            result = self.tools.store_capture(str(req.get('name')), jpeg, req.get('save_dir', ''),
                                              req.get('pose'), scan_text)
            if req.get('task') == 'describe_tile':
                self.pool.submit(self._serve, req)           # the AI query runs in the background
                continue
            self.responses.put({'id': req.get('id'), 'task': 'capture_view', 'ok': True, 'result': result,
                                'elapsed_s': round(time.monotonic() - asked_at, 2)})

    # ---------------------------------------------------------------- requests
    def _on_request(self, msg: String):
        try:
            req = json.loads(msg.data)
            if not isinstance(req, dict):
                raise ValueError('request is not a JSON object')
        except ValueError as exc:
            self.get_logger().error(f'Bad request on {REQUEST_TOPIC}: {exc}')
            self.responses.put({'id': None, 'ok': False, 'error': f'bad request: {exc}'})
            return
        if req.get('task') in ('capture_view', 'describe_tile'):
            if not req.get('name'):
                self.responses.put({'id': req.get('id'), 'task': req.get('task'), 'ok': False, 'error': 'needs "name"'})
            else:
                self.waiting.append((req, time.monotonic()))     # served by the next camera frame
            return
        self.pool.submit(self._serve, req)

    def _serve(self, req: dict):
        self.get_logger().info(f'Request {req.get("id")}: {req.get("task")}')
        reply = self.tools.handle(req)
        if reply['ok']:
            self.get_logger().info(f'Request {req.get("id")} ({req.get("task")}) done in {reply["elapsed_s"]:.1f}s')
        else:
            self.get_logger().error(f'Request {req.get("id")} ({req.get("task")}) failed: {reply["error"]}')
        self.responses.put(reply)

    def _flush(self):
        now = time.monotonic()
        for item in [w for w in self.waiting if now - w[1] > self.image_wait_s]:
            self.waiting.remove(item)
            self.responses.put({'id': item[0].get('id'), 'task': item[0].get('task'), 'ok': False,
                                'error': f'no new image on {self.image_topic} within {self.image_wait_s:g} s'})
        while True:
            try:
                reply = self.responses.get_nowait()
            except queue.Empty:
                break
            self.response_pub.publish(String(data=json.dumps(reply, ensure_ascii=False, default=str)))
            if reply.get('ok') and reply.get('task') in ('describe_views', 'describe_tile', 'place_initial_tile', 'choose_target'):
                self.last_result_pub.publish(String(data=json.dumps(reply.get('result'), ensure_ascii=False, default=str)))

    def _publish_status(self):
        self.status_pub.publish(String(data=json.dumps({
            'ready': True, 'server': self.tools.client.url, 'group_set': self.group_set,
            'have_map': self.map is not None, 'have_image': self.latest_image is not None,
            'have_scan': self.scan is not None, 'queries': self.tools.queries,
            'last_error': self.tools.last_error, 'uptime_s': round(time.monotonic() - self.started, 1)})))

    def destroy_node(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LlmClient()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
