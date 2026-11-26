"""Simulated perception for lite3_sim: stands in for object_detector + qr_scanner, from objects listed in the maze file.

The maze YAML gets an optional `perception:` block (maze frame, metres; or `tile: [i, j]` + `offset_m`):

    perception:
      survivors:                      # M03 (qr: text) / M05 (no qr)
        - {tile: [0, 2], offset_m: [-0.25, 0.0], qr: "My friend is in the market.", qr_offset_m: [0.30, 0.0]}
        - {tile: [4, 2]}
      objects:                        # M01 (type object, e.g. hand_tools) / M02 (type hazard, e.g. flammable)
        - {tile: [2, 2], label: hand_tools}
        - {tile: [1, 0], label: flammable, type: hazard}
      props: auto                     # every typed scene tile ('1'..'8') gets a scene prop at its centre (or a list)
      clutter_props: []               # extra prop spots, e.g. on a cluttered non-scene tile (false prop sightings)

Published like the real nodes (same topics, same JSON fields the reporter and the mission manager read):
  /perception/event        survivor / object / hazard events (type, label, confidence, bbox_xyxy, location, robot_pose,
                           qr_in_view, bbox_h_frac, hazard_certain), one per object per `event_repeat_s`
  /perception/detections   raw detections every tick ({"detections": [{label, confidence, bbox_xyxy}]}): scene props
  /qr/visible              every tick: {"t", "codes": [{camera, text, decoded_ok, map_point, ...}]}
  /qr/detections, /perception/qr_text   once per QR code, when first decoded

Seen = inside the camera's horizontal FOV, within range, and no wall / slit panel / curtain between (maze segments).
A QR code is decoded only closer (qr_range_m) and near the image centre (qr_max_bearing_deg), like a real reader.

    python3 -m lite3_sim.sim_perception --ros-args -p maze:=<maze.yaml> [-p map_origin:=start]
"""
from __future__ import annotations

import json
import math
import time
import uuid

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from lite3_sim import world as W
from lite3_sim.maze import load_maze

# a prop the detector fires on in each kind of scene room (the label does not matter to prop votes, only that it is one)
SCENE_PROP = {'clothes_shop': 'cart', 'fitness_centre': 'cart', 'kitchen': 'kitchen', 'market': 'market_stall',
              'workshop': 'tool_bench', 'clinic': 'medical_kit', 'cleaning_room': 'cleaning_cart',
              'furniture_shop': 'cart'}


class SimPerception(Node):
    def __init__(self):
        super().__init__('sim_perception')
        p = lambda n, d: self.declare_parameter(n, d).value       # noqa: E731
        path = p('maze', '')
        if not path:
            raise RuntimeError('parameter "maze" is required')
        self.maze = load_maze(path)
        with open(path, encoding='utf-8') as fh:
            spec = (yaml.safe_load(fh) or {}).get('perception') or {}
        origin = p('map_origin', 'start')
        self.map_T_maze = W.invert(self.maze.start) if origin == 'start' else (0.0, 0.0, 0.0)
        self.geom = W.Geometry(self.maze, self.map_T_maze)
        self.hfov = math.radians(p('hfov_deg', 81.0))
        self.fx, self.cx, self.img_w, self.img_h = p('fx', 462.6), p('cx', 320.1), 640, 480
        self.cam_x = p('camera_x_m', 0.252)
        self.range_m = p('range_m', 3.0)                 # survivors / objects / props seen this far
        self.survivor_h_m = p('survivor_h_m', 0.22)      # printed survivor picture (bag fullrun_26_11_141104: ~0.21 m)
        self.qr_range_m = p('qr_range_m', 1.2)           # QR decoded only this close ...
        self.qr_bearing = math.radians(p('qr_max_bearing_deg', 30.0))   # ... and this near the image centre
        self.event_repeat_s = p('event_repeat_s', 4.0)
        self.map_frame, self.base_frame = p('map_frame', 'map'), p('base_frame', 'base_link')

        def at(o, key='offset_m'):
            if 'xy' in o:
                x, y = o['xy']
            else:
                x, y = self.maze.tile_centre(tuple(o['tile']))
                dx, dy = o.get(key, [0.0, 0.0])
                x, y = x + dx, y + dy
            return W.compose(self.map_T_maze, (float(x), float(y), 0.0))[:2]

        self.things = []          # dicts: kind (object|hazard|prop|qr), label, xy (map), text, owner
        for k, s in enumerate(spec.get('survivors') or []):
            self.things.append({'kind': 'object', 'label': 'survivor', 'xy': at(s), 'id': f'survivor{k}'})
            if s.get('qr'):
                self.things.append({'kind': 'qr', 'label': 'qr', 'xy': at(s, 'qr_offset_m'), 'text': str(s['qr']),
                                    'id': f'qr{k}'})
        for k, o in enumerate(spec.get('objects') or []):
            self.things.append({'kind': o.get('type', 'object'), 'label': o['label'], 'xy': at(o), 'id': f'obj{k}'})
        props = spec.get('props', 'auto')
        if props == 'auto':
            props = [{'tile': list(t), 'label': SCENE_PROP.get(n, 'cart')} for t, n in self.maze.scene_types.items()]
        for k, o in enumerate(list(props or []) + list(spec.get('clutter_props') or [])):
            self.things.append({'kind': 'prop', 'label': o.get('label', 'cart'), 'xy': at(o), 'id': f'prop{k}'})

        self.ev_pub = self.create_publisher(String, '/perception/event', 10)
        self.det_pub = self.create_publisher(String, '/perception/detections', 10)
        self.qrv_pub = self.create_publisher(String, '/qr/visible', 10)
        self.qrd_pub = self.create_publisher(String, '/qr/detections', 10)
        self.qrt_pub = self.create_publisher(String, '/perception/qr_text', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_event = {}
        self.qr_decoded = set()
        self.last_qr_seen = 0.0
        self.create_timer(1.0 / p('rate_hz', 5.0), self._tick)
        for t in self.things:
            mx, my = W.compose(W.invert(self.map_T_maze), (t['xy'][0], t['xy'][1], 0.0))[:2]
            self.get_logger().info(f"{t['kind']:7} {t['label']:14} maze ({mx:.2f}, {my:.2f}) map "
                                   f"({t['xy'][0]:.2f}, {t['xy'][1]:.2f}) {t.get('text', '')}")

    def _pose(self):
        try:
            tr = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:  # noqa: BLE001
            return None
        q = tr.transform.rotation
        return (tr.transform.translation.x, tr.transform.translation.y,
                math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def _view(self, pose, xy):
        """(range, bearing) from the camera if the spot is in view and in line of sight, else None."""
        cx, cy = pose[0] + self.cam_x * math.cos(pose[2]), pose[1] + self.cam_x * math.sin(pose[2])
        dx, dy = xy[0] - cx, xy[1] - cy
        rng = math.hypot(dx, dy)
        bearing = W.wrap(math.atan2(dy, dx) - pose[2])
        if rng > self.range_m or rng < 0.2 or abs(bearing) > self.hfov / 2:
            return None
        hit, _ = self.geom.raycast(cx, cy, [math.atan2(dy, dx)], rng - 0.03)
        if np.isfinite(hit[0]):
            return None                                  # a wall / panel / curtain in between
        return rng, bearing

    def _bbox(self, rng, bearing, h_m=0.35):
        u = self.cx - self.fx * math.tan(bearing)
        h = min(self.img_h, self.fx * h_m / max(rng, 0.1))
        w = max(10.0, 0.6 * h)
        v0 = self.img_h / 2 - h / 2
        return [int(max(0, u - w / 2)), int(max(0, v0)), int(min(self.img_w, u + w / 2)), int(min(self.img_h, v0 + h))]

    def _tick(self):
        pose = self._pose()
        if pose is None:
            return
        now = time.time()
        st = self.get_clock().now().to_msg()
        robot = {'x': round(pose[0], 2), 'y': round(pose[1], 2), 'yaw_deg': round(math.degrees(pose[2]), 1),
                 'frame': self.map_frame}
        dets, codes = [], []
        for t in self.things:
            v = self._view(pose, t['xy'])
            if v is None:
                continue
            rng, bearing = v
            if t['kind'] == 'qr':
                if rng <= self.qr_range_m and abs(bearing) <= self.qr_bearing:
                    self.last_qr_seen = now
                    first = t['id'] not in self.qr_decoded
                    mp = [round(t['xy'][0], 3), round(t['xy'][1], 3), 0.0]
                    codes.append({'camera': 'front', 'text': t['text'], 'decoded_ok': True, 'method': 'sim',
                                  'distance_m': round(rng, 2), 'bearing_deg': round(math.degrees(bearing), 1),
                                  'map_point': mp, 'already_reported': not first})
                    if first:
                        self.qr_decoded.add(t['id'])
                        self.qrd_pub.publish(String(data=json.dumps(
                            {'text': t['text'], 'decoded': t['text'], 'camera': 'front', 'map_point': mp,
                             'frame': self.map_frame, 'distance_m': round(rng, 2), 'first_seen_unix': now})))
                        self.qrt_pub.publish(String(data=t['text']))
                        self.get_logger().info(f"QR decoded: {t['text']!r} at {rng:.2f} m")
                continue
            bbox = self._bbox(rng, bearing, self.survivor_h_m if t['label'] == 'survivor' else 0.25)
            dets.append({'label': t['label'], 'confidence': 0.8, 'bbox_xyxy': bbox})
            if t['kind'] == 'prop' or now - self.last_event.get(t['id'], 0.0) < self.event_repeat_s:
                continue
            self.last_event[t['id']] = now
            loc = {'bearing_deg': round(math.degrees(bearing), 1), 'range_m': round(rng, 2),
                   'xy': [round(t['xy'][0], 2), round(t['xy'][1], 2)],
                   'position_map': {'x': round(t['xy'][0], 2), 'y': round(t['xy'][1], 2), 'z': 0.0,
                                    'frame': self.map_frame}}
            ev = {'event_id': uuid.uuid4().hex[:8], 'type': t['kind'], 'stamp': {'sec': st.sec, 'nanosec': st.nanosec},
                  'label': t['label'], 'confidence': 0.8, 'bbox_xyxy': bbox, 'image_size': [self.img_w, self.img_h],
                  'location': loc, 'robot_pose': robot, 'simulated': True}
            if t['label'] == 'survivor':
                ev['qr_in_view'] = now - self.last_qr_seen < 3.0
                ev['bbox_h_frac'] = round((bbox[3] - bbox[1]) / self.img_h, 3)
            if t['kind'] == 'hazard':
                ev['hazard_certain'] = True
            self.ev_pub.publish(String(data=json.dumps(ev)))
            self.get_logger().info(f"EVENT {t['kind']}: {t['label']} range {rng:.2f} m bearing "
                                   f"{math.degrees(bearing):+.0f} deg")
        self.qrv_pub.publish(String(data=json.dumps({'t': now, 'codes': codes})))
        self.det_pub.publish(String(data=json.dumps({'detections': dets})))


def main(args=None):
    rclpy.init(args=args)
    node = SimPerception()
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
