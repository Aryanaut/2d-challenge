#!/usr/bin/env python3
"""Replay a rosbag through the explorer's scene-prop votes (lite3_explorer/prop_votes.py) and print the votes per tile.

    source install/setup.bash
    python3 scripts/propvote_replay.py bags/<name>/<name>_0.db3 [--min-frames 8] [--min-score 4.0]

Detections come from /perception/detections when the bag has it (explorer.sh --reporter); otherwise the detector model
is run on /camera/color/image_raw/compressed (bags recorded with --images). The robot pose is map->odom->base_link from
/tf, the map /map, the tile grid /tile_map: the same inputs the mission manager uses live. For every tile it prints the
sightings, when the rule (min frames + min score) was first met, when the lidar flagged it a scene and when the robot
first stood on it.
"""
import argparse
import ast
import bisect
import json
import math
import os
import sqlite3

import numpy as np
from rclpy.serialization import deserialize_message
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from lite3_explorer import prop_votes as PV
from lite3_explorer.map_grid import GridInfo

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'lite3_perception', 'models', 'detector.onnx')


def yaw_q(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('db3')
    ap.add_argument('--min-frames', type=int, default=8)
    ap.add_argument('--min-score', type=float, default=4.0)
    ap.add_argument('--max-range', type=float, default=3.0)
    ap.add_argument('--max-turn-dps', type=float, default=30.0)
    ap.add_argument('--step', type=int, default=2, help='images only: run the detector on every Nth frame')
    a = ap.parse_args()

    c = sqlite3.connect(a.db3)
    ids = {n: i for i, n in c.execute('select id,name from topics')}
    t0 = c.execute('select min(timestamp) from messages').fetchone()[0]

    def rows(topic):
        return c.execute('select timestamp,data from messages where topic_id=? order by timestamp', (ids[topic],))

    for need in ('/tf', '/map', '/tile_map'):
        if need not in ids:
            raise SystemExit(f'bag has no {need}')

    tf = {('map', 'odom'): ([], []), ('odom', 'base_link'): ([], [])}
    for ts, b in rows('/tf'):
        for t in deserialize_message(b, TFMessage).transforms:
            k = (t.header.frame_id, t.child_frame_id)
            if k in tf:
                tf[k][0].append(ts)
                tf[k][1].append((t.transform.translation.x, t.transform.translation.y, yaw_q(t.transform.rotation)))

    def look(k, ts):
        tt, v = tf[k]
        i = bisect.bisect_right(tt, ts) - 1
        return v[i] if i >= 0 else None

    def pose(ts):
        m, o = look(('map', 'odom'), ts), look(('odom', 'base_link'), ts)
        if not m or not o:
            return None
        return (m[0] + math.cos(m[2]) * o[0] - math.sin(m[2]) * o[1],
                m[1] + math.sin(m[2]) * o[0] + math.cos(m[2]) * o[1], m[2] + o[2])

    maps_t, maps = [], []
    for ts, b in rows('/map'):
        m = deserialize_message(b, OccupancyGrid)
        i = m.info
        maps_t.append(ts)
        maps.append((np.array(m.data, np.int16).reshape(i.height, i.width),
                     GridInfo(i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y)))
    tm_t, tm = [], []
    for ts, b in rows('/tile_map'):
        tm_t.append(ts)
        tm.append(json.loads(deserialize_message(b, String).data))
    fx, cx = 462.6, 320.1
    if '/camera/color/camera_info' in ids:
        for ts, b in rows('/camera/color/camera_info'):
            m = deserialize_message(b, CameraInfo)
            fx, cx = m.k[0], m.k[2]
            break

    # (bag time, detections) from the topic, or from the model on the images
    frames = []
    if '/perception/detections' in ids:
        src = '/perception/detections'
        for ts, b in rows(src):
            try:
                frames.append((ts, json.loads(deserialize_message(b, String).data).get('detections') or []))
            except ValueError:
                pass
    elif '/camera/color/image_raw/compressed' in ids:
        import cv2
        import onnxruntime as ort
        src = 'detector run on the images'
        sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        S = inp.shape[2] if isinstance(inp.shape[2], int) else 640
        names = ast.literal_eval(sess.get_modelmeta().custom_metadata_map['names'])
        for n, (ts, b) in enumerate(rows('/camera/color/image_raw/compressed')):
            if n % a.step:
                continue
            img = cv2.imdecode(np.frombuffer(bytes(deserialize_message(b, CompressedImage).data), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            r = min(S / h, S / w)
            nh, nw = round(h * r), round(w * r)
            top, left = (S - nh) // 2, (S - nw) // 2
            cv = np.full((S, S, 3), 114, np.uint8)
            cv[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
            p = sess.run(None, {inp.name: cv2.dnn.blobFromImage(cv, 1 / 255.0, swapRB=True)})[0][0]
            if p.shape[0] > p.shape[1]:
                p = p.T
            sc = p[4:].T
            cl, cf = sc.argmax(1), sc.max(1)
            k = cf >= 0.35
            dets = []
            if k.any():
                x1 = (p[0][k] - p[2][k] / 2 - left) / r
                y1 = (p[1][k] - p[3][k] / 2 - top) / r
                bw, bh = p[2][k] / r, p[3][k] / r
                rects = [[float(q) for q in v] for v in zip(x1, y1, bw, bh)]
                for i in np.array(cv2.dnn.NMSBoxes(rects, cf[k].tolist(), 0.35, 0.45)).flatten():
                    x, y, ww, hh = rects[i]
                    dets.append({'label': names[int(cl[k][i])], 'confidence': float(cf[k][i]),
                                 'bbox_xyxy': [x * 640 / w, y, (x + ww) * 640 / w, y + hh]})
            frames.append((ts, dets))
    else:
        raise SystemExit('bag has neither /perception/detections nor colour images')

    votes, met, last, skipped = {}, {}, None, 0
    for ts, dets in frames:
        dets = [d for d in dets if d.get('label') in PV.DEFAULT_PROPS]
        if not dets:
            continue
        p = pose(ts)
        mi, ti = bisect.bisect_right(maps_t, ts) - 1, bisect.bisect_right(tm_t, ts) - 1
        if p is None or mi < 0 or ti < 0:
            continue
        if last is not None and ts - last[0] < 2e9:
            d = math.atan2(math.sin(p[2] - last[1]), math.cos(p[2] - last[1]))
            if abs(math.degrees(d)) / max((ts - last[0]) / 1e9, 1e-3) > a.max_turn_dps:
                last = (ts, p[2])
                skipped += 1
                continue
        last = (ts, p[2])
        g = tm[ti]['grid']
        yw = math.radians(g['yaw_deg'])

        def tile_of(x, y):
            dx, dy = x - g['origin_xy'][0], y - g['origin_xy'][1]
            u, v = math.cos(yw) * dx + math.sin(yw) * dy, -math.sin(yw) * dx + math.cos(yw) * dy
            return f'{math.floor(u / g["size_m"])},{math.floor(v / g["size_m"])}'
        here = tile_of(p[0], p[1])
        grid, info = maps[mi]
        t = round((ts - t0) / 1e9, 1)
        for x, y, conf, lab, rng in PV.place(dets, p, grid, info, fx=fx, cx=cx, max_range_m=a.max_range):
            tile = tile_of(x, y)
            if tile == here:
                continue
            e = votes.setdefault(tile, {'frames': 0, 'score': 0.0, 'first_t': t, 'min_r': 9.0, 'labels': {}})
            e['frames'] += 1
            e['score'] += conf
            e['min_r'] = min(e['min_r'], rng)
            e['labels'][lab] = e['labels'].get(lab, 0) + 1
            if tile not in met and e['frames'] >= a.min_frames and e['score'] >= a.min_score:
                met[tile] = t

    flag, enter = {}, {}
    for ts, d in zip(tm_t, tm):
        t = round((ts - t0) / 1e9, 1)
        for k, v in d['tiles'].items():
            if v.get('scene') and k not in flag:
                flag[k] = t
        cur = d.get('current_tile')
        if cur and cur not in enter:
            enter[cur] = t

    print(f'source: {src}, {len(frames)} frames, {skipped} skipped while turning; rule: >= {a.min_frames} sightings '
          f'and score >= {a.min_score}\n')
    print(f'{"tile":6} {"sightings":>9} {"score":>6} {"first":>6} {"min_r":>5}  {"rule met":>8}  {"lidar flag":>10}  '
          f'{"entered":>7}  labels')
    for tile in sorted(set(votes) | set(flag), key=lambda k: tuple(map(int, k.split(',')))):
        e = votes.get(tile)
        if e:
            print(f'{tile:6} {e["frames"]:9} {e["score"]:6.1f} {e["first_t"]:6} {e["min_r"]:5.2f}  '
                  f'{str(met.get(tile, "-")):>8}  {str(flag.get(tile, "-")):>10}  {str(enter.get(tile, "-")):>7}  {e["labels"]}')
        else:
            print(f'{tile:6} {"-":>9} {"":6} {"":6} {"":5}  {"-":>8}  {str(flag.get(tile, "-")):>10}  '
                  f'{str(enter.get(tile, "-")):>7}')
    blocked = [k for k in met if k in enter and enter[k] > met[k]]
    if blocked:
        print(f'\nentered AFTER the rule was met (prop_votes would have kept them out): {", ".join(blocked)}')


if __name__ == '__main__':
    main()
