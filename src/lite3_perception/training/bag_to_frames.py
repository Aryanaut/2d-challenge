#!/usr/bin/env python3
"""
Turn a ROS 2 bag of camera images into training frames (sharp, non-duplicate JPEGs).

Usage (on the robot, after `source ~/2d/install/setup.bash` or /opt/ros/<distro>/setup.bash):
    python3 bag_to_frames.py ~/bags/train_01 ~/train_frames
    python3 bag_to_frames.py ~/bags/train_01 ~/train_frames --fps 4 --topic /camera/color/image_raw/compressed

Then zip the output folder and upload it to the Colab training notebook.
Frames are named <bagname>_<n>.jpg so the notebook splits train/val per recording.
"""
import argparse
import os
import re

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image


def storage_id_of(bag_dir):
    meta = os.path.join(bag_dir, "metadata.yaml")
    if os.path.exists(meta) and "mcap" in open(meta).read():
        return "mcap"
    return "sqlite3"


def to_bgr(msg, is_compressed):
    if is_compressed:
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    buf = np.frombuffer(msg.data, np.uint8)
    if msg.encoding in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.step // 3, 3)[:, : msg.width]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
    if msg.encoding in ("yuv422", "yuyv", "yuv422_yuy2"):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUY2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--topic", default="/camera/color/image_raw/compressed")
    ap.add_argument("--fps", type=float, default=4.0, help="frames per second to sample")
    ap.add_argument("--blur_pct", type=float, default=25, help="drop the blurriest N%% of frames")
    ap.add_argument("--dup_thresh", type=float, default=6.0, help="drop frames this similar to the previous kept one")
    a = ap.parse_args()

    bag = os.path.expanduser(a.bag.rstrip("/"))
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]", "", os.path.basename(bag)) or "bag"

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage_id_of(bag)),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                            output_serialization_format="cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if a.topic not in types:
        raise SystemExit(f"topic {a.topic} not in bag. Topics: {list(types)}")
    is_compressed = types[a.topic].endswith("CompressedImage")
    msg_type = CompressedImage if is_compressed else Image
    reader.set_filter(rosbag2_py.StorageFilter(topics=[a.topic]))

    period_ns = int(1e9 / a.fps)
    next_t = None
    saved = []                                  # (path, sharpness, small gray)
    n = 0
    while reader.has_next():
        _, data, t = reader.read_next()
        if next_t is not None and t < next_t:
            continue
        next_t = t + period_ns
        img = to_bgr(deserialize_message(data, msg_type), is_compressed)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(cv2.resize(g, (640, 360)), cv2.CV_64F).var()
        path = os.path.join(out, f"{tag}_{n:05d}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append((path, sharp, cv2.resize(g, (64, 36)).astype(np.float32)))
        n += 1

    if not saved:
        raise SystemExit("no frames read")
    thr = np.percentile([s for _, s, _ in saved], a.blur_pct)
    kept, last = 0, None
    for path, sharp, small in saved:
        if sharp < thr or (last is not None and np.abs(small - last).mean() < a.dup_thresh):
            os.remove(path)
            continue
        kept += 1
        last = small
    print(f"{len(saved)} sampled -> {kept} kept in {out}")


if __name__ == "__main__":
    main()
