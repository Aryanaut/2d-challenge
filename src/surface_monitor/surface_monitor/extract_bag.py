"""Pull labelled training frames out of a ROS 2 bag of the down camera.

Easiest: record one short bag per surface, then extract each whole bag with its label:
  ros2 bag record -o wood_flat1 /down_camera/image_raw        (walk over flat wood, Ctrl+C)
  ros2 run surface_monitor surface_extract wood_flat1 --label wood

Or label time ranges (seconds from the bag start) of one long run:
  ros2 run surface_monitor surface_extract run1 --label wood  --start 0  --end 25
  ros2 run surface_monitor surface_extract run1 --label metal --start 31 --end 48
  (skip a second or two around each transition - those frames are half one surface, half another)

Frames go to data/<label>/<session>/ (session defaults to the bag name).
"""

import argparse
import os
import sys

import cv2
import numpy as np


def image_to_bgr(msg):
    """sensor_msgs/Image or CompressedImage -> BGR numpy (no cv_bridge needed)."""
    if hasattr(msg, "format"):  # CompressedImage
        return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    h, w, step = msg.height, msg.width, msg.step
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        return img if enc == "bgr8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(buf.reshape(h, step)[:, :w], cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422_yuy2", "yuyv", "yuv422"):
        return cv2.cvtColor(buf.reshape(h, step)[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if enc in ("uyvy",):
        return cv2.cvtColor(buf.reshape(h, step)[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f"unsupported encoding {msg.encoding}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--label", required=True, choices=["wood", "metal", "other"])
    ap.add_argument("--topic", default="/down_camera/image_raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--session", help="default: bag folder name (+ time range)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds from bag start")
    ap.add_argument("--end", type=float, default=float("inf"))
    ap.add_argument("--every", type=int, default=3, help="keep every Nth frame")
    args = ap.parse_args(argv)

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        sys.exit("needs a sourced ROS 2 environment (rosbag2_py)")

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=""),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in types:
        sys.exit(f"topic {args.topic} not in bag. Topics: {sorted(types)}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
    msg_type = get_message(types[args.topic])

    session = args.session or os.path.basename(os.path.normpath(args.bag)) + (
        f"_{args.start:g}-{args.end:g}" if args.start > 0 or args.end != float("inf") else "")
    out_dir = os.path.join(args.out, args.label, session)
    os.makedirs(out_dir, exist_ok=True)

    t0, n, saved = None, 0, 0
    while reader.has_next():
        _, data, t_ns = reader.read_next()
        t0 = t_ns if t0 is None else t0
        t = (t_ns - t0) * 1e-9
        if t < args.start:
            continue
        if t > args.end:
            break
        n += 1
        if (n - 1) % args.every:
            continue
        bgr = image_to_bgr(deserialize_message(data, msg_type))
        if bgr is not None:
            cv2.imwrite(os.path.join(out_dir, f"{t_ns}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
