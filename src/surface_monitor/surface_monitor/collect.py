"""Record labelled training frames straight from the webcam (no ROS needed).

Interactive (needs a display, e.g. over Remote Desktop):
  ros2 run surface_monitor surface_collect --camera /dev/v4l/by-id/<webcam> --session run1
     w = record wood   m = record metal   o = record other   space = pause   q = quit

Headless (over SSH): record one surface for a fixed time
  ros2 run surface_monitor surface_collect --camera <dev> --label metal --session stairs1 --seconds 60

Frames go to data/<label>/<session>/ every --every-s seconds while recording.
Move the camera/robot around while recording: different spots, angles, light, some shaking.
"""

import argparse
import os
import sys
import time

import cv2

KEYS = {ord("w"): "wood", ord("m"): "metal", ord("o"): "other"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--session", default=time.strftime("s%m%d_%H%M"))
    ap.add_argument("--label", choices=["wood", "metal", "other"], help="headless mode: record this label")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--every-s", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.camera}")

    label = args.label
    headless = label is not None or not os.environ.get("DISPLAY")
    if headless and label is None:
        sys.exit("no display: give --label wood|metal|other for headless recording")
    t_end = time.time() + args.seconds if headless else None
    last_save, counts = 0.0, {}

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        if label and now - last_save >= args.every_s:
            d = os.path.join(args.out, label, args.session)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{int(now * 1000)}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[label] = counts.get(label, 0) + 1
            last_save = now
        if headless:
            if now >= t_end:
                break
            continue
        view = frame.copy()
        cv2.putText(view, f"REC {label}" if label else "paused (w/m/o to record)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if label else (200, 200, 200), 2)
        cv2.putText(view, str(counts), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("surface_collect", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            label = None
        elif k in KEYS:
            label = KEYS[k]
    print("Saved frames:", counts, "->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
