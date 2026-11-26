"""Find the USB cameras and tell left from right (no ROS needed).

  ros2 run usb_cameras probe_cameras            # list devices, grab one photo from each
  ros2 run usb_cameras probe_cameras --both /dev/video0 /dev/video2   # can both stream at once?

For each /dev/video* it prints whether it gives pictures, the resolution, the stable
/dev/v4l/by-path name, and saves ~/cam_probe/<device>.jpg - open those to see which camera
looks left and which looks right (or hold a hand in front of one camera while probing).
"""

import argparse
import os
import time

import cv2

from .capture import CameraThread, list_video_devices, open_camera


def probe(out_dir, width, height):
    os.makedirs(out_dir, exist_ok=True)
    devices = list_video_devices()
    if not devices:
        print("No /dev/video* devices on this computer.\n"
              "If the webcams are plugged into the robot, they have not been passed through to this "
              "computer yet (ask the instructor), or try other USB ports.")
        return
    print(f"{'device':14s} {'pictures?':10s} {'size':10s} stable name / photo")
    for dev, links in devices:
        cap = open_camera(dev, width, height)
        if cap is None:
            print(f"{dev:14s} {'no':10s} {'-':10s} (busy or not a camera)")
            continue
        frame = None
        t0 = time.time()
        while time.time() - t0 < 3.0:            # first frames are often black: keep a later one
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                if time.time() - t0 > 1.0:
                    break
        cap.release()
        if frame is None:
            print(f"{dev:14s} {'no':10s} {'-':10s} (metadata node - normal: each webcam has two)")
            continue
        name = os.path.basename(dev)
        path = os.path.join(out_dir, f"{name}.jpg")
        cv2.imwrite(path, frame)
        by_path = [l for l in links if "/by-path/" in l]
        print(f"{dev:14s} {'YES':10s} {frame.shape[1]}x{frame.shape[0]:<5d} "
              f"{by_path[0] if by_path else '(no by-path name)'}\n{'':37s}photo: {path}")
    print("\nUse the devices marked YES. Prefer the /dev/v4l/by-path/... names: they stay the same per USB port.")


def both(devs, width, height, seconds=5.0):
    """Stream several cameras at once - checks the USB bus can carry them together."""
    threads = [CameraThread(f"cam{i}", d, width, height, 15, True, log=print) for i, d in enumerate(devs)]
    for t in threads:
        t.start()
    time.sleep(seconds)
    for t in threads:
        t.stop()
    for t in threads:
        fps = t.frames / seconds
        verdict = "OK" if fps >= 5 else "TOO FEW FRAMES (try --width 320 --height 240, or another USB port)"
        print(f"{t.device}: {t.frames} frames in {seconds:.0f} s = {fps:.1f} fps  {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/cam_probe"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--both", nargs="+", metavar="DEVICE", help="stream these devices at the same time")
    args = ap.parse_args(argv)
    if args.both:
        both(args.both, args.width, args.height)
    else:
        probe(args.out, args.width, args.height)


if __name__ == "__main__":
    main()
