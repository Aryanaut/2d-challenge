"""Tune the local classifier thresholds on your real camera.

  # live from the Logitech webcam (no ROS needed); press q to quit
  ros2 run surface_monitor surface_tune --camera /dev/video0

  # on saved photos / frames
  ros2 run surface_monitor surface_tune frames/*.jpg

  # also ask the VLM about one image (checks group ID, client.py path and prompt)
  ros2 run surface_monitor surface_tune img.jpg --vlm --group group07 \
      --client-script ~/scene_understanding/client.py

Hold the camera at its real mounting height over each surface (flat wood, rubble blocks,
wooden steps, metal stairs, lab floor) and note the printed numbers. Set the th_* values in
config/surface_monitor.yaml so each surface lands on the right side of the thresholds.
"""

import argparse
import glob
import os
import sys

import cv2

from .local_classifier import Thresholds, classify_image


def _line(label, f):
    return (f"{label:8s} L={f['L']:6.1f} Lstd={f['Lstd']:5.1f} a={f['a']:5.1f} b={f['b']:5.1f} "
            f"chroma={f['chroma']:5.1f} grad={f['grad']:6.1f} bright={f['bright']:.3f}")


def _default_prompt():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("surface_monitor"), "prompts", "surface_prompt.txt")
    except Exception:
        return os.path.join(os.path.dirname(__file__), "..", "prompts", "surface_prompt.txt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files or glob patterns")
    ap.add_argument("--camera", help="video device, e.g. /dev/video0 or a /dev/v4l/by-id path")
    ap.add_argument("--vlm", action="store_true", help="also query the VLM for each image file")
    ap.add_argument("--server", default="http://192.168.10.141:8000")
    ap.add_argument("--group", default="groupXX")
    ap.add_argument("--client-script", default=os.path.expanduser("~/scene_understanding/client.py"))
    ap.add_argument("--client-python", default=sys.executable)
    ap.add_argument("--prompt-file", default=_default_prompt())
    ap.add_argument("--model", help="also show the trained model's prediction (surface_model.joblib)")
    args = ap.parse_args(argv)
    th = Thresholds()
    model = None
    if args.model:
        from .model_classifier import ModelClassifier
        model = ModelClassifier(args.model)

    def model_str(img):
        if model is None:
            return ""
        lab, conf, _ = model.predict(img)
        return f" | model: {lab} ({conf:.2f})"

    if args.camera:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            sys.exit(f"cannot open {args.camera}")
        has_gui = bool(os.environ.get("DISPLAY"))
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            label, f = classify_image(frame, th)
            print(_line(label, f) + model_str(frame))
            if has_gui:
                cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("surface_tune", frame)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
            else:
                cv2.waitKey(100)
        return

    paths = [p for pat in args.images for p in sorted(glob.glob(pat))]
    if not paths:
        ap.error("give image files or --camera")
    client = None
    if args.vlm:
        from .vlm_client import GatewayClient
        client = GatewayClient(args.client_python, args.client_script, args.server, args.group,
                               args.prompt_file)
        for problem in client.config_problems():
            print(f"VLM config problem: {problem}")
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        label, f = classify_image(img, th)
        print(f"{os.path.basename(path)}: {_line(label, f)}{model_str(img)}")
        if client:
            r = client.query(img)
            print(f"    VLM: ok={r.ok} surface={r.surface} conf={r.confidence:.2f} "
                  f"({r.latency_s:.1f} s) {r.evidence or r.error}")


if __name__ == "__main__":
    main()
