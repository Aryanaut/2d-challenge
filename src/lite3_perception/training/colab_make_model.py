# =====================================================================
#  Make detector.onnx in Google Colab (free GPU), then copy it to the robot.
#  colab.research.google.com -> New notebook -> Runtime > Change runtime type > T4 GPU
#  Paste each "CELL" block into its own cell and run top to bottom.
#  The robot never needs PyTorch: it only runs the exported .onnx file.
# =====================================================================

# ---------------- CELL 1: install ----------------
# !pip install -q ultralytics onnx onnxslim

# =====================================================================
#  OPTION A (today, no labelling): YOLO-World with your own class words,
#  frozen into a normal ONNX model. Good starting point.
# =====================================================================

# ---------------- CELL 2A ----------------
from ultralytics import YOLO

CLASSES = [
    "person", "toy figure", "doll",                          # survivors
    "first aid kit", "red cross sign", "medicine box",       # medical room
    "electrical panel", "warning sign", "fire extinguisher", # hazard
    "stove", "refrigerator", "sink", "pot", "pan", "knife", "kettle", "microwave",  # kitchen
    "bed", "sofa", "toilet", "qr code",
]
model = YOLO("yolov8s-worldv2.pt")
model.set_classes(CLASSES)
model.save("world_custom.pt")
YOLO("world_custom.pt").export(format="onnx", imgsz=480, opset=12, simplify=True)
# -> world_custom.onnx  (rename to detector.onnx)

# =====================================================================
#  OPTION B (better, once you have photos of the real props):
#  train a small YOLO on your own labelled images.
#   1. Take ~30-80 photos per object (phone or robot camera, different angles,
#      distances, lighting) in the practice arena.
#   2. Upload to roboflow.com (free), draw boxes, generate a dataset,
#      Export -> "YOLOv8" format -> "show download code".
#      (Roboflow Universe also has public kitchen / first-aid / hazard-sign sets
#       you can merge in.)
# =====================================================================

# ---------------- CELL 2B ----------------
# !pip install -q roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_KEY")
# ds = rf.workspace("YOUR_WS").project("YOUR_PROJECT").version(1).download("yolov8")
#
# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")                  # n = fastest on the robot CPU; try "s" if accuracy is poor
# model.train(data=f"{ds.location}/data.yaml", epochs=80, imgsz=480, batch=16)
# best = YOLO("runs/detect/train/weights/best.pt")
# print(best.val())                           # check mAP before trusting it
# best.export(format="onnx", imgsz=480, opset=12, simplify=True)
# # -> runs/detect/train/weights/best.onnx  (rename to detector.onnx)

# ---------------- CELL 3: quick sanity test of the ONNX file ----------------
# !pip install -q onnxruntime
# import onnxruntime as ort
# s = ort.InferenceSession("world_custom.onnx")
# print(s.get_inputs()[0].shape, s.get_outputs()[0].shape)
# print(s.get_modelmeta().custom_metadata_map.get("names"))   # class names the robot will use

# ---------------- CELL 4: download ----------------
# from google.colab import files
# files.download("world_custom.onnx")         # or best.onnx
