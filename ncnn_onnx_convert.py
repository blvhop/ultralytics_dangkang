import sys
sys.path.insert(0, '.')
print(sys.path)

from ultralytics import YOLO

model = YOLO('nltk.pt')
model.export(format='onnx', imgsz=640, half=True, opset=12, optimize=True)
