import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from voc_xml import save_voc_xml

model = YOLO('yolov8x.pt')

subfolders = []
for root, dirs, files in os.walk(top="/mnt/hbw/鹤形目/", topdown=True, onerror=None, followlinks=True):
    for d in dirs:
        subfolders.append(os.path.join(root, d))

for subfolder in tqdm(subfolders):
    try:
        # classes: 14: 'bird'
        results = model.predict(source=subfolder, classes=[14], iou=0.45, verbose=False, save=False, stream=False)
    except FileNotFoundError:
        continue

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs

        # 封装预测结果
        detections = []
        for i, cls in enumerate(boxes.cls.tolist()):
            detection = {}
            detection['name'] = os.path.split(subfolder)[-1]
            detection['confidence'] = boxes.conf[i]
            detection['bndbox'] = boxes.xyxy[0].cpu().numpy().astype(int).tolist()
            detections.append(detection)
        # 保存为xml
        save_voc_xml(result.path, Path(result.path).with_suffix(".xml"), detections)