# 东莞古树webapi
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from voc_xml import save_voc_xml

model = YOLO('yolov8l.pt')


# yolov8识别鸟类并生成voc xml标签
subfolders = []
for root, dirs, files in os.walk(top="/root/blvhop/hwxj/images/", topdown=True, onerror=None, followlinks=True):
    for d in dirs:
        subfolders.append(os.path.join(root, d))

for subfolder in tqdm(subfolders):
    try:
        # classes: 0: person 1: bicycle 2: car 3: motorcycle 4: airplane 5: bus 6: train
        results = model.predict(source=subfolder, classes=[0, 1, 2, 3, 4, 5, 6], iou=0.45, verbose=True, save=False, stream=False)
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
        # save_voc_xml(result.path, Path(result.path).with_suffix(".xml"), detections)