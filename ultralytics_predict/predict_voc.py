import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from voc_xml import save_voc_xml

model = YOLO('yolov8x.pt')


# yolov8识别鸟类并生成voc xml标签
subfolders = []
for root, dirs, files in os.walk(top="/root/blvhop/hwxj/images/", topdown=True, onerror=None, followlinks=True):
    for d in dirs:
        subfolders.append(os.path.join(root, d))

for subfolder in tqdm(subfolders):
    try:
        # classes: 14: 'bird'
        results = model.predict(source=subfolder, task='detect', classes=[14], iou=0.45, conf=0.25, verbose=True, save=False, stream=False)
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
            detection['bndbox'] = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            detections.append(detection)
        # 保存为xml
        # save_voc_xml(result.path, Path(result.path).with_suffix(".xml"), detections)