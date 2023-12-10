import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from blvhop.dataset_common import save_voc_xml

model_path = '/root/blvhop/model/model_qn_det_20231209.pt'
model = YOLO('yolov8x.pt')


# yolov8识别鸟类并生成voc xml标签
subfolders = []
TOP = '/data/blvhop/vector_index/dataset/00采集数据-A原图/群鸟-东方白鹳-数据集/'
subfolders.append(TOP)
for root, dirs, files in os.walk(top=TOP, topdown=True, onerror=None, followlinks=True):
    for d in dirs:
        subfolders.append(os.path.join(root, d))

for subfolder in tqdm(subfolders):
    try:
        # classes: 0: person / 1: bicycle / 2: car / 14: bird
        # results = model.predict(source=subfolder, task='detect', classes=[14], iou=0.45, conf=0.25, verbose=True, save=False, stream=False)
        results = model.predict(source=subfolder, task='detect', iou=0.45, conf=0.25, verbose=True, save=False, stream=False)
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
            # 标签名修改为目录名
            # detection['name'] = os.path.split(subfolder)[-1]
            # 标签名修改为中文名
            name = model.names[boxes.cls.cpu().numpy().astype(int)[i]]
            if name == 'person':
                detection['name'] = '人'
            elif name == 'bird':
                detection['name'] = '鸟'
            else:
                detection['name'] = name
            detection['confidence'] = boxes.conf[i]
            detection['bndbox'] = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            detections.append(detection)
        # 保存为xml
        save_voc_xml(result.path, Path(result.path).with_suffix(".xml"), detections)