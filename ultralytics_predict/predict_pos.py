import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from voc_xml import save_voc_xml

model = YOLO('/root/blvhop/model/model_jcnl_pos_20230629.onnx')

# yolov8姿态关键点识别
results = model.predict(source='/data/blvhop/vector_index/dataset/00采集数据-A原图/test', task='pose', iou=0.45, conf=0.25, half=True, verbose=False, save=False, stream=False, boxes=False)

for result in results:
    # Keypoints object can be used index, manipulate and normalize coordinates. The keypoint conversion operation is cached.
    # keypoints = results[0].keypoints  # Masks object
    # keypoints.xy  # x, y keypoints (pixels), (num_dets, num_kpts, 2/3), the last dimension can be 2 or 3, depends the model.
    # keypoints.xyn  # x, y keypoints (normalized), (num_dets, num_kpts, 2/3)
    # keypoints.conf  # confidence score(num_dets, num_kpts) of each keypoint if the last dimension is 3.
    # keypoints.data  # raw keypoints tensor, (num_dets, num_kpts, 2/3)

    xy = result.keypoints.xy
    if xy.shape[0] == 0 or xy.shape[1] == 0:
        # 无数据
        continue

    # 保存预测结果
    img = cv2.imread(result.path)
    for i, points in enumerate(xy):
        if points.size()[0] == 0:
            # 无结果
            continue
        center = points.cpu().numpy().astype('int')[0]
        # 画出圆形标注
        img = cv2.circle(img, center, 20, (0, 10, 255), 2)
        # 写置信度
        img = cv2.putText(img, f'{result.boxes.conf.cpu().numpy()[i]:.2f}', (center[0]+20, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
    img_path = Path(result.path.replace('test', 'test.pos'))
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(img_path)
