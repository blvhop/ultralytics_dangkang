import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from voc_xml import save_voc_xml

model = YOLO('yolov8x-seg.pt')

# yolov8识别鸟类轮廓并且生成掩码图
subfolders = []
for root, dirs, files in os.walk(top="/root/blvhop/dggs_original/", topdown=True, onerror=None, followlinks=True):
    for d in dirs:
        subfolders.append(os.path.join(root, d))

subfolders = ['/root/blvhop/dggs_original/']

for subfolder in tqdm(subfolders):
    try:
        # classes: 14: 'bird'
        # results = model.predict(source=subfolder, classes=[14], iou=0.45, conf=0.25, verbose=False, save=False, stream=False)
        results = model.predict(source=subfolder, iou=0.45, conf=0.25, verbose=False, save=True, exist_ok=True, project=f'runs/seg/', stream=False)
    except FileNotFoundError:
        continue

    for result in results:
        # Segmentation
        # result.masks.data  # masks, (N, H, W)
        # result.masks.xy  # x,y segments (pixels), List[segment] * N
        # result.masks.xyn  # x,y segments (normalized), List[segment] * N

        if result is None or result.masks is None:
            continue

        # 封装预测结果
        # 将结果存为224*224分辨率的图像，鸟形状填充为黑色，背景填充为白色
        for i, points in enumerate(result.masks.xyn):
            points *= 224
            points = points.astype('int')
            # opencv 创建空白图像（使用h w c和数据类型，创建一个空白图像）
            img = np.zeros((224, 224, 3), np.uint8)
            img.fill(255)  # 填充白色背景
            # 多边形轮廓，内容填充黑色
            img = cv2.fillConvexPoly(img, points, (0, 0, 0))
            seg_img_path = Path(result.path.replace('_original', '') + '_' + str(i) + '.jpg')
            seg_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(seg_img_path)
