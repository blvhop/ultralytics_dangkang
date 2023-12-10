import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from voc_xml import save_voc_xml

model_path = '/root/blvhop/model/model_qn_pose_20231209.pt'
model = YOLO(model_path)

# yolov8姿态关键点识别
# 不画框，只打点
results = model.predict(source='/data/blvhop/vector_index/dataset/00采集数据-A原图/群鸟-东方白鹳', iou=0.45, verbose=False, save=True, stream=False, boxes=False)