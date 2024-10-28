import copy
import os
import shutil

import numpy as np
import cv2

from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

model_path = 'model_lottery_20231228.onnx'
model = YOLO(model=model_path)

NEW_W, NEW_H = (640, 640)

TOP_O = '预测原始文件'
OUT_M = '预测中奖文件'
OUT_P = '预测提示文件'
os.makedirs(TOP_O, exist_ok=True)
os.makedirs(OUT_M, exist_ok=True)
os.makedirs(OUT_P, exist_ok=True)
impath_list = []
for root, dirs, files in os.walk(top=TOP_O, topdown=True, onerror=None, followlinks=True):
    for f in files:
        if os.path.splitext(f)[1][1:].lower() in IMG_FORMATS:
            impath_list.append(os.path.join(root, f))
impath_list.sort()

for impath in tqdm(impath_list, desc="图片预测"):
    try:
        im_cv2 = cv2.imdecode(np.fromfile(impath, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = im_cv2.shape[:2]
        if h < w:
            im_cv2_r = cv2.rotate(im_cv2, cv2.ROTATE_90_CLOCKWISE)
        else:
            im_cv2_r = im_cv2
        ratio = min(NEW_H / h, NEW_W / w)
        im_resize = cv2.resize(im_cv2_r, (640, 640), interpolation=cv2.INTER_AREA if ratio <= 1.0 else cv2.INTER_CUBIC)
        results = model.predict(source=im_resize, task='detect', iou=0.45, conf=0.86, half=True, verbose=False, save=False, stream=False)
        result = results[0]
        boxes = result.boxes

        if h < w:
            xyxys = boxes.xyxy.cpu().numpy()
            xyxys[:, [0, 2]] = xyxys[:, [0, 2]] * h / NEW_W
            xyxys[:, [1, 3]] = xyxys[:, [1, 3]] * w / NEW_H
            X1 = xyxys[:, 1]
            Y1 = h - xyxys[:, 2]
            X2 = xyxys[:, 3]
            Y2 = h - xyxys[:, 0]
            xyxys[:, 0] = X1
            xyxys[:, 1] = Y1
            xyxys[:, 2] = X2
            xyxys[:, 3] = Y2
        else:
            xyxys = boxes.xyxy.cpu().numpy()
            xyxys[:, [0, 2]] = xyxys[:, [0, 2]] * w / NEW_W
            xyxys[:, [1, 3]] = xyxys[:, [1, 3]] * h / NEW_H

        names = set()
        duplicate_names = set()
        t_xyxy = None
        t_inner_names = []
        t_inner_xyxys = []
        t_match_names = []
        t_match_xyxys = []
        number_detections = []

        for i, cls in enumerate(boxes.cls.tolist()):
            name = model.predictor.model.names[boxes.cls.cpu().numpy().astype(int)[i]]
            conf = boxes.conf[i]
            xyxy = xyxys[i].astype(int).tolist()

            if name in names:
                duplicate_names.add(name)
            else:
                names.add(name)

            if 'hysb' == name:
                continue
            if 't' == name:
                t_xyxy = copy.copy(xyxy)
                continue

            detection = {}
            detection['name'] = name
            detection['confidence'] = conf
            detection['bndbox'] = xyxy
            number_detections.append(detection)

        if t_xyxy is not None:
            for detection in number_detections:
                name = detection['name']
                xmin, ymin, xmax, ymax = detection['bndbox']
                if xmin > t_xyxy[0] and ymin > t_xyxy[1] and xmax < t_xyxy[2] and ymax < t_xyxy[3]:
                    t_inner_names.append(name)
                    t_inner_xyxys.append([xmin, ymin, xmax, ymax])

        for detection in number_detections:
            name = detection['name']
            xmin, ymin, xmax, ymax = detection['bndbox']
            if name in t_inner_names:
                if xmin > t_xyxy[0] and ymin > t_xyxy[1] and xmax < t_xyxy[2] and ymax < t_xyxy[3]:
                    pass
                else:
                    t_match_names.append(name)
                    t_match_xyxys.append([xmin, ymin, xmax, ymax])

        need_manual = False
        if t_xyxy is None \
                or len(t_inner_names) != 1 \
                or len(number_detections) != 53 \
                or (len(t_match_names) == 0 and len(duplicate_names) > 0):
            need_manual = True

        if need_manual:
            p_impath = OUT_P + impath[len(TOP_O):]
            os.makedirs(os.path.split(p_impath)[0], exist_ok=True)
            shutil.copy2(impath, p_impath)
            continue

        if len(t_match_names) == 0:
            continue

        for x1, y1, x2, y2 in t_match_xyxys:
            cv2.rectangle(im_cv2, (x1, y1), (x2, y2), (0, 255, 0), 3)
        m_impath = OUT_M + impath[len(TOP_O):]
        os.makedirs(os.path.split(m_impath)[0], exist_ok=True)
        cv2.imwrite(m_impath, im_cv2)

    except Exception as e:
        print("ERROR", impath, e)
    continue
