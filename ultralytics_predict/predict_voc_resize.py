import os
import time

import numpy as np
import cv2
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from blvhop.dataset import save_voc_xml, draw_detections

model_path = '/data/blvhop/ultralytics/yolov8x.pt'
model = YOLO(model=model_path)

NEW_W, NEW_H = (640, 640)  # 缩放到的目标尺寸

# yolov8识别并生成voc xml标签
TOP = '/data/blvhop/vector_index/dataset/00采集数据-A原图/test'
impath_list = []
for root, dirs, files in os.walk(top=TOP, topdown=True, onerror=None, followlinks=True):
    for f in files:
        if os.path.splitext(f)[1][1:].lower() in IMG_FORMATS:
            impath_list.append(os.path.join(root, f))
impath_list.sort()
draw_path = Path(f'{TOP}_draw_out_{time.strftime("%Y%m%d_%H%M%S")}')

for impath in tqdm(impath_list, desc="图片预测"):
    try:
        # opencv打开图像
        im_cv2 = cv2.imdecode(np.fromfile(impath, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = im_cv2.shape[:2]  # 图片的真实尺寸 shape [height, width]

        ### 旋转-part1（仅横向图片操作）
        # 若图像是横向图像，则顺时针旋转90°变成总想图像，对带有方向性的数据很有用，只用标注正向图像，然后增强旋转180°，不用额外去标注横向的图像了。
        if h < w:
            # 若图像height < width，则图像顺时针旋转90°
            im_cv2 = cv2.rotate(im_cv2, cv2.ROTATE_90_CLOCKWISE)
        ###

        ### 缩放-part1
        # 图片缩放到640x640再预测，预测完后，还得将结果还原
        # 计算缩放比 Scale ratio (new / old)
        ratio = min(NEW_H / h, NEW_W / w)  # 是否旋转了不影响ratio计算
        im_resize = cv2.resize(im_cv2, (640, 640), interpolation=cv2.INTER_AREA if ratio <= 1.0 else cv2.INTER_CUBIC)
        ###

        # 预测。传入的是图片数据，save参数就没有用了，此时保存的文件名始终为image0.jpg
        results = model.predict(source=im_resize, task='detect', iou=0.45, conf=0.25, verbose=False, save=False, stream=False)

        result = results[0]
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs

        ### 旋转-part2（仅横向图片操作）
        if h < w:
            # 经过旋转过的图像先缩放还原，然后旋转还原，注意此时的缩放还原宽高不仅缩放的图像宽高不一致
            # 预测结果缩放还原，因为旋转了90°，所以h和w在公式中互换
            xyxys = boxes.xyxy.cpu().numpy()
            xyxys[:, [0, 2]] = xyxys[:, [0, 2]] * h / NEW_W
            xyxys[:, [1, 3]] = xyxys[:, [1, 3]] * w / NEW_H
            # 预测结果旋转还原
            # 草稿纸上画一下，可知新的坐标X1=y1,Y1=h-x2; X2=y2, Y2=h-x1
            X1 = xyxys[:, 1]
            Y1 = h - xyxys[:, 2]
            X2 = xyxys[:, 3]
            Y2 = h - xyxys[:, 0]
            xyxys[:, 0] = X1
            xyxys[:, 1] = Y1
            xyxys[:, 2] = X2
            xyxys[:, 3] = Y2
        else:
            # 没有旋转过的图像仅缩放还原即可
            ### 缩放-part2
            # 预测结果坐标缩放还原
            xyxys = boxes.xyxy.cpu().numpy()
            xyxys[:, [0, 2]] = xyxys[:, [0, 2]] * w / NEW_W
            xyxys[:, [1, 3]] = xyxys[:, [1, 3]] * h / NEW_H
            ###
        ###

        # 封装预测结果
        detections = []
        for i, cls in enumerate(boxes.cls.tolist()):
            detection = {}
            detection['name'] = model.names[boxes.cls.cpu().numpy().astype(int)[i]]
            detection['confidence'] = boxes.conf[i]
            detection['bndbox'] = xyxys[i].astype(int).tolist()
            detections.append(detection)
        # 保存为xml
        save_voc_xml(impath, Path(impath).with_suffix(".xml"), detections)

        # 保存预测绘制图片
        draw_file = draw_path / impath[len(TOP) + 1:]  # 生成图片保存路径，注意路径合成时，后面的路径第一个字符不能是/，否则只会以后面的路径作为合成路径名
        if not draw_file.parent.exists():
            os.makedirs(draw_file.parent)
        im = Image.open(impath)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        draw_detections(im, detections, [0.86, 0.50], True)
        im.save(draw_file, format='JPEG')

    except Exception as e:
        print("ERROR", impath, e)
    continue
