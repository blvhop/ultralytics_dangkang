import os
import re

from PIL import Image

# 设置环境变量
# 图像缩放到固定640x640
os.environ["xxx_blvhop_modify_resize"] = '1'
# 自定义小目标绘制
os.environ["xxx_blvhop_modify_pose_draw"] = '1'

from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from blvhop.dataset import save_voc_xml, draw_detections

model_path = '/data/blvhop/ultralytics/yolov8x.pt'
# model_path = '/data/blvhop/model/runs/train/nltk_20241225/weights/last.pt'
# model_path = '/root/blvhop/model/model_qn_det_20231209.pt'
# model_path = '/root/blvhop/model/runs/train/dzyqj_20240302/weights/best.pt'
# model_path = '/root/blvhop/model/runs/train/hwxj_det_20240109/weights/best.pt' # mjzy
# model_path = '/root/blvhop/model/runs/train/dltt_20240419/weights/best.pt'
# model_path = '/data/blvhop/model/runs/train/dltt_20240703/weights/last.onnx'
# model_path = '/data/blvhop/model/runs/train/nltk_20230826/weights/best.pt' # nltk

model = YOLO(model=model_path)

model_type = None
m = re.match('.*/train/(.*?)_.*\d{8}/weights.*', model_path)
if m is not None:
    model_type = m.group(1) # dzyqj / hwxj / jt / dltt
m = re.match('.*dzyqj.*', model_path)
if m is not None:
    model_type = 'dzyqj'

# yolov8识别鸟类并生成voc xml标签
subfolders = []
TOP = '/data/blvhop/nltkhzh'
TOP = '/data/blvhop/dzyqj/00010_0'
TOP = '/data/blvhop/mjzy'
TOP = '/mnt/W/dataset/00采集数据-00综合数据'
TOP = '/root/blvhop/vector_index/dataset/tmp4/ahmc'
TOP = '/root/blvhop/test/dltt/test'
TOP = '/root/blvhop/vector_index/dataset/vidio_split'
subfolders.append(TOP)
for root, dirs, files in os.walk(top=TOP, topdown=True, onerror=None, followlinks=True):
    for d in dirs:
        subfolders.append(os.path.join(root, d))

for subfolder in tqdm(subfolders):
    try:
        # classes: 0: person / 1: bicycle / 2: car / 14: bird
        # results = model.predict(source=subfolder, task='detect', classes=[14], iou=0.45, conf=0.25, verbose=True, save=False, stream=False)
        results = model.predict(source=subfolder, task='detect', half=True, iou=0.45, conf=0.25, verbose=True, save=False, stream=False)
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
            # if detection['name'] not in ('鸟',):
            #     continue
            detection['confidence'] = boxes.conf[i]
            detection['bndbox'] = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            if model_type == 'dzyqj':
                # 如果是电子元器件，向下和向右移动一个像素，不太清楚为什么训练的模型会往左上角去一个像素
                detection['bndbox'] = (boxes.xyxy[i].cpu().numpy().astype(int) + 1).tolist()
            detections.append(detection)
        # 保存为xml
        save_voc_xml(result.path, Path(result.path).with_suffix(".xml"), detections)
        # 保存图片
        if len(detections) > 0:
            im = Image.open(result.path)
            draw_detections(im, detections, [0.0, 0.0], False)
            filepath = Path('/data/blvhop/vector_index/dataset/xxxx_result_tmp' + result.path)
            os.makedirs(filepath.parent, exist_ok=True)
            im.save(filepath)
