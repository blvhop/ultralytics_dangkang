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
from blvhop.dataset_common import save_voc_xml, draw_detections_optimal

model_path = '/root/blvhop/model/runs/train/dl_20240918/weights/last.onnx'

model = YOLO(model=model_path)

model_type = None
m = re.match('.*/train/(.*?)_.*\d{8}/weights.*', model_path)
if m is not None:
    model_type = m.group(1) # dzyqj / hwxj / dl
m = re.match('.*dzyqj.*', model_path)
if m is not None:
    model_type = 'dzyqj'

# yolov8识别鸟类并生成voc xml标签
subfolders = []
TOP = '/data/blvhop/nltkhzh'
TOP = '/data/blvhop/dzyqj/00010_0'
TOP = '/data/blvhop/mjzy'
TOP = '/mnt/W/dataset/00采集数据-00综合数据'
TOP = '/root/blvhop/vector_index/dataset/vidio_split'
subfolders.append(TOP)
for root, dirs, files in os.walk(top=TOP, topdown=True, onerror=None, followlinks=True):
    for d in dirs:
        if model_type == 'dl':
            if root.find('电力铁塔驱鸟') >= 0:
                subfolders.append(os.path.join(root, d))

for subfolder in tqdm(subfolders):
    try:
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
        # 保存图片
        if len(detections) > 0:
            im = Image.open(result.path)
            draw_detections_optimal(im, detections, 0)
            filepath = Path('/data/blvhop/vector_index/dataset' + result.path)
            os.makedirs(filepath.parent, exist_ok=True)
            im.save(filepath)
