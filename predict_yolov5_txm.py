import os

# 设置环境变量
# 图像缩放到固定640x640
os.environ["xxx_modify_resize"] = '0'
# 自定义小目标绘制
os.environ["xxx_modify_pose_draw"] = '0'

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from blvhop.dataset_common import save_voc_xml
from blvhop.dataset_common import IMG_FORMATS, VID_FORMATS
from blvhop.common import get_filelist

model_path = '/data/blvhop/ultralytics/yolov8x.pt'
model = YOLO(model=model_path)

model_path_yolov5 = '/data/blvhop/model/yolov5_txm_20220909_1280_fp32.onnx'
model_yolov5 = torch.hub.load(repo_or_dir=r"/root/blvhop/yolov5", model="custom", path=model_path_yolov5, source='local', force_reload=True)

source = '/data/blvhop/nltkhzh'
filelist = get_filelist(source)

for filepath, _ in tqdm(filelist):
    ext = os.path.splitext(filepath)[1][1:].lower()
    if ext not in IMG_FORMATS:
        continue

    # print(filepath)
    im = Image.open(filepath)
    width, height = im.size
    detections = []

    # yolov8预测
    # classes: 0: person / 1: bicycle / 2: car / 14: bird
    results = model.predict(source=im, task='detect', classes=[14], iou=0.45, conf=0.25, verbose=False, save=False, stream=False)
    result = results[0]
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs

    # 封装预测结果
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
        if detection['name'] not in ('鸟',):
            continue
        # 标签名修改为目录名
        detection['name'] = filepath.split('/')[-2]
        detection['confidence'] = boxes.conf[i]
        detection['bndbox'] = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
        detections.append(detection)

    # yolov5预测
    im = im.resize((1280, 1280), Image.BICUBIC)
    results_yolov5 = model_yolov5(im, size=1280)

    pred = results_yolov5.pandas().xyxyn[0]
    for index, row in pred.iterrows():
        detection = {}
        # 0: '有羽冠', 1: '无羽冠', 2: '嘴黄褐色', 3: '嘴非黄褐色', 4: '脖纯色', 5: '脖非纯色', 6: '背纯色', 7: '背非纯色', 8: '腿黑色', 9: '腿非黑色'
        if row['class'] < 2:
            detection['name'] = '羽冠'
        elif row['class'] < 4:
            detection['name'] = '嘴'
        elif row['class'] < 6:
            detection['name'] = '脖'
        elif row['class'] < 8:
            detection['name'] = '背'
        elif row['class'] < 10:
            detection['name'] = '腿'
        else:
            continue
            detection['name'] = row['name']
        detection['confidence'] = round(row['confidence'], 3)
        detection['bndbox'] = [round(row['xmin']*width), round(row['ymin']*height), round(row['xmax']*width), round(row['ymax']*height)]
        detections.append(detection)

    # 保存为xml
    save_voc_xml(filepath, Path(filepath).with_suffix(".xml"), detections)
