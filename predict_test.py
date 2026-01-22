import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 设置环境变量
# 图像缩放到固定640x640
os.environ["xxx_blvhop_modify_resize"] = '1'
# 自定义小目标绘制
os.environ["xxx_blvhop_modify_pose_draw"] = '1'

from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from blvhop.dataset_common import save_voc_xml

model_path = '/data/blvhop/ultralytics/yolov8x.pt'
# model_path = '/root/blvhop/model/model_qn_det_20231209.pt'
model_path = '/root/blvhop/model/runs/train/hwxj_det_20240109/weights/best.pt' # mjzy
# model_path = '/root/blvhop/model/runs/train/dzyqj_20240222/weights/best.pt'
# model_path = '/root/blvhop/model/runs/train/hawt_20240223/weights/best.pt'
# model_path = '/root/blvhop/model/runs/train/hyzxlg_20240223/weights/best.pt'
# model_path = "/data/blvhop/vector_index/model/model_hwxj_det_20240318.engine"
model_path = '/root/blvhop/model/runs/train/dl_20240918/weights/last.pt'
model_path = '/root/blvhop/ultralytics/nltk.pt'
model_path = '/data/backup/00xxx_dk_imp/产品/后摩大道/workspace/houmo-modelzoo/models/detection/yolov8m/yolov8m_640_640.original.onnx'

model = YOLO(model=model_path, task='detect')

# results = model.predict(source='/mnt/W/root/blvhop/gc_webapi/static/runs/20240226/predict/201040^C1119^270.png', task='detect', half=True, iou=0.45, conf=0.00025, verbose=True, save=True, stream=False)


# yolov8识别鸟类并生成voc xml标签
subfolders = []
TOP = '/data/blvhop/nltkhzh'
TOP = '/data/blvhop/dzyqj/00009_0'
# TOP = '/data/blvhop/hawt'
# TOP = '/data/blvhop/hyzxlg/好招楼'
# TOP = '/mnt/W/usr/local/run/autoCut/images20240219-to-dataset'
# TOP = '/data/blvhop/jt'
# TOP = '/data/blvhop/mjzy'
TOP = '/root/blvhop/vector_index/dataset/vidio_split'
TOP = '/data/backup/00xxx_dk_imp/产品/后摩大道/workspace/houmo-modelzoo/data/datasets/video_demo/'
# TOP = '/root/blvhop/ultralytics'

subfolders.append(TOP)
for root, dirs, files in os.walk(top=TOP, topdown=True, onerror=None, followlinks=True):
    for d in dirs:
        subfolders.append(os.path.join(root, d))

for subfolder in tqdm(subfolders):
    try:
        # classes: 0: person / 1: bicycle / 2: car / 14: bird
        # results = model.predict(source=subfolder, task='detect', classes=[14], iou=0.45, conf=0.25, verbose=True, save=False, stream=False)
        results = model.predict(source=subfolder, task='detect', device='1', half=False, iou=0.45, conf=0.25, verbose=True, save=True, stream=False)
        # print(results)
    except FileNotFoundError:
        continue

