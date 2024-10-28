# https://docs.ultralytics.com/zh/modes/track/#_7
from ultralytics import YOLO

model_path = '/root/blvhop/model/runs/train/djwrj_20240229/weights/best.pt'

model = YOLO(model=model_path)

TOP = '/data/blvhop/vector_index/dataset/AI视频材料-大疆无人机'
TOP = '/data/blvhop/djwrj/DJI_0020_77.jpg'

results = model.track(source=TOP, device='0,1', half=False, iou=0.45, conf=0.25, verbose=True,
                      save=True, save_txt=False, save_conf=False, save_crop=False, save_frames=False,
                      show_labels=False, show_conf=False, line_width=3,
                      stream=False)  # 使用默认追踪器进行追踪

