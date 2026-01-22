import os
# https://docs.ultralytics.com/zh/modes/track/#_7
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# 图像缩放到固定640x640
os.environ["xxx_blvhop_modify_resize"] = '1'
# 自定义小目标绘制
os.environ["xxx_blvhop_modify_pose_draw"] = '1'

from ultralytics import YOLO

model_path = '/root/blvhop/model/model_qn_pose_20231209.pt'
model_path = '/root/blvhop/vector_index/model/model_hwxj_det_20231012.pt'
model_path = '/root/blvhop/model/runs/train/hwxj_det_20240109/weights/best.pt'
model_path = '/root/blvhop/model/runs/train/hwxj_det_20240109/weights/best.pt'
model_path = '/root/blvhop/model/runs/train/sdxnl_det_20241122/weights/best.pt'
model_path = '/root/blvhop/model/runs/train/sdxnl_det_20241122/weights/best.pt'
model_path = '/root/blvhop/ultralytics/yolov8x.pt'
model_path = '/root/blvhop/model/runs/train/hhtk_hxh_202505217/weights/best.pt'
model_path = '/data/backup/00xxx_dk_imp/产品/后摩大道/workspace/houmo-modelzoo_v2.4/models/detection/aisdk/dl.onnx'
# 邮通大厦场站识别模型 1280x1280
model_path = '/data/blvhop/ultralytics/runs/detect/ytds_20250826_202508262/weights/last.pt'
model = YOLO(model=model_path)

TOP = '/data/label-studio/media/local/mjzy/普氏原羚_20190707.mp4'
TOP = '/data/label-studio/media/local/mjzy/小麂_白鹇_狗獾_环颈雉_牛背鹭_水牛_20230514.mp4'
TOP = '/data/blvhop/vector_index/dataset/视频'
# TOP = '/data/blvhop/test.mp4'

# 使用模型进行追踪
# 追踪参数：save是否保存预测视频；save_frames是否保存预测的每一帧；
# 注意，跟踪模型的save必须为True，否则跟踪id将无效，原因有待调试。
results = model.track(
    source=TOP, device='1', half=False, iou=0.45, conf=0.25, verbose=True,
    save=True, save_txt=False, save_conf=False, save_crop=False, save_frames=False,
    show_labels=True, show_conf=True, line_width=3,
    stream=False
)  # 使用默认追踪器进行追踪

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs

# 转换格式为视频帧列表[帧[对象[id, name, conf, xywh],],]
track_result = []
for result in results:
    boxes = result.boxes.cpu().numpy()
    one_frame_result = []
    if boxes.id is not None:
        for i in range(len(boxes.id)):
            id = int(boxes.id[i])
            name = model.names[int(boxes.cls[i])]
            conf = boxes.conf[i]
            # 将yolo的xywhn中心点和宽高的百分比值转换成ls的xywh左上角和宽高百分比*100的值
            xywhn = boxes.xywhn[i] * 100
            xywh = [xywhn[0] - xywhn[2] / 2, xywhn[1] - xywhn[3] / 2, xywhn[2], xywhn[3]]
            one_object_result = [id, name, conf, xywh]
            one_frame_result.append(one_object_result)
    track_result.append(one_frame_result)

print(track_result)

# 生成avi视频后转换为mp4
# ffmpeg -i xxx.avi -c:v libx264 -b:v 4M -c:a aac -b:a 128K -movflags +faststart output.mp4
# 递归转换所有子文件夹的 AVI，Windows 批处理需用 for /r 递归遍历，Linux/macOS 用 find 命令，示例（Linux）：
# find . -name "*.avi" -exec sh -c 'ffmpeg -i "$0" -c:v copy -c:a copy "${0%.avi}.mp4"' {} \;
# find . -name "*.avi" -exec sh -c 'ffmpeg -i "$0" -c:v libx264 -b:v 4M -c:a aac -b:a 128K -movflags +faststart "${0%.avi}.mp4"' {} \;
