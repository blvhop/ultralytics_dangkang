from ultralytics import YOLO

# #########################################################
# ##### 训练
# # 加载模型
# model = YOLO('/root/blvhop/ultralytics/yolov8l.pt')  # load a pretrained model (recommended for training)
# # 训练模型
# # model.train(data='coco128.yaml', epochs=1, batch=1, imgsz=640, optimizer='SGD', lr0=0.01, workers=1)
# model.train(data='hwxj_det.yaml', epochs=1, batch=1, imgsz=640, optimizer='SGD', lr0=0.01, workers=1)
# # model.train(data='hwxj_det.yaml', epochs=1, imgsz=640, optimizer='SGD', lr0=0.01, batch=128, device='0,1', workers=1)
# #########################################################


#########################################################
##### 预测
model_path = '/root/blvhop/vector_index/model/model_hwxj_det_20231012.pt'
model_path = '/root/blvhop/ultralytics/yolov8s-fp16.onnx'
model_path = '/data/blvhop/model/runs/train/dltt_20240703/weights/last.onnx'
model = YOLO(model_path)

TOP = '/data/label-studio/media/local/mjzy/普氏原羚_20190707.mp4'
TOP = '/tmp/ttt.png'
TOP = '/root/blvhop/test/dltt'
results = model.predict(task='detect', source=TOP, half=True, iou=0.45, conf=0.25, verbose=True, save=True, stream=False)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
#########################################################
    print(result)


