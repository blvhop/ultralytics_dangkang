from ultralytics import YOLO

# #########################################################
# ##### 训练
# # 加载模型
# model = YOLO('/root/blvhop/ultralytics/yolov8l.pt')  # load a pretrained model (recommended for training)
# # 训练模型
# # model.train(data='coco128.yaml', epochs=1, batch=1, imgsz=640, optimizer='SGD', lr0=0.01, workers=1)
# model.train(data='hwxj-g.yaml', epochs=1, batch=1, imgsz=640, optimizer='SGD', lr0=0.01, workers=1)
# # model.train(data='hwxj-g.yaml', epochs=1, imgsz=640, optimizer='SGD', lr0=0.01, batch=128, device='0,1', workers=1)
# #########################################################


#########################################################
##### 预测
model = YOLO('/root/blvhop/model/model_hwxj_det_20230702.onnx')

results = model.predict(source='/root/blvhop/test/hwxj', task='detect', iou=0.45, conf=0.25, verbose=True, save=True, stream=False)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
#########################################################


