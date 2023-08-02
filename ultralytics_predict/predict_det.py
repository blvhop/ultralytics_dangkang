from ultralytics import YOLO

#########################################################
##### 训练
# 加载模型
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# 训练模型
model.train(data='coco128.yaml', epochs=100, imgsz=640)
#########################################################


#########################################################
##### 预测
model = YOLO('/root/blvhop/model/model_hwxj_det_20230702.onnx')

results = model.predict(source='/root/blvhop/test/hwxj', iou=0.45, conf=0.25, verbose=True, save=True, stream=False)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
#########################################################


