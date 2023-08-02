from ultralytics import YOLO

# model = YOLO('/root/blvhop/model_detect/model_hwxj_group_20230421.pt')
model = YOLO('yolov8x.pt')

# classes: 14: 'bird'
results = model.predict(source="/root/blvhop/images/", classes=[14], iou=0.45, verbose=False, save=False, stream=False)

print(results)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
