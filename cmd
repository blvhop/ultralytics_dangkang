TYPE=dzyqj

yolo cfg=cfg-dzyqj.yaml detect train model=yolov8m.pt epochs=120 optimizer=SGD lr0=0.01 batch=128 save_period=119 device=0,1 imgsz=640 cache=True data=$TYPE.yaml project=/root/blvhop/model/runs/train/ name=$TYPE\_`date "+%Y%m%d"`

cp /root/blvhop/model/runs/train/dzyqj_20240222/weights/best.pt /root/blvhop/model/model_dzyqj.pt 

yolo export format=onnx half=True device=0 model=/root/blvhop/model/model_dzyqj.pt

cp /root/blvhop/model/model_dzyqj.pt /mnt/W/root/blvhop/gc_webapi/static/

cp /root/blvhop/model/runs/train/hyzxlg_20240223/weights/best.pt /mnt/W/root/blvhop/hyzxlg_webapi/static/model_hyzxlg.pt

yolo export format=engine half=True device=0 model=model_dzyqj.pt
