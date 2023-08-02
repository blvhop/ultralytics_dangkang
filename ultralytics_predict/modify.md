# ultralytics需要修改的内容


    ultralytics/yolo/cfg/default.yaml
    -workspace: 4  # TensorRT: workspace size (GB)
    改动为
    +workspace: 1  # TensorRT: workspace size (GB)

