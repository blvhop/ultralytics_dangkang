# ultralytics需要修改的内容


    ultralytics/yolo/cfg/default.yaml
    -workspace: 4  # TensorRT: workspace size (GB)
    改动为
    +workspace: 1  # TensorRT: workspace size (GB)


    ultralytics/yolo/utils/torch_utils.py
    - arg = 'cuda:0'
    + # arg = 'cuda:0'
    + arg = f'cuda:{devices[i]}'

    ultralytics/yolo/engine/results.py
    self.names = names
    可修改标签
    +# self.names[0] = '人'
