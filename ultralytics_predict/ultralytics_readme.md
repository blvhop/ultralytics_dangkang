# ultralytics需要修改的内容

修改宗旨，加入噪点，加入宽高直接缩放到位。
默认宽高固定比例缩放，感觉在动物识别中没必要固定比例缩放。

此修改适合公司训练的模型；如果是公共模型，则用未修改版本源码。


ultralytics/yolo/cfg/default.yaml

    -workspace: 4  # TensorRT: workspace size (GB)
    改动为
    +workspace: 1  # TensorRT: workspace size (GB)

ultralytics/data/base.py

    1、直接缩放到目标分辨率（不在固定宽高比例）
    2、将放大cv2.INTER_LINEAR方式改为cv2.INTER_CUBIC
    - r = self.imgsz / max(h0, w0)  # ratio
    - if r != 1:  # if sizes are not equal
    -   interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
    -   im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
    + if h0 != self.imgsz or w0 != self.imgsz:
    +   r = self.imgsz / max(h0, w0)  # ratio
    +   interp = cv2.INTER_CUBIC if (self.augment or r > 1) else cv2.INTER_AREA
    +   im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=interp)


ultralytics/data/dataset.py

    将默认scaleup=False不允许放大去掉，即允许放大
    - transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
    + transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz))])

ultralytics/data/augment.py

    安装albumentations，使能albumentations数据增强
    pip install albumentations
    下面三种数据增强概率设为0.1
    - A.Blur(p=0.1),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.1), # 给图像增加高斯噪声。
    或者
    A.OneOf([
        A.Blur(), # 高斯卷积核模糊，但是只有卷积核大小随机。
        # A.IAAAdditiveGaussianNoise(),  # 将高斯噪声添加到输入图像
        A.GaussNoise(),  # 将高斯噪声应用于输入图像。
    ], p=0.2),  # 应用选定变换的概率
    A.ToGray(p=0.1),
    A.ImageCompression(quality_lower=75, p=0.1)]

    class LetterBox:
    改scaleFill=True直接缩放到目标大小
    - def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
    + def __init__(self, new_shape=(640, 640), auto=False, scaleFill=True, scaleup=True, center=True, stride=32):

    def __call__(self, labels=None, image=None):
    区分放大和缩小使用的方式
    - img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    + img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA if r <= 1.0 else cv2.INTER_CUBIC)

ultralytics/models/yolo/detect/predict.py

    def postprocess(self, preds, img, orig_imgs):
        直接等比缩放还原，不需要反向还原LetterBox内容
        - pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        + pred[:, [0, 2]] = pred[:, [0, 2]] * orig_img.shape[1]/img.shape[3]
        + pred[:, [1, 3]] = pred[:, [1, 3]] * orig_img.shape[0]/img.shape[2]

ultralytics/yolo/utils/torch_utils.py

    - arg = 'cuda:0'
    + # arg = 'cuda:0'
    + arg = f'cuda:{devices[i]}'



## 可修改内容备忘

ultralytics/yolo/engine/results.py

    self.names = names
    可修改标签
    +# self.names[0] = '人'

ultralytics/engine/trainer.py

    去掉ni in self.plot_idx条件可打印所有用于训练的图片：train_batch0.jpg
    - if self.args.plots and ni in self.plot_idx:
    + if self.args.plots:
        self.plot_training_samples(batch, ni)

ultralytics/engine/predictor.py
    
    def preprocess(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            # 图片预处理，主要是LetterBox方法
            im = np.stack(self.pre_transform(im))
            ################################
            ##### 保存待预测的图片数据
            os.makedirs('/tmp/blvhop/predictor', exist_ok=True)
            for x in im:
                x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'/tmp/blvhop/predictor/{time.time()}.jpg', x)
            ################################
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)