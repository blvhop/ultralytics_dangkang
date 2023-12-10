import os
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

IMGSZ = 1280

files = [file for file in Path('/root/blvhop/test/hwxj1280code').rglob("*.*")]
for filepath in tqdm(files, ncols=80, desc='遍历文件'):
    ######################################################
    # 全部cv2方式resize对比

    interpolations = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    ]
    interpolation_names = [
        'INTER_NEAREST',
        'INTER_LINEAR',
        'INTER_CUBIC',
        'INTER_AREA',
        'INTER_LANCZOS4',
    ]

    im = cv2.imread(str(filepath))  # BGR 0-255
    h, w, c = im.shape
    r = min(IMGSZ / w, IMGSZ / h)
    w, h = int(r * w), int(r * h)

    for i, interpolation in enumerate(interpolations):
        im_r = cv2.resize(im, (w, h), interpolation=interpolation)
        filepath_r = filepath.parent / f'{filepath.stem}_cv2_{interpolation}_{interpolation_names[i]}.jpg'
        cv2.imwrite(str(filepath_r), im_r, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    ######################################################
    # 全部PIL方式resize对比

    resamples = [
        Image.Resampling.NEAREST, # 默认
        Image.Resampling.BOX,
        Image.Resampling.BILINEAR,
        Image.Resampling.HAMMING,
        Image.Resampling.BICUBIC,
        Image.Resampling.LANCZOS,
    ]

    im = Image.open(filepath)  # RGB # im = im.convert('RGB')
    (w, h) = im.size
    r = min(IMGSZ / w, IMGSZ / h)
    w, h = int(r * w), int(r * h)

    for i, resample in enumerate(resamples):
        im_r = im.resize((w, h), resample)
        filepath_r = filepath.parent / f'{filepath.stem}_pil_{resample}_{resample.name}.jpg'
        im_r.save(str(filepath_r), subsampling=0, quality=100)
