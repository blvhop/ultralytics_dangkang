import os

# 设置环境变量
# 图像缩放到固定640x640
import traceback

os.environ["xxx_modify_resize"] = '0'
# 自定义小目标绘制
os.environ["xxx_modify_pose_draw"] = '0'

from pathlib import Path
from tqdm import tqdm
import cv2
import fitz  # pip install PyMuPDF
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

from blvhop.common import get_filelist
from blvhop.multi import multi_run, queue

pdf_path = r'/data/blvhop/identification/test/DASTRA SERVICES - Statuts constitutifs(1).pdf'
ROOT = '/data/blvhop/identification'
TARGET = f'{ROOT}_result'


def pdf_process(model, device, pdf_path, r_pdf_path):
    with fitz.open(pdf_path) as pdf:
        logger.debug(f"PDF文档页数{pdf.page_count}页 ({pdf_path})")
        if pdf.page_count == 0:
            return

        for pg in range(0, pdf.page_count):
            # logger.info(f"识别PDF文档的第{(pg + 1)}页 ({pdf_path})")
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            im_pil = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            im_cv2 = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

            # classes: 0: person / 1: bicycle / 2: car / 14: bird
            results = model.predict(source=im_cv2, task='detect', device=device, classes=[0], iou=0.45, conf=0.20, verbose=False, save=False, stream=True)

            boxes = next(results).boxes.cpu().numpy()
            detections = []
            for i, xyxy in enumerate(boxes.xyxy):
                xmin, ymin, xmax, ymax = xyxy
                if xmax - xmin < 100 or ymax - ymin < 100:
                    continue
                else:
                    xmin -= 50
                    xmax += 350
                    ymin -= 50
                    ymax += 75
                if model.names[int(boxes.cls[i])] in ('book',):
                    continue
                detection = {'name': model.names[int(boxes.cls[i])], 'confidence': boxes.conf[i], 'bndbox': xyxy.astype(int).tolist(), 'name_det': None, 'confidence_det': None}
                detections.append(detection)
            if len(detections) > 0:
                jpg_draw_file = f'{TARGET}/{r_pdf_path[:-4]}_{pg:02d}.jpg'
                jpg_draw_path = os.path.split(jpg_draw_file)[0]
                os.makedirs(jpg_draw_path, exist_ok=True)
                im_pil = Image.fromarray(cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(im_pil)
                draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)], width=3, fill=(0x00, 0xFF, 0x83))
                im_pil.save(jpg_draw_file, format='JPEG')


###########################################
def single_processor(processor_id, working, queue_in, queue_out):
    """
    单进程业务方法
    processor_id, working, queue_in, queue_out为必要参数；
    后面的参数为业务代码需要的可变参数。
    """
    try:
        import traceback
        from ultralytics import YOLO

        model_path = '/data/blvhop/ultralytics/yolov8x.pt'
        model = YOLO(model=model_path)
        if processor_id < process_count / 3:
            device = '0'
        else:
            device = '0,1'

        while True:
            if not working.value:
                # 数据处理完毕
                break
            try:
                thing = queue_in.get(block=True, timeout=0.2)
            except queue.Empty:
                continue

            ##########################################
            # 开始业务代码
            pdf_path, r_pdf_path = thing
            try:
                pdf_process(model, device, pdf_path, r_pdf_path)
            except Exception as e:
                logger.error(f'ERROR: {pdf_path}, {e}')
                logger.info(traceback.format_exc())
            ##########################################

            ##########################################
            # 添加数据到主进程中处理
            # 前三个参数必须为processor_id, thing，[新增thing列表things_added], 后面可添加任意返回值到主进程处理
            queue_out.put((processor_id, thing, [], "success"))
    except Exception as e:
        logger.error(e)
        err = traceback.format_exc()
        logger.error(err)
        logger.info(thing)

    logger.info(f'{processor_id} finished\n', flush=True, end='')


###########################################
def call_back_in_main(result):
    processor_id, thing, things_added, extra_result = result
    # logger.debug(thing)


###########################################
things = set()
pdf_paths = get_filelist(ROOT)
pdf_paths = sorted(pdf_paths, key=lambda x: x[0])
for pdf_path, r_pdf_path in tqdm(pdf_paths):
    things.add((pdf_path, r_pdf_path))

process_count = 24  # yolox: 24; yolol:
multi_run(single_processor, call_back_in_main, things, process_count)
