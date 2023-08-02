import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def save_voc_xml(jpg_path: str, xml_path: str, detections: list):
    """
    :param jpg_path: 标注图片
    :param xml_path: 标注xml
    :param detections: 标注对象list[json]。
    标注对象组成：
        bndbox: xmin左上角x值, ymin左上角y值, xmax右下角x值, ymax右下角y值
        name: 标签名
        confidence: 置信度（可选，save_voc_xml无需；draw_bbox需要）
    示例：[{"bndbox": [22, 23, 25, 45.2],"name": "马", "confidence": 0.91},{"bndbox": [21, 25, 25, 52],"name": "人"}]
    :return:
    """
    # 保存xml标签
    with open(xml_path, mode='w') as f:
        img = Image.open(jpg_path)

        ###############################################################
        ##### 新建xml
        # 创建Annotation根节点
        root = ET.Element('annotation')

        ET.SubElement(root, 'folder').text = ''
        # 创建filename子节点，无扩展名
        ET.SubElement(root, 'filename').text = ''
        ET.SubElement(root, 'path').text = ''

        # 创建source子节点
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'

        # 创建size子节点
        sizes = ET.SubElement(root, 'size')
        ET.SubElement(sizes, 'width').text = str(img.width)
        ET.SubElement(sizes, 'height').text = str(img.height)
        ET.SubElement(sizes, 'depth').text = str(len(img.getbands()))

        # 创建segmented子节点
        ET.SubElement(root, 'segmented').text = '0'

        for gt_object in detections:
            # 创建object子节点
            e_object = ET.SubElement(root, 'object')
            ET.SubElement(e_object, 'name').text = gt_object['name']
            ET.SubElement(e_object, 'pose').text = 'Unspecified'
            ET.SubElement(e_object, 'truncated').text = '0'
            ET.SubElement(e_object, 'difficult').text = '0'
            e_bndbox = ET.SubElement(e_object, 'bndbox')
            ET.SubElement(e_bndbox, 'xmin').text = str(int(gt_object['bndbox'][0]))
            ET.SubElement(e_bndbox, 'ymin').text = str(int(gt_object['bndbox'][1]))
            ET.SubElement(e_bndbox, 'xmax').text = str(int(gt_object['bndbox'][2]))
            ET.SubElement(e_bndbox, 'ymax').text = str(int(gt_object['bndbox'][3]))

        indent(root)
        content = ET.tostring(root, 'utf-8', 'xml')
        with open(xml_path, mode='wb') as f:
            f.write(content)


# 不同置信度，字体颜色固定，填充不同颜色，
confidence_colors = [
    (0, 255, 131),  # 绿色
    (128, 230, 200),  # 淡蓝色
    (230, 210, 200),  # 淡红色
    (128, 128, 128),  # 灰色
]

# 不同物种不同颜色，颜色区别要大，彩虹颜色
name_colors = {
    '夜鹭1': (0x00, 0xFF, 0x83),
    '夜鹭2': (0x00, 0xFF, 0x83),
    '夜鹭3': (0x00, 0xFF, 0x83),
    '白鹭': (0x22, 0x8B, 0x22),
}


def draw_detections_optimal(img, detections, threshold_cls):
    """
    # 优化过地绘制图像识别框
    # :param img: 图像
    # :param detections: 目标检测数据。新增：detection['confidence']  # 分类置信度可选
    # :param threshold_cls: 分类名称显示置信度，大于此值才会绘制分类名称，否则只打框
    """

    if isinstance(img, np.ndarray):
        # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    font_object = ImageFont.truetype("msyh.ttc", 28)
    for detection in detections:
        name = detection['name']  # 检测标签
        confidence = detection.get('confidence', 0.0)  # 检测置信度
        xmin, ymin, xmax, ymax = detection['bndbox']

        # 设置显示置信度背景填充颜色等级
        if confidence >= 0.89:
            color_index = 0
        elif confidence >= 0.99:
            color_index = 0
        elif confidence >= 0.50:
            color_index = 1
        elif confidence >= 0.35:
            color_index = 2
        else:
            color_index = 3

        # 绘制目标边框
        draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)], width=3, fill=name_colors.get(name, (0xFF, 0x00, 0x00)))

        # 置信度小于阈值不显示名称
        if confidence < threshold_cls:
            continue
        # 宽高小于32不显示名称
        if (xmax - xmin) <= 32 or (ymax - ymin) <= 32:
            continue

        # 计算名称文字大小
        text = f"{name} {int(confidence * 100)}%"
        # DeprecationWarning: textsize is deprecated and willbe removed in Pillow 10 (2023-07-01). Use textbbox or textlength instead.
        # text_width, text_height = draw.textsize(text, font=font_object)
        _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font_object)
        text_ymin = ymin - text_height
        if text_ymin < 0:
            text_ymin = 0
        # 绘制名称背景框
        draw.rectangle([(xmin, text_ymin), (xmin + text_width + 1, text_ymin + text_height)], fill=confidence_colors[color_index])
        # 绘制名称
        draw.text((xmin, text_ymin), text, font=font_object, fill=(25, 25, 25))

#