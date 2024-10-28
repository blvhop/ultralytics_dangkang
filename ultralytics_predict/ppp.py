with fitz.open(pdf_path) as pdf:
    if pdf.page_count != 4:
        logger.warning(f"PDF文档页数不等于4页，仅处理4页({pdf_path})")
    for pg in range(0, pdf.page_count):
        logger.info(f"识别PDF文档的第{(pg + 1)}页({pdf_path})")
        page = pdf[pg]
        mat = fitz.Matrix(2, 2)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        if pm.width > 2000 or pm.height > 2000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        im_pil = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        # 文档方向判定后纠正方向
        im_cv2_lit = cv2.cvtColor(np.array(im_pil.resize((256, 256))), cv2.COLOR_RGB2BGR)
        cls_result = ori.predict(input_data=im_cv2_lit)
        cls_result = next(cls_result)
        if cls_result[0]['scores'][0] >= 0.55:
            angle = cls_result[0]['label_names'][0]
            if angle == '0':
                pass
            elif angle == '90':
                im_pil = im_pil.transpose(Image.ROTATE_90)
            elif angle == '180':
                im_pil = im_pil.transpose(Image.ROTATE_180)
            elif angle == '270':
                im_pil = im_pil.transpose(Image.ROTATE_270)
        else:
            angle = '00'

        # im_pil.save(f'angle_{Path(pdf_path).stem}_{pg}_{angle}.jpg') # debug

        im_cv2 = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        if pg <= 1:
            erase_red(im_cv2)
        imgs.append(im_cv2)

        result = ocr.ocr(im_cv2, cls=True)
        # 过滤只保留高置信度
        result = list(filter(lambda x: x[1][1] >= 0.90, result[0]))

        results.append(result)