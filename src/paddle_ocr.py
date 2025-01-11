import logging

import numpy as np
import pymupdf
from paddleocr import PaddleOCR
from PIL import Image

logger = logging.getLogger("ppocr")
logger.setLevel(logging.INFO)

ocr = PaddleOCR(
    cls_model_dir="./models/paddle/ch_ppocr_mobile_v2.0_cls_infer/",
    det_model_dir="./models/paddle/Multilingual_PP-OCRv3_det_infer/",
    rec_model_dir="./models/paddle/latin_PP-OCRv3_rec_infer/",
    use_angle_cls=True,
    use_gpu=True,
    use_onnx=False,
    lang="fr",
)


def ocr_pdf(page: pymupdf.Page, model: PaddleOCR = ocr):
    mat = pymupdf.Matrix(2, 2)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)

    return model.ocr(np.array(img), cls=False)


if __name__ == "__main__":
    pdf_path = "data/test_pdf.pdf"
    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        result = ocr_pdf(page)
        print(result[0][0])

        break
