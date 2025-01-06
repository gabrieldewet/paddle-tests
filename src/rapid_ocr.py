import logging

import numpy as np
import pymupdf
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

model = RapidOCR(
    cls_model_path="./models/cls_onnx/model.onnx",
    det_model_path="./models/det_onnx/model.onnx",
    rec_model_path="./models/rec_onnx/model.onnx",
)


def ocr_pdf(page, model=model):
    mat = pymupdf.Matrix(2, 2)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)

    return model(np.array(img))


if __name__ == "__main__":
    pdf_path = "data/test_pdf.pdf"
    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        result, elapse = ocr_pdf(page)
        # print(result)
        print(elapse)
        print(result[0])

        break
