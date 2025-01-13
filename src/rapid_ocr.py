import numpy as np
import pymupdf
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from src.settings import N_CPU, USE_GPU

ocr = RapidOCR(
    cls_model_path="./models/rapid_onnx/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx",
    det_model_path="./models/rapid_onnx/Multilingual_PP-OCRv3_det_infer/Multilingual_PP-OCRv3_det_infer.onnx",
    rec_model_path="./models/rapid_onnx/latin_PP-OCRv3_rec_infer/latin_PP-OCRv3_rec_infer.onnx",
    intra_op_num_threads=N_CPU,
    inter_op_num_threads=N_CPU,
    cls_use_cuda=False,
    det_use_cuda=False,
    rec_use_cuda=False,
)


def ocr_pdf(page: pymupdf.Page, model: RapidOCR = ocr):
    mat = pymupdf.Matrix(2, 2)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)

    return model(np.array(img))[0]


if __name__ == "__main__":
    pdf_path = "data/test_pdf.pdf"
    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        result, elapse = ocr_pdf(page)
        # print(result)
        print(elapse)
        print(result[0])

        break
