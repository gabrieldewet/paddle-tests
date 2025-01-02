import cv2
import numpy as np
import pymupdf
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# ocr = PaddleOCR(
#     cls_model_dir="./models/ch_ppocr_mobile_v2.0_cls_infer/",
#     det_model_dir="./models/en_PP-OCRv3_det_infer/",
#     rec_model_dir="./models/en_PP-OCRv3_rec_infer/",
#     use_angle_cls=True,
#     use_gpu=True,
#     use_onnx=False,
#     lang="fr",
# )

ocr = PaddleOCR(
    cls_model_dir="../models/ch_ppocr_mobile_v2.0_cls_infer",
    det_model_dir="../models/ch_PP-OCRv3_det_slim_infer",
    rec_model_dir="../models/ch_PP-OCRv3_rec_slim_infer",
    use_angle_cls=True,
    use_gpu=True,
    use_onnx=False,
    lang="fr",
)

# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/paddleocr.py

pdf_path = "data/test_pdf.pdf"
pdf = pymupdf.open(pdf_path)
for page in pdf:
    mat = pymupdf.Matrix(2, 2)
    pm = page.get_pixmap(matrix=mat, alpha=False)

# Load pdf


result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# imgs = []
# with fitz.open(img_path) as pdf:
#     for page_num in range(pdf.page_count):
#         page = pdf[page_num]
#         mat = fitz.Matrix(2, 2)
#         pm = page.getPixmap(matrix=mat, alpha=False)

#         # if width or height > 2000 pixels, don't enlarge the image
#         if pm.width > 2000 or pm.height > 2000:
#             pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

#         img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         imgs.append(img)
