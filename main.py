from typing import Literal

import pymupdf
import typer


def main(engine: str):
    print(f"Using {engine=}")
    assert engine in ["paddle", "rapid", "onnx"], "Invalid engine"
    if engine == "paddle":
        from src.paddle_ocr import ocr_pdf
    elif engine == "rapid":
        from src.rapid_ocr import ocr_pdf
    elif engine == "onnx":
        from src.onnx_ocr import ocr_pdf

    pdf_path = "data/test_pdf.pdf"
    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        result = ocr_pdf(page)
        print(result)
        break


if __name__ == "__main__":
    typer.run(main)
