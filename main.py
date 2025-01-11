import pymupdf
import typer


def get_backend(back_end: str):
    if back_end == "paddle":
        from src.paddle_ocr import ocr_pdf

        return ocr_pdf
    if back_end == "rapid":
        from src.rapid_ocr import ocr_pdf

        return ocr_pdf
    if back_end == "onnx":
        from src.onnx_ocr import ocr_pdf

        return ocr_pdf


def main(back_end: str):
    print(f"Using {back_end=}")
    assert back_end in ["paddle", "rapid", "onnx"], "Invalid back end - Chose one of 'paddle', 'rapid', 'onnx'"
    processor = get_backend(back_end)

    pdf_path = "data/test_pdf.pdf"
    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        result = processor(page)
        print(result)
        break


if __name__ == "__main__":
    typer.run(main)
