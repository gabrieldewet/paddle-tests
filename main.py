import tracemalloc

import pymupdf
import typer

from src.utils import dummy_computation, get_backend_engine, log_memory, plot_memory_usage

BACKENDS = [
    "paddle",
    "rapid",
    "onnx",
    "dummy",
]


def ocr_page(page: pymupdf.Page, backend: str):
    ocr = get_backend_engine(backend)
    return ocr(page)


def main(mode: str):
    pdf_path = "data/test_pdf.pdf"
    pdf = pymupdf.open(pdf_path)

    if mode == "memtest":
        psutil_memory_log = []
        tracemalloc_memory_log = []

        for backend in BACKENDS:
            print(f"Testing {backend=}...")
            tracemalloc.start()

            for page in pdf:
                log_memory(psutil_memory_log, tracemalloc_memory_log)
                result = dummy_computation(page)
                # result = ocr_page(page, backend)
                log_memory(psutil_memory_log, tracemalloc_memory_log)
                print(f"Page {page.number}: {len(result)}...")

            tracemalloc.stop()

    else:
        backend = mode
        print(f"Using {backend=}")
        assert backend in BACKENDS, "Invalid back end - Chose one of 'paddle', 'rapid', 'onnx', 'dummy'"

        for page in pdf:
            result = ocr_page(page, backend)
            print(result)
            break


if __name__ == "__main__":
    typer.run(main)
