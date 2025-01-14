import os
import time

import pandas as pd
import psutil
import pymupdf
import typer

from src.settings import LABEL, PLATFORM
from src.utils import get_backend_engine, log_memory, make_plots

BACKENDS = [
    "rapid",
    "paddle",
    "onnx",
]


def ocr_page(page: pymupdf.Page, backend: str):
    ocr = get_backend_engine(backend)
    return ocr(page)


def main(mode: str):
    print(f"Running for '{PLATFORM}'")

    pdf_path = "data/test_pdf.pdf"
    pdf_1 = pymupdf.open(pdf_path)
    pdf_2 = pymupdf.open(pdf_path)
    pdf_3 = pymupdf.open(pdf_path)

    if mode == "memtest":
        memory_df = pd.DataFrame(columns=["backend", "page", "memory_usage", "total_memory", "processing_time"])

        for backend in BACKENDS:
            print(f"Testing {backend=}...")

            base_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            for i, pdf_ in enumerate([pdf_1, pdf_2, pdf_3]):
                for page in pdf_:
                    page_num = page.number + i * len(pdf_)

                    start_time = time.time()
                    result = ocr_page(page, backend)
                    processing_time = round(time.time() - start_time, 2)
                    memory_usage = log_memory(base_memory)
                    memory_df = pd.concat(
                        [
                            memory_df,
                            pd.DataFrame(
                                {
                                    "backend": backend,
                                    "page": page_num,
                                    "memory_usage": memory_usage[0],
                                    "total_memory": memory_usage[1],
                                    "processing_time": processing_time,
                                },
                                index=[0],
                            ),
                        ],
                        ignore_index=True,
                    )
                    if page_num % 11 == 0:
                        print(
                            f"Page {page_num:3}: Memory= {memory_usage[0]:5.2f} MB | Cumulative memory= {memory_usage[1]:5.2f} MB | Processing time={processing_time:5.2f} s"
                        )
        memory_df.to_csv(f"metrics/{PLATFORM}/memory_usage_{LABEL}.csv", index=False)
        make_plots(memory_df)

    else:
        backend = mode
        print(f"Using {backend=}")
        assert backend in BACKENDS + ["dummy"], "Invalid back end - Chose one of 'paddle', 'rapid', 'onnx', 'dummy'"

        for page in pdf_1:
            result = ocr_page(page, backend)
            print(result[0])
            break


if __name__ == "__main__":
    typer.run(main)
