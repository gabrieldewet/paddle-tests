import os
import tracemalloc

import pandas as pd
import psutil
import pymupdf
import typer

from src.utils import dummy_computation, get_backend_engine, log_memory, make_plots

BACKENDS = [
    "rapid",
    "paddle",
    "onnx",
]


def ocr_page(page: pymupdf.Page, backend: str):
    ocr = get_backend_engine(backend)
    return ocr(page)


def main(mode: str):
    pdf_path = "data/test_pdf.pdf"
    pdf = pymupdf.open(pdf_path)

    if mode == "memtest":
        memory_df = pd.DataFrame(columns=["backend", "page", "memory_usage_psutil", "memory_usage_tracemalloc"])

        for backend in BACKENDS:
            print(f"Testing {backend=}...")
            tracemalloc.start()

            base_memory_psutil = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            base_memory_tracemalloc = tracemalloc.get_traced_memory()[0] / (1024 * 1024)

            for i, pdf_ in enumerate([pdf, pdf, pdf]):
                for page in pdf:
                    page_num = page.number + i * len(pdf)
                    # result = dummy_computation(page)
                    result = ocr_page(page, backend)
                    memory_usage = log_memory(base_memory_psutil, base_memory_tracemalloc)
                    memory_df = pd.concat(
                        [
                            memory_df,
                            pd.DataFrame(
                                {
                                    "backend": backend,
                                    "page": page_num,
                                    "memory_usage_psutil": memory_usage[0],
                                    "memory_usage_tracemalloc": memory_usage[1],
                                },
                                index=[0],
                            ),
                        ],
                        ignore_index=True,
                    )
                    if page_num % 7 == 0:
                        print(f"Page {page_num}: {len(result)=} | {memory_usage[0]=} MB | {memory_usage[1]=} MB")

            tracemalloc.stop()
        # print(memory_df)
        make_plots(memory_df)

    else:
        backend = mode
        print(f"Using {backend=}")
        assert backend in BACKENDS + ["dummy"], "Invalid back end - Chose one of 'paddle', 'rapid', 'onnx', 'dummy'"

        for page in pdf:
            result = ocr_page(page, backend)
            print(result[0])
            break


if __name__ == "__main__":
    typer.run(main)
