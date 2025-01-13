import os
import tracemalloc

import matplotlib.pyplot as plt
import psutil
import seaborn as sns


# Function to run heavy computation
def dummy_computation(dummy_input=None):
    a = [i for i in range(10000)]
    b = [i**2 for i in range(10000)]
    return a + b


def log_memory(base_memory_psutil, base_memory_tracemalloc):
    process = psutil.Process(os.getpid())
    mem_psutil = process.memory_info().rss / (1024 * 1024)  # in MB

    # Log the memory allocated by Python objects using tracemalloc
    mem_tracmalloc = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
    return mem_psutil - base_memory_psutil, mem_tracmalloc - base_memory_tracemalloc


def plot_memory_usage(memory_df, mode="psutil"):
    # Plot memory usage for all backends in subplots
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create a plot for each backend
    for backend in memory_df["backend"].unique():
        subset = memory_df[memory_df["backend"] == backend]
        if mode == "psutil":
            sns.lineplot(x="page", y="memory_usage_psutil", data=subset, marker="o", label=backend)
        elif mode == "tracemalloc":
            sns.lineplot(x="page", y="memory_usage_tracemalloc", data=subset, marker="o", label=backend)

    plt.title(f"Memory Usage for All Models ({mode})", fontsize=16)
    plt.xlabel("Page Number", fontsize=12)
    plt.ylabel("Memory Usage (MB)", fontsize=12)
    plt.legend(title="Backend", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"./metrics/memory_usage_all_models_{mode}.png")
    # plt.show()


def make_plots(memory_df):
    plot_memory_usage(memory_df, "psutil")
    plot_memory_usage(memory_df, "tracemalloc")


def get_backend_engine(backend: str):
    if backend == "paddle":
        from src.paddle_ocr import ocr_pdf

        return ocr_pdf
    if backend == "rapid":
        from src.rapid_ocr import ocr_pdf

        return ocr_pdf
    if backend == "onnx":
        from src.onnx_ocr import ocr_pdf

        return ocr_pdf
    if backend == "dummy":
        return dummy_computation
