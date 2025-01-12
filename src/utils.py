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


def log_memory_usage_psutil(memory_log):
    # Log the current memory usage in MB using psutil
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    memory_log.append(memory_usage)


def log_memory_usage_tracemalloc(memory_log):
    # Log the memory allocated by Python objects using tracemalloc
    current, peak = tracemalloc.get_traced_memory()
    memory_log.append(current / (1024 * 1024))  # Convert to MB


def log_memory(psutil_memory_log, tracemalloc_memory_log):
    log_memory_usage_psutil(psutil_memory_log)
    log_memory_usage_tracemalloc(tracemalloc_memory_log)


def plot_memory_usage(memory_logs, backends, mode="psutil"):
    # Plot memory usage for all backends in subplots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(backends), 1, figsize=(10, len(backends) * 6), sharex=True)

    for ax, (backend, memory_log) in zip(axes, zip(backends, memory_logs)):
        sns.lineplot(x=range(len(memory_log)), y=memory_log, marker="o", ax=ax)
        ax.set_title(f"Memory Usage for {backend} ({mode})", fontsize=16)
        ax.set_xlabel("Page Number", fontsize=12)
        ax.set_ylabel("Memory Usage (MB)", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"memory_usage_all_models_{mode}.png")
    plt.show()


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
