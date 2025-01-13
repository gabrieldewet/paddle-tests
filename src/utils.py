import os
import platform
import tracemalloc

import matplotlib.pyplot as plt
import psutil
import seaborn as sns
import torch


def get_system():
    if platform.system() == "Darwin":
        return "mac"
    if torch.cuda.is_available():
        return "gpu"
    else:
        return "cpu"


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
    return (
        round(mem_psutil - base_memory_psutil, 3),
        round(mem_tracmalloc - base_memory_tracemalloc, 3),
        round(mem_psutil, 3),
    )


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
    plt.savefig(f"./metrics/{get_system()}/memory_usage_all_models_{mode}.png")
    # plt.show()


def plot_memory_usage_all(memory_df):
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Left axis: Cumulative memory usage
    ax1.set_ylabel("Cumulative Memory Usage (MB)", fontsize=14, color="tab:blue")
    ax1.plot(memory_df.index, memory_df["total_memory"], label="Cumulative Memory Usage", color="tab:blue")
    ax1.set_ylim(0, memory_df["total_memory"].max() * 1.2)  # Set max limit for better visual scaling
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Right axis: Inference time (in seconds)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Inference Time (seconds)", fontsize=14, color="tab:orange")

    ax2.set_ylim(0, memory_df["processing_time"].max() * 1.3)  # Set max limit for better visual scaling
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    boundaries = []
    group_ticks = []
    current_page = 0
    colors = ["tab:red", "tab:green", "tab:purple"]
    for backend, color in zip(memory_df["backend"].unique(), colors):
        subset = memory_df.loc[memory_df["backend"] == backend]
        ax2.bar(
            subset.index,
            subset["processing_time"],
            width=0.5,
            label=f"Inference Time ({backend})",
            color=color,
            alpha=0.6,
        )

        page_count = memory_df.loc[memory_df["backend"] == backend].shape[0]
        group_ticks.append((page_count / 2) + current_page)
        current_page += page_count
        boundaries.append(current_page)

    for boundary in boundaries[:-1]:
        ax1.axvline(x=boundary, color="gray", linestyle="--", linewidth=3)

    # Secondary x-axis
    secax = ax1.secondary_xaxis("top")
    secax.set_xticks(group_ticks)  # Position ticks in the middle of groups
    secax.set_xticklabels(memory_df["backend"].unique())  # Labels for groups
    secax.set_xlabel("Backend", fontsize=14)

    # Add title and legends
    plt.title("Memory Usage and Inference Time for Different Backends", fontsize=16, y=1.1)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"./metrics/{get_system()}/memory_usage_all.png")


def make_plots(memory_df):
    plot_memory_usage(memory_df, "psutil")
    plot_memory_usage(memory_df, "tracemalloc")
    plot_memory_usage_all(memory_df)


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
