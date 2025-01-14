import os

import matplotlib.pyplot as plt
import psutil
import seaborn as sns

from src.settings import LABEL, MULTIPROCESS, PLATFORM


# Function to run heavy computation
def dummy_computation(dummy_input=None):
    a = [i for i in range(10000)]
    b = [i**2 for i in range(10000)]
    return a + b


def log_memory(base_memory):
    process = psutil.Process(os.getpid())
    mem_psutil = process.memory_info().rss / (1024 * 1024)  # in MB

    return (
        round(mem_psutil - base_memory, 3),
        round(mem_psutil, 3),
    )


def plot_memory_usage(memory_df):
    # Plot memory usage for all backends in subplots
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 9))

    # Create a plot for each backend
    for backend in memory_df["backend"].unique():
        subset = memory_df[memory_df["backend"] == backend]
        sns.lineplot(x="page", y="memory_usage", data=subset, marker="o", label=backend)

    plt.title("Memory usage for all models)", fontsize=16)
    plt.xlabel("Page Number", fontsize=12)
    plt.ylabel("Memory Usage (MB)", fontsize=12)
    plt.legend(title="Backend", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"./metrics/{PLATFORM}/memory_usage_all_models_{LABEL}.png")


def plot_memory_usage_all(memory_df):
    fig, ax1 = plt.subplots(figsize=(16, 9))

    # Left axis: Cumulative memory usage
    ax1.set_ylabel("Cumulative Memory Usage (MB)", fontsize=14, color="tab:blue")
    ax1.plot(memory_df.index, memory_df["total_memory"], label="Cumulative Memory Usage", color="tab:blue")
    ax1.set_ylim(0, memory_df["total_memory"].max() * 1.2)  # Set max limit for better visual scaling
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Right axis: Inference time (in seconds)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Inference Time (seconds)", fontsize=14, color="tab:orange")

    ax2.set_ylim(0, memory_df["processing_time"].max() * 1.25)  # Set max limit for better visual scaling
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
    plt.savefig(f"./metrics/{PLATFORM}/memory_usage_all_{LABEL}.png")


def make_plots(memory_df):
    plot_memory_usage(memory_df)
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
