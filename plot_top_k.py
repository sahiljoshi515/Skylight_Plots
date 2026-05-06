import matplotlib.pyplot as plt
import numpy as np

from bar_plot_common import MODEL_FILL_COLORS
from paper_style import apply_paper_rcparams

apply_paper_rcparams("light")

data = {
    "Qwen3.5": {
        1: {
            "sizes": [0.8, 2, 4, 9, 27],
            "scores": [34.455, 43.12166667, 46.845, 49.4, 77.655],
        },
        4: {
            "sizes": [0.8, 2, 4, 9, 27],
            "scores": [52.555, 70.655, 75.82166667, 74.555, 91.16666667],
        },
        16: {
            "sizes": [0.8, 2, 4, 9, 27],
            "scores": [60.12166667, 79.645, 84.33333333, 80.77833333, 93.16666667],
        },
        64: {
            "sizes": [0.8, 2, 4, 9, 27],
            "scores": [65.47833333, 84.58833333, 88.055, 84.61166667, 92.77833333],
        },
        128: {
            "sizes": [0.8, 2, 4, 9, 27],
            "scores": [70.71166667, 87.51166667, 88.77833333, 86.11166667, 92],
        },
    },
}

def plot_bar_by_k():
    model_name = "Qwen3.5"
    topk_data = data[model_name]

    ks = sorted(topk_data.keys())
    sizes = topk_data[ks[0]]["sizes"]

    x = np.arange(len(ks)) * 0.82
    bar_width = 0.13

    fig, ax = plt.subplots(figsize=(12, 5.6))

    colors = MODEL_FILL_COLORS
    bar_annotations = []

    for i, size in enumerate(sizes):
        scores = []

        for k in ks:
            idx = topk_data[k]["sizes"].index(size)
            scores.append(topk_data[k]["scores"][idx])

        offset = (i - (len(sizes) - 1) / 2) * bar_width

        bars = ax.bar(
            x + offset,
            scores,
            width=bar_width,
            color=colors[i % len(colors)],
            edgecolor="#6a6a6a",
            linewidth=0.6,
            label=f"{size}B",
        )

        for bar, yy in zip(bars, scores):
            bar_annotations.append((bar.get_x() + bar.get_width() / 2, yy))

    fig.canvas.draw()
    bar_width_pixels = abs(ax.transData.transform((bar_width, 0))[0] - ax.transData.transform((0, 0))[0])
    annotation_fontsize = 28
    fs = 26
    text_color = "#1f1f1f"

    for xx, yy in bar_annotations:
        ax.text(
            xx,
            yy * 0.5,
            f"{yy:.1f}",
            ha="center",
            va="center",
            fontsize=annotation_fontsize,
            fontweight="bold",
            rotation=90,
            color="#1f1f1f",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=30)

    ax.set_xticks(x, [str(k) for k in ks])
    ax.tick_params(axis="x", labelsize=26)
    ax.tick_params(axis="y", labelsize=26)

    ax.set_ylim(0, 102)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.margins(x=0.02)

    ax.legend(
        # title="Model size",
        ncol=len(sizes),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=15,
        title_fontsize=16,
    )

    ylabel_x_axes = -0.055
    y_top_row = 1.08
    ax.text(
        ylabel_x_axes,
        y_top_row,
        "RULER-HARD-32K",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=text_color,
        zorder=11,
    )

    ax.set_xlabel(r"top-$k$", labelpad=10, fontsize=30)
    fig.subplots_adjust(bottom=0.2, left=0.1, right=0.98, top=0.9)
    plt.show()


plot_bar_by_k()
