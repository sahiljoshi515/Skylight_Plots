import matplotlib.pyplot as plt
import numpy as np

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

    x = np.arange(len(ks))
    bar_width = 0.14

    fig, ax = plt.subplots(figsize=(10, 4.2))

    colors = plt.cm.tab10.colors

    for i, size in enumerate(sizes):
        scores = []

        for k in ks:
            idx = topk_data[k]["sizes"].index(size)
            scores.append(topk_data[k]["scores"][idx])

        offset = (i - (len(sizes) - 1) / 2) * bar_width

        ax.bar(
            x + offset,
            scores,
            width=bar_width,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.6,
            label=f"{size}B",
        )

        for xx, yy in zip(x + offset, scores):
            ax.text(
                xx,
                yy + 2.5,
                f"{yy:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                rotation=0,
            )

    ax.set_ylabel("RULER-HARD-32K performance (%)")

    ax.set_xticks(x, [str(k) for k in ks])
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_ylim(0, 102)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.margins(x=0.02)

    ax.legend(
        title="Model size",
        ncol=len(sizes),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
    )

    ax.set_xlabel(r"top-$k$", labelpad=10)
    fig.subplots_adjust(bottom=0.3, left=0.1, right=0.98, top=0.94)
    plt.show()


plot_bar_by_k()