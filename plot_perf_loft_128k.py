import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

FONT_SCALE = 4.5
BASE_FONT_SIZE = 9

def plot_clean(theme="light"):
    if theme == "dark":
        plt.style.use("dark_background")
        grid_alpha = 0.2
        edge_color = "white"
        text_color = "white"
    else:
        plt.style.use("default")
        grid_alpha = 0.35
        edge_color = "black"
        text_color = "black"

    data = {
        "Llama3": {
            "group": "Standard",
            "sizes": [1, 3, 8],
            "dense": [0.00728, 0.172725, 0.27],
            "sparse_50x": [0, 0.1848333333, 0.184],
            "sparse_5x":  [0.00322, 0.1848333333, 0.214],
        },
        "Qwen2.5": {
            "group": "Standard",
            "sizes": [0.5, 1.5, 3, 7, 14, 32],
            "dense": [0.014, 0.056, 0.032, 0.046, 0.092, 0.08],
            "sparse_50x": [0.012, 0.054, 0.036, 0.0375, 0.104, 0.09],
            "sparse_5x":  [0.014, 0.056, 0.036, 0.048, 0.096, 0.078],
        },
        "Ministral3": {
            "group": "Standard",
            "sizes": [3, 8, 14],
            "dense": [0.216, 0.326, 0.444],
            "sparse_50x": [0.196, 0.35, 0.418],
            "sparse_5x":  [0.196, 0.334, 0.446],
        },
        "Qwen3.5": {
            "group": "Hybrid",
            "sizes": [0.8, 2, 4, 9, 27],
            "dense": [0.0224, 0.162, 0.202, 0.128, 0.542],
            "sparse_50x": [0.0184, 0.142, 0.174, 0.138, 0.522],
            "sparse_5x":  [0.0224, 0.166, 0.186, 0.114, 0.60466],
        },
        # "Gemma3": {
        #     "group": "Hybrid",
        #     "sizes": [1, 4, 12, 27],
        #     "dense": [14.88833333, 44.71166667, 60.8, 85.855],
        #     "sparse_50x": [12.11166667, 42.4, 60.28833333, 85.23333333],
        #     "sparse_5x":  [14.91166667, 44.655, 60.9, 86.31166667],
        # },
    }

    fig, ax = plt.subplots(figsize=(18, 7))

    colors = plt.cm.tab10.colors
    bar_width = 0.72
    bar_step = 0.82
    line_half_width = 0.31
    model_gap = 0.8
    section_gap = 1.5

    model_order = ["Llama3", "Qwen2.5", "Ministral3", "Qwen3.5"]
    annotation_fontsize = BASE_FONT_SIZE * 2

    all_values = []
    for name in model_order:
        all_values.extend(data[name]["dense"])
        all_values.extend(data[name]["sparse_50x"])
        all_values.extend(data[name]["sparse_5x"])
    y_top = max(all_values) * 1.15 + 0.04

    xticks, xticklabels = [], []
    group_centers = []
    section_centers = {"Standard": [], "Hybrid": []}

    x_cursor = 0

    for i, model_name in enumerate(model_order):
        model_data = data[model_name]
        base_color = colors[i % len(colors)]

        sizes = model_data["sizes"]
        # Keep x positions length-exact even when x_cursor is fractional.
        xs = x_cursor + np.arange(len(sizes), dtype=float) * bar_step

        dense = np.array(model_data["dense"])
        s50 = np.array(model_data["sparse_50x"])
        s5 = np.array(model_data["sparse_5x"])

        # ---- Dense bars ----
        ax.bar(
            xs,
            dense,
            width=bar_width,
            color=base_color,
            edgecolor=edge_color,
            linewidth=1.0,
            zorder=2,
        )

        for x, y in zip(xs, dense):
            ax.text(
                x,
                y * 0.5,
                f"{y * 100:.1f}",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color="black",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, boxstyle="round,pad=0.08"),
                zorder=8,
            )

        # ---- 50× ----
        for x, y in zip(xs, s50):
            ax.hlines(
                y,
                x - line_half_width,
                x + line_half_width,
                colors=text_color,
                linestyles="--",
                linewidth=2.2,
                zorder=6,
            )
            ax.scatter(
                x,
                y,
                s=110,
                marker="o",
                color=base_color,
                edgecolor=edge_color,
                linewidth=1.5,
                zorder=7,
            )

        # ---- 5× ----
        for x, y in zip(xs, s5):
            ax.hlines(
                y,
                x - line_half_width,
                x + line_half_width,
                colors=text_color,
                linestyles=":",
                linewidth=2.6,
                zorder=6,
            )
            ax.scatter(
                x,
                y,
                s=110,
                marker="D",
                color=base_color,
                edgecolor=edge_color,
                linewidth=1.5,
                zorder=7,
            )

        # Keep higher-value annotation above lower-value annotation with
        # small adaptive offsets for minimal collision in fractional-scale data.
        for idx, (x, y50, y5) in enumerate(zip(xs, s50, s5)):
            gap = abs(y50 - y5)
            base_offset = 0.012
            extra_offset = max(0.0, 0.02 - gap) * 0.7
            offset = base_offset + extra_offset
            x_shift = 0.03 + (0.015 if (model_name == "Qwen3.5" and idx % 2) else 0.0)

            if y50 >= y5:
                y50_text, va50 = y50 + offset, "bottom"
                y5_text, va5 = y5 - offset, "top"
            else:
                y50_text, va50 = y50 - offset, "top"
                y5_text, va5 = y5 + offset, "bottom"

            # Keep text inside visible y-range.
            y50_text = min(max(y50_text, 0.01), y_top - 0.01)
            y5_text = min(max(y5_text, 0.01), y_top - 0.01)

            ax.text(
                x + x_shift,
                y50_text,
                f"{y50 * 100:.1f}",
                ha="left",
                va=va50,
                fontsize=annotation_fontsize,
                color="black",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.08"),
                zorder=8,
            )
            ax.text(
                x - x_shift,
                y5_text,
                f"{y5 * 100:.1f}",
                ha="right",
                va=va5,
                fontsize=annotation_fontsize,
                color="black",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.08"),
                zorder=8,
            )

        xticks.extend(xs)
        xticklabels.extend([f"{s}B" for s in sizes])
        group_centers.append(xs.mean())
        section_centers[model_data["group"]].append(xs.mean())

        x_cursor += len(sizes) * bar_step + model_gap
        if model_name == "Ministral3":
            x_cursor += section_gap

    # ---- Section labels ----
    for section, centers in section_centers.items():
        ax.text(
            np.mean(centers),
            1.02,
            section,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=BASE_FONT_SIZE * 3.0,
            fontweight="bold",
        )

    # Divider
    ax.axvline(
        (group_centers[2] + group_centers[3]) / 2,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
    )

    ax.set_ylabel("LOFT-128K Performance (%)", fontsize=BASE_FONT_SIZE * FONT_SCALE)
    # ax.set_xlabel("Model Size grouped by Family", fontsize=BASE_FONT_SIZE * FONT_SCALE, labelpad=35)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=BASE_FONT_SIZE * 2.2, rotation=35, ha="right")

    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE * FONT_SCALE)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.grid(True, axis="y", linestyle="--", alpha=grid_alpha)
    ax.set_ylim(0, y_top)

    # ---- Legends ----
    model_handles = [
        Patch(facecolor=colors[i % len(colors)], edgecolor=edge_color, label=name)
        for i, name in enumerate(model_order)
    ]

    measurement_handles = [
        Patch(facecolor="gray", label="Dense"),
        Line2D([0], [0], color=text_color, linestyle="--", marker="o", markersize=12, linewidth=2.4, label="50×"),
        Line2D([0], [0], color=text_color, linestyle=":", marker="D", markersize=12, linewidth=2.8, label="5×"),
    ]

    legend1 = ax.legend(
        handles=model_handles,
        # title="Model Family",
        # title_fontsize=BASE_FONT_SIZE * 1.5,
        loc="upper center",
        bbox_to_anchor=(0.3, 1.18),
        ncol=5,
        fontsize=BASE_FONT_SIZE * 2,
    )

    legend2 = ax.legend(
        handles=measurement_handles,
        title="Sparsity",
        title_fontsize=BASE_FONT_SIZE * 1.8,
        loc="upper center",
        bbox_to_anchor=(0.85, 1.18),
        ncol=3,
        fontsize=BASE_FONT_SIZE * 1.55,
        handlelength=2.2,
        handleheight=1.4,
    )

    ax.add_artist(legend1)

    plt.subplots_adjust(top=0.75, bottom=0.18)
    plt.show()


plot_clean("light")
plot_clean("dark")