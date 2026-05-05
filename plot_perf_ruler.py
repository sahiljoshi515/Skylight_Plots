import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from bar_plot_common import MODEL_FILL_COLORS, setup_mpl_fonts, theme_colors
from paper_style import apply_paper_rcparams


def plot_clean(theme="light"):
    apply_paper_rcparams(theme)
    tc = theme_colors(theme)
    grid_alpha = tc["grid_alpha"]
    edge_color = tc["edge_color"]
    text_color = tc["text_color"]
    dense_text_color = tc["dense_text_color"]
    color_50x = tc["color_s1"]
    color_5x = tc["color_s2"]

    data = {
        "Llama3": {
            "group": "Standard",
            "sizes": [1, 3, 8],
            "dense": [37.47, 66.25, 87.89],
            "sparse_50x": [25.38, 41.57, 87.18],
            "sparse_5x":  [35.89, 59, 86.9],
        },
        "Qwen2.5": {
            "group": "Standard",
            "sizes": [0.5, 1.5, 3, 7, 14, 32, 72],
            "dense": [33.33333333, 60.1, 64.53333333, 76.91166667, 82.76666667, 86.22166667, 88.38833333],
            "sparse_50x": [24.4, 37.655, 59.01166667, 72.51166667, 75.82166667, 82.5, 85.445],
            "sparse_5x":  [33.18833333, 59.96666667, 64.645, 77.72166667, 82.28833333, 85.66666667, 88.66666667],
        },
        "Ministral3": {
            "group": "Standard",
            "sizes": [3, 8, 14],
            "dense": [86.16666667, 89.27833333, 91.11166667],
            "sparse_50x": [76.445, 87.83333333, 90.22166667],
            "sparse_5x":  [85.83333333, 88.945, 91.11166667],
        },
        "Qwen3.5": {
            "group": "Hybrid",
            "sizes": [0.8, 2, 4, 9, 27],
            "dense": [81.855, 87.255, 91.245, 89.11166667, 92.77833333],
            "sparse_50x": [79.41166667, 87.645, 91.91166667, 88.27833333, 91],
            "sparse_5x":  [81.7, 87.31166667, 91.07833333, 89.27833333, 91.334],
        },
        "Gemma3": {
            "group": "Hybrid",
            "sizes": [1, 4, 12, 27],
            "dense": [14.88833333, 44.71166667, 60.8, 85.855],
            "sparse_50x": [12.11166667, 42.4, 60.28833333, 85.23333333],
            "sparse_5x":  [14.91166667, 44.655, 60.9, 86.31166667],
        },
    }

    fig, ax = plt.subplots(figsize=(16, 5.2))

    bar_width = 0.70
    bar_step = 0.90
    line_half_width = 0.26
    model_gap = 1.05

    model_order = ["Llama3", "Qwen2.5", "Ministral3", "Qwen3.5", "Gemma3"]

    xticks, xticklabels = [], []
    group_centers = []
    section_centers = {"Standard": [], "Hybrid": []}
    x_last_ministral = None
    x_first_qwen35 = None

    x_cursor = 0
    trans_blended = blended_transform_factory(ax.transData, ax.transAxes)

    fs = 18
    fs_ylabel = 24
    setup_mpl_fonts(fs, fs_ylabel)
    # Same y for every dense label (performance % scale), just above the axis
    dense_label_y = 1.0

    for i, model_name in enumerate(model_order):
        model_data = data[model_name]
        base_color = MODEL_FILL_COLORS[i % len(MODEL_FILL_COLORS)]

        sizes = model_data["sizes"]
        xs = x_cursor + np.arange(len(sizes), dtype=float) * bar_step

        dense = np.array(model_data["dense"])
        s50 = np.array(model_data["sparse_50x"])
        s5 = np.array(model_data["sparse_5x"])

        ax.bar(
            xs,
            dense,
            width=bar_width,
            color=base_color,
            edgecolor=edge_color,
            linewidth=0.9,
            zorder=2,
        )

        for x, d in zip(xs, dense):
            ax.text(
                x,
                dense_label_y,
                f"{d:.1f}",
                ha="center",
                va="bottom",
                fontsize=fs,
                color=dense_text_color,
                fontweight="normal",
                zorder=10,
            )

        for x, y in zip(xs, s50):
            ax.hlines(
                y,
                x - line_half_width,
                x + line_half_width,
                colors=color_50x,
                linestyles="--",
                linewidth=1.8,
                alpha=0.85,
                zorder=6,
            )
            ax.scatter(
                x,
                y,
                s=220,
                marker="x",
                c=color_50x,
                linewidths=1.2,
                zorder=7,
            )

        for x, y in zip(xs, s5):
            ax.hlines(
                y,
                x - line_half_width,
                x + line_half_width,
                colors=color_5x,
                linestyles=":",
                linewidth=2.0,
                alpha=0.9,
                zorder=6,
            )
            ax.scatter(
                x,
                y,
                s=120,
                marker="+",
                c=color_5x,
                linewidths=2.0,
                zorder=7,
            )

        for x, y50, y5 in zip(xs, s50, s5):
            gap = abs(y50 - y5)
            off_lo = 11 if gap > 2.5 else 13
            off_hi = 11 if gap > 2.5 else 13
            if y50 <= y5:
                ax.annotate(
                    f"{y50:.1f}",
                    xy=(x, y50),
                    xytext=(0, -off_lo),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fs,
                    color=color_50x,
                    zorder=11,
                )
                ax.annotate(
                    f"{y5:.1f}",
                    xy=(x, y5),
                    xytext=(0, off_hi),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=fs,
                    color=color_5x,
                    zorder=11,
                )
            else:
                ax.annotate(
                    f"{y5:.1f}",
                    xy=(x, y5),
                    xytext=(0, -off_lo),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fs,
                    color=color_5x,
                    zorder=11,
                )
                ax.annotate(
                    f"{y50:.1f}",
                    xy=(x, y50),
                    xytext=(0, off_hi),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=fs,
                    color=color_50x,
                    zorder=11,
                )

        gc = float(xs.mean())
        ax.text(
            gc,
            -0.17,
            model_name,
            transform=trans_blended,
            ha="center",
            va="top",
            fontsize=fs,
            fontweight="bold",
            color=text_color,
        )

        xticks.extend(xs)
        xticklabels.extend([f"{s}B" for s in sizes])
        group_centers.append(gc)
        section_centers[model_data["group"]].append(gc)

        if model_name == "Ministral3":
            x_last_ministral = float(xs[-1])
        if model_name == "Qwen3.5":
            x_first_qwen35 = float(xs[0])

        x_cursor += len(sizes) * bar_step + model_gap

    if x_last_ministral is not None and x_first_qwen35 is not None:
        divider_x = 0.5 * (x_last_ministral + x_first_qwen35)
        ax.axvline(
            divider_x,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.55,
        )

    ax.set_ylabel("Accuracy (%)", color=text_color, fontsize=fs_ylabel, labelpad=16)
    ax.yaxis.label.set_fontsize(fs_ylabel)
    ylabel_x_axes = -0.055
    ax.yaxis.set_label_coords(ylabel_x_axes, 0.5)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=42, ha="right", fontsize=fs, color=text_color)

    ax.tick_params(axis="y", labelsize=fs, colors=text_color)
    ax.tick_params(axis="x", labelsize=fs, pad=3, colors=text_color)

    ax.grid(True, axis="y", linestyle="--", alpha=grid_alpha)
    ax.set_ylim(0, 100)
    ax.margins(x=0.01, y=0.05)

    measurement_handles = [
        Patch(facecolor="#c8c8c8", edgecolor=edge_color, linewidth=0.8, label="Dense"),
        Line2D(
            [0],
            [0],
            color=color_50x,
            linestyle="--",
            marker="x",
            markersize=12,
            markerfacecolor=color_50x,
            markeredgecolor=color_50x,
            linewidth=1.6,
            label="50×",
        ),
        Line2D(
            [0],
            [0],
            color=color_5x,
            linestyle=":",
            marker="+",
            markersize=11,
            markerfacecolor=color_5x,
            markeredgecolor=color_5x,
            markeredgewidth=2.0,
            linewidth=1.8,
            label="5×",
        ),
    ]

    mean_std = float(np.mean(section_centers["Standard"]))
    mean_hyb = float(np.mean(section_centers["Hybrid"]))
    mid_legend_x = 0.5 * (mean_std + mean_hyb)
    trans_top = blended_transform_factory(ax.transData, ax.transAxes)
    y_top_row = 1.17

    leg_face = "white" if theme == "light" else "#2a2a2a"
    ax.legend(
        handles=measurement_handles,
        ncol=3,
        loc="center",
        bbox_to_anchor=(mid_legend_x, y_top_row),
        bbox_transform=trans_top,
        frameon=True,
        fancybox=False,
        edgecolor="0.55",
        facecolor=leg_face,
        framealpha=0.98,
        fontsize=fs,
        columnspacing=1.2,
        handletextpad=0.5,
        borderpad=0.28,
    )

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

    ax.text(
        mean_std,
        y_top_row,
        "Standard",
        transform=trans_top,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=text_color,
        zorder=11,
    )
    ax.text(
        mean_hyb,
        y_top_row,
        "Hybrid",
        transform=trans_top,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=text_color,
        zorder=11,
    )

    plt.subplots_adjust(bottom=0.24, left=0.13, right=0.99, top=0.76)
    plt.show()


plot_clean("light")
# plot_clean("dark")
