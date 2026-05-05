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
    color_va = tc["color_s1"]
    color_oracle = tc["color_s2"]

    data = {
        "Llama3": {
            "group": "Standard",
            "sizes": [1, 3, 8],
            "dense": [37.47, 66.25, 87.89],
            "vattention": [36.78, 59.25, 87.18],
            "oracle_topk": [25.38, 41.57, 74.1],
        },
        "Qwen2.5": {
            "group": "Standard",
            "sizes": [0.5, 1.5, 3, 7, 14],
            "dense": [33.33333333, 60.1, 64.53333333, 76.91166667, 82.76666667],
            "vattention": [31.43333333, 56.92166667, 63.755, 77.355, 82.67833333],
            "oracle_topk": [24.4, 37.655, 59.01166667, 72.51166667, 75.82166667],
        },
    }

    fig, ax = plt.subplots(figsize=(14, 5.2))
    bar_width = 0.70
    bar_step = 0.90
    line_half_width = 0.26
    model_gap = 1.05
    model_order = ["Llama3", "Qwen2.5"]

    xticks, xticklabels = [], []
    group_centers = []
    section_centers = {"Standard": [], "Hybrid": []}

    x_cursor = 0
    trans_blended = blended_transform_factory(ax.transData, ax.transAxes)
    fs = 18
    fs_ylabel = 24
    setup_mpl_fonts(fs, fs_ylabel)
    dense_label_y = 1.0

    for i, model_name in enumerate(model_order):
        model_data = data[model_name]
        base_color = MODEL_FILL_COLORS[i % len(MODEL_FILL_COLORS)]
        sizes = model_data["sizes"]
        xs = x_cursor + np.arange(len(sizes), dtype=float) * bar_step
        dense = np.array(model_data["dense"])
        vatt = np.array(model_data["vattention"])
        oracle = np.array(model_data["oracle_topk"])

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
                f"{d:.2f}",
                ha="center",
                va="bottom",
                fontsize=fs,
                color=dense_text_color,
                fontweight="normal",
                zorder=10,
            )

        for x, y in zip(xs, vatt):
            ax.hlines(
                y,
                x - line_half_width,
                x + line_half_width,
                colors=color_va,
                linestyles="--",
                linewidth=1.8,
                alpha=0.85,
                zorder=6,
            )
            ax.scatter(x, y, s=220, marker="x", c=color_va, linewidths=1.2, zorder=7)

        for x, y in zip(xs, oracle):
            ax.hlines(
                y,
                x - line_half_width,
                x + line_half_width,
                colors=color_oracle,
                linestyles=":",
                linewidth=2.0,
                alpha=0.9,
                zorder=6,
            )
            ax.scatter(x, y, s=120, marker="+", c=color_oracle, linewidths=2.0, zorder=7)

        for x, yv, yo in zip(xs, vatt, oracle):
            gap = abs(yv - yo)
            off_lo = 11 if gap > 2.5 else 13
            off_hi = 11 if gap > 2.5 else 13
            if yv <= yo:
                ax.annotate(
                    f"{yv:.2f}",
                    xy=(x, yv),
                    xytext=(0, -off_lo),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fs,
                    color=color_va,
                    zorder=11,
                )
                ax.annotate(
                    f"{yo:.2f}",
                    xy=(x, yo),
                    xytext=(0, off_hi),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=fs,
                    color=color_oracle,
                    zorder=11,
                )
            else:
                ax.annotate(
                    f"{yo:.2f}",
                    xy=(x, yo),
                    xytext=(0, -off_lo),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fs,
                    color=color_oracle,
                    zorder=11,
                )
                ax.annotate(
                    f"{yv:.2f}",
                    xy=(x, yv),
                    xytext=(0, off_hi),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=fs,
                    color=color_va,
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

        x_cursor += len(sizes) * bar_step + model_gap

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
            color=color_va,
            linestyle="--",
            marker="x",
            markersize=12,
            markerfacecolor=color_va,
            markeredgecolor=color_va,
            linewidth=1.6,
            label="vAttention",
        ),
        Line2D(
            [0],
            [0],
            color=color_oracle,
            linestyle=":",
            marker="+",
            markersize=11,
            markerfacecolor=color_oracle,
            markeredgecolor=color_oracle,
            markeredgewidth=2.0,
            linewidth=1.8,
            label="OracleTopK",
        ),
    ]

    std_centers = section_centers["Standard"]
    hyb_centers = section_centers["Hybrid"]
    mean_std = float(np.mean(std_centers)) if std_centers else float("nan")
    mean_hyb = float(np.mean(hyb_centers)) if hyb_centers else float("nan")
    if hyb_centers:
        mid_legend_x = 0.5 * (mean_std + mean_hyb)
    else:
        mid_legend_x = mean_std
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
        columnspacing=1.0,
        handletextpad=0.45,
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
    if hyb_centers:
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
