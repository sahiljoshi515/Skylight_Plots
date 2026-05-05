import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
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
            "dense": [0.01922, 0.2631, 0.53932],
            "sparse_50x": [0, 0.21688, 0.37668],
            "sparse_5x": [0.01322, 0.25532, 0.47666],
        },
        "Qwen2.5": {
            "group": "Standard",
            "sizes": [0.5, 1.5, 3, 7, 14, 32, 72],
            "dense": [0.022, 0.05734, 0.128, 0.24534, 0.294, 0.36934, 0.3138],
            "sparse_50x": [0.02, 0.054, 0.122, 0.20866, 0.27534, 0.38132, 0.2678],
            "sparse_5x": [0.026, 0.06334, 0.13268, 0.23868, 0.292, 0.38066, 0.2528],
        },
        "Ministral3": {
            "group": "Standard",
            "sizes": [3, 8, 14],
            "dense": [0.33266, 0.55866, 0.562],
            "sparse_50x": [0.26732, 0.50134, 0.57134],
            "sparse_5x": [0.31, 0.538, 0.58232],
        },
        "Qwen3.5": {
            "group": "Hybrid",
            "sizes": [0.8, 2, 4, 9, 27],
            "dense": [0.03, 0.30868, 0.25132, 0.19334, 0.628],
            "sparse_50x": [0.022, 0.284, 0.20934, 0.29134, 0.578],
            "sparse_5x": [0.026, 0.316, 0.24, 0.22, 0.61466],
        },
    }

    fig, ax = plt.subplots(figsize=(16, 5.2))
    bar_width = 0.70
    bar_step = 0.90
    line_half_width = 0.26
    model_gap = 1.05
    model_order = ["Llama3", "Qwen2.5", "Ministral3", "Qwen3.5"]

    all_values = []
    for name in model_order:
        all_values.extend(data[name]["dense"])
        all_values.extend(data[name]["sparse_50x"])
        all_values.extend(data[name]["sparse_5x"])
    y_top = max(all_values) * 1.18 + 0.05
    dense_label_y = max(0.002, min(y_top * 0.02, max(all_values) * 0.04))

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
                f"{d * 100:.1f}",
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
            ax.scatter(x, y, s=220, marker="x", c=color_50x, linewidths=1.2, zorder=7)

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
            ax.scatter(x, y, s=120, marker="+", c=color_5x, linewidths=2.0, zorder=7)

        for x, y50, y5 in zip(xs, s50, s5):
            gap = abs(y50 - y5) * 100
            off_lo = 11 if gap > 2.5 else 13
            off_hi = 11 if gap > 2.5 else 13
            if y50 <= y5:
                ax.annotate(
                    f"{y50 * 100:.1f}",
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
                    f"{y5 * 100:.1f}",
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
                    f"{y5 * 100:.1f}",
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
                    f"{y50 * 100:.1f}",
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
        ax.axvline(
            0.5 * (x_last_ministral + x_first_qwen35),
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
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.grid(True, axis="y", linestyle="--", alpha=grid_alpha)
    ax.set_ylim(0, y_top)
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
        "LOFT-32K",
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
