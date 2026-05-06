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

    tasks = ["AIME2025", "SWEBench"]
    model_name = "Qwen3.5-27B"
    dense = np.array([86.67, 53.0])
    s5 = np.array([86.67, 53.09])
    s50 = np.array([86.67, 51.8])

    fig, ax = plt.subplots(figsize=(16, 5.2))

    bar_width = 0.70
    bar_step = 1.20
    line_half_width = 0.26
    xticks, xticklabels = [], []
    trans_blended = blended_transform_factory(ax.transData, ax.transAxes)

    fs = 54
    fs_ylabel = 72
    setup_mpl_fonts(fs, fs_ylabel)
    axis_fs = fs / 2
    axis_label_fs = fs_ylabel / 2
    header_fs = fs / 2
    legend_fs = fs / 2
    annotation_fs = fs / 1.2
    # Same y for every dense label (performance % scale), just above the axis
    dense_label_y = 1.0

    bar_colors = [
        MODEL_FILL_COLORS[0 % len(MODEL_FILL_COLORS)],
        MODEL_FILL_COLORS[1 % len(MODEL_FILL_COLORS)],
    ]
    xs = np.arange(len(tasks), dtype=float) * bar_step

    ax.bar(
        xs,
        dense,
        width=bar_width,
        color=bar_colors,
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
            fontsize=annotation_fs,
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
                    fontsize=annotation_fs,
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
                    fontsize=annotation_fs,
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
                    fontsize=annotation_fs,
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
                    fontsize=annotation_fs,
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
        fontsize=header_fs,
        fontweight="bold",
        color=text_color,
    )

    xticks.extend(xs)
    xticklabels.extend(tasks)

    ax.set_ylabel("Accuracy (%)", color=text_color, fontsize=axis_label_fs, labelpad=16)
    ax.yaxis.label.set_fontsize(axis_label_fs)
    ylabel_x_axes = -0.055
    ax.yaxis.set_label_coords(ylabel_x_axes, 0.5)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=42, ha="right", fontsize=axis_fs, color=text_color)

    ax.tick_params(axis="y", labelsize=axis_fs, colors=text_color)
    ax.tick_params(axis="x", labelsize=axis_fs, pad=3, colors=text_color)

    ax.grid(True, axis="y", linestyle="--", alpha=grid_alpha)
    ax.set_ylim(0, 100)
    ax.margins(x=0.01, y=0.05)

    measurement_handles = [
        Patch(facecolor=bar_colors[0], edgecolor=edge_color, linewidth=0.8, label="Dense"),
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

    mid_legend_x = gc
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
        fontsize=legend_fs,
        columnspacing=1.2,
        handletextpad=0.5,
        borderpad=0.28,
    )

    ax.text(
        ylabel_x_axes,
        y_top_row,
        "Longform Generation",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=header_fs,
        fontweight="bold",
        color=text_color,
        zorder=11,
    )

    plt.subplots_adjust(bottom=0.24, left=0.13, right=0.99, top=0.76)
    plt.show()


plot_clean("light")
# plot_clean("dark")
