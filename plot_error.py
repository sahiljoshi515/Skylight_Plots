import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from bar_plot_common import MODEL_FILL_COLORS, setup_mpl_fonts, theme_colors
from paper_style import apply_paper_rcparams


def plot_output_error(theme="light"):
    apply_paper_rcparams(theme)
    tc = theme_colors(theme)
    grid_alpha = tc["grid_alpha"]
    edge_color = tc["edge_color"]
    text_color = tc["text_color"]
    dense_text_color = tc["dense_text_color"]

    data = {
        "Llama3": {
            "group": "Standard",
            "sizes": [1, 3, 8],
            "error_50x": [0.104929, 0.1406562, 0.114568],
        },
        "Qwen2.5": {
            "group": "Standard",
            "sizes": [0.5, 1.5, 3, 7, 14, 32, 72],
            "error_50x": [0.053332, 0.0449, 0.0596, 0.05522, 0.07602, 0.09422, 0.105],
        },
        "Ministral3": {
            "group": "Standard",
            "sizes": [3, 8, 14],
            "error_50x": [0.136, 0.13658, 0.1216],
        },
        "Qwen3.5": {
            "group": "Hybrid",
            "sizes": [0.8, 2, 4, 9, 27],
            "error_50x": [0.0341478, 0.0448, 0.04112, 0.045652, 0.06296],
        },
    }

    fig, ax = plt.subplots(figsize=(16, 5.2))
    bar_width = 0.70
    bar_step = 0.90
    model_gap = 1.05
    model_order = ["Llama3", "Qwen2.5", "Ministral3", "Qwen3.5"]

    max_val = float(np.max(np.abs([v for m in data.values() for v in m["error_50x"]])))
    y_top = max_val * 1.4
    dense_label_y = max(max_val * 0.025, max_val * 0.02)

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
        errors = np.array(model_data["error_50x"])

        ax.bar(
            xs,
            errors,
            width=bar_width,
            color=base_color,
            edgecolor=edge_color,
            linewidth=0.9,
            zorder=3,
        )

        for x, err in zip(xs, errors):
            ax.text(
                x,
                dense_label_y,
                f"{err:.3f}",
                ha="center",
                va="bottom",
                fontsize=fs,
                color=dense_text_color,
                fontweight="normal",
                zorder=10,
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

    ax.axhline(0, color=text_color, linewidth=1.5, alpha=0.8)

    ax.set_ylabel("Error", color=text_color, fontsize=fs_ylabel, labelpad=16)
    ax.yaxis.label.set_fontsize(fs_ylabel)
    ylabel_x_axes = -0.055
    ax.yaxis.set_label_coords(ylabel_x_axes, 0.5)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=42, ha="right", fontsize=fs, color=text_color)
    ax.tick_params(axis="y", labelsize=fs, colors=text_color)
    ax.tick_params(axis="x", labelsize=fs, pad=3, colors=text_color)

    ax.grid(True, axis="y", linestyle="--", alpha=grid_alpha)
    ax.set_ylim(0, y_top)
    ax.margins(x=0.01, y=0.05)

    measurement_handles = [
        Patch(
            facecolor="#c8c8c8",
            edgecolor=edge_color,
            linewidth=0.8,
            label="50× sparsity",
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
        ncol=1,
        loc="center",
        bbox_to_anchor=(mid_legend_x, y_top_row),
        bbox_transform=trans_top,
        frameon=True,
        fancybox=False,
        edgecolor="0.55",
        facecolor=leg_face,
        framealpha=0.98,
        fontsize=fs,
        borderpad=0.35,
    )

    ax.text(
        ylabel_x_axes,
        y_top_row,
        "50× output error",
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


plot_output_error("light")
# plot_output_error("dark")
