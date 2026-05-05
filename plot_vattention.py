import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
            "dense": [37.47, 66.25, 87.89],
            "vattention": [36.78, 59.25, 87.18],
            "oracle_topk":  [25.38, 41.57, 74.1],
        },
        "Qwen2.5": {
            "group": "Standard",
            "sizes": [0.5, 1.5, 3, 7, 14],
            "dense": [33.33333333, 60.1, 64.53333333, 76.91166667, 82.76666667],
            "vattention": [31.43333333, 56.92166667, 63.755, 77.355, 82.67833333],
            "oracle_topk":  [24.4, 37.655, 59.01166667, 72.51166667, 75.82166667],
        },
        "Qwen3.5": {
            "group": "Hybrid",
            "sizes": [0.8, 2, 4, 9],
            "dense": [81.855, 87.255, 91.245, 89.11166667],
            "vattention": [81.86666667, 87.03333333, 90.96666667, 88.72166667],
            "oracle_topk":  [79.41166667, 87.645, 91.91166667, 88.27833333],
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

    model_order = ["Llama3", "Qwen2.5", "Qwen3.5"]
    annotation_fontsize = BASE_FONT_SIZE * 2

    all_values = []
    for name in model_order:
        all_values.extend(data[name]["dense"])
        all_values.extend(data[name]["vattention"])
        all_values.extend(data[name]["oracle_topk"])
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
        vatt = np.array(model_data["vattention"])
        oracle = np.array(model_data["oracle_topk"])

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
                f"{y:.2f}",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color="black",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, boxstyle="round,pad=0.08"),
                zorder=8,
            )

        # ---- vAttention ----
        for x, y in zip(xs, vatt):
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

        # ---- OracleTopK ----
        for x, y in zip(xs, oracle):
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
        for idx, (x, y_vatt, y_oracle) in enumerate(zip(xs, vatt, oracle)):
            gap = abs(y_vatt - y_oracle)
            base_offset = 0.012
            extra_offset = max(0.0, 0.02 - gap) * 0.7
            offset = base_offset + extra_offset
            x_shift = 0.03 + (0.015 if (model_name == "Qwen3.5" and idx % 2) else 0.0)

            if y_vatt >= y_oracle:
                y_vatt_text, va_vatt = y_vatt + offset, "bottom"
                y_oracle_text, va_oracle = y_oracle - offset, "top"
            else:
                y_vatt_text, va_vatt = y_vatt - offset, "top"
                y_oracle_text, va_oracle = y_oracle + offset, "bottom"

            # Keep text inside visible y-range.
            y_vatt_text = min(max(y_vatt_text, 0.01), y_top - 0.01)
            y_oracle_text = min(max(y_oracle_text, 0.01), y_top - 0.01)

            ax.text(
                x + x_shift,
                y_vatt_text,
                f"{y_vatt:.2f}",
                ha="left",
                va=va_vatt,
                fontsize=annotation_fontsize,
                color="black",
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.08"),
                zorder=8,
            )
            ax.text(
                x - x_shift,
                y_oracle_text,
                f"{y_oracle:.2f}",
                ha="right",
                va=va_oracle,
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
        if not centers:
            continue
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

    # Divider between Standard and Hybrid groups (if both are present)
    standard_centers = section_centers.get("Standard", [])
    hybrid_centers = section_centers.get("Hybrid", [])
    divider_x = None
    if standard_centers and hybrid_centers:
        divider_x = (standard_centers[-1] + hybrid_centers[0]) / 2
    elif len(group_centers) >= 2:
        mid = len(group_centers) // 2
        divider_x = (group_centers[mid - 1] + group_centers[mid]) / 2

    if divider_x is not None:
        ax.axvline(
            divider_x,
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
        )

    ax.set_ylabel("RULER-HARD-32K Performance", fontsize=BASE_FONT_SIZE * FONT_SCALE)
    # ax.set_xlabel("Model Size grouped by Family", fontsize=BASE_FONT_SIZE * FONT_SCALE, labelpad=35)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=BASE_FONT_SIZE * 2.2, rotation=35, ha="right")

    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE * FONT_SCALE)

    ax.grid(True, axis="y", linestyle="--", alpha=grid_alpha)
    ax.set_ylim(0, 100)

    # ---- Legends ----
    model_handles = [
        Patch(facecolor=colors[i % len(colors)], edgecolor=edge_color, label=name)
        for i, name in enumerate(model_order)
    ]

    measurement_handles = [
        Patch(facecolor="gray", label="Dense"),
        Line2D([0], [0], color=text_color, linestyle="--", marker="o", markersize=12, linewidth=2.4, label="vAttention"),
        Line2D([0], [0], color=text_color, linestyle=":", marker="D", markersize=12, linewidth=2.8, label="OracleTopK"),
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