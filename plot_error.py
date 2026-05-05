import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

FONT_SCALE = 4
BASE_FONT_SIZE = 8

def plot_output_error(theme="light"):
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

    # -----------------------------
    # 🔥 PUT YOUR ERROR VALUES HERE
    # -----------------------------
    data = {
        "Llama3": {
            "group": "Standard",
            "sizes": [1, 3, 8],
            "error_50x": [0.104929, 0.1406562, 0.114568 ],   # <-- YOUR VALUES
        },
        "Qwen2.5": {
            "group": "Standard",
            "sizes": [0.5, 1.5, 3, 7, 14, 32, 72],
            "error_50x": [0.053332, 0.0449, 0.0596, 0.05522, 0.07602, 0.09422, 0.105],  # <-- YOUR VALUES
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
        # "Gemma3": {
        #     "group": "Hybrid",
        #     "sizes": [1, 4, 12, 27],
        #     "error_50x": [2.7, 2.3, 0.5, 0.6],
        # },
    }

    fig, ax = plt.subplots(figsize=(18, 7))

    colors = plt.cm.tab10.colors
    bar_width = 0.55
    model_gap = 1.5
    section_gap = 3

    model_order = ["Llama3", "Qwen2.5", "Ministral3", "Qwen3.5"]

    xticks, xticklabels = [], []
    group_centers = []
    section_centers = {"Standard": [], "Hybrid": []}

    x_cursor = 0

    for i, model_name in enumerate(model_order):
        model_data = data[model_name]
        base_color = colors[i % len(colors)]

        sizes = model_data["sizes"]
        xs = np.arange(x_cursor, x_cursor + len(sizes))
        errors = np.array(model_data["error_50x"])

        # ---- Error bars ----
        ax.bar(
            xs,
            errors,
            width=bar_width,
            color=base_color,
            edgecolor=edge_color,
            linewidth=1.0,
            zorder=3,
        )

        xticks.extend(xs)
        xticklabels.extend([f"{s}B" for s in sizes])
        group_centers.append(xs.mean())
        section_centers[model_data["group"]].append(xs.mean())

        # Model label
        ax.text(
            xs.mean(),
            -0.12,
            model_name,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=BASE_FONT_SIZE * 2.0,
            fontweight="bold",
        )

        x_cursor += len(sizes) + model_gap

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
            fontsize=BASE_FONT_SIZE * 2.8,
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

    # Zero line (important for error plots)
    ax.axhline(
        0,
        color=text_color,
        linewidth=1.5,
        alpha=0.8,
    )

    ax.set_ylabel("Output Error (50× Sparsity)", fontsize=BASE_FONT_SIZE * FONT_SCALE)
    ax.set_xlabel("Model Size grouped by Family", fontsize=BASE_FONT_SIZE * FONT_SCALE, labelpad=35)

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xticklabels,
        fontsize=BASE_FONT_SIZE * 2.0,
        rotation=35,
        ha="right",
    )

    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE * FONT_SCALE)
    ax.grid(True, axis="y", linestyle="--", alpha=grid_alpha)

    # Auto scale
    max_val = np.max(np.abs([v for m in data.values() for v in m["error_50x"]]))
    ax.set_ylim(0, max_val * 1.4)

    # Legend
    model_handles = [
        Patch(facecolor=colors[i % len(colors)], edgecolor=edge_color, label=name)
        for i, name in enumerate(model_order)
    ]

    ax.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=5,
        fontsize=BASE_FONT_SIZE * 2.2,
    )

    plt.subplots_adjust(top=0.78, bottom=0.25)
    plt.show()


plot_output_error("light")
plot_output_error("dark")