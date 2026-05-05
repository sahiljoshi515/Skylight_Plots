import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

from paper_style import apply_paper_rcparams

apply_paper_rcparams("light")

data = {
    "Llama3": {
        "date": "2024-04",
        "dense": 87.89,
        "sparse": {
            "50×": 74.10,
            "20×": 83.83,
            "10×": 86.37,
            "5×": 86.90,
        },
    },
    "Qwen2.5": {
        "date": "2024-09",
        "dense": 88.38833333,
        "sparse": {
            "50×": 85.445,
            "20×": 87.38833333,
            "10×": 88.055,
            "5×": 88.66666667,
        },
    },
    "Gemma3": {
        "date": "2025-03",
        "dense": 85.855,
        "sparse": {
            "50×": 85.23333333,
            "20×": 85.88833333,
            "10×": 86.53333333,
            "5×": 86.31166667,
        },
    },
    "Ministral3": {
        "date": "2025-12",
        "dense": 91.11166667,
        "sparse": {
            "50×": 90.22166667,
            "20×": 90.77833333,
            "10×": 91.27833333,
            "5×": 91.11166667,
        },
    },
    "Qwen3.5": {
        "date": "2026-02",
        "dense": 92.77833333,
        "sparse": {
            "50×": 91,
            "20×": 97.8,
            "10×": 91,
            "5×": 91.334,
        },
    },
}

def plot_relative_sparse_performance():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y_min, y_max = 80, 110
    label_y = y_min + 0.8
    placed_annotation_positions = []

    # Collision box in data units (x-axis is matplotlib date numbers in days).
    label_collision_dx = 24
    label_collision_dy = 1.0

    # Try nearby slots around each point so labels spread in multiple directions.
    # (x offset in days, y offset in performance units, horizontal align, vertical align)
    candidate_offsets = [
        (0, 0.80, "center", "bottom"),
        (0, -1.00, "center", "top"),
        (10, 0.80, "left", "bottom"),
        (-10, 0.80, "right", "bottom"),
        (10, -1.00, "left", "top"),
        (-10, -1.00, "right", "top"),
        (16, 0.10, "left", "center"),
        (-16, 0.10, "right", "center"),
        (0, 1.45, "center", "bottom"),
        (0, -1.55, "center", "top"),
    ]

    sparsity_order = ["50×", "20×", "10×", "5×"]
    markers = {
        "50×": "o",
        "20×": "s",
        "10×": "^",
        "5×": "D",
    }

    colors = plt.cm.tab10.colors
    model_colors = {
        model: colors[i % len(colors)]
        for i, model in enumerate(data.keys())
    }

    # Small date offsets so multiple sparsity points do not overlap exactly.
    date_offsets = {
        "50×": -18,
        "20×": -6,
        "10×": 6,
        "5×": 18,
    }

    for model, d in data.items():
        base_date = datetime.strptime(d["date"], "%Y-%m")
        dense = d["dense"]

        for sparsity in sparsity_order:
            sparse_value = d["sparse"].get(sparsity)

            if sparse_value is None:
                continue

            rel_perf = (sparse_value / dense) * 100
            plot_date = mdates.date2num(base_date) + date_offsets[sparsity]

            ax.scatter(
                plot_date,
                rel_perf,
                s=160,
                marker=markers[sparsity],
                color=model_colors[model],
                edgecolors="black",
                linewidths=1.1,
                zorder=3,
                label=sparsity if model == list(data.keys())[0] else None,
            )

            # Pick the first non-colliding slot around the point.
            chosen_x, chosen_y = plot_date, rel_perf + 0.80
            chosen_ha, chosen_va = "center", "bottom"
            best_score = -1.0

            for dx, dy, ha, va in candidate_offsets:
                test_x = plot_date + dx
                test_y = rel_perf + dy

                # Keep labels inside plot area margins.
                if test_y < y_min + 0.6 or test_y > y_max - 0.6:
                    continue

                min_separation = float("inf")
                has_collision = False
                for placed_x, placed_y in placed_annotation_positions:
                    sep_x = abs(test_x - placed_x)
                    sep_y = abs(test_y - placed_y)
                    if sep_x <= label_collision_dx and sep_y <= label_collision_dy:
                        has_collision = True
                        break
                    # Larger is better; normalize by collision box size.
                    score = (sep_x / label_collision_dx) + (sep_y / label_collision_dy)
                    if score < min_separation:
                        min_separation = score

                if not has_collision:
                    chosen_x, chosen_y = test_x, test_y
                    chosen_ha, chosen_va = ha, va
                    best_score = float("inf")
                    break

                if min_separation > best_score:
                    best_score = min_separation
                    chosen_x, chosen_y = test_x, test_y
                    chosen_ha, chosen_va = ha, va

            annotation_text = ax.text(
                chosen_x,
                chosen_y,
                f"{rel_perf:.1f}",
                ha=chosen_ha,
                va=chosen_va,
                fontsize=7.5,
                fontweight="normal",
                color="black",
                zorder=5,
            )
            annotation_text.set_path_effects([
                pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)
            ])
            placed_annotation_positions.append((chosen_x, chosen_y))

        # Model label below the cluster
        ax.text(
            mdates.date2num(datetime.strptime(d["date"], "%Y-%m")),
            label_y,
            model,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.axhline(
        100,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.set_xlabel("Model release date", labelpad=10)
    ax.set_ylabel("Relative sparse vs. dense performance (%)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(
        title="Sparsity",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
    )

    plt.subplots_adjust(bottom=0.28, left=0.1, right=0.98, top=0.94)
    plt.show()


plot_relative_sparse_performance()