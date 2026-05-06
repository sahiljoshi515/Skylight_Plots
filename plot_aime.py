def plot_clean():
    import matplotlib.pyplot as plt
    import numpy as np

    model_name = "Qwen3.5-27B"
    task_name = "AIME2025"

    dense = 86.67
    labels = ["5x", "10x", "20x", "50x"]
    values = np.array([86.67, 90.0, 86.0, 83.0])

    x = np.arange(len(labels))
    width = 0.62
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 6))

    bar_colors = [colors[1], colors[2], colors[3], colors[0]]
    bars = ax.bar(x, values, width, color=bar_colors, alpha=0.5)

    ax.axhline(
        dense,
        color=colors[4],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"dense ({dense:.2f})",
    )

    ax.set_title(f"{model_name} on {task_name}", fontsize=28)
    ax.set_ylabel("Score", fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=26)
    ax.tick_params(axis="y", labelsize=26)

    ax.set_ylim(80, 92)
    ax.legend(loc="upper right", fontsize=28)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=28)

    plt.tight_layout()
    plt.show()


plot_clean()
