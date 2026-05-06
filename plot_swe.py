def plot_clean():
    import matplotlib.pyplot as plt
    import numpy as np

    configs = ["resolved", "unresolved", "empty_patch", "error"]
    top2 = np.array([57, 24, 26, 3])
    top20 = np.array([60, 19, 34, 0])
    dense = np.array([76, 34, 4, 0])

    x = np.arange(len(configs))
    width = 0.25
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width,
        top2,
        width,
        label="50x",
        color=colors[0],
        alpha=0.5,
    )
    bars2 = ax.bar(
        x,
        top20,
        width,
        label="5x",
        color=colors[1],
        alpha=0.5,
    )
    bars3 = ax.bar(
        x + width,
        dense,
        width,
        label="dense",
        color=colors[2],
        alpha=0.5,
    )

    ax.set_title("Qwen3.5-27B on Django SWEBench", fontsize=28)
    ax.set_ylabel("# Instances", fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=26)
    ax.tick_params(axis="y", labelsize=26)

    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", fontsize=28)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.bar_label(bars1, padding=3, fontsize=28)
    ax.bar_label(bars2, padding=3, fontsize=28)
    ax.bar_label(bars3, padding=3, fontsize=28)

    plt.tight_layout()
    plt.show()


plot_clean()
