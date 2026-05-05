"""Publication-oriented matplotlib defaults and small layout helpers."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_rcparams(theme: str = "light") -> None:
    plt.style.use("default")
    if theme == "dark":
        plt.style.use("dark_background")

    base = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
    mpl.rcParams.update(base)


def annotate_three_values_above_bar(
    ax,
    x: float,
    y0: float,
    y1: float,
    y2: float,
    *,
    y_scale: float = 1.0,
    decimals: int = 1,
    fontsize: float = 7.0,
    color: str = "#1a1a1a",
    offset_pts: float = 14.0,
) -> None:
    """One compact line above the cluster: val0 · val1 · val2 (matches legend order)."""
    d0 = y0 * y_scale
    d1 = y1 * y_scale
    d2 = y2 * y_scale
    fmt = f"{{:.{decimals}f}}"
    sep = "\u2009·\u2009"
    text = f"{fmt.format(d0)}{sep}{fmt.format(d1)}{sep}{fmt.format(d2)}"
    ymax = max(y0, y1, y2)
    ax.annotate(
        text,
        xy=(x, ymax),
        xytext=(0, offset_pts),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
    )


def place_two_row_legend_below_ax(
    ax,
    row1_handles,
    row2_handles,
    *,
    row2_title: str | None = "Sparsity",
    ncol1: int = 5,
    ncol2: int = 3,
    bottom_margin: float = 0.44,
) -> None:
    """Two legend rows below the x-axis label (axes coordinates; avoids xlabel clash)."""
    leg1 = ax.legend(
        handles=row1_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol1,
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        framealpha=0.95,
        columnspacing=1.05,
        handletextpad=0.55,
        borderpad=0.4,
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=row2_handles,
        title=row2_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=ncol2,
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        framealpha=0.95,
        columnspacing=1.05,
        handletextpad=0.55,
        borderpad=0.4,
    )
    plt.subplots_adjust(bottom=bottom_margin, left=0.09, right=0.99, top=0.91)
