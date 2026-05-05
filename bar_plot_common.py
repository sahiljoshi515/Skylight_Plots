"""Shared styling for benchmark bar plots (aligned with plot_perf_ruler template)."""

import matplotlib as mpl

MODEL_FILL_COLORS = [
    "#cfe2f3",
    "#fde9d9",
    "#e2efd9",
    "#f5dede",
    "#e8e0f2",
]


def theme_colors(theme: str) -> dict:
    if theme == "dark":
        return {
            "grid_alpha": 0.2,
            "edge_color": "#444444",
            "text_color": "white",
            "dense_text_color": "#f5f5f5",
            "color_s1": "#64b5f6",
            "color_s2": "#ef5350",
        }
    return {
        "grid_alpha": 0.45,
        "edge_color": "#6a6a6a",
        "text_color": "#2a2a2a",
        "dense_text_color": "#1e1e1e",
        "color_s1": "#1565c0",
        "color_s2": "#c62828",
    }


def setup_mpl_fonts(fs: int, fs_ylabel: int) -> None:
    mpl.rcParams.update(
        {
            "axes.labelsize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "legend.fontsize": fs,
            "font.size": fs,
        }
    )
