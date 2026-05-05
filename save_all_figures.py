"""Run every plot_*.py script in this folder and save the figures.

The plot scripts use plt.show() to display figures interactively.
This runner patches plt.show() so each call writes the current figure(s)
to ./figures/<script_name>_<index>.png instead of opening a window.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)

SCRIPTS = [
    "plot_error.py",
    "plot_perf_loft_128k.py",
    "plot_perf_loft_32K.py",
    "plot_perf_ruler.py",
    "plot_scatter.py",
    "plot_top_k.py",
    "plot_vattention.py",
]


def run_script(script_name: str) -> list[Path]:
    stem = Path(script_name).stem
    counter = {"i": 0}
    saved: list[Path] = []

    original_show = plt.show

    def saving_show(*_args, **_kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            counter["i"] += 1
            out_path = OUT_DIR / f"{stem}_{counter['i']}.png"
            fig.savefig(
                out_path,
                dpi=200,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
            saved.append(out_path)
            print(f"  saved {out_path.relative_to(HERE)}")
            plt.close(fig)

    plt.show = saving_show
    try:
        print(f"Running {script_name}...")
        runpy.run_path(str(HERE / script_name), run_name="__main__")
    finally:
        plt.show = original_show
        plt.close("all")

    return saved


def main() -> int:
    os.chdir(HERE)
    sys.path.insert(0, str(HERE))

    # Usage: python save_all_figures.py [script_name ...]
    # e.g. python save_all_figures.py plot_perf_ruler
    #      python save_all_figures.py plot_perf_ruler.py
    args = sys.argv[1:]
    if args:
        to_run: list[str] = []
        for a in args:
            name = a if a.endswith(".py") else f"{a}.py"
            if name not in to_run:
                to_run.append(name)
    else:
        to_run = list(SCRIPTS)

    all_saved: list[Path] = []
    for script in to_run:
        path = HERE / script
        if not path.exists():
            print(f"skip (missing): {script}")
            continue
        all_saved.extend(run_script(script))

    print()
    print(f"Done. Wrote {len(all_saved)} figure(s) to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
