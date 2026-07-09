#!/usr/bin/env python3
"""Render the influence--covariate correlations as per-objective heatmap panels.

Reads the correlation table produced by ``run_arena_influence_heatmap.py`` and
draws one small heatmap per match-specification objective (rows = action type,
columns = match-level covariate), sharing a single diverging colour bar centred
at zero. This is the heatmap counterpart of the horizontal grouped-bar figure
and matches the per-objective panels shown in the thesis companion.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "new_result" / "arena_ablation_reset"

COVARIATES = ["Match count", "Bridge var.", "Closeness\n(ln-gap)", "Surprise"]

# Display order + labels for the five objective families.
OBJECTIVE_ORDER = [
    ("skill_gap__gpt-4-1106-preview__vs__gpt-4-0125-preview", "Skill gap\n(Top-1)"),
    ("ci_boundary__top1_top2__gpt-4-1106-preview__vs__gpt-4-0125-preview", "CI boundary\n(Top-1 vs 2)"),
    ("kendall_tau__fitted_ranking__T0.5", "Kendall $\\tau$"),
    ("player_uncertainty__gpt-4-1106-preview", "Player\nuncertainty"),
    ("trace_uncertainty", "Trace\nuncertainty"),
]

# Action row order + short labels (flip highlighted in the narrative).
ACTION_ORDER = ["drop", "flip", "add_all_pairs", "add_all_outcomes", "add_weighted"]
ACTION_LABELS = {
    "drop": "Drop",
    "flip": "Flip",
    "add_all_pairs": "Add-pair",
    "add_all_outcomes": "Add-out",
    "add_weighted": "Add-w",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="arena55k")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.input_dir / f"{args.dataset}__influence_covariate_correlation__1sn.csv"
    df = pd.read_csv(csv_path)

    n = len(OBJECTIVE_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(2.35 * n + 1.1, 3.4), sharey=True)
    cbar_ax = fig.add_axes([0.925, 0.24, 0.014, 0.54])

    for j, (ax, (obj_key, obj_label)) in enumerate(zip(axes, OBJECTIVE_ORDER)):
        sub = df[df["objective_key"] == obj_key].set_index("action")
        mat = sub.reindex(ACTION_ORDER)[COVARIATES]
        mat.index = [ACTION_LABELS[a] for a in ACTION_ORDER]
        sns.heatmap(
            mat,
            ax=ax,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            annot=False,
            linewidths=0.5,
            linecolor="white",
            cbar=(j == 0),
            cbar_ax=(cbar_ax if j == 0 else None),
            cbar_kws={"label": "Spearman $\\rho$"},
        )
        # Draw the cell values manually: seaborn's built-in annot mis-colours /
        # drops labels here, so place each number with contrast-aware colour.
        values = mat.to_numpy()
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                v = values[r, c]
                if not np.isfinite(v):
                    continue
                txt_color = "white" if abs(v) >= 0.55 else "#12233b"
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                )
        ax.set_title(obj_label, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        ax.tick_params(axis="y", labelsize=8, rotation=0)
        # Emphasise the Flip row (structurally most influential action).
        ax.add_patch(
            plt.Rectangle((0, 1), len(COVARIATES), 1, fill=False, edgecolor="#1A2233", lw=1.4)
        )

    fig.subplots_adjust(left=0.06, right=0.905, top=0.86, bottom=0.20, wspace=0.15)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}__influence_covariate_correlation_heatmap__1sn"
    for ext in ("pdf", "png"):
        out = args.output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
