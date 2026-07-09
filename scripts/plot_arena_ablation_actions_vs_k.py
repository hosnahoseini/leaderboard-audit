#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from clean_bt_rank.plotting import ACTION_LABEL_MAP, ACTION_PALETTE, use_paper_rc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a NeurIPS-style #actions vs k plot from arena ablation results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "new_result" / "arena_ablation_reset" / "arena55k_topk_actions_needed_summary.csv",
        help="Summary CSV produced by run_arena_ablation_reset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "new_result" / "arena_ablation_reset",
        help="Directory for output PDF/PNG files.",
    )
    parser.add_argument(
        "--stem",
        default="arena55k_topk_actions_needed_curve_neurips",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional explicit plot title. Defaults to the dataset_name column when present.",
    )
    return parser.parse_args()


def _variant_order(df: pd.DataFrame) -> list[str]:
    preferred = ["flip", "drop", "add_outcomes", "add_weighted", "add_pairs"]
    present = df["variant"].drop_duplicates().tolist()
    ordered = [variant for variant in preferred if variant in present]
    ordered.extend([variant for variant in present if variant not in ordered])
    return ordered


def make_plot(summary_df: pd.DataFrame, *, title: str | None = None) -> plt.Figure:
    use_paper_rc()

    df = summary_df.copy()
    if "n_actions_plot" not in df.columns:
        df["n_actions_plot"] = df["n_actions"]
    df["variant_label"] = df["variant"].map(ACTION_LABEL_MAP).fillna(df["variant"]).str.replace("\n", " ", regex=False)

    ordered_variants = _variant_order(df)
    fig, ax = plt.subplots(figsize=(6.8, 3.7))

    for variant in ordered_variants:
        grp = df.loc[df["variant"] == variant].sort_values("k")
        color = ACTION_PALETTE.get(variant, "#264653")
        ax.plot(
            grp["k"].to_numpy(dtype=float),
            grp["n_actions_plot"].to_numpy(dtype=float),
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.3,
            zorder=3,
        )

        last = grp.iloc[-1]
        ax.annotate(
            last["variant_label"],
            xy=(float(last["k"]), float(last["n_actions_plot"])),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=9.5,
            va="center",
            ha="left",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    k_values = sorted(df["k"].unique().tolist())
    ax.set_xticks(k_values)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Top-k boundary $k$")
    ax.set_ylabel("# actions needed")
    if title is None and "dataset_name" in df.columns and len(df):
        title = str(df["dataset_name"].iloc[0])
    ax.set_title(title or "Actions Needed vs Top-k", pad=10)

    ax.grid(True, which="major", axis="y", color="#d0d0d0", linewidth=0.75, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color="#ececec", linewidth=0.55, alpha=0.9)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)
    ax.set_axisbelow(True)

    y_min = max(0.8, float(df["n_actions_plot"].min()) * 0.8)
    y_max = float(df["n_actions_plot"].max()) * 1.35
    ax.set_ylim(y_min, y_max)

    # Reserve space for direct labels on the right.
    ax.set_xlim(min(k_values), max(k_values) * 1.7)

    fig.tight_layout(pad=0.5)
    return fig


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(input_path)
    fig = make_plot(summary_df, title=args.title)
    fig.savefig(output_dir / f"{args.stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / f"{args.stem}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(output_dir / f"{args.stem}.pdf")
    print(output_dir / f"{args.stem}.png")


if __name__ == "__main__":
    main()
