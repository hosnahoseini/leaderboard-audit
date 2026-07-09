#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_feature_summary_heatmap")

import matplotlib as mpl

mpl.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clean_bt_rank.plotting import use_paper_rc  # noqa: E402

DEFAULT_INPUT = (
    ROOT
    / "notebooks"
    / "artifacts"
    / "player_kendall_influence_plots"
    / "player_influence_joint_newton_abs_per_dataset_feature_summary_plot_data.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "notebooks" / "artifacts" / "player_kendall_influence_plots"
DEFAULT_PLAYER_STATS_DIR = DEFAULT_OUTPUT_DIR

ROW_ORDER = ["Pearson |corr|", "Spearman |corr|", "Q4-Q1 mean z", "Q4-Q1 Cohen d"]
FEATURE_ORDER = ["skill", "degree", "bridge_var", "closeness", "surprise"]

# Features shown in the published heatmap. `skill` is summarized but not plotted:
# it is the fitted BT parameter rather than a graph covariate, and its Q4-Q1 mean z
# (~-2.6) would otherwise dominate the shared color scale.
PUBLISHED_FEATURES = ["degree", "bridge_var", "closeness", "surprise"]

FEATURE_LABELS = {
    "skill": "Skill",
    "degree": "Degree",
    "bridge_var": "Bridge\nvariance",
    "closeness": "Closeness",
    "surprise": "Surprise",
}
TARGET_COLUMN = "player_influence_joint_newton_abs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a NeurIPS-style heatmap summarizing feature-vs-influence associations."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--player-stats-dir",
        type=Path,
        default=DEFAULT_PLAYER_STATS_DIR,
        help="Directory containing per-dataset *__player_kendall_player_stats.csv files used to backfill missing features.",
    )
    parser.add_argument(
        "--dataset",
        default="all_datasets",
        help="Dataset key to plot, or 'all_datasets' to average across datasets.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_ORDER,
        default=PUBLISHED_FEATURES,
        help="Feature columns to plot, in order. Defaults to the published set (skill excluded).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional output filename stem without suffix. Defaults to '<dataset>__player_influence_joint_newton_abs_feature_summary_neurips'.",
    )
    parser.add_argument(
        "--write-summary-csv",
        type=Path,
        default=None,
        help="Optional path to write the merged per-dataset feature summary table, including backfilled features such as skill.",
    )
    return parser.parse_args()


def _safe_abs_corr(feature: pd.Series, target: pd.Series, *, method: str) -> float:
    valid = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(valid) < 2:
        return float("nan")
    corr = valid["feature"].corr(valid["target"], method=method)
    return float(abs(corr)) if pd.notna(corr) else float("nan")


def _safe_cohen_d(top: pd.Series, bottom: pd.Series) -> float:
    top_vals = top.dropna().to_numpy(dtype=float)
    bottom_vals = bottom.dropna().to_numpy(dtype=float)
    if min(len(top_vals), len(bottom_vals)) < 2:
        return float("nan")
    top_var = float(np.var(top_vals, ddof=1))
    bottom_var = float(np.var(bottom_vals, ddof=1))
    pooled_num = (len(top_vals) - 1) * top_var + (len(bottom_vals) - 1) * bottom_var
    pooled_den = len(top_vals) + len(bottom_vals) - 2
    if pooled_den <= 0:
        return float("nan")
    pooled_std = float(np.sqrt(pooled_num / pooled_den))
    if pooled_std == 0.0:
        return 0.0
    return float((np.mean(top_vals) - np.mean(bottom_vals)) / pooled_std)


def _quartile_labels(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    labels = pd.Series(np.nan, index=values.index, dtype=float)
    if len(valid) < 4:
        return labels
    ranked = valid.rank(method="first")
    labels.loc[valid.index] = pd.qcut(ranked, 4, labels=False).astype(float)
    return labels


def _summarize_player_stats(player_stats: pd.DataFrame, *, dataset: str) -> list[dict[str, float | str]]:
    if TARGET_COLUMN not in player_stats.columns:
        return []

    target = player_stats[TARGET_COLUMN].astype(float)
    target_std = float(target.std(ddof=0))
    if target_std == 0.0 or not np.isfinite(target_std):
        target_z = pd.Series(np.nan, index=player_stats.index, dtype=float)
    else:
        target_z = (target - float(target.mean())) / target_std

    rows: list[dict[str, float | str]] = []
    for feature in FEATURE_ORDER:
        if feature not in player_stats.columns:
            continue
        feature_values = player_stats[feature].astype(float)
        quartiles = _quartile_labels(feature_values)
        top_mask = quartiles == 3
        bottom_mask = quartiles == 0
        rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "Pearson |corr|": _safe_abs_corr(feature_values, target, method="pearson"),
                "Spearman |corr|": _safe_abs_corr(feature_values, target, method="spearman"),
                "Q4-Q1 mean z": float(target_z.loc[top_mask].mean() - target_z.loc[bottom_mask].mean()),
                "Q4-Q1 Cohen d": _safe_cohen_d(target.loc[top_mask], target.loc[bottom_mask]),
            }
        )
    return rows


def _build_summary_from_player_stats_dir(player_stats_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for path in sorted(player_stats_dir.glob("*__player_kendall_player_stats.csv")):
        dataset_key = path.name[: -len("__player_kendall_player_stats.csv")]
        player_stats = pd.read_csv(path)
        rows.extend(_summarize_player_stats(player_stats, dataset=dataset_key))
    return pd.DataFrame(rows, columns=["dataset", "feature", *ROW_ORDER])


def _load_summary(input_csv: Path, player_stats_dir: Path) -> pd.DataFrame:
    input_df = pd.read_csv(input_csv) if input_csv.exists() else pd.DataFrame(columns=["dataset", "feature", *ROW_ORDER])
    computed_df = _build_summary_from_player_stats_dir(player_stats_dir)
    if computed_df.empty:
        return input_df

    if input_df.empty:
        return computed_df

    combined = pd.concat([input_df, computed_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["dataset", "feature"], keep="first")
    return combined


def _build_matrix(df: pd.DataFrame, dataset: str, features: list[str] | None = None) -> pd.DataFrame:
    features = list(features) if features else PUBLISHED_FEATURES
    if dataset == "all_datasets":
        plot_df = df.groupby("feature", as_index=False)[ROW_ORDER].mean(numeric_only=True)
    else:
        plot_df = df.loc[df["dataset"] == dataset, ["feature", *ROW_ORDER]].copy()
        if plot_df.empty:
            available = ", ".join(sorted(df["dataset"].unique()))
            raise ValueError(f"Unknown dataset '{dataset}'. Available datasets: {available}")
    plot_df = plot_df.set_index("feature").reindex(features)
    if plot_df.isnull().any().any():
        raise ValueError(f"Missing values found while building plot matrix for dataset '{dataset}'.")
    return plot_df.T


def _save(fig: plt.Figure, output_stem: Path) -> None:
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def render_heatmap(matrix: pd.DataFrame, *, title: str) -> plt.Figure:
    use_paper_rc()
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "xtick.labelsize": 19,
            "ytick.labelsize": 25,
            "figure.titlesize": 22,
            "figure.dpi": 180,
            "savefig.dpi": 400,
        }
    )

    values = matrix.to_numpy(dtype=float)
    vlim = max(1.0, float(np.nanmax(np.abs(values))))

    fig_width = max(10.8, 2.2 * matrix.shape[1] + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.6))
    im = ax.imshow(values, cmap="RdYlBu_r", vmin=-vlim, vmax=vlim, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([FEATURE_LABELS.get(col, col) for col in matrix.columns], rotation=0, ha="center")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.tick_params(axis="x", length=0, pad=18)
    ax.tick_params(axis="y", length=0, pad=14)

    # Thin white separators make the grid look cleaner in print without heavy borders.
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color=(1.0, 1.0, 1.0, 0.8), linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = values[row_idx, col_idx]
            text_color = "#1a1a1a" if abs(value) < 0.75 * vlim else "white"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="semibold",
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Value", rotation=90, labelpad=16)
    cbar.ax.tick_params(labelsize=18, width=0.8, length=5)
    cbar.outline.set_linewidth(0.8)

    ax.set_title(title, pad=18)
    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    args = parse_args()
    df = _load_summary(args.input_csv, args.player_stats_dir)
    if args.write_summary_csv is not None:
        args.write_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.write_summary_csv, index=False)
        print(f"Saved {args.write_summary_csv}")
    matrix = _build_matrix(df, args.dataset, args.features)

    output_stem = args.output_stem
    if output_stem is None:
        output_stem = f"{args.dataset}__player_influence_joint_newton_abs_feature_summary_neurips"
    output_path = args.output_dir.resolve() / output_stem
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = "All Datasets Summary" if args.dataset == "all_datasets" else args.dataset.replace("_", " ").title()
    fig = render_heatmap(matrix, title=title)
    _save(fig, output_path)


if __name__ == "__main__":
    main()
