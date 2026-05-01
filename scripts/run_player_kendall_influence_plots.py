#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_player_kendall_plots")

import matplotlib as mpl

mpl.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT, ROOT / "IsRankingRobust"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clean_bt_rank import (  # noqa: E402
    BattleDataset,
    BradleyTerryModel,
    available_hf_battle_datasets,
    compute_player_influence,
    compute_player_statistics,
    drop_player_and_measure_effect,
    kendall_tau_match_influence_report,
    load_named_battle_data,
    ranking_from_model,
)
from clean_bt_rank.player_influence import _fit_frame_with_indices, _refit_from_fitted_frame  # noqa: E402
from clean_bt_rank.plotting import use_paper_rc  # noqa: E402

KENDALL_TEMPERATURE = 0.1
INFLUENCE_METHOD = "1sn"

HIGH_RANK_COLORS = ["#E63946", "#D62828", "#F4A261"]
LOW_RANK_COLORS = ["#1D3557", "#457B9D", "#8ECAE6"]
NEUTRAL_LINE = "#9AA0A6"
HIST_COLOR = "#D9D9D9"
EDGE_COLOR = "#222222"


@dataclass
class DatasetBundle:
    key: str
    display_name: str
    raw: pd.DataFrame
    fit_kwargs: dict
    n_matches: int
    n_players: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot player Kendall influence distribution markers and most-influential-player rank shifts across datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=sorted(available_hf_battle_datasets()),
        help="Dataset keys to run. Default: all available datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "notebooks" / "artifacts" / "player_kendall_influence_plots",
        help="Directory for plot outputs. Default: notebooks/artifacts/player_kendall_influence_plots.",
    )
    parser.add_argument(
        "--distribution-top-n",
        type=int,
        default=3,
        help="Number of highest and lowest influence players to mark in the distribution plots.",
    )
    parser.add_argument(
        "--rank-top-k",
        type=int,
        default=20,
        help="Show players whose before or after rank is within the top-k ranks in the rank-shift plot.",
    )
    return parser.parse_args()


def _style_axis(ax: plt.Axes, *, grid_axis: str) -> None:
    ax.grid(True, axis=grid_axis, color="#C7C7C7", linewidth=0.7, alpha=0.7)
    other_axis = "x" if grid_axis == "y" else "y"
    ax.grid(False, axis=other_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)
    ax.set_axisbelow(True)


def _short_label(name: str, max_len: int = 30) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _save_figure(fig: plt.Figure, path_stem: Path) -> None:
    pdf_path = path_stem.with_suffix(".pdf")
    png_path = path_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    print(f"Saved {path}")


def _load_dataset_bundle(dataset_key: str) -> DatasetBundle:
    loaded = load_named_battle_data(dataset_key)
    raw = loaded.battle_frame.copy()
    players = pd.unique(pd.concat([raw["model_a"], raw["model_b"]], ignore_index=True))
    return DatasetBundle(
        key=loaded.key,
        display_name=loaded.display_name,
        raw=raw,
        fit_kwargs=dict(loaded.fit_kwargs),
        n_matches=int(len(raw)),
        n_players=int(len(players)),
    )


def _resolve_dataset_bundles(dataset_keys: list[str]) -> list[DatasetBundle]:
    bundles = [_load_dataset_bundle(dataset_key) for dataset_key in dataset_keys]
    bundles.sort(key=lambda bundle: (bundle.n_matches, bundle.n_players, bundle.key))
    return bundles


def _plot_distribution_markers(
    *,
    dataset_name: str,
    dataset_slug: str,
    player_stats: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    numeric_cols = [
        col
        for col in ["skill", "rating", "degree", "bridge_var", "closeness", "surprise"]
        if col in player_stats.columns and pd.api.types.is_numeric_dtype(player_stats[col])
    ]
    if not numeric_cols:
        print(f"Skipping distribution markers for {dataset_slug}: no numeric player-stat columns found.")
        return

    top_n = max(1, min(top_n, len(player_stats)))
    highest = player_stats.nlargest(top_n, "player_influence_joint_newton_abs")
    lowest = player_stats.nsmallest(top_n, "player_influence_joint_newton_abs")
    influence = player_stats["player_influence_joint_newton_abs"].to_numpy(dtype=float)

    n_cols = min(3, len(numeric_cols))
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.7 * n_rows),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, numeric_cols):
        values = player_stats[col].astype(float).to_numpy()
        valid = np.isfinite(values)
        valid_corr = valid & np.isfinite(influence)
        bins = min(36, max(12, int(np.sqrt(valid.sum()))))
        weights = np.full(valid.sum(), 1.0 / max(valid.sum(), 1), dtype=float)

        ax.hist(
            values[valid],
            bins=bins,
            weights=weights,
            color=HIST_COLOR,
            edgecolor="white",
            linewidth=0.6,
        )
        for rank, (_, row) in enumerate(highest.iterrows()):
            ax.axvline(float(row[col]), color=HIGH_RANK_COLORS[rank % len(HIGH_RANK_COLORS)], lw=2.4, zorder=3)
        for rank, (_, row) in enumerate(lowest.iterrows()):
            ax.axvline(float(row[col]), color=LOW_RANK_COLORS[rank % len(LOW_RANK_COLORS)], lw=2.4, zorder=3)

        if valid_corr.sum() >= 2:
            pearson_r = float(np.corrcoef(values[valid_corr], influence[valid_corr])[0, 1])
            spearman_r = float(
                pd.Series(values[valid_corr]).corr(pd.Series(influence[valid_corr]), method="spearman")
            )
            corr_text = f"r={pearson_r:.2f}, ρ={spearman_r:.2f}"
        else:
            corr_text = "r=NA, ρ=NA"

        ax.set_title(col.replace("_", " ").title(), pad=8, fontweight="semibold")
        ax.text(
            0.02,
            0.98,
            corr_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Share of players")
        _style_axis(ax, grid_axis="y")

    for ax in axes[len(numeric_cols) :]:
        ax.set_visible(False)

    handles = []
    for rank, (_, row) in enumerate(highest.iterrows()):
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color=HIGH_RANK_COLORS[rank % len(HIGH_RANK_COLORS)],
                lw=2.4,
                label=f"High |τ| #{rank + 1}: {_short_label(str(row['player']))}",
            )
        )
    for rank, (_, row) in enumerate(lowest.iterrows()):
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color=LOW_RANK_COLORS[rank % len(LOW_RANK_COLORS)],
                lw=2.4,
                label=f"Low |τ| #{rank + 1}: {_short_label(str(row['player']))}",
            )
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(3, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.suptitle(
        f"{dataset_name}: player-stat distributions with extreme |Kendall τ| markers",
        y=1.08,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935), pad=0.95)
    _save_figure(fig, output_dir / f"{dataset_slug}__player_kendall_influence_distribution_markers")


def _plot_rank_shift(
    *,
    dataset_name: str,
    dataset_slug: str,
    raw: pd.DataFrame,
    bt_model: BradleyTerryModel,
    player_stats: pd.DataFrame,
    output_dir: Path,
    rank_top_k: int,
) -> dict:
    most_influential_player = (
        player_stats.sort_values("player_influence_joint_newton_abs", ascending=False).iloc[0]["player"]
    )
    fit_frame = _fit_frame_with_indices(bt_model)
    filtered_frame = fit_frame.loc[
        (fit_frame["model_a"] != most_influential_player) & (fit_frame["model_b"] != most_influential_player)
    ].reset_index(drop=True)
    refit_after_drop = _refit_from_fitted_frame(bt_model, filtered_frame)

    before_remaining = [player for player in ranking_from_model(bt_model) if player != most_influential_player]
    after_remaining = ranking_from_model(refit_after_drop)
    drop_result = drop_player_and_measure_effect(
        bt_model,
        player=most_influential_player,
        player_stats=player_stats,
        temperature=KENDALL_TEMPERATURE,
    )

    rank_shift = pd.DataFrame({"player": before_remaining})
    rank_shift["before_rank"] = np.arange(1, len(rank_shift) + 1)
    rank_shift["after_rank"] = rank_shift["player"].map({player: i + 1 for i, player in enumerate(after_remaining)})
    rank_shift["rank_change"] = rank_shift["before_rank"] - rank_shift["after_rank"]
    rank_shift["direction"] = np.where(
        rank_shift["rank_change"] > 0,
        "up",
        np.where(rank_shift["rank_change"] < 0, "down", "unchanged"),
    )
    rank_shift = rank_shift.sort_values(["after_rank", "before_rank"]).reset_index(drop=True)
    csv_path = output_dir / f"{dataset_slug}_most_influential_rank_shift.csv"
    rank_shift.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    focus = rank_shift.sort_values("before_rank").copy()
    focus["display_order"] = np.arange(len(focus), 0, -1)
    plot_shift = focus.loc[focus[["before_rank", "after_rank"]].min(axis=1) <= rank_top_k].copy()
    if plot_shift.empty:
        print(f"Skipping rank-shift plot for {dataset_slug}: no rows within top-{rank_top_k}.")
        return {
            "most_influential_player": str(most_influential_player),
            "rank_shift_csv": str(csv_path),
            "rank_shift_plot_pdf": None,
            "rank_shift_plot_png": None,
            "n_matches_dropped": int(raw["model_a"].shape[0] - len(filtered_frame) // 2),
            "players_with_rank_change": int((rank_shift["rank_change"] != 0).sum()),
            "predicted_delta_joint_newton": float(drop_result.predicted_delta_joint_newton),
            "actual_smooth_delta": float(drop_result.actual_smooth_delta),
            "actual_rank_tau": float(drop_result.actual_rank_tau),
            "joint_newton_abs_influence": float(
                player_stats.set_index("player").loc[most_influential_player, "player_influence_joint_newton_abs"]
            ),
        }

    plot_shift = plot_shift.sort_values("display_order", ascending=True)
    fig_height = max(5.4, 0.42 * len(plot_shift))
    fig, ax = plt.subplots(figsize=(11.8, fig_height))

    for _, row in plot_shift.iterrows():
        ax.plot(
            [row["before_rank"], row["after_rank"]],
            [row["display_order"], row["display_order"]],
            color=NEUTRAL_LINE,
            linewidth=2.0,
            alpha=0.9,
            zorder=1,
        )

    ax.scatter(
        plot_shift["before_rank"],
        plot_shift["display_order"],
        color=LOW_RANK_COLORS[0],
        s=78,
        alpha=0.5,
        edgecolor="white",
        linewidth=0.7,
        label="Before drop",
        zorder=3,
    )
    ax.scatter(
        plot_shift["after_rank"],
        plot_shift["display_order"],
        color=HIGH_RANK_COLORS[0],
        s=78,
        alpha=0.5,
        edgecolor="white",
        linewidth=0.7,
        label="After drop",
        zorder=3,
    )

    ax.set_yticks(plot_shift["display_order"].to_numpy())
    ax.set_yticklabels([_short_label(str(name), max_len=34) for name in plot_shift["player"]], fontsize=12)
    ax.tick_params(axis="y", length=0, pad=9)
    plt.setp(ax.get_yticklabels(), ha="right")

    max_label_x = float(plot_shift[["before_rank", "after_rank"]].max().max())
    for _, row in plot_shift.iterrows():
        if int(row["rank_change"]) != 0:
            ax.text(
                max(row["before_rank"], row["after_rank"]) + 0.35,
                row["display_order"],
                f"{int(row['rank_change']):+d}",
                va="center",
                ha="left",
                fontsize=12,
                fontweight="semibold",
                color=EDGE_COLOR,
            )
            max_label_x = max(max_label_x, max(row["before_rank"], row["after_rank"]) + 1.0)

    n_matches_dropped = int(raw["model_a"].shape[0] - len(filtered_frame) // 2)
    influence_value = float(
        player_stats.set_index("player").loc[most_influential_player, "player_influence_joint_newton_abs"]
    )
    subtitle = (
        f"dropped player: {_short_label(str(most_influential_player), max_len=42)}"
        f" | matches dropped: {n_matches_dropped}"
        f" | predicted Δτ: {drop_result.predicted_delta_joint_newton:.3f}"
        f" | actual τ: {drop_result.actual_rank_tau:.3f}"
        f" | |τ|-influence: {influence_value:.3f}"
    )

    fig.suptitle(
        f"{dataset_name}: ranking shift after dropping the most influential player",
        y=0.988,
        fontweight="semibold",
    )
    fig.text(0.20, 0.958, subtitle, ha="left", va="top", fontsize=11.5)
    ax.set_xlabel("Rank among remaining players")
    ax.set_ylabel("Player")
    ax.set_ylim(plot_shift["display_order"].min() - 0.6, plot_shift["display_order"].max() + 0.6)
    ax.set_xlim(0.5, max(rank_top_k + 1.4, max_label_x))
    _style_axis(ax, grid_axis="x")
    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0.01, 0.02), borderaxespad=0.0)
    fig.tight_layout(rect=(0.20, 0.03, 0.99, 0.93), pad=0.95)
    plot_stem = output_dir / f"{dataset_slug}_player_kendall_most_influential_rank_shift"
    _save_figure(fig, plot_stem)
    return {
        "most_influential_player": str(most_influential_player),
        "rank_shift_csv": str(csv_path),
        "rank_shift_plot_pdf": str(plot_stem.with_suffix(".pdf")),
        "rank_shift_plot_png": str(plot_stem.with_suffix(".png")),
        "n_matches_dropped": n_matches_dropped,
        "players_with_rank_change": int((rank_shift["rank_change"] != 0).sum()),
        "predicted_delta_joint_newton": float(drop_result.predicted_delta_joint_newton),
        "actual_smooth_delta": float(drop_result.actual_smooth_delta),
        "actual_rank_tau": float(drop_result.actual_rank_tau),
        "joint_newton_abs_influence": influence_value,
    }


def run_dataset(
    bundle: DatasetBundle,
    *,
    output_dir: Path,
    distribution_top_n: int,
    rank_top_k: int,
) -> dict:
    dataset = BattleDataset.from_dataframe(bundle.raw)
    bt_model = BradleyTerryModel.from_dataset(dataset, **bundle.fit_kwargs).fit()

    ranking = ranking_from_model(bt_model)
    match_report = kendall_tau_match_influence_report(
        bt_model,
        ranking=ranking,
        temperature=KENDALL_TEMPERATURE,
        method=INFLUENCE_METHOD,
    )
    player_influence = compute_player_influence(
        bt_model,
        match_influence_report=match_report,
        temperature=KENDALL_TEMPERATURE,
        method=INFLUENCE_METHOD,
    )
    player_stats = compute_player_statistics(
        bt_model,
        player_influence=player_influence,
        match_influence_report=match_report,
        temperature=KENDALL_TEMPERATURE,
        method=INFLUENCE_METHOD,
    )

    print(f"Plotting {bundle.key} ({bundle.display_name}) | matches={bundle.n_matches} players={bundle.n_players}")
    player_stats_path = output_dir / f"{bundle.key}__player_kendall_player_stats.csv"
    match_report_path = output_dir / f"{bundle.key}__player_kendall_match_report.csv"
    player_stats.to_csv(player_stats_path, index=False)
    match_report.to_csv(match_report_path, index=False)
    print(f"Saved {player_stats_path}")
    print(f"Saved {match_report_path}")

    _plot_distribution_markers(
        dataset_name=bundle.display_name,
        dataset_slug=bundle.key,
        player_stats=player_stats,
        output_dir=output_dir,
        top_n=distribution_top_n,
    )
    rank_shift_summary = _plot_rank_shift(
        dataset_name=bundle.display_name,
        dataset_slug=bundle.key,
        raw=bundle.raw,
        bt_model=bt_model,
        player_stats=player_stats,
        output_dir=output_dir,
        rank_top_k=rank_top_k,
    )
    plot_inputs_path = output_dir / f"{bundle.key}__player_kendall_plot_inputs.json"
    plot_inputs = {
        "dataset_key": bundle.key,
        "dataset_display_name": bundle.display_name,
        "n_matches": bundle.n_matches,
        "n_players": bundle.n_players,
        "fit_kwargs": bundle.fit_kwargs,
        "kendall_temperature": KENDALL_TEMPERATURE,
        "influence_method": INFLUENCE_METHOD,
        "distribution_top_n": distribution_top_n,
        "rank_top_k": rank_top_k,
        "player_stats_csv": str(player_stats_path),
        "match_report_csv": str(match_report_path),
        "distribution_columns": [
            col
            for col in ["skill", "rating", "degree", "bridge_var", "closeness", "surprise"]
            if col in player_stats.columns and pd.api.types.is_numeric_dtype(player_stats[col])
        ],
        "distribution_plot_pdf": str((output_dir / f"{bundle.key}__player_kendall_influence_distribution_markers").with_suffix(".pdf")),
        "distribution_plot_png": str((output_dir / f"{bundle.key}__player_kendall_influence_distribution_markers").with_suffix(".png")),
        "rank_shift_summary": rank_shift_summary,
    }
    _write_json(plot_inputs_path, plot_inputs)
    return {
        "dataset_key": bundle.key,
        "dataset_display_name": bundle.display_name,
        "n_matches": bundle.n_matches,
        "n_players": bundle.n_players,
        "player_stats_csv": str(player_stats_path),
        "match_report_csv": str(match_report_path),
        "plot_inputs_json": str(plot_inputs_path),
        **rank_shift_summary,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_keys = set(available_hf_battle_datasets())
    unknown = sorted(set(args.datasets) - valid_keys)
    if unknown:
        raise SystemExit(f"Unknown dataset keys: {unknown}. Valid: {sorted(valid_keys)}")

    use_paper_rc()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )

    bundles = _resolve_dataset_bundles(list(args.datasets))
    ordered_keys = tuple(bundle.key for bundle in bundles)

    print(f"Running datasets: {ordered_keys}")
    print(f"Output dir: {output_dir}")
    dataset_summaries = []
    for bundle in bundles:
        dataset_summaries.append(
            run_dataset(
            bundle,
            output_dir=output_dir,
            distribution_top_n=args.distribution_top_n,
            rank_top_k=args.rank_top_k,
        )
        )
    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "dataset_order": [bundle.key for bundle in bundles],
        "dataset_sizes": [
            {
                "dataset_key": bundle.key,
                "dataset_display_name": bundle.display_name,
                "n_matches": bundle.n_matches,
                "n_players": bundle.n_players,
            }
            for bundle in bundles
        ],
        "config": {
            "distribution_top_n": args.distribution_top_n,
            "rank_top_k": args.rank_top_k,
            "kendall_temperature": KENDALL_TEMPERATURE,
            "influence_method": INFLUENCE_METHOD,
        },
        "dataset_outputs": dataset_summaries,
    }
    _write_json(output_dir / "player_kendall_influence_plot_run_metadata.json", run_metadata)
    print("Done.")


if __name__ == "__main__":
    main()
