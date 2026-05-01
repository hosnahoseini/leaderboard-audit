#!/usr/bin/env python3
"""Run player-level Kendall tau influence analysis for every named HF dataset and save plots.

This mirrors ``notebooks/player_kendall_tau_influence_analysis.ipynb`` (same outputs as the
notebook's savefig/to_csv paths, with ``DATASET_SLUG__`` prefix where the notebook uses it).

Usage (from repo root, with the package importable)::

    python -m clean_bt_rank.experiments.run_player_kendall_tau_all_datasets

Or restrict keys::

    python -m clean_bt_rank.experiments.run_player_kendall_tau_all_datasets --datasets arena55k vision_arena
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.use("agg")

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ARTIFACTS = ROOT / "notebooks" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

from clean_bt_rank import (  # noqa: E402
    BattleDataset,
    BradleyTerryModel,
    available_hf_battle_datasets,
    compute_player_influence,
    compute_player_influence_correlations,
    compute_player_statistics,
    drop_player_and_measure_effect,
    kendall_tau_match_influence_report,
    load_named_battle_data,
    ranking_from_model,
    run_extreme_skill_ablation,
)
from clean_bt_rank.player_influence import _fit_frame_with_indices, _refit_from_fitted_frame  # noqa: E402

KENDALL_TEMPERATURE = 0.1
INFLUENCE_METHOD = "1sn"


def _plot_pearson_spearman_vs_metrics(
    rows: pd.DataFrame, title: str, ax: plt.Axes, *, feature_order: list[str]
) -> None:
    plot_df = rows.set_index("feature").loc[feature_order].reset_index()
    y = np.arange(len(plot_df))
    h = 0.35
    ax.barh(y - h / 2, plot_df["pearson_r"], height=h, label="Pearson r", color="C0")
    ax.barh(y + h / 2, plot_df["spearman_rho"], height=h, label="Spearman ρ", color="C1")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature"])
    ax.axvline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel("Correlation with influence")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)


def run_one_dataset(dataset_key: str) -> None:
    loaded = load_named_battle_data(dataset_key)
    raw = loaded.battle_frame.copy()
    dataset_slug = loaded.key
    display_name = loaded.display_name

    dataset = BattleDataset.from_dataframe(raw)
    fit_kwargs = dict(loaded.fit_kwargs)
    bt_model = BradleyTerryModel.from_dataset(dataset, **fit_kwargs).fit()
    reference_ranking = ranking_from_model(bt_model)

    match_report = kendall_tau_match_influence_report(
        bt_model,
        ranking=reference_ranking,
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
    correlations = compute_player_influence_correlations(player_stats)
    ablation = run_extreme_skill_ablation(
        bt_model,
        player_stats,
        temperature=KENDALL_TEMPERATURE,
    )

    top_k = bt_model.n_players_

    correlation_joint_newton_abs = correlations.loc[
        correlations["target"] == "player_influence_joint_newton_abs",
        ["feature", "pearson_r", "pearson_pvalue", "spearman_rho", "spearman_pvalue", "n"],
    ].sort_values("spearman_rho", ascending=False)

    correlation_joint_newton_signed = correlations.loc[
        correlations["target"] == "player_influence_joint_newton",
        ["feature", "pearson_r", "pearson_pvalue", "spearman_rho", "spearman_pvalue", "n"],
    ].sort_values("spearman_rho", ascending=False)

    _metric_feature_order = sorted(correlation_joint_newton_abs["feature"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    _plot_pearson_spearman_vs_metrics(
        correlation_joint_newton_abs,
        "Joint-Newton |influence| vs player metrics",
        axes[0],
        feature_order=_metric_feature_order,
    )
    _plot_pearson_spearman_vs_metrics(
        correlation_joint_newton_signed,
        "Joint-Newton signed influence vs player metrics",
        axes[1],
        feature_order=_metric_feature_order,
    )
    fig.tight_layout()
    p_corr = (
        ARTIFACTS
        / f"{dataset_slug}__player_kendall_joint_newton_influence_metric_correlations.png"
    )
    fig.savefig(p_corr, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", p_corr)

    _paper = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    mpl.rcParams.update(_paper)

    top_n = min(3, len(player_stats))
    _plot_cols = [
        c
        for c in [
            "skill",
            "rating",
            "degree",
            "bridge_var",
            "closeness",
            "surprise",
        ]
        if c in player_stats.columns and pd.api.types.is_numeric_dtype(player_stats[c])
    ]

    _var_summary = (
        player_stats[_plot_cols].agg(["min", "max", "mean"]).T.rename(columns={"mean": "avg"})
    )
    _var_summary.to_csv(ARTIFACTS / f"{dataset_slug}__player_stat_variable_summary.csv")
    print("Wrote", ARTIFACTS / f"{dataset_slug}__player_stat_variable_summary.csv")

    most_rows = player_stats.nlargest(top_n, "player_influence_joint_newton_abs")
    least_rows = player_stats.nsmallest(top_n, "player_influence_joint_newton_abs")
    _inf_arr = player_stats["player_influence_joint_newton_abs"].to_numpy(dtype=float)
    _red = ["#7B241C", "#C0392B", "#F5B7B1"]
    _blue = ["#1B4F72", "#2874A6", "#A9CCE3"]

    n = len(_plot_cols)
    _nc = 3
    _nr = int(np.ceil(n / _nc))
    fig, axes = plt.subplots(_nr, _nc, figsize=(7.2, 2.15 * _nr), sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, _plot_cols):
        s = player_stats[col].astype(float)
        nbins = min(40, max(10, int(np.sqrt(len(player_stats)))))
        n_all = max(len(player_stats), 1)
        s_arr = s.to_numpy()
        valid_corr = np.isfinite(s_arr) & np.isfinite(_inf_arr)
        s_c, inf_c = s_arr[valid_corr], _inf_arr[valid_corr]
        valid = np.isfinite(s_arr)
        s_v = s_arr[valid]
        w = np.full(s_v.size, 1.0 / n_all, dtype=float)
        ax.hist(s_v, bins=nbins, weights=w, density=False, color="#d9d9d9", edgecolor="white", linewidth=0.4)
        for rank, (_, r) in enumerate(most_rows.iterrows()):
            ax.axvline(float(r[col]), color=_red[rank], lw=1.5, zorder=3)
        for rank, (_, r) in enumerate(least_rows.iterrows()):
            ax.axvline(float(r[col]), color=_blue[rank], lw=1.5, zorder=3)
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("count / N players")
        if s_c.size >= 2:
            pr = float(np.corrcoef(s_c, inf_c)[0, 1])
            spr = float(pd.Series(s_c).corr(pd.Series(inf_c), method="spearman"))
        else:
            pr = spr = float("nan")
        ax.set_title(f"r={pr:.3f}, ρ={spr:.3f}", fontsize=8, loc="left")

    handles = []
    for rank, (_, r) in enumerate(most_rows.iterrows()):
        name = str(r["player"])
        short = name[:24] + ("…" if len(name) > 24 else "")
        handles.append(
            mpl.lines.Line2D([], [], color=_red[rank], lw=1.5, label=f"High |τ| #{rank + 1}: {short}")
        )
    for rank, (_, r) in enumerate(least_rows.iterrows()):
        name = str(r["player"])
        short = name[:24] + ("…" if len(name) > 24 else "")
        handles.append(
            mpl.lines.Line2D([], [], color=_blue[rank], lw=1.5, label=f"Low |τ| #{rank + 1}: {short}")
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.suptitle(
        f"Player-stat distributions: top {top_n} vs bottom {top_n} by |Kendall τ influence|",
        y=1.12,
    )
    for j in range(len(_plot_cols), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _out = ARTIFACTS / f"{dataset_slug}__player_kendall_influence_distribution_markers.pdf"
    fig.savefig(_out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(_out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("Saved", _out)

    most_influential_player = player_stats.sort_values(
        "player_influence_joint_newton_abs", ascending=False
    ).iloc[0]["player"]
    fit_frame = _fit_frame_with_indices(bt_model)
    filtered_frame = fit_frame.loc[
        (fit_frame["model_a"] != most_influential_player) & (fit_frame["model_b"] != most_influential_player)
    ].reset_index(drop=True)
    refit_after_drop = _refit_from_fitted_frame(bt_model, filtered_frame)

    before_remaining = [player for player in ranking_from_model(bt_model) if player != most_influential_player]
    after_remaining = ranking_from_model(refit_after_drop)
    player_drop_result = drop_player_and_measure_effect(
        bt_model,
        player=most_influential_player,
        player_stats=player_stats,
        temperature=KENDALL_TEMPERATURE,
    )
    print(
        "drop_effect",
        {
            "predicted_joint_newton_delta": player_drop_result.predicted_delta_joint_newton,
            "actual_smooth_delta": player_drop_result.actual_smooth_delta,
            "actual_rank_tau": player_drop_result.actual_rank_tau,
        },
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
    rank_shift_path_csv = ARTIFACTS / f"{dataset_slug}_most_influential_rank_shift.csv"
    rank_shift.to_csv(rank_shift_path_csv, index=False)

    focus = rank_shift.sort_values(["before_rank"]).copy()
    focus["display_order"] = np.arange(len(focus), 0, -1)
    plot_shift = focus.loc[focus[["before_rank", "after_rank"]].min(axis=1) <= top_k].copy()

    def _short_label(name: str, max_len: int = 36) -> str:
        s = str(name)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    fig, ax = plt.subplots(figsize=(12, 12))
    for _, row in plot_shift.iterrows():
        ax.plot(
            [row["before_rank"], row["after_rank"]],
            [row["display_order"], row["display_order"]],
            color="#9aa0a6",
            linewidth=1.8,
            alpha=0.9,
        )

    ax.scatter(plot_shift["before_rank"], plot_shift["display_order"], color="#1f77b4", s=80, label="Before drop", zorder=3)
    ax.scatter(plot_shift["after_rank"], plot_shift["display_order"], color="#d62728", s=80, label="After drop", zorder=3)

    _plot_y = plot_shift.sort_values("display_order", ascending=True)
    ax.set_yticks(_plot_y["display_order"].to_numpy())
    ax.set_yticklabels([_short_label(p) for p in _plot_y["player"]], fontsize=9)
    ax.tick_params(axis="y", length=0)
    plt.setp(ax.get_yticklabels(), ha="right")

    for _, row in plot_shift.iterrows():
        if row["rank_change"] != 0:
            ax.text(
                max(row["before_rank"], row["after_rank"]) + 0.35,
                row["display_order"],
                f"{int(row['rank_change']):+d}",
                va="center",
                fontsize=10,
                ha="left",
            )

    _n_matches_dropped = raw["model_a"].shape[0] - len(filtered_frame) // 2
    ax.set_title(
        f"{display_name}: Ranking Shift After Dropping {most_influential_player} "
        f"(matches dropped: {_n_matches_dropped})"
    )
    ax.set_xlabel("Rank among remaining players")
    ax.set_ylabel("Player")
    ymin, ymax = plot_shift["display_order"].min(), plot_shift["display_order"].max()
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_xlim(0.5, min(top_k + 2, len(before_remaining) + 0.5))
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.subplots_adjust(left=0.22)
    rank_shift_path = ARTIFACTS / f"{dataset_slug}_player_kendall_most_influential_rank_shift.png"
    fig.savefig(rank_shift_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved", rank_shift_path, rank_shift_path_csv)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    plot_specs = [
        ("skill", f"{display_name}: Joint-Newton Player Influence vs Skill"),
        ("degree", f"{display_name}: Joint-Newton Player Influence vs Degree"),
        ("bridge_var", f"{display_name}: Joint-Newton Player Influence vs Opponent Skill Variance"),
    ]
    for ax_, (feature, title) in zip(axes, plot_specs):
        sns.scatterplot(
            data=player_stats,
            x=feature,
            y="player_influence_joint_newton_abs",
            hue="player",
            palette="tab10",
            s=100,
            ax=ax_,
            legend=False,
        )
        ax_.set_title(title)
        ax_.set_ylabel("Player influence (joint-Newton absolute)")
        ax_.set_xlabel(feature)

    fig.tight_layout()
    scatter_path = ARTIFACTS / f"{dataset_slug}_player_kendall_influence_scatter_grid.png"
    fig.savefig(scatter_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved", scatter_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_players = player_stats.nlargest(top_k, "player_influence_joint_newton_abs").sort_values(
        "player_influence_joint_newton_abs"
    )
    sns.barplot(
        data=top_players,
        x="player_influence_joint_newton_abs",
        y="player",
        hue="player",
        palette="viridis",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title(f"{display_name}: Top {top_k} Players by Absolute Joint-Newton Kendall Influence")
    ax.set_xlabel("Player influence (joint-Newton absolute)")
    ax.set_ylabel("")
    fig.tight_layout()
    bar_path = ARTIFACTS / f"{dataset_slug}_player_kendall_influence_top_players.png"
    fig.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved", bar_path)

    plot_ablation = ablation.melt(
        id_vars="player",
        value_vars=[
            "predicted_delta_sum",
            "predicted_delta_joint_newton",
            "actual_smooth_delta",
            "actual_rank_tau_delta",
        ],
        var_name="metric",
        value_name="delta",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_ablation, x="player", y="delta", hue="metric", palette="Set2", ax=ax)
    ax.set_title(f"{display_name}: Summed vs Joint-Newton Predicted Player-Drop Effect")
    ax.set_xlabel("Dropped player")
    ax.set_ylabel("Delta")
    ax.axhline(0.0, color="black", linewidth=1)
    fig.tight_layout()
    ablation_path = ARTIFACTS / f"{dataset_slug}_player_kendall_ablation_comparison.png"
    fig.savefig(ablation_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved", ablation_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Subset of dataset keys (default: all named HF datasets).",
    )
    args = parser.parse_args()
    all_keys = tuple(sorted(available_hf_battle_datasets()))
    if args.datasets:
        unknown = set(args.datasets) - set(all_keys)
        if unknown:
            raise SystemExit(f"Unknown dataset keys: {sorted(unknown)}. Valid: {all_keys}")
        keys = tuple(args.datasets)
    else:
        keys = all_keys

    sns.set_theme(style="whitegrid", context="talk")
    print("Running for:", keys)
    for k in keys:
        print("===", k, "===")
        run_one_dataset(k)
    print("Done. Artifacts under:", ARTIFACTS)


if __name__ == "__main__":
    main()
