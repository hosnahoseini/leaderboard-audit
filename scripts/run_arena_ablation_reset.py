#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT, ROOT / "tests", ROOT / "IsRankingRobust"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clean_bt_rank import (
    ACTION_LABEL_MAP,
    BattleDataset,
    BradleyTerryModel,
    load_named_battle_data,
    plot_variant_actions_needed,
    ranking_from_model,
    use_paper_rc,
)
from clean_bt_rank.iterative_actions import gap_based_objective_search_across_player_pairs
from verify_against_baseline import baseline_matchups_as_player_pairs

VARIANTS = [
    {"variant": "drop", "action": "drop", "candidate_mode": None},
    {"variant": "flip", "action": "flip", "candidate_mode": None},
    {"variant": "add_pairs", "action": "add", "candidate_mode": "all_pairs"},
    {"variant": "add_outcomes", "action": "add", "candidate_mode": "all_outcomes"},
    {"variant": "add_weighted", "action": "add", "candidate_mode": "weighted"},
]
DEFAULT_K_VALUES = (1, 3, 5, 10, 20, 40)
use_paper_rc()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Arena top-k ablation-reset sweep.")
    parser.add_argument("--dataset", default="arena55k", help="Dataset key. Default: arena55k.")
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="Top-k boundaries to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "new_result" / "arena_ablation_reset",
        help="Directory for CSV/PDF/PNG outputs.",
    )
    parser.add_argument(
        "--max-action-fraction",
        type=float,
        default=0.01,
        help="Max alpha as ceil(fraction * n_original_matches).",
    )
    return parser.parse_args()


def compute_max_actions(n_rows: int, fraction: float) -> int:
    return max(1, int(np.ceil(float(fraction) * int(n_rows))))


def build_named_dataset_model(dataset_key: str) -> dict[str, object]:
    loaded = load_named_battle_data(dataset_key)
    battle = loaded.battle_frame.copy()
    dataset = BattleDataset.from_dataframe(battle, weighted_symmetric_ties=True)
    bt_model = BradleyTerryModel.from_dataset(dataset, **loaded.fit_kwargs).fit()
    return {
        "dataset_key": loaded.key,
        "dataset_name": loaded.display_name,
        "raw": battle,
        "dataset": dataset,
        "bt_model": bt_model,
    }


def sorted_topk_pairs(bt_model: BradleyTerryModel, k: int) -> list[tuple[str, str]]:
    return [(player_a, player_b) for player_a, player_b, _ in baseline_matchups_as_player_pairs(bt_model, k)]


def run_variant(bt_model: BradleyTerryModel, pairs: list[tuple[str, str]], spec: dict[str, object], *, max_actions: int) -> dict[str, object]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = gap_based_objective_search_across_player_pairs(
            bt_model,
            pairs,
            spec["action"],
            start_alpha=1,
            max_alpha=max_actions,
            recompute_mode="refit",
            influence_method="1sn",
            candidate_mode=spec["candidate_mode"],
        )
    nested = result["result"]
    return {
        "variant": spec["variant"],
        "action": spec["action"],
        "candidate_mode": spec["candidate_mode"],
        "player_pair": None if result["player_pair"] is None else tuple(result["player_pair"]),
        "met": bool(result["met"]),
        "n_actions": np.nan if result["alpha"] is None else int(result["alpha"]),
        "initial_value": np.nan if nested is None else float(nested["initial_value"]),
        "final_value": np.nan if nested is None else float(nested["final_value"]),
    }


def save_k_curve_plot(summary_df: pd.DataFrame, output_dir: Path, *, dataset_key: str, dataset_name: str) -> None:
    plot_df = summary_df.copy()
    plot_df["variant_label"] = plot_df["variant"].map(ACTION_LABEL_MAP).fillna(plot_df["variant"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for variant, grp in plot_df.groupby("variant_label", sort=False):
        ordered = grp.sort_values("k")
        ax.plot(
            ordered["k"].to_numpy(dtype=int),
            ordered["n_actions_plot"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            label=str(variant),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(plot_df["k"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Top-k boundary")
    ax.set_ylabel("Actions needed")
    ax.set_title(f"{dataset_name}: actions needed to change top-k")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset_key}_topk_actions_needed_curve.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / f"{dataset_key}_topk_actions_needed_curve.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    built = build_named_dataset_model(str(args.dataset))
    requested_k_values = sorted({int(k) for k in args.k_values})
    n_models = int(len(ranking_from_model(built["bt_model"])))
    k_values = [k for k in requested_k_values if 1 <= k < n_models]
    invalid = [k for k in requested_k_values if k not in k_values]
    if not k_values:
        raise ValueError(f"No valid k values remain for n_models={n_models}; requested: {requested_k_values}")
    if invalid:
        print(
            f"Skipping invalid k values for {built['dataset_key']} with n_models={n_models}: {invalid}",
            flush=True,
        )

    max_actions = compute_max_actions(len(built["raw"]), args.max_action_fraction)
    all_rows: list[dict[str, object]] = []

    for k in k_values:
        pairs = sorted_topk_pairs(built["bt_model"], k)
        rows = []
        for spec in VARIANTS:
            print({"dataset_key": built["dataset_key"], "k": k, **spec, "max_actions": max_actions}, flush=True)
            rows.append(run_variant(built["bt_model"], pairs, spec, max_actions=max_actions))

        summary_df = pd.DataFrame(rows)
        summary_df.insert(0, "dataset_key", built["dataset_key"])
        summary_df.insert(1, "dataset_name", built["dataset_name"])
        summary_df.insert(2, "k", int(k))
        summary_df.insert(3, "pair_count", int(len(pairs)))
        summary_df.insert(4, "max_actions", int(max_actions))
        summary_df["variant_label"] = summary_df["variant"].map(ACTION_LABEL_MAP)
        summary_df["n_actions_plot"] = summary_df["n_actions"].fillna(max_actions + 1)
        summary_df["success_label"] = np.where(
            summary_df["met"],
            summary_df["n_actions"].astype("Int64").astype(str),
            f">{max_actions}",
        )
        summary_df.to_csv(output_dir / f"{built['dataset_key']}_k{k}_actions_needed.csv", index=False)

        fig, _ = plot_variant_actions_needed(summary_df, title=f"{built['dataset_name']} (k={k})", max_actions=max_actions)
        fig.savefig(output_dir / f"{built['dataset_key']}_k{k}_actions_needed.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(output_dir / f"{built['dataset_key']}_k{k}_actions_needed.png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        all_rows.extend(summary_df.to_dict(orient="records"))

    final_df = pd.DataFrame(all_rows).sort_values(["k", "variant"]).reset_index(drop=True)
    final_df.to_csv(output_dir / f"{built['dataset_key']}_topk_actions_needed_summary.csv", index=False)
    save_k_curve_plot(final_df, output_dir, dataset_key=built["dataset_key"], dataset_name=built["dataset_name"])
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset": built["dataset_key"],
                "dataset_name": built["dataset_name"],
                "k_values": k_values,
                "max_action_fraction": float(args.max_action_fraction),
                "max_actions": int(max_actions),
                "n_models": n_models,
                "n_original_matches": int(len(built["raw"])),
            },
            indent=2,
        )
    )
    print(f"Saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
