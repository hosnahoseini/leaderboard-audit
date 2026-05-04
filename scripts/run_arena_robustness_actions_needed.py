from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

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
    available_hf_battle_datasets,
    load_named_battle_data,
    plot_variant_actions_needed,
    ranking_from_model,
    use_paper_rc,
)
from clean_bt_rank.iterative_actions import gap_based_objective_search_across_player_pairs
from package.RankAMIP.logistic import isRankingRobust
from verify_against_baseline import baseline_matchups_as_player_pairs


VARIANTS = [
    {"variant": "drop", "action": "drop", "candidate_mode": None},
    {"variant": "flip", "action": "flip", "candidate_mode": None},
    {"variant": "add_pairs", "action": "add", "candidate_mode": "all_pairs"},
    {"variant": "add_outcomes", "action": "add", "candidate_mode": "all_outcomes"},
    {"variant": "add_weighted", "action": "add", "candidate_mode": "weighted"},
]
use_paper_rc()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-1 robustness action-needed analysis across datasets.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=sorted(available_hf_battle_datasets()),
        help="Dataset keys to run. Default: all available datasets.",
    )
    parser.add_argument(
        "--dataset-order",
        choices=("alpha", "size_asc", "size_desc"),
        default="size_asc",
        help="How to order datasets before running the batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "notebooks" / "artifacts" / "top1_robustness_action_needed",
        help="Directory for CSV/PDF outputs.",
    )
    parser.add_argument(
        "--max-action-fraction",
        type=float,
        default=None,
        help="Optional global max alpha fraction override. If omitted, use per-dataset defaults.",
    )
    parser.add_argument(
        "--assert-baseline-match",
        action="store_true",
        help="Fail if the drop-action comparison against isRankingRobust does not match exactly.",
    )
    return parser.parse_args()


def compute_max_actions(n_rows: int, fraction: float) -> int:
    return max(1, int(np.ceil(float(fraction) * int(n_rows))))


def resolve_dataset_order(dataset_keys: list[str], order_mode: str) -> tuple[list[str], list[tuple[str, int]]]:
    keys = [str(key) for key in dataset_keys]
    if order_mode == "alpha":
        ordered = sorted(keys)
        return ordered, []

    size_rows: list[tuple[str, int]] = []
    for key in keys:
        loaded = load_named_battle_data(key)
        size_rows.append((key, int(len(loaded.battle_frame))))

    reverse = order_mode == "size_desc"
    size_rows.sort(key=lambda item: (item[1], item[0]), reverse=reverse)
    ordered = [key for key, _ in size_rows]
    return ordered, size_rows


def max_action_fraction_for_dataset(dataset_key: str, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    if dataset_key in {"arena55k", "nba_elo_top50", "llm_judge_arena"}:
        return 0.01
    return 0.10


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
        "X": dataset.design_matrix(),
        "y": dataset.outcomes.copy(),
    }


def sorted_top1_pairs(bt_model: BradleyTerryModel) -> list[tuple[str, str]]:
    return [(player_a, player_b) for player_a, player_b, _ in baseline_matchups_as_player_pairs(bt_model, 1)]


def baseline_free_index_to_name(bt_model: BradleyTerryModel, free_idx: Optional[int]) -> str:
    if free_idx is None:
        return bt_model.competitor_names_[bt_model.reference_player]
    full_idx = free_idx if free_idx < bt_model.reference_player else free_idx + 1
    return bt_model.competitor_names_[full_idx]


def _json_safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def serialize_selected_matches(selected_matches: Optional[pd.DataFrame]) -> list[dict[str, object]]:
    if selected_matches is None or selected_matches.empty:
        return []
    records = selected_matches.to_dict(orient="records")
    return [{str(k): _json_safe_value(v) for k, v in record.items()} for record in records]


def selected_matches_long_frame(
    dataset_key: str,
    dataset_name: str,
    variant_row: dict[str, object],
) -> pd.DataFrame:
    records = variant_row["selected_matches_payload"]
    if not records:
        return pd.DataFrame()
    player_pair = variant_row["player_pair"]
    player_a = None if player_pair is None else player_pair[0]
    player_b = None if player_pair is None else player_pair[1]
    rows = []
    for selected_order, record in enumerate(records):
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "variant": variant_row["variant"],
                "action": variant_row["action"],
                "candidate_mode": variant_row["candidate_mode"],
                "target_player_a": player_a,
                "target_player_b": player_b,
                "selected_order": selected_order,
                **record,
            }
        )
    return pd.DataFrame(rows)


def run_variant(
    bt_model: BradleyTerryModel,
    pairs: list[tuple[str, str]],
    spec: dict[str, object],
    *,
    max_actions: int,
) -> dict[str, object]:
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
    selected_matches_payload = serialize_selected_matches(None if nested is None else nested["selected_matches"])
    return {
        "variant": spec["variant"],
        "action": spec["action"],
        "candidate_mode": spec["candidate_mode"],
        "player_pair": None if result["player_pair"] is None else tuple(result["player_pair"]),
        "met": bool(result["met"]),
        "n_actions": np.nan if result["alpha"] is None else int(result["alpha"]),
        "initial_value": np.nan if nested is None else float(nested["initial_value"]),
        "final_value": np.nan if nested is None else float(nested["final_value"]),
        "selected_match_count": len(selected_matches_payload),
        "selected_row_uids": []
        if nested is None or "row_uid" not in nested["selected_matches"].columns
        else [int(x) for x in nested["selected_matches"]["row_uid"].tolist()],
        "selected_matches_payload": selected_matches_payload,
        "selected_matches_json": json.dumps(selected_matches_payload, sort_keys=True),
        "search_result": result,
    }


def compare_drop_to_baseline(
    bt_model: BradleyTerryModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_actions: int,
) -> pd.DataFrame:
    pairs = sorted_top1_pairs(bt_model)
    with contextlib.redirect_stdout(io.StringIO()):
        ours = gap_based_objective_search_across_player_pairs(
            bt_model,
            pairs,
            "drop",
            start_alpha=1,
            max_alpha=max_actions,
            recompute_mode="refit",
            influence_method="1sn",
        )
    baseline = (-1, -1, -1, -1, [-1])
    baseline_alpha = None
    for alpha in range(1, max_actions + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            candidate = isRankingRobust(1, alpha, X, y)
        if candidate[0] != -1:
            baseline = candidate
            baseline_alpha = alpha
            break
    baseline_met = baseline[0] != -1
    baseline_pair = None if not baseline_met else (
        baseline_free_index_to_name(bt_model, baseline[0]),
        baseline_free_index_to_name(bt_model, baseline[1]),
    )
    ours_nested = ours["result"]
    row = {
        "baseline_alpha": baseline_alpha,
        "our_alpha": None if ours["alpha"] is None else int(ours["alpha"]),
        "alpha_match": baseline_alpha == ours["alpha"],
        "baseline_met": baseline_met,
        "our_met": bool(ours["met"]),
        "met_match": baseline_met == bool(ours["met"]),
        "baseline_pair": baseline_pair,
        "our_pair": None if ours["player_pair"] is None else tuple(ours["player_pair"]),
        "pair_match": baseline_pair == (None if ours["player_pair"] is None else tuple(ours["player_pair"])),
        "baseline_initial_value": np.nan if not baseline_met else float(baseline[2]),
        "our_initial_value": np.nan if ours_nested is None else float(ours_nested["initial_value"]),
        "initial_match": (not baseline_met and ours_nested is None)
        or np.isclose(float(baseline[2]), float(ours_nested["initial_value"]), atol=1e-12, rtol=0.0),
        "baseline_final_value": np.nan if not baseline_met else float(baseline[3]),
        "our_final_value": np.nan if ours_nested is None else float(ours_nested["final_value"]),
        "final_match": (not baseline_met and ours_nested is None)
        or np.isclose(float(baseline[3]), float(ours_nested["final_value"]), atol=1e-12, rtol=0.0),
        "baseline_indices": [] if not baseline_met else [int(x) for x in baseline[4]],
        "our_indices": [] if ours_nested is None else [int(x) for x in ours_nested["selected_matches"]["row_uid"].tolist()],
    }
    row["indices_match"] = row["baseline_indices"] == row["our_indices"]
    return pd.DataFrame([row])


def plot_actions_needed(summary_df: pd.DataFrame, dataset_name: str, stem: str, *, max_actions: int, output_dir: Path) -> Path:
    fig, _ = plot_variant_actions_needed(summary_df, title=dataset_name, max_actions=max_actions)
    pdf_path = output_dir / f"{stem}_top1_actions_needed.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path


def run_dataset(dataset_key: str, *, output_dir: Path, max_action_fraction: float) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    built = build_named_dataset_model(dataset_key)
    bt_model = built["bt_model"]
    ranking = ranking_from_model(bt_model)
    pairs = sorted_top1_pairs(bt_model)
    fraction = max_action_fraction_for_dataset(dataset_key, max_action_fraction)
    max_actions = compute_max_actions(len(built["raw"]), fraction)
    rows = []
    selected_match_frames = []
    for spec in VARIANTS:
        print({"dataset_key": dataset_key, **spec, "max_actions": max_actions}, flush=True)
        out = run_variant(bt_model, pairs, spec, max_actions=max_actions)
        selected_match_frames.append(selected_matches_long_frame(built["dataset_key"], built["dataset_name"], out))
        rows.append({k: v for k, v in out.items() if k not in {"search_result", "selected_matches_payload"}})
    summary_df = pd.DataFrame(rows)
    selected_matches_df = (
        pd.concat([frame for frame in selected_match_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in selected_match_frames)
        else pd.DataFrame()
    )
    pdf_path = plot_actions_needed(summary_df, built["dataset_name"], built["dataset_key"], max_actions=max_actions, output_dir=output_dir)
    baseline_df = compare_drop_to_baseline(bt_model, built["X"], built["y"], max_actions=max_actions)
    meta = {
        "dataset_key": built["dataset_key"],
        "dataset_name": built["dataset_name"],
        "n_original_matches": int(len(built["raw"])),
        "max_action_fraction": float(fraction),
        "max_actions": int(max_actions),
        "n_fitted_rows": int(built["dataset"].n_matches),
        "n_models": int(len(ranking)),
        "top1": ranking[0],
        "top2": ranking[1],
        "pdf_path": str(pdf_path),
    }
    return summary_df, meta, baseline_df, selected_matches_df


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_keys, size_rows = resolve_dataset_order(list(args.datasets), args.dataset_order)
    print("Resolved dataset order:", flush=True)
    for idx, dataset_key in enumerate(dataset_keys, start=1):
        size_suffix = ""
        for key, n_rows in size_rows:
            if key == dataset_key:
                size_suffix = f" ({n_rows} battle rows)"
                break
        print(f"{idx:>2}. {dataset_key}{size_suffix}", flush=True)
    all_rows = []
    all_baseline_rows = []
    all_selected_match_rows = []

    for dataset_key in dataset_keys:
        print(f"Running {dataset_key} ...", flush=True)
        summary_df, meta, baseline_df, selected_matches_df = run_dataset(
            dataset_key,
            output_dir=output_dir,
            max_action_fraction=args.max_action_fraction,
        )
        summary_df.to_csv(output_dir / f"{meta['dataset_key']}_top1_actions_needed.csv", index=False)
        selected_matches_df.to_csv(output_dir / f"{meta['dataset_key']}_top1_actions_selected_matches.csv", index=False)

        summary_plot_ready = summary_df.copy()
        summary_plot_ready.insert(0, "dataset_key", meta["dataset_key"])
        summary_plot_ready.insert(1, "dataset_name", meta["dataset_name"])
        summary_plot_ready.insert(2, "max_actions", meta["max_actions"])
        summary_plot_ready["variant_label"] = summary_plot_ready["variant"].map(ACTION_LABEL_MAP)
        summary_plot_ready["actions_needed_plot"] = summary_plot_ready["n_actions"].fillna(meta["max_actions"] + 1)
        summary_plot_ready["success_label"] = np.where(
            summary_plot_ready["met"],
            summary_plot_ready["n_actions"].astype("Int64").astype(str),
            f">{meta['max_actions']}",
        )
        summary_plot_ready.to_csv(output_dir / f"{meta['dataset_key']}_top1_actions_needed_plot_ready.csv", index=False)

        for row in summary_df.to_dict(orient="records"):
            all_rows.append({**meta, **row})

        baseline_out = baseline_df.copy()
        baseline_out.insert(0, "dataset_key", meta["dataset_key"])
        baseline_out.insert(1, "dataset_name", meta["dataset_name"])
        baseline_out.insert(2, "max_actions", meta["max_actions"])
        baseline_out.to_csv(output_dir / f"{meta['dataset_key']}_drop_vs_isrankingrobust.csv", index=False)
        all_baseline_rows.extend(baseline_out.to_dict(orient="records"))
        all_selected_match_rows.extend(selected_matches_df.to_dict(orient="records"))

    final_table = pd.DataFrame(all_rows)[
        [
            "dataset_key",
            "dataset_name",
            "max_actions",
            "variant",
            "action",
            "candidate_mode",
            "player_pair",
            "met",
            "n_actions",
            "initial_value",
            "final_value",
            "selected_match_count",
            "selected_row_uids",
            "selected_matches_json",
            "n_original_matches",
            "n_fitted_rows",
            "n_models",
            "top1",
            "top2",
            "pdf_path",
        ]
    ]
    final_table.to_csv(output_dir / "all_datasets_top1_actions_needed.csv", index=False)

    plot_ready_table = final_table.copy()
    plot_ready_table["variant_label"] = plot_ready_table["variant"].map(ACTION_LABEL_MAP)
    plot_ready_table["actions_needed_plot"] = plot_ready_table["n_actions"].fillna(plot_ready_table["max_actions"] + 1)
    plot_ready_table["success_label"] = np.where(
        plot_ready_table["met"],
        plot_ready_table["n_actions"].astype("Int64").astype(str),
        ">" + plot_ready_table["max_actions"].astype(int).astype(str),
    )
    plot_ready_table.to_csv(output_dir / "all_datasets_top1_actions_needed_plot_ready.csv", index=False)

    baseline_table = pd.DataFrame(all_baseline_rows)
    baseline_table.to_csv(output_dir / "all_datasets_drop_vs_isrankingrobust.csv", index=False)

    selected_matches_table = pd.DataFrame(all_selected_match_rows)
    selected_matches_table.to_csv(output_dir / "all_datasets_top1_actions_selected_matches.csv", index=False)

    if args.assert_baseline_match:
        required = ["alpha_match", "met_match", "pair_match", "initial_match", "final_match", "indices_match"]
        if not bool(baseline_table[required].to_numpy(dtype=bool).all()):
            raise AssertionError("Baseline comparison mismatch detected.")

    print(f"Saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
