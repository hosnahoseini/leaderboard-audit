from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2]
for path in [ROOT / "src", ROOT, ROOT / "IsRankingRobust"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DownloadConfig, load_dataset

from clean_bt_rank import (
    ACTION_LABEL_MAP,
    BattleDataset,
    BradleyTerryModel,
    available_hf_battle_datasets,
    compute_confidence_intervals,
    load_named_battle_data,
    plot_ci_ranking,
    plot_variant_actions_needed,
    ranking_from_model,
    use_paper_rc,
)
from clean_bt_rank.datasets import _load_cached_hf_frame
from clean_bt_rank.iterative_actions import (
    apply_action_on_top_alpha_influential_matches,
    compute_all_action_influences,
    refit_model_with_action,
)
from clean_bt_rank.objectives import SkillGapObjective
from package.RankAMIP.data_script import make_BT_design_matrix


VARIANTS = [
    {"variant": "drop", "action": "drop", "candidate_mode": None},
    {"variant": "flip", "action": "flip", "candidate_mode": None},
    {"variant": "add_pairs", "action": "add", "candidate_mode": "all_pairs"},
    {"variant": "add_outcomes", "action": "add", "candidate_mode": "all_outcomes"},
    {"variant": "add_weighted", "action": "add", "candidate_mode": "weighted"},
]
use_paper_rc()

MODEL_CACHE: dict[str, dict[str, object]] = {}


def compute_max_actions(n_rows: int, fraction: float) -> int:
    return max(1, int(np.ceil(float(fraction) * int(n_rows))))


def load_arena55k_raw_dataframe() -> pd.DataFrame:
    df = _load_cached_hf_frame("lmarena-ai/arena-human-preference-55k", "train")
    if df is None:
        ds = load_dataset(
            "lmarena-ai/arena-human-preference-55k",
            split="train",
            download_config=DownloadConfig(local_files_only=True),
        )
        df = ds.to_pandas()
    return df[["model_a", "model_b", "winner_model_a", "winner_tie"]].copy()


def make_bt_design_matrix_wtd(raw: pd.DataFrame):
    winner_fwd = raw["winner_model_a"].copy().astype(float)
    winner_rev = (1 - raw["winner_model_a"]).astype(float)
    tie_mask = raw["winner_tie"] == 1
    winner_fwd[tie_mask] = 1.0
    winner_rev[tie_mask] = 1.0
    combined = pd.concat(
        [
            pd.DataFrame({"model_a": raw["model_a"], "model_b": raw["model_b"], "winner": winner_fwd.astype(int)}),
            pd.DataFrame({"model_a": raw["model_b"], "model_b": raw["model_a"], "winner": winner_rev.astype(int)}),
        ],
        ignore_index=True,
    )
    return make_BT_design_matrix(combined)


def build_arena55k_verified_model() -> dict[str, object]:
    raw = load_arena55k_raw_dataframe()
    X, y, player_to_id = make_bt_design_matrix_wtd(raw)
    battle = raw.rename(columns={"winner_model_a": "winner"}).assign(
        winner=lambda df: np.where(df["winner_tie"] == 1, "tie", np.where(df["winner"] == 1, "model_a", "model_b"))
    )[["model_a", "model_b", "winner"]]
    dataset = BattleDataset.from_dataframe(
        battle,
        competitors=[name for name, _ in sorted(player_to_id.items(), key=lambda item: item[1])],
        weighted_symmetric_ties=True,
    )
    bt_model = BradleyTerryModel(
        dataset.design_matrix(),
        dataset.outcomes,
        competitor_names=dataset.competitors,
        reference_player=0,
        hessian_ridge=0.0,
    ).fit()
    bt_model.match_frame_ = dataset.frame.copy()
    return {
        "dataset_key": "arena55k",
        "dataset_name": "Chatbot Arena 55k",
        "raw": raw,
        "dataset": dataset,
        "bt_model": bt_model,
    }


def build_named_dataset_model(dataset_key: str) -> dict[str, object]:
    if dataset_key in MODEL_CACHE:
        return MODEL_CACHE[dataset_key]
    if dataset_key == "arena55k":
        built = build_arena55k_verified_model()
        MODEL_CACHE[dataset_key] = built
        return built
    loaded = load_named_battle_data(dataset_key)
    dataset = BattleDataset.from_dataframe(loaded.battle_frame.copy(), weighted_symmetric_ties=True)
    bt_model = BradleyTerryModel.from_dataset(dataset, **loaded.fit_kwargs).fit()
    built = {
        "dataset_key": loaded.key,
        "dataset_name": loaded.display_name,
        "raw": loaded.battle_frame.copy(),
        "dataset": dataset,
        "bt_model": bt_model,
    }
    MODEL_CACHE[dataset_key] = built
    return built


def clear_model_cache(dataset_key: str | None = None) -> None:
    if dataset_key is None:
        MODEL_CACHE.clear()
        return
    MODEL_CACHE.pop(str(dataset_key), None)


def load_named_battle_row_count(dataset_key: str) -> int:
    loaded = load_named_battle_data(dataset_key)
    return int(len(loaded.battle_frame))


def ci_summary_frame(
    bt_model: BradleyTerryModel,
    *,
    ci_method: str,
    ci_alpha: float,
) -> pd.DataFrame:
    ci_df = compute_confidence_intervals(bt_model, method=ci_method, alpha=ci_alpha).to_frame(bt_model.competitor_names_)
    ci_df = ci_df.reset_index(drop=True)
    ci_df.insert(0, "rank", np.arange(1, len(ci_df) + 1, dtype=int))
    return ci_df


def export_gao_ci_ranking(
    bt_model: BradleyTerryModel,
    dataset_name: str,
    stem: str,
    *,
    output_dir: Path,
    ci_method: str,
    ci_alpha: float,
) -> tuple[pd.DataFrame, Path, Path, Path]:
    ci_df = ci_summary_frame(bt_model, ci_method=ci_method, ci_alpha=ci_alpha)
    csv_path = output_dir / f"{stem}_gao_ci_ranking.csv"
    pdf_path = output_dir / f"{stem}_gao_ci_ranking.pdf"
    png_path = output_dir / f"{stem}_gao_ci_ranking.png"
    ci_df.to_csv(csv_path, index=False)

    fig, _ = plot_ci_ranking(ci_df, dataset_name=dataset_name)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return ci_df, csv_path, pdf_path, png_path


def preview_boundary(ci_df: pd.DataFrame, k: Optional[int]) -> Optional[str]:
    if k is None or k < 1 or k >= len(ci_df):
        return None
    left = ci_df.iloc[k - 1]["competitor"]
    right = ci_df.iloc[k]["competitor"]
    return f"{left} vs {right}"


def intervals_overlap(lower_a: float, upper_a: float, lower_b: float, upper_b: float) -> bool:
    return max(float(lower_a), float(lower_b)) <= min(float(upper_a), float(upper_b))


def _json_safe_value(value: Any):
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


def serialize_frame_records(frame: Optional[pd.DataFrame]) -> list[dict[str, object]]:
    if frame is None or frame.empty:
        return []
    records = frame.to_dict(orient="records")
    return [{str(k): _json_safe_value(v) for k, v in record.items()} for record in records]


def pair_ci_status(
    bt_model: BradleyTerryModel,
    top_player: str,
    outside_player: str,
    *,
    ci_method: str,
    ci_alpha: float,
) -> dict[str, object]:
    ci_df = ci_summary_frame(bt_model, ci_method=ci_method, ci_alpha=ci_alpha).set_index("competitor")
    top_row = ci_df.loc[top_player]
    outside_row = ci_df.loc[outside_player]
    overlap = intervals_overlap(top_row["ci_lower"], top_row["ci_upper"], outside_row["ci_lower"], outside_row["ci_upper"])
    outsider_above = float(outside_row["ci_lower"]) > float(top_row["ci_upper"])
    top_above = float(top_row["ci_lower"]) > float(outside_row["ci_upper"])
    return {
        "top_player": top_player,
        "outside_player": outside_player,
        "top_rank": int(top_row["rank"]),
        "outside_rank": int(outside_row["rank"]),
        "gap": float(bt_model.reported_gap(top_player, outside_player)),
        "overlap": bool(overlap),
        "top_above": bool(top_above),
        "outsider_above": bool(outsider_above),
        "top_rating": float(top_row["rating"]),
        "outside_rating": float(outside_row["rating"]),
        "top_ci_lower": float(top_row["ci_lower"]),
        "top_ci_upper": float(top_row["ci_upper"]),
        "outside_ci_lower": float(outside_row["ci_lower"]),
        "outside_ci_upper": float(outside_row["ci_upper"]),
    }


def ci_membership_target_met(initial_gap: float, final_status: dict[str, object]) -> bool:
    gap_sign_changed = float(final_status["gap"]) <= 0.0 if float(initial_gap) >= 0.0 else float(final_status["gap"]) >= 0.0
    return bool(gap_sign_changed and final_status["outsider_above"] and not final_status["overlap"])


def predicted_gap_target_met(initial_gap: float, predicted_gap: float) -> bool:
    return bool(float(predicted_gap) <= 0.0 if float(initial_gap) >= 0.0 else float(predicted_gap) >= 0.0)


def build_ci_overlap_pairs(
    bt_model: BradleyTerryModel,
    k: int,
    *,
    ci_method: str,
    ci_alpha: float,
) -> pd.DataFrame:
    ci_df = ci_summary_frame(bt_model, ci_method=ci_method, ci_alpha=ci_alpha)
    top_df = ci_df.iloc[:k].copy()
    outside_df = ci_df.iloc[k:].copy()
    rows = []
    for _, top_row in top_df.iterrows():
        for _, outside_row in outside_df.iterrows():
            overlap = intervals_overlap(top_row["ci_lower"], top_row["ci_upper"], outside_row["ci_lower"], outside_row["ci_upper"])
            if not overlap:
                continue
            rows.append(
                {
                    "top_player": str(top_row["competitor"]),
                    "outside_player": str(outside_row["competitor"]),
                    "top_rank": int(top_row["rank"]),
                    "outside_rank": int(outside_row["rank"]),
                    "initial_gap": float(bt_model.reported_gap(top_row["competitor"], outside_row["competitor"])),
                    "top_ci_lower": float(top_row["ci_lower"]),
                    "top_ci_upper": float(top_row["ci_upper"]),
                    "outside_ci_lower": float(outside_row["ci_lower"]),
                    "outside_ci_upper": float(outside_row["ci_upper"]),
                    "initial_overlap": True,
                }
            )
    return pd.DataFrame(rows)


def boundary_overlap_table(ci_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in range(1, len(ci_df)):
        top_row = ci_df.iloc[k - 1]
        outside_row = ci_df.iloc[k]
        overlap_amount = min(float(top_row["ci_upper"]), float(outside_row["ci_upper"])) - max(
            float(top_row["ci_lower"]), float(outside_row["ci_lower"])
        )
        rows.append(
            {
                "k": int(k),
                "top_player": str(top_row["competitor"]),
                "outside_player": str(outside_row["competitor"]),
                "top_rank": int(top_row["rank"]),
                "outside_rank": int(outside_row["rank"]),
                "boundary_gap": float(top_row["rating"]) - float(outside_row["rating"]),
                "boundary_overlap_amount": float(max(0.0, overlap_amount)),
                "boundary_has_overlap": bool(overlap_amount > 0.0),
            }
        )
    return pd.DataFrame(rows)


def choose_meaningful_k(
    bt_model: BradleyTerryModel,
    *,
    ci_method: str,
    ci_alpha: float,
) -> dict[str, object]:
    ci_df = ci_summary_frame(bt_model, ci_method=ci_method, ci_alpha=ci_alpha)
    boundary_df = boundary_overlap_table(ci_df)
    if boundary_df.empty:
        raise ValueError("Need at least two competitors to choose a top-k boundary.")

    scored_rows = []
    for row in boundary_df.to_dict(orient="records"):
        pair_table = build_ci_overlap_pairs(bt_model, int(row["k"]), ci_method=ci_method, ci_alpha=ci_alpha)
        scored_rows.append(
            {
                **row,
                "eligible_pair_count": int(len(pair_table)),
            }
        )
    scored_df = pd.DataFrame(scored_rows)

    overlapping_boundary = scored_df.loc[scored_df["boundary_has_overlap"]].copy()
    if not overlapping_boundary.empty:
        chosen = (
            overlapping_boundary.sort_values(
                ["boundary_overlap_amount", "eligible_pair_count", "boundary_gap", "k"],
                ascending=[False, False, True, True],
            )
            .iloc[0]
            .to_dict()
        )
        chosen["k_selection_rule"] = "max_adjacent_boundary_overlap"
        return chosen

    overlapping_any = scored_df.loc[scored_df["eligible_pair_count"] > 0].copy()
    if not overlapping_any.empty:
        chosen = (
            overlapping_any.sort_values(
                ["eligible_pair_count", "boundary_gap", "k"],
                ascending=[False, True, True],
            )
            .iloc[0]
            .to_dict()
        )
        chosen["k_selection_rule"] = "max_cross_boundary_overlap_pairs"
        return chosen

    chosen = scored_df.sort_values(["boundary_gap", "k"], ascending=[True, True]).iloc[0].to_dict()
    chosen["k_selection_rule"] = "min_adjacent_gap_no_overlap_fallback"
    return chosen


def selected_matches_long_frame(
    dataset_key: str,
    dataset_name: str,
    variant_row: dict[str, object],
) -> pd.DataFrame:
    records = variant_row["selected_matches_payload"]
    if not records:
        return pd.DataFrame()
    player_pair = variant_row["player_pair"]
    top_player = None if player_pair is None else player_pair[0]
    outside_player = None if player_pair is None else player_pair[1]
    rows = []
    for selected_order, record in enumerate(records):
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "k": int(variant_row["k"]),
                "variant": variant_row["variant"],
                "action": variant_row["action"],
                "candidate_mode": variant_row["candidate_mode"],
                "target_top_player": top_player,
                "target_outside_player": outside_player,
                "selected_order": selected_order,
                **record,
            }
        )
    return pd.DataFrame(rows)


def remaining_matches_long_frame(
    dataset_key: str,
    dataset_name: str,
    variant_row: dict[str, object],
) -> pd.DataFrame:
    records = variant_row["remaining_matches_payload"]
    if not records:
        return pd.DataFrame()
    player_pair = variant_row["player_pair"]
    top_player = None if player_pair is None else player_pair[0]
    outside_player = None if player_pair is None else player_pair[1]
    rows = []
    for remaining_order, record in enumerate(records):
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "k": int(variant_row["k"]),
                "variant": variant_row["variant"],
                "action": variant_row["action"],
                "candidate_mode": variant_row["candidate_mode"],
                "target_top_player": top_player,
                "target_outside_player": outside_player,
                "remaining_order": remaining_order,
                **record,
            }
        )
    return pd.DataFrame(rows)


def run_variant(
    bt_model: BradleyTerryModel,
    pair_table: pd.DataFrame,
    spec: dict[str, object],
    *,
    k: int,
    max_actions: int,
    ci_method: str,
    ci_alpha: float,
) -> dict[str, object]:
    cached_reports: dict[tuple[str, str], pd.DataFrame] = {}
    for alpha in range(1, max_actions + 1):
        for pair in pair_table.to_dict(orient="records"):
            top_player = pair["top_player"]
            outside_player = pair["outside_player"]
            objective = SkillGapObjective(top_player, outside_player)
            key = (top_player, outside_player)
            if key not in cached_reports:
                cached_reports[key] = compute_all_action_influences(
                    bt_model,
                    objective,
                    spec["action"],
                    influence_method="1sn",
                    candidate_mode=spec["candidate_mode"],
                )
            report = cached_reports[key]
            if alpha > len(report):
                continue
            initial_gap = float(pair["initial_gap"])
            sort_ascending = initial_gap > 0.0
            with contextlib.redirect_stdout(io.StringIO()):
                approximate_result = apply_action_on_top_alpha_influential_matches(
                    bt_model,
                    objective,
                    report,
                    alpha,
                    spec["action"],
                    recompute_mode="approximate",
                    sort_ascending=sort_ascending,
                    group_by_match=True,
                )
            predicted_gap = float(approximate_result["final_value"])
            if not predicted_gap_target_met(initial_gap, predicted_gap):
                continue
            selected = approximate_result["selected_matches"]
            if selected is None or selected.empty:
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    updated_model = refit_model_with_action(bt_model, selected, spec["action"])
            except ValueError:
                continue
            final_status = pair_ci_status(
                updated_model,
                top_player,
                outside_player,
                ci_method=ci_method,
                ci_alpha=ci_alpha,
            )
            selected_matches_payload = serialize_frame_records(selected)
            remaining_matches_payload = serialize_frame_records(updated_model.match_frame_)
            met = ci_membership_target_met(initial_gap, final_status)
            if met:
                final_ranking = ranking_from_model(updated_model)
                return {
                    "k": int(k),
                    "variant": spec["variant"],
                    "action": spec["action"],
                    "candidate_mode": spec["candidate_mode"],
                    "player_pair": (top_player, outside_player),
                    "met": True,
                    "n_actions": int(alpha),
                    "max_actions": int(max_actions),
                    "initial_value": initial_gap,
                    "predicted_final_value": predicted_gap,
                    "final_value": float(final_status["gap"]),
                    "initial_overlap": True,
                    "final_overlap": bool(final_status["overlap"]),
                    "final_outsider_above": bool(final_status["outsider_above"]),
                    "final_top_above": bool(final_status["top_above"]),
                    "gap_sign_changed": bool(float(final_status["gap"]) <= 0.0 if initial_gap >= 0.0 else float(final_status["gap"]) >= 0.0),
                    "selected_match_count": len(selected_matches_payload),
                    "selected_row_uids": [] if "row_uid" not in selected.columns else [int(x) for x in selected["row_uid"].tolist()],
                    "selected_matches_payload": selected_matches_payload,
                    "selected_matches_json": json.dumps(selected_matches_payload, sort_keys=True),
                    "remaining_match_count": int(len(updated_model.match_frame_)),
                    "remaining_matches_payload": remaining_matches_payload,
                    "remaining_row_uids": []
                    if "row_uid" not in updated_model.match_frame_.columns
                    else [int(x) for x in updated_model.match_frame_["row_uid"].tolist()],
                    "remaining_matches_json": json.dumps(remaining_matches_payload[:25], sort_keys=True),
                    "top_rank": int(pair["top_rank"]),
                    "outside_rank": int(pair["outside_rank"]),
                    "final_top1": final_ranking[0],
                    "final_top2": final_ranking[1],
                }
    return {
        "k": int(k),
        "variant": spec["variant"],
        "action": spec["action"],
        "candidate_mode": spec["candidate_mode"],
        "player_pair": None,
        "met": False,
        "n_actions": np.nan,
        "max_actions": int(max_actions),
        "initial_value": np.nan,
        "predicted_final_value": np.nan,
        "final_value": np.nan,
        "initial_overlap": np.nan,
        "final_overlap": np.nan,
        "final_outsider_above": np.nan,
        "final_top_above": np.nan,
        "gap_sign_changed": np.nan,
        "selected_match_count": 0,
        "selected_row_uids": [],
        "selected_matches_payload": [],
        "selected_matches_json": "[]",
        "remaining_match_count": 0,
        "remaining_matches_payload": [],
        "remaining_row_uids": [],
        "remaining_matches_json": "[]",
        "top_rank": np.nan,
        "outside_rank": np.nan,
        "final_top1": None,
        "final_top2": None,
    }


def plot_actions_needed(summary_df: pd.DataFrame, dataset_name: str, stem: str, *, k: int, output_dir: Path) -> Path:
    fig, _ = plot_variant_actions_needed(
        summary_df,
        title=f"{dataset_name} (k={k})",
        max_actions=int(summary_df["max_actions"].iloc[0]),
    )
    pdf_path = output_dir / f"{stem}_k{k}_ci_aware_actions_needed.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path


def run_dataset(
    dataset_key: str,
    *,
    output_dir: Path,
    max_action_fraction: float,
    ci_method: str,
    ci_alpha: float,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    built = build_named_dataset_model(dataset_key)
    bt_model = built["bt_model"]
    ranking = ranking_from_model(bt_model)
    max_actions = compute_max_actions(len(built["raw"]), max_action_fraction)
    ci_df, ranking_csv_path, ranking_pdf_path, ranking_png_path = export_gao_ci_ranking(
        bt_model,
        built["dataset_name"],
        built["dataset_key"],
        output_dir=output_dir,
        ci_method=ci_method,
        ci_alpha=ci_alpha,
    )
    k_info = choose_meaningful_k(bt_model, ci_method=ci_method, ci_alpha=ci_alpha)
    k = int(k_info["k"])
    if not (1 <= k < len(ranking)):
        raise ValueError(f"k={k} is invalid for {dataset_key}; expected 1 <= k < {len(ranking)}.")
    pair_table = build_ci_overlap_pairs(bt_model, k, ci_method=ci_method, ci_alpha=ci_alpha)
    pair_csv_path = output_dir / f"{built['dataset_key']}_k{k}_eligible_ci_pairs.csv"
    pair_table.to_csv(pair_csv_path, index=False)

    variant_rows = [
        run_variant(
            bt_model,
            pair_table,
            spec,
            k=k,
            max_actions=max_actions,
            ci_method=ci_method,
            ci_alpha=ci_alpha,
        )
        for spec in VARIANTS
    ]
    selected_match_frames = [selected_matches_long_frame(built["dataset_key"], built["dataset_name"], row) for row in variant_rows]
    remaining_match_frames = [remaining_matches_long_frame(built["dataset_key"], built["dataset_name"], row) for row in variant_rows]
    rows = [{k: v for k, v in row.items() if k not in {"selected_matches_payload", "remaining_matches_payload"}} for row in variant_rows]
    summary_df = pd.DataFrame(rows)
    selected_matches_df = (
        pd.concat([frame for frame in selected_match_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in selected_match_frames)
        else pd.DataFrame()
    )
    remaining_matches_df = (
        pd.concat([frame for frame in remaining_match_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in remaining_match_frames)
        else pd.DataFrame()
    )
    pdf_path = plot_actions_needed(summary_df, built["dataset_name"], built["dataset_key"], k=k, output_dir=output_dir)
    meta = {
        "dataset_key": built["dataset_key"],
        "dataset_name": built["dataset_name"],
        "k": k,
        "k_selection_rule": str(k_info["k_selection_rule"]),
        "boundary_has_overlap": bool(k_info["boundary_has_overlap"]),
        "boundary_overlap_amount": float(k_info["boundary_overlap_amount"]),
        "boundary_gap": float(k_info["boundary_gap"]),
        "k_boundary_top_player": str(k_info["top_player"]),
        "k_boundary_outside_player": str(k_info["outside_player"]),
        "n_original_matches": int(len(built["raw"])),
        "n_fitted_rows": int(built["dataset"].n_matches),
        "n_models": int(len(ranking)),
        "top1": ranking[0],
        "top2": ranking[1],
        "max_actions": int(max_actions),
        "ranking_csv_path": str(ranking_csv_path),
        "ranking_pdf_path": str(ranking_pdf_path),
        "ranking_png_path": str(ranking_png_path),
        "ranking_boundary_preview": preview_boundary(ci_df, k),
        "eligible_pair_count": int(len(pair_table)),
        "pair_csv_path": str(pair_csv_path),
        "selected_matches_csv_path": str(output_dir / f"{built['dataset_key']}_k{k}_ci_aware_selected_matches.csv"),
        "remaining_matches_csv_path": str(output_dir / f"{built['dataset_key']}_k{k}_ci_aware_remaining_matches.csv"),
        "pdf_path": str(pdf_path),
    }
    return summary_df, meta, pair_table, selected_matches_df, remaining_matches_df


def run_ci_aware_batch(
    *,
    dataset_keys: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    max_action_fraction: float = 0.04,
    ci_method: str = "gao_local",
    ci_alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    dataset_keys = sorted(available_hf_battle_datasets()) if dataset_keys is None else list(dataset_keys)
    output_dir = ROOT / "notebooks" / "artifacts" / "ci_aware_top1_robustness_action_needed" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_rows = []
    all_rows = []
    all_pair_rows = []
    all_selected_match_rows = []
    all_remaining_match_rows = []

    for dataset_key in dataset_keys:
        print(f"Running {dataset_key} ...")
        summary_df, meta, pair_table, selected_matches_df, remaining_matches_df = run_dataset(
            dataset_key,
            output_dir=output_dir,
            max_action_fraction=max_action_fraction,
            ci_method=ci_method,
            ci_alpha=ci_alpha,
        )
        summary_df.to_csv(output_dir / f"{meta['dataset_key']}_k{meta['k']}_ci_aware_actions_needed.csv", index=False)
        selected_matches_df.to_csv(output_dir / f"{meta['dataset_key']}_k{meta['k']}_ci_aware_selected_matches.csv", index=False)
        remaining_matches_df.to_csv(output_dir / f"{meta['dataset_key']}_k{meta['k']}_ci_aware_remaining_matches.csv", index=False)

        summary_plot_ready = summary_df.copy()
        if "dataset_key" not in summary_plot_ready.columns:
            summary_plot_ready.insert(0, "dataset_key", meta["dataset_key"])
        if "dataset_name" not in summary_plot_ready.columns:
            summary_plot_ready.insert(1, "dataset_name", meta["dataset_name"])
        if "k" not in summary_plot_ready.columns:
            summary_plot_ready.insert(2, "k", meta["k"])
        summary_plot_ready["variant_label"] = summary_plot_ready["variant"].map(ACTION_LABEL_MAP)
        summary_plot_ready["actions_needed_plot"] = summary_plot_ready["n_actions"].fillna(summary_plot_ready["max_actions"] + 1)
        summary_plot_ready["success_label"] = np.where(
            summary_plot_ready["met"],
            summary_plot_ready["n_actions"].astype("Int64").astype(str),
            ">" + summary_plot_ready["max_actions"].astype(int).astype(str),
        )
        summary_plot_ready.to_csv(output_dir / f"{meta['dataset_key']}_k{meta['k']}_ci_aware_actions_needed_plot_ready.csv", index=False)

        ranking_rows.append(
            {
                "dataset_key": meta["dataset_key"],
                "dataset_name": meta["dataset_name"],
                "k": meta["k"],
                "k_selection_rule": meta["k_selection_rule"],
                "boundary_has_overlap": meta["boundary_has_overlap"],
                "boundary_overlap_amount": meta["boundary_overlap_amount"],
                "boundary_gap": meta["boundary_gap"],
                "k_boundary_top_player": meta["k_boundary_top_player"],
                "k_boundary_outside_player": meta["k_boundary_outside_player"],
                "n_models": meta["n_models"],
                "max_actions": meta["max_actions"],
                "boundary_preview": meta["ranking_boundary_preview"],
                "eligible_pair_count": meta["eligible_pair_count"],
                "ranking_csv_path": meta["ranking_csv_path"],
                "ranking_pdf_path": meta["ranking_pdf_path"],
                "ranking_png_path": meta["ranking_png_path"],
            }
        )
        for row in summary_df.to_dict(orient="records"):
            all_rows.append({**meta, **row})
        if not pair_table.empty:
            pair_out = pair_table.copy()
            pair_out.insert(0, "dataset_key", meta["dataset_key"])
            pair_out.insert(1, "dataset_name", meta["dataset_name"])
            pair_out.insert(2, "k", meta["k"])
            all_pair_rows.extend(pair_out.to_dict(orient="records"))
        all_selected_match_rows.extend(selected_matches_df.to_dict(orient="records"))
        all_remaining_match_rows.extend(remaining_matches_df.to_dict(orient="records"))

    ranking_index = pd.DataFrame(ranking_rows)
    ranking_index.to_csv(output_dir / "all_datasets_gao_ci_ranking_index.csv", index=False)

    final_table = pd.DataFrame(all_rows)[
        [
            "dataset_key",
            "dataset_name",
            "k",
            "k_selection_rule",
            "boundary_has_overlap",
            "boundary_overlap_amount",
            "boundary_gap",
            "k_boundary_top_player",
            "k_boundary_outside_player",
            "variant",
            "action",
            "candidate_mode",
            "player_pair",
            "met",
            "n_actions",
            "max_actions",
            "initial_value",
            "final_value",
            "initial_overlap",
            "final_overlap",
            "final_outsider_above",
            "final_top_above",
            "gap_sign_changed",
            "selected_match_count",
            "selected_row_uids",
            "selected_matches_json",
            "remaining_match_count",
            "remaining_row_uids",
            "remaining_matches_json",
            "top_rank",
            "outside_rank",
            "final_top1",
            "final_top2",
            "n_original_matches",
            "n_fitted_rows",
            "n_models",
            "top1",
            "top2",
            "ranking_boundary_preview",
            "eligible_pair_count",
            "pair_csv_path",
            "selected_matches_csv_path",
            "remaining_matches_csv_path",
            "ranking_csv_path",
            "ranking_pdf_path",
            "ranking_png_path",
            "pdf_path",
        ]
    ]
    final_table.to_csv(output_dir / "all_datasets_ci_aware_actions_needed.csv", index=False)

    plot_ready_table = final_table.copy()
    plot_ready_table["variant_label"] = plot_ready_table["variant"].map(ACTION_LABEL_MAP)
    plot_ready_table["actions_needed_plot"] = plot_ready_table["n_actions"].fillna(plot_ready_table["max_actions"] + 1)
    plot_ready_table["success_label"] = np.where(
        plot_ready_table["met"],
        plot_ready_table["n_actions"].astype("Int64").astype(str),
        ">" + plot_ready_table["max_actions"].astype(int).astype(str),
    )
    plot_ready_table.to_csv(output_dir / "all_datasets_ci_aware_actions_needed_plot_ready.csv", index=False)

    pair_table_all = pd.DataFrame(all_pair_rows)
    if not pair_table_all.empty:
        pair_table_all.to_csv(output_dir / "all_datasets_eligible_ci_pairs.csv", index=False)

    selected_matches_table = pd.DataFrame(all_selected_match_rows)
    selected_matches_table.to_csv(output_dir / "all_datasets_ci_aware_selected_matches.csv", index=False)

    remaining_matches_table = pd.DataFrame(all_remaining_match_rows)
    remaining_matches_table.to_csv(output_dir / "all_datasets_ci_aware_remaining_matches.csv", index=False)

    return {
        "ranking_index": ranking_index,
        "final_table": final_table,
        "plot_ready_table": plot_ready_table,
        "pair_table_all": pair_table_all,
        "selected_matches_table": selected_matches_table,
        "remaining_matches_table": remaining_matches_table,
    }
