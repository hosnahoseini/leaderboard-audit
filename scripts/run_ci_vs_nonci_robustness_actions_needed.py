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
for path in [ROOT / "src", ROOT, ROOT / "IsRankingRobust"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clean_bt_rank import ACTION_LABEL_MAP, available_hf_battle_datasets, plot_ranking_before_after, plot_variant_actions_needed, use_paper_rc
from clean_bt_rank.ci_aware_actions_needed import (
    VARIANTS,
    build_ci_overlap_pairs,
    build_named_dataset_model,
    choose_meaningful_k,
    ci_membership_target_met,
    compute_max_actions,
    export_gao_ci_ranking,
    pair_ci_status,
    predicted_gap_target_met,
    remaining_matches_long_frame,
    selected_matches_long_frame,
)
from clean_bt_rank.experiments._common import ranking_frame
from clean_bt_rank.iterative_actions import (
    apply_action_on_top_alpha_influential_matches,
    compute_all_action_influences,
    refit_model_with_action,
)
from clean_bt_rank.objectives import CIBoundaryObjective, CIStrictGapObjective, SkillGapObjective
from clean_bt_rank.player_influence import ranking_from_model


use_paper_rc()
ACTIVE_VARIANTS = list(VARIANTS)


def set_active_variants(variant_names: Optional[list[str]]) -> None:
    global ACTIVE_VARIANTS
    if not variant_names:
        ACTIVE_VARIANTS = list(VARIANTS)
        return
    requested = {str(name) for name in variant_names}
    ACTIVE_VARIANTS = [spec for spec in VARIANTS if str(spec["variant"]) in requested]
    if not ACTIVE_VARIANTS:
        raise ValueError(f"No variants selected from {sorted(requested)}.")


def resolve_dataset_order(dataset_keys: list[str], order_mode: str) -> tuple[list[str], list[tuple[str, int]]]:
    keys = [str(key) for key in dataset_keys]
    if order_mode == "alpha":
        ordered = sorted(keys)
        return ordered, []

    size_rows: list[tuple[str, int]] = []
    for key in keys:
        built = build_named_dataset_model(key)
        size_rows.append((key, int(len(built["raw"]))))

    reverse = order_mode == "size_desc"
    size_rows.sort(key=lambda item: (item[1], item[0]), reverse=reverse)
    ordered = [key for key, _ in size_rows]
    return ordered, size_rows


def find_smallest_runnable_dataset_key(candidate_keys: list[str]) -> tuple[str, list[tuple[str, str]]]:
    failures: list[tuple[str, str]] = []
    size_rows: list[tuple[int, str]] = []
    for key in candidate_keys:
        try:
            built = build_named_dataset_model(key)
        except Exception as exc:
            failures.append((key, str(exc)))
            continue
        size_rows.append((int(len(built["raw"])), key))
    if not size_rows:
        failure_blob = "; ".join(f"{key}: {msg}" for key, msg in failures)
        raise RuntimeError(f"No runnable dataset found. Failures: {failure_blob}")
    size_rows.sort(key=lambda row: (row[0], row[1]))
    return size_rows[0][1], failures


def choose_low_middle_high_valid_ks(
    bt_model,
    *,
    ci_method: str,
    ci_alpha: float,
) -> list[dict[str, object]]:
    ranking = ranking_frame(bt_model, ci_method=ci_method).sort_values("rank").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for k in range(1, len(ranking)):
        top_player = str(ranking.iloc[k - 1]["competitor"])
        outside_player = str(ranking.iloc[k]["competitor"])
        objective = CIStrictGapObjective(top_player, outside_player, ci_method=ci_method, alpha=ci_alpha)
        initial_value = float(objective.value(bt_model))
        if initial_value < 0.0:
            continue
        rows.append(
            {
                "k": int(k),
                "top_player": top_player,
                "outside_player": outside_player,
                "initial_value": initial_value,
            }
        )
    valid_df = pd.DataFrame(rows)
    if valid_df.empty:
        raise RuntimeError("No valid k found where the strict CI objective is initially unmet.")

    n_valid = len(valid_df)
    regions = [
        ("low", valid_df.iloc[: max(1, n_valid // 3)].copy()),
        ("middle", valid_df.iloc[n_valid // 3 : max(n_valid // 3 + 1, 2 * n_valid // 3)].copy()),
        ("high", valid_df.iloc[max(2 * n_valid // 3, 0) :].copy()),
    ]
    chosen: list[dict[str, object]] = []
    used_ks: set[int] = set()
    for label, frame in regions:
        if frame.empty:
            continue
        row = frame.sort_values(["initial_value", "k"], ascending=[True, True]).iloc[0].to_dict()
        if int(row["k"]) in used_ks:
            continue
        row["k_selection_rule"] = "low_mid_high_valid"
        row["region"] = label
        used_ks.add(int(row["k"]))
        chosen.append(row)
    if len(chosen) < 3:
        for row in valid_df.sort_values(["initial_value", "k"], ascending=[True, True]).to_dict(orient="records"):
            if int(row["k"]) in used_ks:
                continue
            row["k_selection_rule"] = "low_mid_high_valid_fallback"
            row["region"] = "fallback"
            used_ks.add(int(row["k"]))
            chosen.append(row)
            if len(chosen) == 3:
                break
    if len(chosen) < 3:
        raise RuntimeError("Could not identify three distinct valid k values.")
    return sorted(chosen, key=lambda row: int(row["k"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CI-aware vs point-estimate top-k actions-needed comparison across datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=tuple(sorted(available_hf_battle_datasets())),
        help="Dataset keys to evaluate.",
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
        default=ROOT / "notebooks" / "artifacts" / "ci_vs_nonci_actions_needed",
        help="Directory for CSV/PDF/PNG outputs.",
    )
    parser.add_argument(
        "--max-action-fraction",
        type=float,
        default=0.04,
        help="Per-dataset max alpha as ceil(fraction * n_original_matches). Default: 0.04.",
    )
    parser.add_argument(
        "--ci-method",
        default="gao_local",
        help="Confidence interval backend used to choose k and evaluate CI-aware success.",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.05,
        help="Confidence interval alpha. Default: 0.05.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print dataset order, then exit.",
    )
    parser.add_argument(
        "--ci-objective",
        choices=("boundary", "strict"),
        default="boundary",
        help="CI-aware objective: legacy frozen boundary objective or strict outsider-worst-vs-insider-best objective.",
    )
    parser.add_argument(
        "--k-selection-mode",
        choices=("meaningful", "low_mid_high_valid"),
        default="meaningful",
        help="How to choose k. 'low_mid_high_valid' selects three valid k values from low/middle/high ranking regions.",
    )
    parser.add_argument(
        "--smallest-runnable",
        action="store_true",
        help="Ignore --datasets and run only the smallest dataset that is loadable in the current environment.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[str(spec["variant"]) for spec in VARIANTS],
        default=None,
        help="Optional subset of action variants to evaluate.",
    )
    return parser.parse_args()


def save_ranking_before_after_artifacts(
    *,
    output_dir: Path,
    dataset_key: str,
    dataset_name: str,
    method_key: str,
    variant_key: str,
    k: int,
    target_player: Optional[str],
    context_players: Optional[list[str]] = None,
    plot_context_size: int = 10,
    ranking_before_df: pd.DataFrame,
    ranking_after_df: pd.DataFrame,
) -> dict[str, str]:
    stem = f"{dataset_key}_k{k}_{method_key}_{variant_key}_ranking_before_after"
    before_csv = output_dir / f"{stem}_before.csv"
    after_csv = output_dir / f"{stem}_after.csv"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"

    ranking_before_plot_df = ranking_before_df.copy()
    ranking_after_plot_df = ranking_after_df.copy()
    if context_players:
        before_idx = ranking_before_df.set_index("competitor")
        after_idx = ranking_after_df.set_index("competitor")
        valid_players = [player for player in context_players if player in before_idx.index and player in after_idx.index]
        if valid_players:
            all_ranks = []
            for player in valid_players:
                all_ranks.append(int(before_idx.loc[player, "rank"]))
                all_ranks.append(int(after_idx.loc[player, "rank"]))
            rank_lo = min(all_ranks)
            rank_hi = max(all_ranks)
            width = max(1, int(plot_context_size))
            span = rank_hi - rank_lo + 1
            extra = max(0, width - span)
            pad_left = extra // 2
            pad_right = extra - pad_left
            start_rank = max(1, rank_lo - pad_left)
            end_rank = start_rank + width - 1
            if end_rank > len(ranking_before_df):
                end_rank = len(ranking_before_df)
                start_rank = max(1, end_rank - width + 1)
            visible_players = ranking_before_df.loc[
                ranking_before_df["rank"].between(start_rank, end_rank),
                "competitor",
            ].tolist()
            ranking_before_plot_df = ranking_before_df.loc[ranking_before_df["competitor"].isin(visible_players)].copy()
            ranking_after_plot_df = ranking_after_df.loc[ranking_after_df["competitor"].isin(visible_players)].copy()

    ranking_before_df.to_csv(before_csv, index=False)
    ranking_after_df.to_csv(after_csv, index=False)

    fig, _ = plot_ranking_before_after(
        ranking_before_plot_df,
        ranking_after_plot_df,
        target_player=target_player,
        k=k,
        title=f"{dataset_name}: {method_key} {variant_key}",
        show_ci=True,
    )
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {
        "ranking_before_csv_path": str(before_csv),
        "ranking_after_csv_path": str(after_csv),
        "ranking_before_after_pdf_path": str(pdf_path),
        "ranking_before_after_png_path": str(png_path),
    }


def run_nonci_variant(
    *,
    bt_model,
    top_player: str,
    outside_player: str,
    spec: dict[str, object],
    k: int,
    max_actions: int,
    dataset_key: str,
    dataset_name: str,
    output_dir: Path,
) -> dict[str, object]:
    objective = SkillGapObjective(top_player, outside_player)
    report = compute_all_action_influences(
        bt_model,
        objective,
        spec["action"],
        influence_method="1sn",
        candidate_mode=spec["candidate_mode"],
    )
    initial_gap = float(bt_model.reported_gap(top_player, outside_player))
    sort_ascending = initial_gap > 0.0
    ranking_before_df = ranking_frame(bt_model, ci_method="gao_local")

    for alpha in range(1, max_actions + 1):
        if alpha > len(report):
            continue
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
        final_gap = float(updated_model.reported_gap(top_player, outside_player))
        met = predicted_gap_target_met(initial_gap, final_gap)
        if met:
            ranking_after_df = ranking_frame(updated_model, ci_method="gao_local")
            ranking_artifacts = save_ranking_before_after_artifacts(
                output_dir=output_dir,
                dataset_key=dataset_key,
                dataset_name=dataset_name,
                method_key="nonci",
                variant_key=str(spec["variant"]),
                k=k,
                target_player=outside_player,
                context_players=[top_player, outside_player],
                ranking_before_df=ranking_before_df,
                ranking_after_df=ranking_after_df,
            )
            selected_payload = selected.to_dict(orient="records")
            remaining_payload = updated_model.match_frame_.to_dict(orient="records")
            return {
                "k": int(k),
                "method": "nonci",
                "variant": spec["variant"],
                "action": spec["action"],
                "candidate_mode": spec["candidate_mode"],
                "player_pair": (top_player, outside_player),
                "met": True,
                "n_actions": int(alpha),
                "max_actions": int(max_actions),
                "initial_value": initial_gap,
                "predicted_final_value": predicted_gap,
                "final_value": final_gap,
                "initial_overlap": np.nan,
                "final_overlap": np.nan,
                "final_outsider_above": np.nan,
                "final_top_above": np.nan,
                "gap_sign_changed": bool(final_gap <= 0.0 if initial_gap >= 0.0 else final_gap >= 0.0),
                "selected_match_count": int(len(selected_payload)),
                "selected_row_uids": [] if "row_uid" not in selected.columns else [int(x) for x in selected["row_uid"].tolist()],
                "selected_matches_payload": selected_payload,
                "selected_matches_json": json.dumps(selected_payload, sort_keys=True, default=str),
                "remaining_match_count": int(len(updated_model.match_frame_)),
                "remaining_matches_payload": remaining_payload,
                "remaining_row_uids": []
                if "row_uid" not in updated_model.match_frame_.columns
                else [int(x) for x in updated_model.match_frame_["row_uid"].tolist()],
                "remaining_matches_json": json.dumps(remaining_payload[:25], sort_keys=True, default=str),
                "top_rank": int(ranking_after_df.loc[ranking_after_df["competitor"] == top_player, "rank"].iloc[0]),
                "outside_rank": int(ranking_after_df.loc[ranking_after_df["competitor"] == outside_player, "rank"].iloc[0]),
                "final_top1": ranking_from_model(updated_model)[0],
                "final_top2": ranking_from_model(updated_model)[1],
                **ranking_artifacts,
            }

    return {
        "k": int(k),
        "method": "nonci",
        "variant": spec["variant"],
        "action": spec["action"],
        "candidate_mode": spec["candidate_mode"],
        "player_pair": (top_player, outside_player),
        "met": False,
        "n_actions": np.nan,
        "max_actions": int(max_actions),
        "initial_value": initial_gap,
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
        "ranking_before_csv_path": None,
        "ranking_after_csv_path": None,
        "ranking_before_after_pdf_path": None,
        "ranking_before_after_png_path": None,
    }


def run_strict_ci_variant_with_artifacts(
    *,
    bt_model,
    top_player: str,
    outside_player: str,
    spec: dict[str, object],
    k: int,
    max_actions: int,
    ci_method: str,
    ci_alpha: float,
    dataset_key: str,
    dataset_name: str,
    output_dir: Path,
) -> dict[str, object]:
    objective = CIStrictGapObjective(
        top_player,
        outside_player,
        ci_method=ci_method,
        alpha=ci_alpha,
    )
    report = compute_all_action_influences(
        bt_model,
        objective,
        spec["action"],
        influence_method="1sn",
        candidate_mode=spec["candidate_mode"],
    )
    ranking_before_df = ranking_frame(bt_model, ci_method=ci_method)
    initial_value = float(objective.value(bt_model))

    for alpha in range(1, max_actions + 1):
        if alpha > len(report):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            approximate_result = apply_action_on_top_alpha_influential_matches(
                bt_model,
                objective,
                report,
                alpha,
                spec["action"],
                recompute_mode="approximate",
                sort_ascending=True,
                group_by_match=True,
            )
        predicted_value = float(approximate_result["final_value"])
        if predicted_value >= 0.0:
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
        final_value = float(objective.value(updated_model))
        met = bool(final_value < 0.0 and final_status["outsider_above"] and not final_status["overlap"])
        if not met:
            continue
        ranking_after_df = ranking_frame(updated_model, ci_method=ci_method)
        ranking_artifacts = save_ranking_before_after_artifacts(
            output_dir=output_dir,
            dataset_key=dataset_key,
            dataset_name=dataset_name,
            method_key="strict_ci",
            variant_key=str(spec["variant"]),
            k=k,
            target_player=outside_player,
            context_players=[top_player, outside_player],
            ranking_before_df=ranking_before_df,
            ranking_after_df=ranking_after_df,
        )
        selected_payload = selected.to_dict(orient="records")
        remaining_payload = updated_model.match_frame_.to_dict(orient="records")
        return {
            "k": int(k),
            "method": "strict_ci",
            "variant": spec["variant"],
            "action": spec["action"],
            "candidate_mode": spec["candidate_mode"],
            "player_pair": (top_player, outside_player),
            "met": True,
            "n_actions": int(alpha),
            "max_actions": int(max_actions),
            "initial_value": initial_value,
            "predicted_final_value": predicted_value,
            "final_value": final_value,
            "initial_overlap": bool(initial_value <= 0.0),
            "final_overlap": bool(final_status["overlap"]),
            "final_outsider_above": bool(final_status["outsider_above"]),
            "final_top_above": bool(final_status["top_above"]),
            "gap_sign_changed": bool(float(final_status["gap"]) <= 0.0),
            "selected_match_count": int(len(selected_payload)),
            "selected_row_uids": [] if "row_uid" not in selected.columns else [int(x) for x in selected["row_uid"].tolist()],
            "selected_matches_payload": selected_payload,
            "selected_matches_json": json.dumps(selected_payload, sort_keys=True, default=str),
            "remaining_match_count": int(len(updated_model.match_frame_)),
            "remaining_matches_payload": remaining_payload,
            "remaining_row_uids": []
            if "row_uid" not in updated_model.match_frame_.columns
            else [int(x) for x in updated_model.match_frame_["row_uid"].tolist()],
            "remaining_matches_json": json.dumps(remaining_payload[:25], sort_keys=True, default=str),
            "top_rank": int(ranking_after_df.loc[ranking_after_df["competitor"] == top_player, "rank"].iloc[0]),
            "outside_rank": int(ranking_after_df.loc[ranking_after_df["competitor"] == outside_player, "rank"].iloc[0]),
            "final_top1": ranking_from_model(updated_model)[0],
            "final_top2": ranking_from_model(updated_model)[1],
            **ranking_artifacts,
        }

    return {
        "k": int(k),
        "method": "strict_ci",
        "variant": spec["variant"],
        "action": spec["action"],
        "candidate_mode": spec["candidate_mode"],
        "player_pair": (top_player, outside_player),
        "met": False,
        "n_actions": np.nan,
        "max_actions": int(max_actions),
        "initial_value": initial_value,
        "predicted_final_value": np.nan,
        "final_value": np.nan,
        "initial_overlap": bool(initial_value <= 0.0),
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
        "ranking_before_csv_path": None,
        "ranking_after_csv_path": None,
        "ranking_before_after_pdf_path": None,
        "ranking_before_after_png_path": None,
    }


def run_ci_variant_with_artifacts(
    *,
    bt_model,
    pair_table: pd.DataFrame,
    spec: dict[str, object],
    k: int,
    max_actions: int,
    ci_method: str,
    ci_alpha: float,
    dataset_key: str,
    dataset_name: str,
    output_dir: Path,
) -> dict[str, object]:
    objective = CIBoundaryObjective(
        bt_model,
        k,
        ci_method=ci_method,
        alpha=ci_alpha,
        freeze_ranking=True,
    )
    cached_reports: dict[str, pd.DataFrame] = {}
    ranking_before_df = ranking_frame(bt_model, ci_method="gao_local")
    initial_boundary_margin = float(objective.value(bt_model))
    top_player = str(objective.player_a_name)
    outside_player = str(objective.player_b_name)

    for alpha in range(1, max_actions + 1):
        report_key = str(spec["variant"])
        if report_key not in cached_reports:
            cached_reports[report_key] = compute_all_action_influences(
                bt_model,
                objective,
                spec["action"],
                influence_method="1sn",
                candidate_mode=spec["candidate_mode"],
            )
        report = cached_reports[report_key]
        if alpha > len(report):
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            approximate_result = apply_action_on_top_alpha_influential_matches(
                bt_model,
                objective,
                report,
                alpha,
                spec["action"],
                recompute_mode="approximate",
                sort_ascending=True,
                group_by_match=True,
            )
        predicted_margin = float(approximate_result["final_value"])
        if predicted_margin >= 0.0:
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
        final_boundary_margin = float(objective.value(updated_model))
        met = bool(final_boundary_margin < 0.0 and final_status["outsider_above"] and not final_status["overlap"])
        if not met:
            continue

        ranking_after_df = ranking_frame(updated_model, ci_method="gao_local")
        ranking_artifacts = save_ranking_before_after_artifacts(
            output_dir=output_dir,
            dataset_key=dataset_key,
            dataset_name=dataset_name,
            method_key="ci_aware",
            variant_key=str(spec["variant"]),
            k=k,
            target_player=outside_player,
            context_players=[top_player, outside_player],
            ranking_before_df=ranking_before_df,
            ranking_after_df=ranking_after_df,
        )
        selected_payload = selected.to_dict(orient="records")
        remaining_payload = updated_model.match_frame_.to_dict(orient="records")
        return {
            "k": int(k),
            "method": "ci_aware",
            "variant": spec["variant"],
            "action": spec["action"],
            "candidate_mode": spec["candidate_mode"],
            "player_pair": (top_player, outside_player),
            "met": True,
            "n_actions": int(alpha),
            "max_actions": int(max_actions),
            "initial_value": initial_boundary_margin,
            "predicted_final_value": predicted_margin,
            "final_value": final_boundary_margin,
            "initial_overlap": bool(initial_boundary_margin <= 0.0),
            "final_overlap": bool(final_status["overlap"]),
            "final_outsider_above": bool(final_status["outsider_above"]),
            "final_top_above": bool(final_status["top_above"]),
            "gap_sign_changed": bool(float(final_status["gap"]) <= 0.0),
            "selected_match_count": int(len(selected_payload)),
            "selected_row_uids": [] if "row_uid" not in selected.columns else [int(x) for x in selected["row_uid"].tolist()],
            "selected_matches_payload": selected_payload,
            "selected_matches_json": json.dumps(selected_payload, sort_keys=True, default=str),
            "remaining_match_count": int(len(updated_model.match_frame_)),
            "remaining_matches_payload": remaining_payload,
            "remaining_row_uids": []
            if "row_uid" not in updated_model.match_frame_.columns
            else [int(x) for x in updated_model.match_frame_["row_uid"].tolist()],
            "remaining_matches_json": json.dumps(remaining_payload[:25], sort_keys=True, default=str),
            "top_rank": int(k),
            "outside_rank": int(k + 1),
            "final_top1": ranking_from_model(updated_model)[0],
            "final_top2": ranking_from_model(updated_model)[1],
            **ranking_artifacts,
        }

    return {
        "k": int(k),
        "method": "ci_aware",
        "variant": spec["variant"],
        "action": spec["action"],
        "candidate_mode": spec["candidate_mode"],
        "player_pair": None,
        "met": False,
        "n_actions": np.nan,
        "max_actions": int(max_actions),
        "initial_value": initial_boundary_margin,
        "predicted_final_value": np.nan,
        "final_value": np.nan,
        "initial_overlap": bool(initial_boundary_margin <= 0.0),
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
        "top_rank": int(k),
        "outside_rank": int(k + 1),
        "final_top1": None,
        "final_top2": None,
        "ranking_before_csv_path": None,
        "ranking_after_csv_path": None,
        "ranking_before_after_pdf_path": None,
        "ranking_before_after_png_path": None,
    }


def run_ci_method(
    *,
    built: dict[str, object],
    output_dir: Path,
    ci_method: str,
    ci_alpha: float,
    max_action_fraction: float,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bt_model = built["bt_model"]
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
    pair_table = build_ci_overlap_pairs(bt_model, k, ci_method=ci_method, ci_alpha=ci_alpha)
    pair_csv_path = output_dir / f"{built['dataset_key']}_k{k}_eligible_ci_pairs.csv"
    pair_table.to_csv(pair_csv_path, index=False)

    variant_rows = [
        run_ci_variant_with_artifacts(
                bt_model=bt_model,
                pair_table=pair_table,
                spec=spec,
                k=k,
                max_actions=max_actions,
                ci_method=ci_method,
                ci_alpha=ci_alpha,
                dataset_key=str(built["dataset_key"]),
                dataset_name=str(built["dataset_name"]),
                output_dir=output_dir,
            )
        for spec in ACTIVE_VARIANTS
    ]
    rows = [{key: value for key, value in row.items() if key not in {"selected_matches_payload", "remaining_matches_payload"}} for row in variant_rows]
    summary_df = pd.DataFrame(rows)
    summary_df["method_label"] = "CI-aware"

    selected_frames = []
    remaining_frames = []
    for row in variant_rows:
        selected_frame = selected_matches_long_frame(built["dataset_key"], built["dataset_name"], row)
        if not selected_frame.empty:
            selected_frame.insert(3, "method", "ci_aware")
            selected_frames.append(selected_frame)
        remaining_frame = remaining_matches_long_frame(built["dataset_key"], built["dataset_name"], row)
        if not remaining_frame.empty:
            remaining_frame.insert(3, "method", "ci_aware")
            remaining_frames.append(remaining_frame)

    selected_matches_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    remaining_matches_df = pd.concat(remaining_frames, ignore_index=True) if remaining_frames else pd.DataFrame()

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
        "n_models": int(len(ranking_from_model(bt_model))),
        "top1": ranking_from_model(bt_model)[0],
        "top2": ranking_from_model(bt_model)[1],
        "max_actions": int(max_actions),
        "ranking_csv_path": str(ranking_csv_path),
        "ranking_pdf_path": str(ranking_pdf_path),
        "ranking_png_path": str(ranking_png_path),
        "eligible_pair_count": int(len(pair_table)),
        "pair_csv_path": str(pair_csv_path),
    }
    return summary_df, meta, pair_table, selected_matches_df, remaining_matches_df


def run_strict_ci_method(
    *,
    built: dict[str, object],
    output_dir: Path,
    ci_method: str,
    ci_alpha: float,
    max_action_fraction: float,
    k: int,
    top_player: str,
    outside_player: str,
    region: str,
    k_selection_rule: str,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    bt_model = built["bt_model"]
    max_actions = compute_max_actions(len(built["raw"]), max_action_fraction)
    ci_df, ranking_csv_path, ranking_pdf_path, ranking_png_path = export_gao_ci_ranking(
        bt_model,
        built["dataset_name"],
        built["dataset_key"],
        output_dir=output_dir,
        ci_method=ci_method,
        ci_alpha=ci_alpha,
    )
    variant_rows = [
        run_strict_ci_variant_with_artifacts(
            bt_model=bt_model,
            top_player=top_player,
            outside_player=outside_player,
            spec=spec,
            k=k,
            max_actions=max_actions,
            ci_method=ci_method,
            ci_alpha=ci_alpha,
            dataset_key=str(built["dataset_key"]),
            dataset_name=str(built["dataset_name"]),
            output_dir=output_dir,
        )
        for spec in ACTIVE_VARIANTS
    ]
    rows = [{key: value for key, value in row.items() if key not in {"selected_matches_payload", "remaining_matches_payload"}} for row in variant_rows]
    summary_df = pd.DataFrame(rows)
    summary_df["method_label"] = "Strict CI-aware"

    selected_frames = []
    remaining_frames = []
    for row in variant_rows:
        selected_frame = selected_matches_long_frame(built["dataset_key"], built["dataset_name"], row)
        if not selected_frame.empty:
            selected_frame.insert(3, "method", "strict_ci")
            selected_frames.append(selected_frame)
        remaining_frame = remaining_matches_long_frame(built["dataset_key"], built["dataset_name"], row)
        if not remaining_frame.empty:
            remaining_frame.insert(3, "method", "strict_ci")
            remaining_frames.append(remaining_frame)

    selected_matches_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    remaining_matches_df = pd.concat(remaining_frames, ignore_index=True) if remaining_frames else pd.DataFrame()

    objective = CIStrictGapObjective(top_player, outside_player, ci_method=ci_method, alpha=ci_alpha)
    meta = {
        "dataset_key": built["dataset_key"],
        "dataset_name": built["dataset_name"],
        "k": int(k),
        "k_selection_rule": str(k_selection_rule),
        "region": str(region),
        "boundary_has_overlap": bool(float(objective.value(bt_model)) <= 0.0),
        "boundary_overlap_amount": np.nan,
        "boundary_gap": float(bt_model.reported_gap(top_player, outside_player)),
        "k_boundary_top_player": str(top_player),
        "k_boundary_outside_player": str(outside_player),
        "n_original_matches": int(len(built["raw"])),
        "n_fitted_rows": int(built["dataset"].n_matches),
        "n_models": int(len(ranking_from_model(bt_model))),
        "top1": ranking_from_model(bt_model)[0],
        "top2": ranking_from_model(bt_model)[1],
        "max_actions": int(max_actions),
        "ranking_csv_path": str(ranking_csv_path),
        "ranking_pdf_path": str(ranking_pdf_path),
        "ranking_png_path": str(ranking_png_path),
        "eligible_pair_count": 1,
        "pair_csv_path": None,
        "ci_ranking_rows": int(len(ci_df)),
    }
    return summary_df, meta, selected_matches_df, remaining_matches_df


def run_nonci_method(
    *,
    built: dict[str, object],
    output_dir: Path,
    k: int,
    top_player: str,
    outside_player: str,
    max_action_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bt_model = built["bt_model"]
    max_actions = compute_max_actions(len(built["raw"]), max_action_fraction)
    variant_rows = [
        run_nonci_variant(
            bt_model=bt_model,
            top_player=top_player,
            outside_player=outside_player,
            spec=spec,
            k=k,
            max_actions=max_actions,
            dataset_key=str(built["dataset_key"]),
            dataset_name=str(built["dataset_name"]),
            output_dir=output_dir,
        )
        for spec in ACTIVE_VARIANTS
    ]
    rows = [{key: value for key, value in row.items() if key not in {"selected_matches_payload", "remaining_matches_payload"}} for row in variant_rows]
    summary_df = pd.DataFrame(rows)
    summary_df["method_label"] = "Point estimate"

    selected_frames = []
    remaining_frames = []
    for row in variant_rows:
        selected_frame = selected_matches_long_frame(built["dataset_key"], built["dataset_name"], row)
        if not selected_frame.empty:
            selected_frame.insert(3, "method", "nonci")
            selected_frames.append(selected_frame)
        remaining_frame = remaining_matches_long_frame(built["dataset_key"], built["dataset_name"], row)
        if not remaining_frame.empty:
            remaining_frame.insert(3, "method", "nonci")
            remaining_frames.append(remaining_frame)

    selected_matches_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    remaining_matches_df = pd.concat(remaining_frames, ignore_index=True) if remaining_frames else pd.DataFrame()
    return summary_df, selected_matches_df, remaining_matches_df


def save_method_plot(summary_df: pd.DataFrame, dataset_name: str, dataset_key: str, method_key: str, k: int, output_dir: Path) -> str:
    fig, _ = plot_variant_actions_needed(
        summary_df,
        title=f"{dataset_name} ({method_key}, k={k})",
        max_actions=int(summary_df["max_actions"].iloc[0]),
    )
    pdf_path = output_dir / f"{dataset_key}_k{k}_{method_key}_actions_needed.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(pdf_path)


def main() -> None:
    args = parse_args()
    set_active_variants(args.variants)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smallest_runnable:
        smallest_key, failures = find_smallest_runnable_dataset_key(list(args.datasets))
        datasets, size_rows = resolve_dataset_order([smallest_key], args.dataset_order)
        if failures:
            print("Skipped non-runnable datasets while resolving the smallest runnable dataset:")
            for key, msg in failures:
                print(f"- {key}: {msg}")
    else:
        datasets, size_rows = resolve_dataset_order(list(args.datasets), args.dataset_order)
    print("Resolved dataset order:")
    for idx, dataset_key in enumerate(datasets, start=1):
        size_suffix = ""
        for key, n_rows in size_rows:
            if key == dataset_key:
                size_suffix = f" ({n_rows} battle rows)"
                break
        print(f"{idx:>2}. {dataset_key}{size_suffix}")

    run_metadata = {
        "datasets": datasets,
        "dataset_order": args.dataset_order,
        "size_rows": size_rows,
        "max_action_fraction": args.max_action_fraction,
        "ci_method": args.ci_method,
        "ci_alpha": args.ci_alpha,
        "ci_objective": args.ci_objective,
        "k_selection_mode": args.k_selection_mode,
        "smallest_runnable": args.smallest_runnable,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    if args.dry_run:
        return

    k_rows = []
    ci_rows = []
    nonci_rows = []
    comparison_rows = []
    all_pair_rows = []
    all_selected_rows = []
    all_remaining_rows = []

    for dataset_key in datasets:
        print(f"Running {dataset_key} ...", flush=True)
        built = build_named_dataset_model(dataset_key)

        if args.k_selection_mode == "low_mid_high_valid":
            k_targets = choose_low_middle_high_valid_ks(
                built["bt_model"],
                ci_method=args.ci_method,
                ci_alpha=args.ci_alpha,
            )
        else:
            k_info = choose_meaningful_k(built["bt_model"], ci_method=args.ci_method, ci_alpha=args.ci_alpha)
            k_targets = [
                {
                    "k": int(k_info["k"]),
                    "top_player": str(k_info["top_player"]),
                    "outside_player": str(k_info["outside_player"]),
                    "initial_value": np.nan,
                    "k_selection_rule": str(k_info["k_selection_rule"]),
                    "region": "meaningful",
                }
            ]

        for target in k_targets:
            k = int(target["k"])
            top_player = str(target["top_player"])
            outside_player = str(target["outside_player"])
            region = str(target.get("region", "na"))
            k_selection_rule = str(target.get("k_selection_rule", args.k_selection_mode))

            if args.ci_objective == "strict":
                ci_summary_df, meta, ci_selected_df, ci_remaining_df = run_strict_ci_method(
                    built=built,
                    output_dir=output_dir,
                    ci_method=args.ci_method,
                    ci_alpha=args.ci_alpha,
                    max_action_fraction=args.max_action_fraction,
                    k=k,
                    top_player=top_player,
                    outside_player=outside_player,
                    region=region,
                    k_selection_rule=k_selection_rule,
                )
                pair_table = pd.DataFrame(
                    [
                        {
                            "top_player": top_player,
                            "outside_player": outside_player,
                            "top_rank": int(k),
                            "outside_rank": int(k + 1),
                            "initial_gap": float(built["bt_model"].reported_gap(top_player, outside_player)),
                            "initial_overlap": bool(float(target.get("initial_value", np.nan)) <= 0.0),
                        }
                    ]
                )
            else:
                ci_summary_df, meta, pair_table, ci_selected_df, ci_remaining_df = run_ci_method(
                    built=built,
                    output_dir=output_dir,
                    ci_method=args.ci_method,
                    ci_alpha=args.ci_alpha,
                    max_action_fraction=args.max_action_fraction,
                )
                k = int(meta["k"])
                top_player = str(meta["k_boundary_top_player"])
                outside_player = str(meta["k_boundary_outside_player"])

            nonci_summary_df, nonci_selected_df, nonci_remaining_df = run_nonci_method(
                built=built,
                output_dir=output_dir,
                k=k,
                top_player=top_player,
                outside_player=outside_player,
                max_action_fraction=args.max_action_fraction,
            )

            ci_method_key = "strict_ci" if args.ci_objective == "strict" else "ci_aware"
            ci_plot_path = save_method_plot(ci_summary_df, str(meta["dataset_name"]), str(meta["dataset_key"]), ci_method_key, int(k), output_dir)
            nonci_plot_path = save_method_plot(nonci_summary_df, str(meta["dataset_name"]), str(meta["dataset_key"]), "nonci", int(k), output_dir)

            ci_summary_df = ci_summary_df.copy()
            nonci_summary_df = nonci_summary_df.copy()
            ci_summary_df.insert(0, "dataset_key", meta["dataset_key"])
            ci_summary_df.insert(1, "dataset_name", meta["dataset_name"])
            ci_summary_df["region"] = region
            nonci_summary_df.insert(0, "dataset_key", meta["dataset_key"])
            nonci_summary_df.insert(1, "dataset_name", meta["dataset_name"])
            nonci_summary_df["region"] = region
            ci_summary_df["method_plot_pdf_path"] = ci_plot_path
            nonci_summary_df["method_plot_pdf_path"] = nonci_plot_path

            ci_summary_path = output_dir / f"{meta['dataset_key']}_k{k}_{ci_method_key}_actions_needed.csv"
            nonci_summary_path = output_dir / f"{meta['dataset_key']}_k{k}_nonci_actions_needed.csv"
            ci_summary_df.to_csv(ci_summary_path, index=False)
            nonci_summary_df.to_csv(nonci_summary_path, index=False)

            if not ci_selected_df.empty:
                ci_selected_path = output_dir / f"{meta['dataset_key']}_k{k}_{ci_method_key}_selected_matches.csv"
                ci_selected_df.to_csv(ci_selected_path, index=False)
                all_selected_rows.extend(ci_selected_df.to_dict(orient="records"))
            if not ci_remaining_df.empty:
                ci_remaining_path = output_dir / f"{meta['dataset_key']}_k{k}_{ci_method_key}_remaining_matches.csv"
                ci_remaining_df.to_csv(ci_remaining_path, index=False)
                all_remaining_rows.extend(ci_remaining_df.to_dict(orient="records"))
            if not nonci_selected_df.empty:
                nonci_selected_path = output_dir / f"{meta['dataset_key']}_k{k}_nonci_selected_matches.csv"
                nonci_selected_df.to_csv(nonci_selected_path, index=False)
                all_selected_rows.extend(nonci_selected_df.to_dict(orient="records"))
            if not nonci_remaining_df.empty:
                nonci_remaining_path = output_dir / f"{meta['dataset_key']}_k{k}_nonci_remaining_matches.csv"
                nonci_remaining_df.to_csv(nonci_remaining_path, index=False)
                all_remaining_rows.extend(nonci_remaining_df.to_dict(orient="records"))

            if not pair_table.empty:
                pair_out = pair_table.copy()
                pair_out.insert(0, "dataset_key", meta["dataset_key"])
                pair_out.insert(1, "dataset_name", meta["dataset_name"])
                pair_out.insert(2, "k", k)
                pair_out["region"] = region
                all_pair_rows.extend(pair_out.to_dict(orient="records"))

            k_rows.append(
                {
                    "dataset_key": meta["dataset_key"],
                    "dataset_name": meta["dataset_name"],
                    "k": k,
                    "region": region,
                    "k_selection_rule": meta["k_selection_rule"],
                    "boundary_has_overlap": meta["boundary_has_overlap"],
                    "boundary_overlap_amount": meta["boundary_overlap_amount"],
                    "boundary_gap": meta["boundary_gap"],
                    "k_boundary_top_player": top_player,
                    "k_boundary_outside_player": outside_player,
                    "eligible_pair_count": meta["eligible_pair_count"],
                    "n_original_matches": meta["n_original_matches"],
                    "n_fitted_rows": meta["n_fitted_rows"],
                    "n_models": meta["n_models"],
                    "ranking_csv_path": meta["ranking_csv_path"],
                    "ranking_pdf_path": meta["ranking_pdf_path"],
                    "ranking_png_path": meta["ranking_png_path"],
                    "pair_csv_path": meta["pair_csv_path"],
                }
            )
            ci_rows.extend(ci_summary_df.to_dict(orient="records"))
            nonci_rows.extend(nonci_summary_df.to_dict(orient="records"))

            merge_cols = ["dataset_key", "dataset_name", "region", "k", "variant", "action", "candidate_mode"]
            comparison_df = ci_summary_df.merge(
                nonci_summary_df,
                on=merge_cols,
                suffixes=("_ci", "_nonci"),
            )
            comparison_df["action_count_delta_nonci_minus_ci"] = comparison_df["n_actions_nonci"] - comparison_df["n_actions_ci"]
            comparison_rows.extend(comparison_df.to_dict(orient="records"))

    pd.DataFrame(k_rows).to_csv(output_dir / "all_datasets_k_selection.csv", index=False)
    ci_filename = "all_datasets_strict_ci_actions_needed.csv" if args.ci_objective == "strict" else "all_datasets_ci_aware_actions_needed.csv"
    pd.DataFrame(ci_rows).to_csv(output_dir / ci_filename, index=False)
    pd.DataFrame(nonci_rows).to_csv(output_dir / "all_datasets_nonci_actions_needed.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "all_datasets_ci_vs_nonci_action_comparison.csv", index=False)
    if all_pair_rows:
        pd.DataFrame(all_pair_rows).to_csv(output_dir / "all_datasets_eligible_ci_pairs.csv", index=False)
    if all_selected_rows:
        pd.DataFrame(all_selected_rows).to_csv(output_dir / "all_datasets_selected_matches.csv", index=False)
    if all_remaining_rows:
        pd.DataFrame(all_remaining_rows).to_csv(output_dir / "all_datasets_remaining_matches.csv", index=False)

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
