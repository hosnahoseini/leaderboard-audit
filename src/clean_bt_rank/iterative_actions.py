from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .bt_model import BradleyTerryModel
from .conditions import ObjectiveCondition
from .objectives import Objective, SkillGapObjective, compute_objective_action_influence
from .reporting import make_objective_influence_report

from clean_bt_rank.conditions import make_threshold_condition

VALID_ACTIONS = {"drop", "add", "flip"}
VALID_RECOMPUTE_MODES = {"refit", "approximate"}
VALID_CANDIDATE_MODES = {"all_pairs", "all_outcomes", "weighted"}


def _normalize_action(action: str) -> str:
    normalized = action.lower()
    if normalized not in VALID_ACTIONS:
        raise ValueError("action must be 'drop', 'add', or 'flip'.")
    return normalized


def _normalize_recompute_mode(recompute_mode: str) -> str:
    normalized = recompute_mode.lower()
    if normalized == "approx":
        normalized = "approximate"
    if normalized not in VALID_RECOMPUTE_MODES:
        raise ValueError("recompute_mode must be 'refit' or 'approximate'.")
    return normalized


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def _coerce_influence_report(
    influence_scores: pd.DataFrame | np.ndarray | list[float],
) -> pd.DataFrame:
    if isinstance(influence_scores, pd.DataFrame):
        report = influence_scores.copy()
    else:
        report = pd.DataFrame({"influence": np.asarray(influence_scores, dtype=float)})

    if "influence" not in report.columns:
        raise ValueError("influence_scores must contain an 'influence' column.")
    if "row_uid" not in report.columns:
        report = report.reset_index(drop=True)
        report["row_uid"] = np.arange(len(report))
    return report


def _uses_forward_reverse_pairs(report: pd.DataFrame) -> bool:
    return "match_id" in report.columns and "match_copy" in report.columns


def _normalize_pair_for_baseline_semantics(
    bt_model: BradleyTerryModel,
    player_a: int | str,
    player_b: int | str,
) -> tuple[int | str, int | str]:
    a_idx = int(bt_model.resolve_player(player_a))
    b_idx = int(bt_model.resolve_player(player_b))
    if a_idx == bt_model.reference_player and b_idx != bt_model.reference_player:
        return player_b, player_a
    return player_a, player_b


def _select_top_alpha_matches(
    report: pd.DataFrame,
    alpha: int,
    sort_ascending: bool = False,
    *,
    group_by_match: bool = True,
) -> pd.DataFrame:
    limit = max(0, int(alpha))
    if limit == 0 or report.empty:
        return report.head(0).copy()

    influence = report["influence"].to_numpy(dtype=float)
    order = np.argsort(influence)
    if not sort_ascending:
        order = order[::-1]

    ranked = report.iloc[order].reset_index(drop=True)
    if not group_by_match or not _uses_forward_reverse_pairs(ranked):
        return ranked.head(limit).reset_index(drop=True)

    top_match_ids = ranked["match_id"].drop_duplicates().head(limit).to_list()
    selected = ranked.loc[ranked["match_id"].isin(top_match_ids)].copy()
    return selected.reset_index(drop=True)


def _selection_limit(report: pd.DataFrame) -> int:
    if report.empty:
        return 0
    if _uses_forward_reverse_pairs(report):
        return int(report["match_id"].nunique())
    return len(report)


def _logical_selection_size(selected: pd.DataFrame) -> int:
    if selected.empty:
        return 0
    if _uses_forward_reverse_pairs(selected):
        return int(selected["match_id"].nunique())
    return int(len(selected))


def _match_frame_with_row_uid(bt_model: BradleyTerryModel) -> pd.DataFrame:
    if bt_model.match_frame_ is None:
        frame = pd.DataFrame({"match_id": np.arange(len(np.asarray(bt_model.X, dtype=float)))})
    else:
        frame = bt_model.match_frame_.copy()
    if "row_uid" not in frame.columns:
        frame = frame.reset_index(drop=True)
        frame["row_uid"] = np.arange(len(frame))
    return frame


def _refit_model(
    bt_model: BradleyTerryModel,
    X: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
) -> BradleyTerryModel:
    updated = BradleyTerryModel(
        X,
        y,
        competitor_names=bt_model.competitor_names_,
        **bt_model._config_kwargs,
    ).fit()
    updated.match_frame_ = frame.reset_index(drop=True)
    return updated


def refit_model_with_action(
    bt_model: BradleyTerryModel,
    selected_matches: pd.DataFrame,
    action: str,
) -> BradleyTerryModel:
    action = _normalize_action(action)
    X = np.asarray(bt_model.X, dtype=float).copy()
    y = np.asarray(bt_model.y, dtype=float).copy()
    frame = _match_frame_with_row_uid(bt_model)

    if action in {"drop", "flip"}:
        chosen_uids = set(int(value) for value in selected_matches["row_uid"].to_list())
        mask = frame["row_uid"].isin(chosen_uids).to_numpy()
        if action == "drop":
            X = X[~mask]
            y = y[~mask]
            frame = frame.loc[~mask].reset_index(drop=True)
        else:
            y[mask] = 1.0 - y[mask]
            if "winner" in frame.columns:
                frame.loc[mask, "winner"] = y[mask]
            if "outcome" in frame.columns:
                frame.loc[mask, "outcome"] = y[mask]
    else:
        if "_candidate_x" not in selected_matches.columns:
            raise ValueError("For action='add', selected_matches must contain '_candidate_x'.")
        add_x = np.vstack(selected_matches["_candidate_x"].to_list()).astype(float)
        add_y = (
            selected_matches["outcome"].to_numpy(dtype=float)
            if "outcome" in selected_matches.columns
            else np.ones(len(selected_matches), dtype=float)
        )
        X = np.vstack([X, add_x])
        y = np.concatenate([y, add_y])
        frame = pd.concat([frame, selected_matches.copy()], ignore_index=True)

    if len(X) < bt_model.n_params_ + 1:
        raise ValueError("Not enough rows remain to refit the Bradley-Terry model.")
    return _refit_model(bt_model, X, y, frame)


def apply_action_on_top_alpha_influential_matches(
    bt_model: BradleyTerryModel,
    objective: Objective,
    influence_scores: pd.DataFrame | np.ndarray | list[float],
    alpha: int,
    action: str,
    *,
    recompute_mode: str = "refit",
    sort_ascending: bool = False,
    group_by_match: bool = True,
) -> dict[str, object]:
    action = _normalize_action(action)
    recompute_mode = _normalize_recompute_mode(recompute_mode)

    initial_value = float(objective.value(bt_model))
    report = _coerce_influence_report(influence_scores)
    selected = _select_top_alpha_matches(
        report,
        alpha,
        sort_ascending=sort_ascending,
        group_by_match=group_by_match,
    )
    if selected.empty:
        return {
            "initial_value": initial_value,
            "final_value": initial_value,
            "n_applied": 0,
            "selected_matches": selected,
            "objective_met": None,
        }

    if recompute_mode == "approximate":
        return {
            "initial_value": initial_value,
            "final_value": initial_value + float(selected["influence"].sum()),
            "n_applied": _logical_selection_size(selected),
            "selected_matches": selected,
            "objective_met": None,
        }

    try:
        updated = refit_model_with_action(bt_model, selected, action)
        final_value = float(objective.value(updated))
    except ValueError:
        final_value = float("nan")

    return {
        "initial_value": initial_value,
        "final_value": final_value,
        "n_applied": _logical_selection_size(selected),
        "selected_matches": selected,
        "objective_met": None,
    }


def find_minimum_alpha_to_meet_objective(
    bt_model: BradleyTerryModel,
    objective: Objective,
    influence_scores: pd.DataFrame | np.ndarray | list[float],
    action: str,
    target_condition: ObjectiveCondition,
    *,
    start_alpha: int = 1,
    max_alpha: int | None = None,
    recompute_mode: str = "refit",
) -> dict[str, object]:
    report = _coerce_influence_report(influence_scores).sort_values("influence", ascending=False).reset_index(drop=True)
    if report.empty:
        return {"met": False, "alpha": None, "result": None}

    alpha = max(1, int(start_alpha))
    max_selectable = _selection_limit(report)
    limit = max_selectable if max_alpha is None else min(max_selectable, int(max_alpha))
    while alpha <= limit:
        result = apply_action_on_top_alpha_influential_matches(
            bt_model,
            objective,
            report,
            alpha,
            action,
            recompute_mode=recompute_mode,
        )
        if target_condition.is_met(float(result["final_value"])):
            return {"met": True, "alpha": alpha, "result": result}
        alpha += 1

    return {"met": False, "alpha": None, "result": None}


def gap_based_objective_search_across_player_pairs(
    bt_model: BradleyTerryModel,
    player_pairs: list[tuple[int | str, int | str]],
    action: str,
    *,
    start_alpha: int = 1,
    recompute_mode: str = "refit",
    influence_method: str = "1sn",
    candidate_mode: str | None = "all_pairs",
    max_alpha: int | None = None,
) -> dict[str, object]:
    action = _normalize_action(action)
    cached_reports: dict[tuple[int | str, int | str], pd.DataFrame] = {}
    alpha = max(1, int(start_alpha))
    max_allowed = max_alpha
    
    while alpha <= max_allowed:
        print(alpha)
        for raw_player_a, raw_player_b in player_pairs:
            player_a, player_b = _normalize_pair_for_baseline_semantics(bt_model, raw_player_a, raw_player_b)
            print(player_a, player_b)
            objective = SkillGapObjective(player_a=player_a, player_b=player_b)
            report = cached_reports.setdefault(
                (player_a, player_b),
                compute_all_action_influences(
                    bt_model,
                    objective,
                    action,
                    influence_method=influence_method,
                    candidate_mode=candidate_mode,
                ),
            )

            max_selectable = _selection_limit(report)
            if alpha > max_selectable:
                continue
            gap = float(bt_model.reported_gap(player_a, player_b))
            target_condition = make_threshold_condition(0.0, ">=") if gap < 0 else make_threshold_condition(0.0, "<=")
            sort_ascending = gap > 0

            result = apply_action_on_top_alpha_influential_matches(
                bt_model,
                objective,
                report,
                alpha,
                action,
                recompute_mode=recompute_mode,
                sort_ascending=sort_ascending,
                group_by_match=True,
            )
            print("result['final_value']", result["final_value"])
            if target_condition.is_met(float(result["final_value"])):
                return {
                    "met": True,
                    "alpha": alpha,
                    "player_pair": (player_a, player_b),
                    "result": result,
                    "target_condition": target_condition,
                }

        alpha += 1

    return {
        "met": False,
        "alpha": None,
        "player_pair": None,
        "result": None,
    }


def _design_row(bt_model: BradleyTerryModel, player_a: int, player_b: int) -> np.ndarray:
    row = np.zeros(bt_model.n_params_, dtype=float)
    if player_a != bt_model.reference_player:
        row[player_a - (1 if player_a > bt_model.reference_player else 0)] = 1.0
    if player_b != bt_model.reference_player:
        row[player_b - (1 if player_b > bt_model.reference_player else 0)] = -1.0
    return row


def _build_add_candidate_report(
    bt_model: BradleyTerryModel,
    candidate_mode: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    skills = bt_model.full_skills()
    rows: list[dict[str, object]] = []
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    match_id_start = (
        int(bt_model.match_frame_["match_id"].max()) + 1
        if bt_model.match_frame_ is not None and "match_id" in bt_model.match_frame_.columns and len(bt_model.match_frame_)
        else int(len(np.asarray(bt_model.X, dtype=float)))
    )
    next_match_id = match_id_start

    def append_candidate_pair(winner: int, loser: int, probability: float) -> None:
        nonlocal next_match_id
        match_id = next_match_id
        next_match_id += 1

        rows.append(
            {
                "model_a": bt_model.competitor_names_[winner],
                "model_b": bt_model.competitor_names_[loser],
                "outcome": 1.0,
                "candidate_probability": probability,
                "match_id": match_id,
                "match_copy": "forward",
            }
        )
        x_rows.append(_design_row(bt_model, winner, loser))
        y_rows.append(1.0)

        rows.append(
            {
                "model_a": bt_model.competitor_names_[loser],
                "model_b": bt_model.competitor_names_[winner],
                "outcome": 0.0,
                "candidate_probability": probability,
                "match_id": match_id,
                "match_copy": "reverse",
            }
        )
        x_rows.append(_design_row(bt_model, loser, winner))
        y_rows.append(0.0)

    if candidate_mode == "all_pairs":
        for player_a in range(bt_model.n_players_):
            for player_b in range(player_a + 1, bt_model.n_players_):
                winner, loser = (player_a, player_b) if skills[player_a] >= skills[player_b] else (player_b, player_a)
                append_candidate_pair(winner, loser, 1.0)
    else:
        for player_a in range(bt_model.n_players_):
            for player_b in range(player_a + 1, bt_model.n_players_):
                probability_ab = float(_sigmoid(skills[player_a] - skills[player_b]))
                probability_ba = float(_sigmoid(skills[player_b] - skills[player_a]))
                append_candidate_pair(player_a, player_b, probability_ab)
                append_candidate_pair(player_b, player_a, probability_ba)

    X_new = np.vstack(x_rows).astype(float)
    y_new = np.asarray(y_rows, dtype=float)
    return pd.DataFrame(rows), X_new, y_new


def compute_all_action_influences(
    bt_model: BradleyTerryModel,
    objective: Objective,
    action: str,
    *,
    influence_method: str = "1sn",
    candidate_mode: str | None = None,
) -> pd.DataFrame:
    action = _normalize_action(action)

    if action == "drop":
        influence = compute_objective_action_influence(bt_model, objective, action="drop", method=influence_method)
        report = make_objective_influence_report(bt_model, objective, influence, influence_name="influence")
        report["row_uid"] = np.arange(len(report))
        return report

    if action == "flip":
        drop_influence = compute_objective_action_influence(bt_model, objective, action="drop", method=influence_method)
        add_flipped = compute_objective_action_influence(
            bt_model,
            objective,
            action="add",
            method=influence_method,
            X_new=np.asarray(bt_model.X, dtype=float),
            y_new=1.0 - np.asarray(bt_model.y, dtype=float),
        )
        report = make_objective_influence_report(
            bt_model,
            objective,
            drop_influence + add_flipped,
            influence_name="influence",
        )
        report["row_uid"] = np.arange(len(report))
        return report

    normalized_mode = (candidate_mode or "all_pairs").lower()
    if normalized_mode not in VALID_CANDIDATE_MODES:
        raise ValueError("candidate_mode must be 'all_pairs', 'all_outcomes', or 'weighted'.")

    report, X_new, y_new = _build_add_candidate_report(bt_model, normalized_mode)
    influence = compute_objective_action_influence(
        bt_model,
        objective,
        action="add",
        method=influence_method,
        X_new=X_new,
        y_new=y_new,
    )
    report["_candidate_x"] = list(X_new)
    report["raw_influence"] = influence
    report["influence"] = influence
    if normalized_mode == "weighted":
        report["influence"] = report["influence"] * report["candidate_probability"]
    report["influence_abs"] = report["influence"].abs()
    report["row_uid"] = np.arange(len(report))
    return report


def run_iterative_action_until_target(
    bt_model: BradleyTerryModel,
    objective: Objective,
    action: str,
    target_condition: ObjectiveCondition,
    *,
    influence_scores: pd.DataFrame | np.ndarray | list[float] | None = None,
    influence_method: str = "1sn",
    candidate_mode: str | None = None,
    start_alpha: int = 1,
    max_alpha: int | None = None,
    recompute_mode: str = "refit",
) -> dict[str, object]:
    report = (
        compute_all_action_influences(
            bt_model,
            objective,
            action,
            influence_method=influence_method,
            candidate_mode=candidate_mode,
        )
        if influence_scores is None
        else influence_scores
    )
    return find_minimum_alpha_to_meet_objective(
        bt_model,
        objective,
        report,
        action,
        target_condition,
        start_alpha=start_alpha,
        max_alpha=max_alpha,
        recompute_mode=recompute_mode,
    )


def run_iterative_actions_for_objectives(
    bt_model: BradleyTerryModel,
    objectives: list[Objective],
    action: str,
    target_condition: ObjectiveCondition,
    *,
    influence_method: str = "1sn",
    candidate_mode: str | None = None,
    start_alpha: int = 1,
    max_alpha: int | None = None,
    recompute_mode: str = "refit",
) -> list[dict[str, object]]:
    results = []
    for objective in objectives:
        results.append(
            run_iterative_action_until_target(
                bt_model,
                objective,
                action,
                target_condition,
                influence_method=influence_method,
                candidate_mode=candidate_mode,
                start_alpha=start_alpha,
                max_alpha=max_alpha,
                recompute_mode=recompute_mode,
            )
        )
    return results


def run_objective_curve_steps(
    bt_model: BradleyTerryModel,
    objective: Objective,
    action: str,
    influence_scores: pd.DataFrame | np.ndarray | list[float],
    *,
    steps: int,
    recompute_mode: str = "refit",
) -> pd.DataFrame:
    report = _coerce_influence_report(influence_scores).sort_values("influence", ascending=False).reset_index(drop=True)
    rows = [{"step": 0, "objective_value": float(objective.value(bt_model))}]
    for alpha in range(1, min(int(steps), len(report)) + 1):
        result = apply_action_on_top_alpha_influential_matches(
            bt_model,
            objective,
            report,
            alpha,
            action,
            recompute_mode=recompute_mode,
        )
        rows.append({"step": alpha, "objective_value": float(result["final_value"])})
    return pd.DataFrame(rows)


closest_top_k_matchups = apply_action_on_top_alpha_influential_matches
