from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .bt_model import BradleyTerryModel
from .objectives import Objective, compute_objective_action_influence
from .reporting import make_objective_influence_report


def build_action_candidate_report(
    bt_model: BradleyTerryModel,
    objective: Objective,
    action: str,
    *,
    influence_method: str = "1sn",
    candidate_mode: str | None = None,
) -> pd.DataFrame:
    action = action.lower()
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
            add_flipped - drop_influence,
            influence_name="influence",
        )
        report["row_uid"] = np.arange(len(report))
        return report

    if action != "add":
        raise ValueError("action must be 'drop', 'flip', or 'add'.")

    candidate_mode = (candidate_mode or "all_pairs").lower()
    if candidate_mode not in {"all_pairs", "all_outcomes", "weighted"}:
        raise ValueError("candidate_mode must be 'all_pairs', 'all_outcomes', or 'weighted'.")

    skills = bt_model.full_skills()
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    rows: list[dict[str, object]] = []

    def design_row(player_a: int, player_b: int) -> np.ndarray:
        row = np.zeros(bt_model.n_params_, dtype=float)
        if player_a != bt_model.reference_player:
            row[player_a - (1 if player_a > bt_model.reference_player else 0)] = 1.0
        if player_b != bt_model.reference_player:
            row[player_b - (1 if player_b > bt_model.reference_player else 0)] = -1.0
        return row

    if candidate_mode == "all_pairs":
        for player_a in range(bt_model.n_players_):
            for player_b in range(player_a + 1, bt_model.n_players_):
                if skills[player_a] >= skills[player_b]:
                    winner, loser = player_a, player_b
                else:
                    winner, loser = player_b, player_a
                rows.append(
                    {
                        "model_a": bt_model.competitor_names_[winner],
                        "model_b": bt_model.competitor_names_[loser],
                        "outcome": 1.0,
                        "candidate_probability": 1.0,
                    }
                )
                x_rows.append(design_row(winner, loser))
                y_rows.append(1.0)
    else:
        for player_a in range(bt_model.n_players_):
            for player_b in range(bt_model.n_players_):
                if player_a == player_b:
                    continue
                probability = 1.0 / (1.0 + np.exp(-(skills[player_a] - skills[player_b])))
                rows.append(
                    {
                        "model_a": bt_model.competitor_names_[player_a],
                        "model_b": bt_model.competitor_names_[player_b],
                        "outcome": 1.0,
                        "candidate_probability": probability,
                    }
                )
                x_rows.append(design_row(player_a, player_b))
                y_rows.append(1.0)

    x_new = np.vstack(x_rows).astype(float)
    y_new = np.asarray(y_rows, dtype=float)
    influence = compute_objective_action_influence(
        bt_model,
        objective,
        action="add",
        method=influence_method,
        X_new=x_new,
        y_new=y_new,
    )
    report = pd.DataFrame(rows)
    report["_candidate_x"] = list(x_new)
    report["raw_influence"] = influence
    report["influence"] = influence
    if candidate_mode == "weighted":
        report["influence"] = report["influence"] * report["candidate_probability"]
    report["influence_abs"] = report["influence"].abs()
    report["row_uid"] = np.arange(len(report))
    return report


def refit_after_action_subset(
    bt_model: BradleyTerryModel,
    selected_matches: pd.DataFrame,
    action: str,
) -> BradleyTerryModel:
    x = np.asarray(bt_model.X, dtype=float).copy()
    y = np.asarray(bt_model.y, dtype=float).copy()
    frame = bt_model.match_frame_.copy() if bt_model.match_frame_ is not None else pd.DataFrame({"match_id": np.arange(len(x))})
    if "row_uid" not in frame.columns:
        frame = frame.reset_index(drop=True)
        frame["row_uid"] = np.arange(len(frame))

    action = action.lower()
    if action in {"drop", "flip"}:
        chosen_uids = set(int(v) for v in selected_matches["row_uid"].to_list())
        mask = frame["row_uid"].isin(chosen_uids).to_numpy()
        if action == "drop":
            x = x[~mask]
            y = y[~mask]
            frame = frame.loc[~mask].reset_index(drop=True)
        else:
            y[mask] = 1.0 - y[mask]
            if "winner" in frame.columns:
                frame.loc[mask, "winner"] = y[mask]
            if "outcome" in frame.columns:
                frame.loc[mask, "outcome"] = y[mask]
    elif action == "add":
        if "_candidate_x" not in selected_matches.columns:
            raise ValueError("selected_matches must contain '_candidate_x' for action='add'.")
        add_x = np.vstack(selected_matches["_candidate_x"].to_list()).astype(float)
        add_y = (
            selected_matches["outcome"].to_numpy(dtype=float)
            if "outcome" in selected_matches.columns
            else np.ones(len(selected_matches), dtype=float)
        )
        x = np.vstack([x, add_x])
        y = np.concatenate([y, add_y])
        frame = pd.concat([frame, selected_matches.copy()], ignore_index=True)
    else:
        raise ValueError("action must be 'drop', 'flip', or 'add'.")

    updated = BradleyTerryModel(
        x,
        y,
        competitor_names=bt_model.competitor_names_,
        reference_player=bt_model.reference_player,
        scale=bt_model.scale,
        base=bt_model.base,
        init_rating=bt_model.init_rating,
        anchor_player=bt_model.anchor_player,
        anchor_rating=bt_model.anchor_rating,
        hessian_ridge=bt_model.hessian_ridge,
    ).fit()
    updated.match_frame_ = frame.reset_index(drop=True)
    return updated


def evaluate_action_subset_delta(
    bt_model: BradleyTerryModel,
    objective: Objective,
    selected_matches: pd.DataFrame,
    action: str,
) -> float:
    if selected_matches.empty:
        return 0.0
    initial_value = float(objective.value(bt_model))
    updated = refit_after_action_subset(bt_model, selected_matches, action)
    return float(objective.value(updated) - initial_value)


def truncate_candidate_report(
    report: pd.DataFrame,
    *,
    max_items: int | None = None,
    sort_column: str = "influence_abs",
) -> pd.DataFrame:
    work = report.copy()
    if max_items is None or max_items >= len(work):
        return work.reset_index(drop=True)
    if sort_column not in work.columns:
        raise ValueError(f"sort_column {sort_column!r} is not present in the report.")
    return work.sort_values(sort_column, ascending=False).head(int(max_items)).reset_index(drop=True)


def estimate_match_shapley_values(
    bt_model: BradleyTerryModel,
    objective: Objective,
    action: str,
    *,
    report: pd.DataFrame | None = None,
    influence_method: str = "1sn",
    candidate_mode: str | None = None,
    num_permutations: int = 32,
    random_seed: int = 0,
    max_items: int | None = None,
    sort_column: str = "influence_abs",
    shapley_column: str = "shapley_value",
) -> pd.DataFrame:
    if num_permutations <= 0:
        raise ValueError("num_permutations must be positive.")

    base_report = (
        build_action_candidate_report(
            bt_model,
            objective,
            action,
            influence_method=influence_method,
            candidate_mode=candidate_mode,
        )
        if report is None
        else report.copy()
    )
    pool = truncate_candidate_report(base_report, max_items=max_items, sort_column=sort_column)
    pool["_shapley_local_id"] = np.arange(len(pool))

    rng = np.random.default_rng(random_seed)
    shapley = np.zeros(len(pool), dtype=float)
    cache: dict[frozenset[int], float] = {frozenset(): 0.0}

    local_ids = pool["_shapley_local_id"].to_numpy(dtype=int)
    for _ in range(int(num_permutations)):
        perm = rng.permutation(local_ids)
        coalition: list[int] = []
        prev_val = 0.0
        for local_id in perm:
            coalition.append(int(local_id))
            key = frozenset(coalition)
            if key not in cache:
                selected = pool.loc[sorted(key)].reset_index(drop=True)
                cache[key] = evaluate_action_subset_delta(bt_model, objective, selected, action)
            next_val = float(cache[key])
            shapley[int(local_id)] += next_val - prev_val
            prev_val = next_val

    pool[shapley_column] = shapley / float(num_permutations)
    pool[f"{shapley_column}_abs"] = pool[shapley_column].abs()
    return pool.drop(columns="_shapley_local_id")


def make_strategy_report(
    report: pd.DataFrame,
    *,
    score_column: str,
    minimize: bool,
    random_seed: int | None = None,
) -> pd.DataFrame:
    work = report.copy()
    if score_column not in work.columns:
        raise ValueError(f"score_column {score_column!r} is not present in the report.")
    if random_seed is None:
        score = work[score_column].to_numpy(dtype=float)
        score = -score if minimize else score
    else:
        rng = np.random.default_rng(random_seed)
        score = rng.random(len(work))
    work["selection_score"] = score
    work["influence"] = work["selection_score"]
    return work


def make_strategy_reports(
    report: pd.DataFrame,
    *,
    minimize: bool,
    shapley_column: str = "shapley_value",
    random_seed: int = 0,
) -> dict[str, pd.DataFrame]:
    return {
        "influence": make_strategy_report(report, score_column="influence", minimize=minimize),
        "shapley": make_strategy_report(report, score_column=shapley_column, minimize=minimize),
        "random": make_strategy_report(report, score_column="influence", minimize=minimize, random_seed=random_seed),
    }


def objective_curves_by_strategy(
    bt_model: BradleyTerryModel,
    objective: Objective,
    action: str,
    strategy_reports: dict[str, pd.DataFrame],
    *,
    steps: int,
    apply_fn,
) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    for strategy_name, selection_report in strategy_reports.items():
        rows = [{"step": 0, "objective_value": float(objective.value(bt_model))}]
        max_steps = min(int(steps), len(selection_report))
        for alpha in range(1, max_steps + 1):
            result = apply_fn(
                bt_model,
                objective,
                selection_report,
                alpha,
                action,
                recompute_mode="refit",
            )
            rows.append({"step": alpha, "objective_value": float(result["final_value"])})
        curves[strategy_name] = pd.DataFrame(rows)
    return curves
