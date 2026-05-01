from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bt_model import BradleyTerryModel
from .objectives import Objective, compute_objective_action_influence
from .reporting import make_objective_influence_report


@dataclass
class AddCandidates:
    frame: pd.DataFrame  # columns: model_a, model_b, outcome, candidate_probability, player_a_index, player_b_index
    X: np.ndarray        # shape (n_candidates, n_params)


def build_add_candidates(bt_model: BradleyTerryModel, *, mode: str = "all_pairs") -> AddCandidates:
    """
    Build the full set of add-action candidates for a fitted BT model.

    Modes:
    - ``all_pairs``: one row per unordered pair (expected winner has outcome=1.0)
    - ``all_outcomes``: one row per ordered pair (both directions, outcome=1.0)
    - ``all_outcomes_weighted``: same as all_outcomes; caller weights influence by probability
    """
    mode = mode.lower()
    if mode not in {"all_pairs", "all_outcomes", "all_outcomes_weighted"}:
        raise ValueError("mode must be 'all_pairs', 'all_outcomes', or 'all_outcomes_weighted'.")

    bt_model._require_fit()
    skills = bt_model.full_skills()
    x_rows: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    def design_row(player_a: int, player_b: int) -> np.ndarray:
        row = np.zeros(bt_model.n_params_, dtype=float)
        if player_a != bt_model.reference_player:
            row[player_a - (1 if player_a > bt_model.reference_player else 0)] = 1.0
        if player_b != bt_model.reference_player:
            row[player_b - (1 if player_b > bt_model.reference_player else 0)] = -1.0
        return row

    if mode == "all_pairs":
        for pa in range(bt_model.n_players_):
            for pb in range(pa + 1, bt_model.n_players_):
                winner, loser = (pa, pb) if skills[pa] >= skills[pb] else (pb, pa)
                prob = 1.0 / (1.0 + np.exp(-(skills[winner] - skills[loser])))
                rows.append({
                    "model_a": bt_model.competitor_names_[winner],
                    "model_b": bt_model.competitor_names_[loser],
                    "outcome": 1.0,
                    "candidate_probability": prob,
                    "player_a_index": winner,
                    "player_b_index": loser,
                })
                x_rows.append(design_row(winner, loser))
    else:
        for pa in range(bt_model.n_players_):
            for pb in range(bt_model.n_players_):
                if pa == pb:
                    continue
                prob = 1.0 / (1.0 + np.exp(-(skills[pa] - skills[pb])))
                rows.append({
                    "model_a": bt_model.competitor_names_[pa],
                    "model_b": bt_model.competitor_names_[pb],
                    "outcome": 1.0,
                    "candidate_probability": prob,
                    "player_a_index": pa,
                    "player_b_index": pb,
                })
                x_rows.append(design_row(pa, pb))

    return AddCandidates(
        frame=pd.DataFrame(rows),
        X=np.vstack(x_rows).astype(float),
    )


def compute_influence(
    bt_model: BradleyTerryModel,
    objective: Objective,
    action: str,
    *,
    method: str = "1sn",
    candidate_mode: str = "all_pairs",
) -> pd.DataFrame:
    """
    Compute per-candidate influence scores for a given action type.

    Returns a DataFrame:
    - add: model_a, model_b, outcome, candidate_probability, _candidate_x, influence
    - drop/flip: match frame columns + influence, row_uid
    """
    action = action.lower()

    if action == "drop":
        influence = compute_objective_action_influence(bt_model, objective, action="drop", method=method)
        report = make_objective_influence_report(bt_model, objective, influence, influence_name="influence")
        if "row_uid" not in report.columns:
            report["row_uid"] = np.arange(len(report))
        return report

    if action == "flip":
        drop_inf = compute_objective_action_influence(bt_model, objective, action="drop", method=method)
        add_flipped = compute_objective_action_influence(
            bt_model,
            objective,
            action="add",
            method=method,
            X_new=np.asarray(bt_model.X, dtype=float),
            y_new=1.0 - np.asarray(bt_model.y, dtype=float),
        )
        report = make_objective_influence_report(
            bt_model, objective, add_flipped + drop_inf, influence_name="influence"
        )
        if "row_uid" not in report.columns:
            report["row_uid"] = np.arange(len(report))
        return report

    if action != "add":
        raise ValueError("action must be 'add', 'drop', or 'flip'.")

    build_mode = "all_outcomes" if candidate_mode == "all_outcomes_weighted" else candidate_mode
    candidates = build_add_candidates(bt_model, mode=build_mode)
    influence_vals = compute_objective_action_influence(
        bt_model,
        objective,
        action="add",
        method=method,
        X_new=candidates.X,
        y_new=np.ones(len(candidates.frame), dtype=float),
    )
    report = candidates.frame.copy()
    report["_candidate_x"] = list(candidates.X)
    report["influence"] = influence_vals
    if candidate_mode == "all_outcomes_weighted":
        report["influence"] = report["influence"] * report["candidate_probability"]
    return report
