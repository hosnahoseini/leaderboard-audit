from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..actions import build_add_candidates, compute_influence
from ..bt_model import BradleyTerryModel
from ..objectives import SkillGapObjective
from ._common import (
    ExperimentState,
    apply_add_candidate,
    apply_drop_row,
    apply_flip_row,
    fit_model_from_state,
    make_state,
    ranking_frame,
)


@dataclass
class TopKManipulationResult:
    policy: str
    action: str
    target_player: str
    direction: str
    k: int
    success: bool
    budget: int
    history: pd.DataFrame
    selected_actions: pd.DataFrame
    ranking_before: pd.DataFrame
    ranking_after: pd.DataFrame

    def summary_dict(self) -> dict[str, object]:
        final = self.history.sort_values("step").iloc[-1]
        return {
            "policy": self.policy,
            "action": self.action,
            "target_player": self.target_player,
            "direction": self.direction,
            "k": self.k,
            "success": self.success,
            "steps_used": int(final["step"]),
            "budget": self.budget,
            "initial_rank": int(self.history.iloc[0]["target_rank"]),
            "final_rank": int(final["target_rank"]),
        }


def run_topk_manipulation_experiment(
    bt_model: BradleyTerryModel,
    *,
    target_player: str,
    k: int,
    direction: str,
    action: str,
    policy: str = "influence",
    influence_method: str = "1sn",
    add_candidate_mode: str = "all_outcomes",
    budget: int = 10,
    random_seed: int = 0,
) -> TopKManipulationResult:
    bt_model._require_fit()
    direction = direction.lower()
    action = action.lower()
    policy = policy.lower()
    if direction not in {"promote", "demote"}:
        raise ValueError("direction must be 'promote' or 'demote'.")
    if action not in {"add", "drop", "flip"}:
        raise ValueError("action must be 'add', 'drop', or 'flip'.")
    if policy not in {"influence", "random"}:
        raise ValueError("policy must be 'influence' or 'random'.")
    add_candidate_mode = add_candidate_mode.lower()
    if add_candidate_mode not in {"all_outcomes", "all_outcomes_weighted", "all_pairs"}:
        raise ValueError("add_candidate_mode must be 'all_outcomes', 'all_outcomes_weighted', or 'all_pairs'.")

    current_model = bt_model
    state = make_state(bt_model)
    rng = np.random.default_rng(random_seed)
    history_rows = [_topk_history_row(current_model, target_player, k, direction, step=0)]
    selected_rows: list[dict[str, object]] = []
    used_add_keys: set[tuple[str, str, float]] = set()
    used_row_uids: set[int] = set()
    ranking_before = ranking_frame(bt_model)

    for step in range(1, budget + 1):
        if _topk_success(current_model, target_player, k, direction):
            break

        objective = _target_boundary_objective(current_model, target_player, k, direction)
        candidate_mode = add_candidate_mode if action == "add" else "all_outcomes"
        report = compute_influence(
            current_model,
            objective,
            action=action,
            method=influence_method,
            candidate_mode=candidate_mode,
        )
        eligible = _filter_topk_candidates(
            report,
            action=action,
            used_add_keys=used_add_keys,
            used_row_uids=used_row_uids,
        )
        if eligible.empty:
            break

        if policy == "influence":
            eligible = eligible.copy()
            eligible["selection_score"] = _selection_score(eligible["influence"].to_numpy(dtype=float), direction)
            chosen = eligible.sort_values("selection_score", ascending=False).iloc[0].copy()
        else:
            chosen = eligible.iloc[int(rng.integers(0, len(eligible)))].copy()
            chosen["selection_score"] = np.nan

        selected_rows.append(_selected_action_row(chosen, step=step, policy=policy))
        state = _apply_topk_action(state, chosen, action=action)

        if action == "add":
            used_add_keys.add((str(chosen["model_a"]), str(chosen["model_b"]), float(chosen.get("outcome", 1.0))))
        else:
            used_row_uids.add(int(chosen["row_uid"]))

        current_model = fit_model_from_state(bt_model, state)
        history_rows.append(_topk_history_row(current_model, target_player, k, direction, step=step))

    return TopKManipulationResult(
        policy=policy,
        action=action,
        target_player=target_player,
        direction=direction,
        k=k,
        success=_topk_success(current_model, target_player, k, direction),
        budget=budget,
        history=pd.DataFrame(history_rows),
        selected_actions=pd.DataFrame(selected_rows),
        ranking_before=ranking_before,
        ranking_after=ranking_frame(current_model),
    )


def run_topk_manipulation_benchmark(
    bt_model: BradleyTerryModel,
    *,
    target_player: str,
    k: int,
    direction: str,
    budget: int = 10,
    influence_method: str = "1sn",
    add_candidate_mode: str = "all_outcomes",
    n_random_trials: int = 5,
    random_seed: int = 0,
) -> tuple[dict[tuple[str, str], TopKManipulationResult | list[TopKManipulationResult]], pd.DataFrame, pd.DataFrame]:
    results: dict[tuple[str, str], TopKManipulationResult | list[TopKManipulationResult]] = {}
    summary_rows: list[dict[str, object]] = []
    history_frames: list[pd.DataFrame] = []

    for action in ["add", "drop", "flip"]:
        influence_result = run_topk_manipulation_experiment(
            bt_model,
            target_player=target_player,
            k=k,
            direction=direction,
            action=action,
            policy="influence",
            influence_method=influence_method,
            add_candidate_mode=add_candidate_mode,
            budget=budget,
            random_seed=random_seed,
        )
        results[(action, "influence")] = influence_result
        summary_rows.append(influence_result.summary_dict())
        history_frames.append(_attach_history(influence_result.history, action=action, policy="influence", trial=0))

        random_results = [
            run_topk_manipulation_experiment(
                bt_model,
                target_player=target_player,
                k=k,
                direction=direction,
                action=action,
                policy="random",
                influence_method=influence_method,
                add_candidate_mode=add_candidate_mode,
                budget=budget,
                random_seed=random_seed + trial,
            )
            for trial in range(n_random_trials)
        ]
        results[(action, "random")] = random_results
        for trial, result in enumerate(random_results):
            summary_rows.append(result.summary_dict() | {"trial": trial})
            history_frames.append(_attach_history(result.history, action=action, policy="random", trial=trial))

    return results, pd.DataFrame(summary_rows), pd.concat(history_frames, ignore_index=True)


def _selection_score(influence: np.ndarray, direction: str) -> np.ndarray:
    return influence if direction == "promote" else -influence


def _topk_success(model: BradleyTerryModel, target_player: str, k: int, direction: str) -> bool:
    ranking = ranking_frame(model)
    rank = int(ranking.loc[ranking["competitor"] == target_player, "rank"].iloc[0])
    return rank <= k if direction == "promote" else rank > k


def _target_boundary_objective(model: BradleyTerryModel, target_player: str, k: int, direction: str) -> SkillGapObjective:
    ranking = ranking_frame(model)
    target_rank = int(ranking.loc[ranking["competitor"] == target_player, "rank"].iloc[0])
    if direction == "promote":
        if target_rank <= k:
            boundary_player = str(ranking.loc[ranking["rank"] == k, "competitor"].iloc[0])
        else:
            boundary_player = str(ranking.loc[ranking["rank"] == k, "competitor"].iloc[0])
        return SkillGapObjective(target_player, boundary_player)

    boundary_rank = k + 1
    if target_rank > k:
        boundary_player = str(ranking.loc[ranking["rank"] == boundary_rank, "competitor"].iloc[0])
    else:
        boundary_player = str(ranking.loc[ranking["rank"] == boundary_rank, "competitor"].iloc[0])
    return SkillGapObjective(target_player, boundary_player)


def _filter_topk_candidates(
    report: pd.DataFrame,
    *,
    action: str,
    used_add_keys: set[tuple[str, str, float]],
    used_row_uids: set[int],
) -> pd.DataFrame:
    report = report.copy()

    if action == "add":
        keys = list(zip(report["model_a"].astype(str), report["model_b"].astype(str), report["outcome"].astype(float)))
        keep = np.array([key not in used_add_keys for key in keys], dtype=bool)
        return report.loc[keep].reset_index(drop=True)

    if "row_uid" not in report.columns:
        report["row_uid"] = np.arange(len(report))
    report = report.loc[~report["row_uid"].astype(int).isin(used_row_uids)].copy()
    return report.reset_index(drop=True)


def _apply_topk_action(state: ExperimentState, chosen: pd.Series, *, action: str) -> ExperimentState:
    if action == "add":
        return apply_add_candidate(state, chosen)
    if action == "drop":
        return apply_drop_row(state, int(chosen["row_uid"]))
    if action == "flip":
        return apply_flip_row(state, int(chosen["row_uid"]))
    raise ValueError(f"Unsupported action {action!r}.")


def _topk_history_row(
    model: BradleyTerryModel,
    target_player: str,
    k: int,
    direction: str,
    *,
    step: int,
) -> dict[str, object]:
    ranking = ranking_frame(model)
    target_rank = int(ranking.loc[ranking["competitor"] == target_player, "rank"].iloc[0])
    target_rating = float(ranking.loc[ranking["competitor"] == target_player, "rating"].iloc[0])
    boundary_k_player = str(ranking.loc[ranking["rank"] == k, "competitor"].iloc[0])
    boundary_k_rating = float(ranking.loc[ranking["rank"] == k, "rating"].iloc[0])
    boundary_kplus1_player = str(ranking.loc[ranking["rank"] == k + 1, "competitor"].iloc[0])
    boundary_kplus1_rating = float(ranking.loc[ranking["rank"] == k + 1, "rating"].iloc[0])
    if direction == "promote":
        objective_value = float(model.reported_gap(target_player, boundary_k_player))
    else:
        objective_value = float(model.reported_gap(target_player, boundary_kplus1_player))

    return {
        "step": step,
        "target_player": target_player,
        "target_rank": target_rank,
        "target_rating": target_rating,
        "objective_value": objective_value,
        "boundary_k_player": boundary_k_player,
        "boundary_k_rating": boundary_k_rating,
        "boundary_kplus1_player": boundary_kplus1_player,
        "boundary_kplus1_rating": boundary_kplus1_rating,
        "success": _topk_success(model, target_player, k, direction),
    }


def _selected_action_row(chosen: pd.Series, *, step: int, policy: str) -> dict[str, object]:
    row = chosen.to_dict()
    row["step"] = step
    row["policy"] = policy
    return row


def _attach_history(frame: pd.DataFrame, *, action: str, policy: str, trial: int) -> pd.DataFrame:
    out = frame.copy()
    out["action"] = action
    out["policy"] = policy
    out["trial"] = trial
    return out
