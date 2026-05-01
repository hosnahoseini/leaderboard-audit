#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from clean_bt_rank import SkillGapObjective, available_hf_battle_datasets, use_paper_rc
from clean_bt_rank.ci_aware_actions_needed import build_named_dataset_model
from clean_bt_rank.experiments._common import apply_add_candidate, fit_model_from_state, make_state, ranking_frame
from clean_bt_rank.objectives import compute_objective_action_influence

POLICY_COLORS = {
    "influence_pair": "#264653",
    "rigging": "#8D99AE",
    "promote": "#2A9D8F",
    "demote": "#E76F51",
}


@dataclass
class TrialResult:
    dataset_key: str
    dataset_name: str
    target_player: str
    direction: str
    k: int
    method: str
    policy: str
    seed: int
    success: bool
    actions_used: int
    rows_added: int
    exposures_seen: int
    initial_rank: int
    final_rank: int
    final_metric_value: float

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_key": self.dataset_key,
            "dataset_name": self.dataset_name,
            "target_player": self.target_player,
            "direction": self.direction,
            "k": self.k,
            "method": self.method,
            "policy": self.policy,
            "trial": self.seed,
            "success": self.success,
            "actions_used": self.actions_used,
            "rows_added": self.rows_added,
            "exposures_seen": self.exposures_seen,
            "initial_rank": self.initial_rank,
            "final_rank": self.final_rank,
            "final_metric_value": self.final_metric_value,
        }


@dataclass
class DecisionRecord:
    dataset_key: str
    dataset_name: str
    target_player: str
    direction: str
    k: int
    trial: int
    exposure_index: int
    model_a: str
    model_b: str
    both_active: bool
    influence_decision: str
    rigging_decision: str
    same_decision: bool
    influence_rows_added: int
    rigging_rows_added: int
    influence_success_after_step: bool
    rigging_success_after_step: bool
    influence_rank_after_step: int
    rigging_rank_after_step: int

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_key": self.dataset_key,
            "dataset_name": self.dataset_name,
            "target_player": self.target_player,
            "direction": self.direction,
            "k": self.k,
            "trial": self.trial,
            "exposure_index": self.exposure_index,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "both_active": self.both_active,
            "influence_decision": self.influence_decision,
            "rigging_decision": self.rigging_decision,
            "same_decision": self.same_decision,
            "influence_rows_added": self.influence_rows_added,
            "rigging_rows_added": self.rigging_rows_added,
            "influence_success_after_step": self.influence_success_after_step,
            "rigging_success_after_step": self.rigging_success_after_step,
            "influence_rank_after_step": self.influence_rank_after_step,
            "rigging_rank_after_step": self.rigging_rank_after_step,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Rigging-style versus influence-style decision rules under the same exposed pair stream."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(available_hf_battle_datasets()),
        help="Dataset keys to evaluate. Default: all available datasets.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Fixed top-k boundary. If omitted, sample three k values from the top, middle, and lower ranking regions for each dataset.",
    )
    parser.add_argument(
        "--n-targets-per-direction",
        type=int,
        default=3,
        help="How many boundary-near targets to include for promote and demote on each dataset.",
    )
    parser.add_argument(
        "--tasks-per-dataset",
        type=int,
        default=0,
        help="Maximum number of boundary-near tasks to keep for each chosen k. Use 0 to keep all available tasks.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=120,
        help="Maximum exposed pairs per paired trial.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of paired exposed-stream trials per target.",
    )
    parser.add_argument(
        "--rigging-mode",
        choices=["omni_bt_diff", "target_only", "omni_on"],
        default="omni_bt_diff",
        help="Rigging decision rule to compare against the influence decision rule.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Relative sampling weight for the target player in the exposed-pair stream.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "notebooks" / "artifacts" / "topk_vs_rigging",
        help="Directory for CSV/PDF/PNG outputs.",
    )
    parser.add_argument(
        "--replot-from-csv",
        action="store_true",
        help="Rebuild comparison figures from existing CSV outputs in --output-dir without rerunning trials.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume from existing CSV checkpoints in --output-dir and skip datasets already marked completed.",
    )
    parser.add_argument(
        "--skip-failed-datasets",
        action="store_true",
        help="If a dataset cannot be loaded or evaluated, record the failure in dataset_progress.csv and continue.",
    )
    return parser.parse_args()


def _style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#c7c7c7", linewidth=0.7, alpha=0.7)
    off_axis = "x" if grid_axis == "y" else "y"
    ax.grid(False, axis=off_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)
    ax.set_axisbelow(True)


def _savefig_both(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)


def _dataset_short_label(dataset_name: str) -> str:
    mapping = {
        "Chatbot Arena 55k": "Arena55k",
        "Chatbot Arena LLM Judges": "LLM Judges",
        "ATP Top-10 Matchups (2020-2024)": "ATP Top-10",
        "NBA Elo Top-50 Teams": "NBA Elo Top-50",
        "MT-Bench Human Judgments": "MT-Bench Human",
        "Vision Arena": "Vision Arena",
        "WebDev Arena": "WebDev Arena",
    }
    return mapping.get(str(dataset_name), str(dataset_name))


def choose_stratified_random_ks(
    ranking: pd.DataFrame,
    *,
    n_targets_per_direction: int,
    rng: np.random.Generator,
) -> list[int]:
    min_k = n_targets_per_direction
    max_k = len(ranking) - n_targets_per_direction
    if max_k < min_k:
        raise ValueError(
            f"Need at least {2 * n_targets_per_direction} ranked items to define "
            f"{n_targets_per_direction} promote and {n_targets_per_direction} demote targets around k."
        )

    valid_ks = np.arange(min_k, max_k + 1, dtype=int)
    regions = [chunk for chunk in np.array_split(valid_ks, 3) if len(chunk) > 0]
    if len(regions) < 3:
        raise ValueError(f"Need at least three valid k regions, found only {len(regions)}.")
    return [int(rng.choice(region)) for region in regions]


def boundary_targets(ranking: pd.DataFrame, *, k: int, n_targets_per_direction: int) -> list[dict[str, object]]:
    ranking = ranking.sort_values("rank").reset_index(drop=True)
    tasks: list[dict[str, object]] = []

    promote_rows = ranking.loc[ranking["rank"].between(k + 1, min(len(ranking), k + n_targets_per_direction))]
    for _, row in promote_rows.iterrows():
        tasks.append({"direction": "promote", "target_player": str(row["competitor"]), "initial_rank": int(row["rank"])})

    demote_rows = ranking.loc[ranking["rank"].between(max(1, k - n_targets_per_direction + 1), k)].sort_values("rank", ascending=False)
    for _, row in demote_rows.iterrows():
        tasks.append({"direction": "demote", "target_player": str(row["competitor"]), "initial_rank": int(row["rank"])})

    return tasks


def select_tasks(tasks: list[dict[str, object]], *, k: int, max_tasks: int) -> list[dict[str, object]]:
    if max_tasks <= 0 or len(tasks) <= max_tasks:
        return list(tasks)

    def _sort_key(task: dict[str, object]) -> tuple[float, int, int, str]:
        initial_rank = int(task["initial_rank"])
        direction = str(task["direction"])
        distance = abs(initial_rank - (k + 0.5))
        direction_priority = 0 if direction == "demote" else 1
        return (distance, direction_priority, initial_rank, str(task["target_player"]))

    return sorted(tasks, key=_sort_key)[:max_tasks]


def _design_row(bt_model, model_a: str, model_b: str) -> np.ndarray:
    name_to_idx = {str(name): idx for idx, name in enumerate(bt_model.competitor_names_)}
    pa = name_to_idx[str(model_a)]
    pb = name_to_idx[str(model_b)]
    row = np.zeros(bt_model.n_params_, dtype=float)
    if pa != bt_model.reference_player:
        row[pa - (1 if pa > bt_model.reference_player else 0)] = 1.0
    if pb != bt_model.reference_player:
        row[pb - (1 if pb > bt_model.reference_player else 0)] = -1.0
    return row


def _current_objective(model, target_player: str, k: int, direction: str) -> tuple[SkillGapObjective, float]:
    ranking = ranking_frame(model)
    target_rank = int(ranking.loc[ranking["competitor"] == target_player, "rank"].iloc[0])
    if direction == "promote":
        boundary_rank = k if target_rank > k else min(len(ranking), k + 1)
        boundary_player = str(ranking.loc[ranking["rank"] == boundary_rank, "competitor"].iloc[0])
        objective = SkillGapObjective(target_player, boundary_player)
        value = float(model.reported_gap(target_player, boundary_player))
        return objective, value

    boundary_rank = (k + 1) if target_rank <= k else k
    boundary_player = str(ranking.loc[ranking["rank"] == boundary_rank, "competitor"].iloc[0])
    objective = SkillGapObjective(target_player, boundary_player)
    value = float(-model.reported_gap(target_player, boundary_player))
    return objective, value


def _rigging_reward(model, target_player: str, direction: str) -> float:
    ranking = ranking_frame(model)
    target_rank = int(ranking.loc[ranking["competitor"] == target_player, "rank"].iloc[0])
    n_rows = len(ranking)
    if direction == "promote":
        anchor_rank = target_rank - 1 if target_rank > 1 else min(n_rows, target_rank + 1)
        anchor_player = str(ranking.loc[ranking["rank"] == anchor_rank, "competitor"].iloc[0])
        return float(model.reported_gap(target_player, anchor_player))
    anchor_rank = target_rank + 1 if target_rank < n_rows else max(1, target_rank - 1)
    anchor_player = str(ranking.loc[ranking["rank"] == anchor_rank, "competitor"].iloc[0])
    return float(-model.reported_gap(target_player, anchor_player))


def _target_rating(model, target_player: str) -> float:
    ranking = ranking_frame(model)
    return float(ranking.loc[ranking["competitor"] == target_player, "rating"].iloc[0])


def _omni_on_reward(model, target_player: str, model_a: str, model_b: str, decision: str) -> float:
    base = 10.0
    scale = 400.0
    k_factor = 4.0
    ra = _target_rating(model, model_a)
    rb = _target_rating(model, model_b)
    rt = _target_rating(model, target_player)
    ea = 1.0 / (1.0 + base ** ((rb - ra) / scale))
    eb = 1.0 / (1.0 + base ** ((ra - rb) / scale))
    if decision == "model_a":
        ra1 = ra + k_factor * eb
        rb1 = rb - k_factor * eb
    elif decision == "model_b":
        ra1 = ra - k_factor * ea
        rb1 = rb + k_factor * ea
    elif decision == "tie":
        ra1 = ra - 0.5 * k_factor * (ea - eb)
        rb1 = rb + 0.5 * k_factor * (ea - eb)
    elif decision == "remove":
        ra1 = ra
        rb1 = rb
    else:
        raise ValueError(f"Unsupported decision {decision!r}.")
    return float(
        1.0 / (1.0 + base ** ((ra1 - rt) / scale))
        + 1.0 / (1.0 + base ** ((rb1 - rt) / scale))
    )


def _omni_anchor_context(model, target_player: str, direction: str) -> tuple[str, float]:
    ranking = ranking_frame(model)
    target_rank = int(ranking.loc[ranking["competitor"] == target_player, "rank"].iloc[0])
    n_rows = len(ranking)
    if direction == "promote":
        anchor_rank = target_rank - 1 if target_rank > 1 else min(n_rows, target_rank + 1)
    else:
        anchor_rank = target_rank + 1 if target_rank < n_rows else max(1, target_rank - 1)
    anchor_player = str(ranking.loc[ranking["rank"] == anchor_rank, "competitor"].iloc[0])
    return anchor_player, _omni_reward_against_anchor(model, target_player, anchor_player, direction, "omni_bt_diff")


def _omni_reward_against_anchor(model, target_player: str, anchor_player: str, direction: str, rigging_mode: str) -> float:
    if rigging_mode == "omni_bt_abs":
        value = _target_rating(model, target_player)
        return float(value if direction == "promote" else -value)
    gap = float(model.reported_gap(target_player, anchor_player))
    return float(gap if direction == "promote" else -gap)


def _topk_success(model, target_player: str, k: int, direction: str) -> tuple[bool, int]:
    ranking = ranking_frame(model)
    rank = int(ranking.loc[ranking["competitor"] == target_player, "rank"].iloc[0])
    return (rank <= k if direction == "promote" else rank > k), rank


def _sample_pair(players: list[str], target_player: str, beta: float, rng: np.random.Generator) -> tuple[str, str]:
    weights = np.ones(len(players), dtype=float)
    weights[players.index(target_player)] = float(beta)
    probs = weights / weights.sum()
    idx = rng.choice(len(players), size=2, replace=False, p=probs)
    return str(players[int(idx[0])]), str(players[int(idx[1])])


def _candidate_row(model, model_a: str, model_b: str) -> pd.Series:
    return pd.Series({"model_a": model_a, "model_b": model_b, "outcome": 1.0, "_candidate_x": _design_row(model, model_a, model_b)})


def _apply_decision(state, model, model_a: str, model_b: str, decision: str) -> tuple[object, int]:
    if decision == "remove":
        return state, 0
    if decision == "model_a":
        return apply_add_candidate(state, _candidate_row(model, model_a, model_b)), 1
    if decision == "model_b":
        return apply_add_candidate(state, _candidate_row(model, model_b, model_a)), 1
    if decision == "tie":
        tied = apply_add_candidate(state, _candidate_row(model, model_a, model_b))
        return apply_add_candidate(tied, _candidate_row(model, model_b, model_a)), 2
    raise ValueError(f"Unsupported decision {decision!r}.")


def _target_only_decision(model_a: str, model_b: str, target_player: str, direction: str) -> str:
    if target_player not in {model_a, model_b}:
        return "remove"
    if direction == "promote":
        return "model_a" if model_a == target_player else "model_b"
    return "model_b" if model_a == target_player else "model_a"


def _objective_delta_for_decision(model, objective: SkillGapObjective, direction: str, model_a: str, model_b: str, decision: str) -> float:
    if decision == "remove":
        return 0.0
    sign = 1.0 if direction == "promote" else -1.0
    if decision == "model_a":
        x_new = np.asarray([_design_row(model, model_a, model_b)], dtype=float)
        y_new = np.asarray([1.0], dtype=float)
        value = compute_objective_action_influence(model, objective, action="add", method="1sn", X_new=x_new, y_new=y_new)
        return float(sign * value[0])
    if decision == "model_b":
        x_new = np.asarray([_design_row(model, model_b, model_a)], dtype=float)
        y_new = np.asarray([1.0], dtype=float)
        value = compute_objective_action_influence(model, objective, action="add", method="1sn", X_new=x_new, y_new=y_new)
        return float(sign * value[0])
    if decision == "tie":
        x_new = np.asarray([_design_row(model, model_a, model_b), _design_row(model, model_b, model_a)], dtype=float)
        y_new = np.asarray([1.0, 1.0], dtype=float)
        value = compute_objective_action_influence(model, objective, action="add", method="1sn", X_new=x_new, y_new=y_new)
        return float(sign * np.sum(value))
    raise ValueError(f"Unsupported decision {decision!r}.")


def choose_influence_decision(model, target_player: str, k: int, direction: str, model_a: str, model_b: str) -> str:
    objective, _ = _current_objective(model, target_player, k, direction)
    decisions = ["remove", "model_a", "model_b", "tie"]
    scores = {decision: _objective_delta_for_decision(model, objective, direction, model_a, model_b, decision) for decision in decisions}
    return max(decisions, key=lambda decision: (scores[decision], decision != "remove"))


def choose_rigging_decision(model, target_player: str, direction: str, model_a: str, model_b: str, rigging_mode: str):
    if rigging_mode == "target_only":
        return _target_only_decision(model_a, model_b, target_player, direction)
    if target_player in {model_a, model_b}:
        return _target_only_decision(model_a, model_b, target_player, direction)
    if rigging_mode == "omni_on":
        decisions = ["model_a", "model_b", "tie", "remove"]
        rewards = [_omni_on_reward(model, target_player, model_a, model_b, decision) for decision in decisions]
        return decisions[int(np.argmax(rewards))]
    if rigging_mode not in {"omni_bt_diff", "omni_bt_abs"}:
        raise ValueError(f"Unsupported rigging_mode {rigging_mode!r}.")
    if rigging_mode == "omni_bt_diff":
        anchor_player, _ = _omni_anchor_context(model, target_player, direction)
        objective = SkillGapObjective(target_player, anchor_player)
        decisions = ["remove", "model_a", "model_b", "tie"]
        scores = {
            decision: _objective_delta_for_decision(model, objective, direction, model_a, model_b, decision)
            for decision in decisions
        }
        return max(decisions, key=lambda decision: (scores[decision], decision != "remove"))

    anchor_player, baseline_reward = _omni_anchor_context(model, target_player, direction)
    best_decision = "remove"
    best_reward = baseline_reward
    for decision in ("model_a", "model_b", "tie"):
        trial_state, _ = _apply_decision(make_state(model), model, model_a, model_b, decision)
        trial_model = fit_model_from_state(model, trial_state)
        reward = _omni_reward_against_anchor(trial_model, target_player, anchor_player, direction, rigging_mode)
        if reward > best_reward:
            best_reward = reward
            best_decision = decision
    return best_decision


def paired_trial_winner(influence_row: pd.Series, rigging_row: pd.Series) -> str:
    inf_success = bool(influence_row["success"])
    rig_success = bool(rigging_row["success"])
    if inf_success != rig_success:
        return "influence_pair" if inf_success else "rigging"

    comparisons = [
        ("exposures_seen", True),
        ("actions_used", True),
        ("rows_added", True),
        ("final_rank", True),
        ("final_metric_value", False),
    ]
    for metric, smaller_is_better in comparisons:
        inf_value = float(influence_row[metric])
        rig_value = float(rigging_row[metric])
        if np.isclose(inf_value, rig_value):
            continue
        if smaller_is_better:
            return "influence_pair" if inf_value < rig_value else "rigging"
        return "influence_pair" if inf_value > rig_value else "rigging"
    return "tie"


def run_paired_trial(
    bt_model,
    *,
    dataset_key: str,
    dataset_name: str,
    target_player: str,
    k: int,
    direction: str,
    rigging_mode: str,
    beta: float,
    budget: int,
    seed: int,
) -> tuple[TrialResult, TrialResult, list[DecisionRecord]]:
    players = [str(name) for name in bt_model.competitor_names_]
    rng = np.random.default_rng(seed)

    states = {"rigging": make_state(bt_model), "influence_pair": make_state(bt_model)}
    models = {"rigging": bt_model, "influence_pair": bt_model}
    initial_rank = int(ranking_frame(bt_model).loc[lambda df: df["competitor"] == target_player, "rank"].iloc[0])
    done = {"rigging": False, "influence_pair": False}
    exposures = {"rigging": 0, "influence_pair": 0}
    actions = {"rigging": 0, "influence_pair": 0}
    rows_added = {"rigging": 0, "influence_pair": 0}

    decision_records: list[DecisionRecord] = []

    for exposure_index in range(1, budget + 1):
        for method in ("rigging", "influence_pair"):
            if not done[method]:
                success, _ = _topk_success(models[method], target_player, k, direction)
                done[method] = success
        if all(done.values()):
            break

        model_a, model_b = _sample_pair(players, target_player, beta, rng)
        both_active = not done["rigging"] and not done["influence_pair"]
        step_payload = {
            "rigging": {"decision": None, "added": 0, "success": done["rigging"], "rank": int(ranking_frame(models["rigging"]).loc[lambda df: df["competitor"] == target_player, "rank"].iloc[0])},
            "influence_pair": {"decision": None, "added": 0, "success": done["influence_pair"], "rank": int(ranking_frame(models["influence_pair"]).loc[lambda df: df["competitor"] == target_player, "rank"].iloc[0])},
        }
        for method in ("rigging", "influence_pair"):
            if done[method]:
                step_payload[method]["decision"] = "already_done"
                continue
            exposures[method] += 1
            if method == "rigging":
                decision = choose_rigging_decision(models[method], target_player, direction, model_a, model_b, rigging_mode)
            else:
                decision = choose_influence_decision(models[method], target_player, k, direction, model_a, model_b)
            next_state, added = _apply_decision(states[method], models[method], model_a, model_b, decision)
            states[method] = next_state
            step_payload[method]["decision"] = decision
            step_payload[method]["added"] = added
            if added > 0:
                actions[method] += 1
                rows_added[method] += added
                models[method] = fit_model_from_state(bt_model, states[method])
            success, _ = _topk_success(models[method], target_player, k, direction)
            done[method] = success
            step_payload[method]["success"] = success
            step_payload[method]["rank"] = int(
                ranking_frame(models[method]).loc[lambda df: df["competitor"] == target_player, "rank"].iloc[0]
            )

        decision_records.append(
            DecisionRecord(
                dataset_key=dataset_key,
                dataset_name=dataset_name,
                target_player=target_player,
                direction=direction,
                k=k,
                trial=seed,
                exposure_index=exposure_index,
                model_a=model_a,
                model_b=model_b,
                both_active=both_active,
                influence_decision=str(step_payload["influence_pair"]["decision"]),
                rigging_decision=str(step_payload["rigging"]["decision"]),
                same_decision=str(step_payload["influence_pair"]["decision"]) == str(step_payload["rigging"]["decision"]),
                influence_rows_added=int(step_payload["influence_pair"]["added"]),
                rigging_rows_added=int(step_payload["rigging"]["added"]),
                influence_success_after_step=bool(step_payload["influence_pair"]["success"]),
                rigging_success_after_step=bool(step_payload["rigging"]["success"]),
                influence_rank_after_step=int(step_payload["influence_pair"]["rank"]),
                rigging_rank_after_step=int(step_payload["rigging"]["rank"]),
            )
        )

    out: dict[str, TrialResult] = {}
    for method, policy in (("rigging", rigging_mode), ("influence_pair", "pairwise_influence")):
        success, final_rank = _topk_success(models[method], target_player, k, direction)
        if method == "rigging":
            if rigging_mode in {"omni_bt_diff", "omni_bt_abs"}:
                anchor_player, _ = _omni_anchor_context(models[method], target_player, direction)
                metric_value = _omni_reward_against_anchor(models[method], target_player, anchor_player, direction, rigging_mode)
            else:
                metric_value = _rigging_reward(models[method], target_player, direction)
        else:
            _, metric_value = _current_objective(models[method], target_player, k, direction)
        out[method] = TrialResult(
            dataset_key=dataset_key,
            dataset_name=dataset_name,
            target_player=target_player,
            direction=direction,
            k=k,
            method=method,
            policy=policy,
            seed=seed,
            success=success,
            actions_used=actions[method],
            rows_added=rows_added[method],
            exposures_seen=exposures[method],
            initial_rank=initial_rank,
            final_rank=final_rank,
            final_metric_value=float(metric_value),
        )
    return out["rigging"], out["influence_pair"], decision_records


def build_task_frame(all_rows: pd.DataFrame) -> pd.DataFrame:
    task_cols = ["dataset_key", "dataset_name", "direction", "target_player", "initial_rank", "k"]
    return all_rows[task_cols].drop_duplicates().sort_values(["dataset_key", "direction", "initial_rank"]).reset_index(drop=True)


def summarize_comparison(all_rows: pd.DataFrame, *, budget: int) -> pd.DataFrame:
    group_cols = ["dataset_key", "dataset_name", "direction", "target_player", "initial_rank", "k", "trial"]
    paired = (
        all_rows.pivot_table(
            index=group_cols,
            columns="method",
            values=["success", "actions_used", "rows_added", "exposures_seen", "final_rank"],
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    paired = paired.reset_index()

    rows = []
    task_cols = ["dataset_key", "dataset_name", "direction", "target_player", "initial_rank", "k"]
    for key, grp in paired.groupby(task_cols, sort=False):
        inf_exp = np.where(grp["success_influence_pair"], grp["exposures_seen_influence_pair"], budget + 1)
        rig_exp = np.where(grp["success_rigging"], grp["exposures_seen_rigging"], budget + 1)
        inf_act = np.where(grp["success_influence_pair"], grp["actions_used_influence_pair"], budget + 1)
        rig_act = np.where(grp["success_rigging"], grp["actions_used_rigging"], budget + 1)
        inf_rows = np.where(grp["success_influence_pair"], grp["rows_added_influence_pair"], budget + 1)
        rig_rows = np.where(grp["success_rigging"], grp["rows_added_rigging"], budget + 1)
        rows.append(
            {
                **dict(zip(task_cols, key)),
                "influence_success_rate": float(np.mean(grp["success_influence_pair"].to_numpy(dtype=float))),
                "rigging_success_rate": float(np.mean(grp["success_rigging"].to_numpy(dtype=float))),
                "influence_exposures_median": float(np.median(inf_exp)),
                "rigging_exposures_median": float(np.median(rig_exp)),
                "influence_actions_median": float(np.median(inf_act)),
                "rigging_actions_median": float(np.median(rig_act)),
                "influence_rows_added_median": float(np.median(inf_rows)),
                "rigging_rows_added_median": float(np.median(rig_rows)),
                "p_rigging_exposures_ge_influence": float(np.mean(rig_exp >= inf_exp)),
                "p_rigging_actions_ge_influence": float(np.mean(rig_act >= inf_act)),
                "median_exposure_advantage_rigging_minus_influence": float(np.median(rig_exp - inf_exp)),
                "median_action_advantage_rigging_minus_influence": float(np.median(rig_act - inf_act)),
                "median_rows_added_advantage_rigging_minus_influence": float(np.median(rig_rows - inf_rows)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "direction", "initial_rank"]).reset_index(drop=True)


def summarize_decision_agreement(decision_df: pd.DataFrame) -> pd.DataFrame:
    task_cols = ["dataset_key", "dataset_name", "direction", "target_player", "initial_rank", "k"]
    if decision_df.empty:
        return pd.DataFrame(
            columns=task_cols
            + [
                "n_exposed_pairs_total",
                "n_jointly_predicted_pairs",
                "same_decision_rate_when_both_active",
                "n_same_decision_when_both_active",
                "n_different_decision_when_both_active",
            ]
        )
    rows = []
    for key, grp in decision_df.groupby(task_cols, dropna=False, sort=False):
        joint = grp.loc[grp["both_active"]]
        n_joint = int(len(joint))
        n_same = int(joint["same_decision"].sum()) if n_joint else 0
        rows.append(
            {
                **dict(zip(task_cols, key)),
                "n_exposed_pairs_total": int(len(grp)),
                "n_jointly_predicted_pairs": n_joint,
                "same_decision_rate_when_both_active": float(joint["same_decision"].mean()) if n_joint else float("nan"),
                "n_same_decision_when_both_active": n_same,
                "n_different_decision_when_both_active": n_joint - n_same,
            }
        )
    grouped = pd.DataFrame(rows)
    return grouped.sort_values(["dataset_key", "direction", "initial_rank"]).reset_index(drop=True)


def summarize_trial_outcomes(all_rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_key", "dataset_name", "direction", "target_player", "initial_rank", "k", "trial"]
    paired = (
        all_rows.pivot_table(
            index=group_cols,
            columns="method",
            values=["success", "actions_used", "rows_added", "exposures_seen", "final_rank", "final_metric_value"],
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    winner_rows = []
    for _, row in paired.iterrows():
        influence_row = pd.Series(
            {
                "success": row["success_influence_pair"],
                "actions_used": row["actions_used_influence_pair"],
                "rows_added": row["rows_added_influence_pair"],
                "exposures_seen": row["exposures_seen_influence_pair"],
                "final_rank": row["final_rank_influence_pair"],
                "final_metric_value": row["final_metric_value_influence_pair"],
            }
        )
        rigging_row = pd.Series(
            {
                "success": row["success_rigging"],
                "actions_used": row["actions_used_rigging"],
                "rows_added": row["rows_added_rigging"],
                "exposures_seen": row["exposures_seen_rigging"],
                "final_rank": row["final_rank_rigging"],
                "final_metric_value": row["final_metric_value_rigging"],
            }
        )
        winner = paired_trial_winner(influence_row, rigging_row)
        winner_rows.append(
            {
                **{col: row[col] for col in group_cols},
                "trial_winner": winner,
                "influence_success": bool(row["success_influence_pair"]),
                "rigging_success": bool(row["success_rigging"]),
                "influence_exposures_seen": int(row["exposures_seen_influence_pair"]),
                "rigging_exposures_seen": int(row["exposures_seen_rigging"]),
                "influence_actions_used": int(row["actions_used_influence_pair"]),
                "rigging_actions_used": int(row["actions_used_rigging"]),
                "influence_rows_added": int(row["rows_added_influence_pair"]),
                "rigging_rows_added": int(row["rows_added_rigging"]),
            }
        )
    return pd.DataFrame(winner_rows).sort_values(["dataset_key", "direction", "initial_rank", "trial"]).reset_index(drop=True)


def summarize_dataset_actions(all_rows: pd.DataFrame, trial_summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        all_rows.groupby(["dataset_key", "dataset_name", "direction", "method"], dropna=False)
        .agg(
            n_runs=("trial", "size"),
            n_success_runs=("success", "sum"),
            success_rate=("success", "mean"),
            mean_exposures_seen=("exposures_seen", "mean"),
            median_exposures_seen=("exposures_seen", "median"),
            mean_actions_used=("actions_used", "mean"),
            median_actions_used=("actions_used", "median"),
            mean_rows_added=("rows_added", "mean"),
            median_rows_added=("rows_added", "median"),
        )
        .reset_index()
    )
    win_counts = (
        trial_summary_df.groupby(["dataset_key", "direction", "trial_winner"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "influence_pair" not in win_counts.columns:
        win_counts["influence_pair"] = 0
    if "rigging" not in win_counts.columns:
        win_counts["rigging"] = 0
    if "tie" not in win_counts.columns:
        win_counts["tie"] = 0
    grouped = grouped.merge(win_counts, on=["dataset_key", "direction"], how="left")
    grouped["influence_trial_wins"] = grouped["influence_pair"].fillna(0).astype(int)
    grouped["rigging_trial_wins"] = grouped["rigging"].fillna(0).astype(int)
    grouped["tied_trials"] = grouped["tie"].fillna(0).astype(int)
    grouped = grouped.drop(columns=["influence_pair", "rigging", "tie"])

    dataset_overall = (
        all_rows.groupby(["dataset_key", "dataset_name", "method"], dropna=False)
        .agg(
            n_runs=("trial", "size"),
            n_success_runs=("success", "sum"),
            success_rate=("success", "mean"),
            mean_exposures_seen=("exposures_seen", "mean"),
            median_exposures_seen=("exposures_seen", "median"),
            mean_actions_used=("actions_used", "mean"),
            median_actions_used=("actions_used", "median"),
            mean_rows_added=("rows_added", "mean"),
            median_rows_added=("rows_added", "median"),
        )
        .reset_index()
    )
    dataset_overall.insert(2, "direction", "all")
    dataset_win_counts = (
        trial_summary_df.groupby(["dataset_key", "trial_winner"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "influence_pair" not in dataset_win_counts.columns:
        dataset_win_counts["influence_pair"] = 0
    if "rigging" not in dataset_win_counts.columns:
        dataset_win_counts["rigging"] = 0
    if "tie" not in dataset_win_counts.columns:
        dataset_win_counts["tie"] = 0
    dataset_overall = dataset_overall.merge(dataset_win_counts, on="dataset_key", how="left")
    dataset_overall["influence_trial_wins"] = dataset_overall["influence_pair"].fillna(0).astype(int)
    dataset_overall["rigging_trial_wins"] = dataset_overall["rigging"].fillna(0).astype(int)
    dataset_overall["tied_trials"] = dataset_overall["tie"].fillna(0).astype(int)
    dataset_overall = dataset_overall.drop(columns=["influence_pair", "rigging", "tie"])

    overall = (
        all_rows.groupby("method", dropna=False)
        .agg(
            n_runs=("trial", "size"),
            n_success_runs=("success", "sum"),
            success_rate=("success", "mean"),
            mean_exposures_seen=("exposures_seen", "mean"),
            median_exposures_seen=("exposures_seen", "median"),
            mean_actions_used=("actions_used", "mean"),
            median_actions_used=("actions_used", "median"),
            mean_rows_added=("rows_added", "mean"),
            median_rows_added=("rows_added", "median"),
        )
        .reset_index()
    )
    overall.insert(0, "dataset_key", "all_datasets")
    overall.insert(1, "dataset_name", "All datasets")
    overall.insert(2, "direction", "all")
    overall["influence_trial_wins"] = int((trial_summary_df["trial_winner"] == "influence_pair").sum())
    overall["rigging_trial_wins"] = int((trial_summary_df["trial_winner"] == "rigging").sum())
    overall["tied_trials"] = int((trial_summary_df["trial_winner"] == "tie").sum())
    return pd.concat([grouped, dataset_overall, overall], ignore_index=True)


def summarize_dataset_actions_both_success(all_rows: pd.DataFrame, trial_summary_df: pd.DataFrame) -> pd.DataFrame:
    if trial_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset_name",
                "direction",
                "method",
                "n_runs",
                "n_success_runs",
                "n_joint_success_trials",
                "success_rate",
                "mean_exposures_seen",
                "median_exposures_seen",
                "mean_actions_used",
                "median_actions_used",
                "mean_rows_added",
                "median_rows_added",
                "influence_trial_wins",
                "rigging_trial_wins",
                "tied_trials",
            ]
        )

    success_pairs = trial_summary_df.loc[
        trial_summary_df["influence_success"] & trial_summary_df["rigging_success"],
        ["dataset_key", "dataset_name", "direction", "target_player", "initial_rank", "k", "trial"],
    ].copy()
    if success_pairs.empty:
        empty = summarize_dataset_actions(all_rows.iloc[0:0], trial_summary_df.iloc[0:0])
        empty["n_joint_success_trials"] = pd.Series(dtype=int)
        return empty

    merge_cols = ["dataset_key", "dataset_name", "direction", "target_player", "initial_rank", "k", "trial"]
    filtered = all_rows.merge(success_pairs, on=merge_cols, how="inner")
    out = summarize_dataset_actions(filtered, trial_summary_df.loc[
        trial_summary_df["influence_success"] & trial_summary_df["rigging_success"]
    ].copy())

    joint_counts = (
        success_pairs.groupby(["dataset_key", "dataset_name", "direction"], dropna=False)
        .size()
        .reset_index(name="n_joint_success_trials")
    )
    joint_counts_all = (
        success_pairs.groupby(["dataset_key", "dataset_name"], dropna=False)
        .size()
        .reset_index(name="n_joint_success_trials")
    )
    joint_counts_all.insert(2, "direction", "all")
    overall_joint = pd.DataFrame(
        [{"dataset_key": "all_datasets", "dataset_name": "All datasets", "direction": "all", "n_joint_success_trials": int(len(success_pairs))}]
    )
    joint_counts = pd.concat([joint_counts, joint_counts_all, overall_joint], ignore_index=True)
    return out.merge(joint_counts, on=["dataset_key", "dataset_name", "direction"], how="left")


def summarize_targets(all_rows: pd.DataFrame, trial_summary_df: pd.DataFrame, *, budget: int) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset_name",
                "direction",
                "target_player",
                "initial_rank",
                "k",
                "n_trials",
                "influence_success_count",
                "rigging_success_count",
                "influence_success_rate",
                "rigging_success_rate",
                "influence_exposures_median",
                "rigging_exposures_median",
                "influence_actions_median",
                "rigging_actions_median",
                "influence_rows_added_median",
                "rigging_rows_added_median",
                "influence_trial_wins",
                "rigging_trial_wins",
                "tied_trials",
                "median_exposure_advantage_rigging_minus_influence",
                "median_action_advantage_rigging_minus_influence",
                "median_rows_added_advantage_rigging_minus_influence",
            ]
        )

    merged = trial_summary_df.copy()
    rows = []
    group_cols = ["dataset_key", "dataset_name", "direction", "target_player", "initial_rank", "k"]
    for key, grp in merged.groupby(group_cols, sort=False, dropna=False):
        inf_exp = np.where(grp["influence_success"], grp["influence_exposures_seen"], budget + 1)
        rig_exp = np.where(grp["rigging_success"], grp["rigging_exposures_seen"], budget + 1)
        inf_act = np.where(grp["influence_success"], grp["influence_actions_used"], budget + 1)
        rig_act = np.where(grp["rigging_success"], grp["rigging_actions_used"], budget + 1)
        inf_rows = np.where(grp["influence_success"], grp["influence_rows_added"], budget + 1)
        rig_rows = np.where(grp["rigging_success"], grp["rigging_rows_added"], budget + 1)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_trials": int(len(grp)),
                "influence_success_count": int(grp["influence_success"].sum()),
                "rigging_success_count": int(grp["rigging_success"].sum()),
                "influence_success_rate": float(np.mean(grp["influence_success"].to_numpy(dtype=float))),
                "rigging_success_rate": float(np.mean(grp["rigging_success"].to_numpy(dtype=float))),
                "influence_exposures_median": float(np.median(inf_exp)),
                "rigging_exposures_median": float(np.median(rig_exp)),
                "influence_actions_median": float(np.median(inf_act)),
                "rigging_actions_median": float(np.median(rig_act)),
                "influence_rows_added_median": float(np.median(inf_rows)),
                "rigging_rows_added_median": float(np.median(rig_rows)),
                "influence_trial_wins": int((grp["trial_winner"] == "influence_pair").sum()),
                "rigging_trial_wins": int((grp["trial_winner"] == "rigging").sum()),
                "tied_trials": int((grp["trial_winner"] == "tie").sum()),
                "median_exposure_advantage_rigging_minus_influence": float(np.median(rig_exp - inf_exp)),
                "median_action_advantage_rigging_minus_influence": float(np.median(rig_act - inf_act)),
                "median_rows_added_advantage_rigging_minus_influence": float(np.median(rig_rows - inf_rows)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "direction", "initial_rank"]).reset_index(drop=True)


def _slugify_target(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return slug or "target"


def _task_labels(task_df: pd.DataFrame) -> list[str]:
    return [f"{row['dataset_key']} | {row['direction']} | r{int(row['initial_rank'])} {row['target_player']}" for _, row in task_df.iterrows()]


def plot_paired_metric(all_rows: pd.DataFrame, *, metric: str, budget: int, output_path: Path):
    task_df = build_task_frame(all_rows)
    labels = _task_labels(task_df)
    y_positions = np.arange(len(task_df), dtype=float)
    fig_height = max(4.0, 0.42 * len(task_df))
    fig, ax = plt.subplots(figsize=(10.0, fig_height))

    for y, (_, task) in zip(y_positions, task_df.iterrows()):
        task_mask = (
            (all_rows["dataset_key"] == task["dataset_key"])
            & (all_rows["direction"] == task["direction"])
            & (all_rows["target_player"] == task["target_player"])
            & (all_rows["k"] == task["k"])
        )
        sub = all_rows.loc[task_mask].pivot_table(index="trial", columns="method", values=["success", metric], aggfunc="first")
        inf_metric = np.where(sub[("success", "influence_pair")], sub[(metric, "influence_pair")], budget + 1)
        rig_metric = np.where(sub[("success", "rigging")], sub[(metric, "rigging")], budget + 1)
        jitter = np.linspace(-0.16, 0.16, max(len(sub), 1))
        for idx, (xi, xr) in enumerate(zip(inf_metric, rig_metric)):
            yy = y + jitter[idx]
            ax.plot([xi, xr], [yy, yy], color="#B8C0CC", linewidth=1.0, alpha=0.7, zorder=1)
        ax.scatter(inf_metric, y + jitter[: len(sub)], s=22, color=POLICY_COLORS["influence_pair"], zorder=3, label=None)
        ax.scatter(rig_metric, y + jitter[: len(sub)], s=22, color=POLICY_COLORS["rigging"], zorder=3, label=None)

    ax.axvline(budget + 1, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    if metric == "exposures_seen":
        ax.set_xlabel("Exposed pairs to reach top-k")
    elif metric == "actions_used":
        ax.set_xlabel("Non-abstained exposed pairs to reach top-k")
    else:
        ax.set_xlabel("BT rows added to reach top-k")
    ax.set_title("Paired exposed-stream comparison")
    _style_axis(ax, grid_axis="x")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POLICY_COLORS["influence_pair"], markersize=6, label="Influence decision"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POLICY_COLORS["rigging"], markersize=6, label="Rigging decision"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    _savefig_both(fig, output_path)
    plt.close(fig)


def plot_advantage(summary_df: pd.DataFrame, *, output_path: Path):
    plot_df = summary_df.copy().sort_values(["dataset_key", "direction", "initial_rank"]).reset_index(drop=True)
    x = np.arange(len(plot_df), dtype=float)
    colors = [POLICY_COLORS[str(v)] for v in plot_df["direction"]]
    fig, ax = plt.subplots(figsize=(max(7.2, 0.75 * len(plot_df)), 3.6))
    bars = ax.bar(x, plot_df["median_exposure_advantage_rigging_minus_influence"], color=colors, edgecolor="#222222", linewidth=0.6)
    ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Median rigging exposures minus influence exposures")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row.dataset_key}\n{row.direction}\nr{int(row.initial_rank)}" for row in plot_df.itertuples()])
    ax.set_title("Positive means rigging needs more exposed pairs")
    _style_axis(ax, grid_axis="y")
    for bar, value in zip(bars, plot_df["median_exposure_advantage_rigging_minus_influence"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _savefig_both(fig, output_path)
    plt.close(fig)


def _trial_advantage_error_frame(trial_summary_df: pd.DataFrame) -> pd.DataFrame:
    if trial_summary_df.empty:
        return pd.DataFrame(
            columns=["dataset_key", "direction", "target_player", "initial_rank", "k", "advantage_std", "advantage_sem"]
        )
    work = trial_summary_df.copy()
    work["trial_exposure_advantage"] = (
        work["rigging_exposures_seen"].astype(float) - work["influence_exposures_seen"].astype(float)
    )
    grouped = (
        work.groupby(["dataset_key", "direction", "target_player", "initial_rank", "k"], dropna=False)["trial_exposure_advantage"]
        .agg(["std", "count"])
        .reset_index()
        .rename(columns={"std": "advantage_std"})
    )
    grouped["advantage_std"] = grouped["advantage_std"].fillna(0.0)
    grouped["advantage_sem"] = grouped["advantage_std"] / np.sqrt(np.clip(grouped["count"].astype(float), 1.0, None))
    return grouped


def plot_advantage_vertical(summary_df: pd.DataFrame, *, output_path: Path, trial_summary_df: pd.DataFrame | None = None):
    plot_df = summary_df.copy().sort_values(["dataset_key", "direction", "initial_rank"]).reset_index(drop=True)
    if trial_summary_df is not None and not trial_summary_df.empty:
        plot_df = plot_df.merge(
            _trial_advantage_error_frame(trial_summary_df),
            on=["dataset_key", "direction", "target_player", "initial_rank", "k"],
            how="left",
        )
    else:
        plot_df["advantage_std"] = 0.0
        plot_df["advantage_sem"] = 0.0
    x = np.arange(len(plot_df), dtype=float)
    colors = [POLICY_COLORS[str(v)] for v in plot_df["direction"]]
    labels = [f"{row.dataset_key}\n{row.direction}\nr{int(row.initial_rank)}" for row in plot_df.itertuples()]
    fig, ax = plt.subplots(figsize=(max(7.2, 0.7 * len(plot_df)), 4.2))
    bars = ax.bar(
        x,
        plot_df["median_exposure_advantage_rigging_minus_influence"],
        yerr=plot_df["advantage_std"].fillna(0.0).to_numpy(dtype=float),
        capsize=4,
        error_kw={"elinewidth": 1.0, "ecolor": "#3c3c3c"},
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
        width=0.82,
    )
    ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Rigging minus influence\nmedian exposed pairs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, linespacing=0.95)
    ax.set_title("Rigging exposure advantage by task", pad=10)
    _style_axis(ax, grid_axis="y")
    ax.tick_params(axis="x", pad=7)
    y_values = plot_df["median_exposure_advantage_rigging_minus_influence"].to_numpy(dtype=float)
    pad = max(2.0, 0.04 * max(np.max(np.abs(y_values)), 1.0))
    for bar, value in zip(bars, y_values):
        va = "bottom" if value >= 0 else "top"
        y = value + pad * (1 if value >= 0 else -1)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.1f}", ha="center", va=va, fontsize=9)
    handles = [
        Line2D([0], [0], color=POLICY_COLORS["promote"], linewidth=8, label="Promote"),
        Line2D([0], [0], color=POLICY_COLORS["demote"], linewidth=8, label="Demote"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(pad=0.75)
    _savefig_both(fig, output_path)
    plt.close(fig)


def plot_dataset_actions(dataset_action_df: pd.DataFrame, *, output_path: Path):
    plot_df = dataset_action_df.loc[
        (dataset_action_df["dataset_key"] != "all_datasets") & (dataset_action_df["direction"] == "all")
    ].copy()
    if plot_df.empty:
        return
    order = plot_df.groupby("dataset_name")["mean_actions_used"].mean().sort_values().index.tolist()
    methods = ["influence_pair", "rigging"]
    colors = [POLICY_COLORS["influence_pair"], POLICY_COLORS["rigging"]]
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.6 * len(order) + 1.0)))
    y = np.arange(len(order), dtype=float)
    width = 0.34
    for idx, (method, color) in enumerate(zip(methods, colors)):
        sub = (
            plot_df.loc[plot_df["method"] == method, ["dataset_name", "mean_actions_used"]]
            .set_index("dataset_name")
            .reindex(order)
        )
        ax.barh(y + (idx - 0.5) * width, sub["mean_actions_used"].to_numpy(dtype=float), height=width, color=color, label=method.replace("_", " "))
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Mean actions used per run")
    ax.set_title("Per-dataset action count comparison")
    _style_axis(ax, grid_axis="x")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _savefig_both(fig, output_path)
    plt.close(fig)


def _all_runs_error_frame(all_rows: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame(columns=["dataset_key", "direction", "method", f"{metric}_std", f"{metric}_sem"])
    grouped = (
        all_rows.groupby(["dataset_key", "direction", "method"], dropna=False)[metric]
        .agg(["std", "count"])
        .reset_index()
        .rename(columns={"std": f"{metric}_std"})
    )
    grouped[f"{metric}_std"] = grouped[f"{metric}_std"].fillna(0.0)
    grouped[f"{metric}_sem"] = grouped[f"{metric}_std"] / np.sqrt(np.clip(grouped["count"].astype(float), 1.0, None))
    return grouped


def plot_dataset_actions_vertical(
    dataset_action_df: pd.DataFrame,
    *,
    output_path: Path,
    all_rows: pd.DataFrame | None = None,
    direction: str = "all",
):
    plot_df = dataset_action_df.loc[
        (dataset_action_df["dataset_key"] != "all_datasets") & (dataset_action_df["direction"] == direction)
    ].copy()
    if plot_df.empty:
        return
    if all_rows is not None and not all_rows.empty:
        error_rows = _all_runs_error_frame(all_rows, metric="actions_used")
        if direction == "all":
            error_rows = (
                all_rows.groupby(["dataset_key", "method"], dropna=False)["actions_used"]
                .agg(["std", "count"])
                .reset_index()
                .rename(columns={"std": "actions_used_std"})
            )
            error_rows.insert(1, "direction", "all")
            error_rows["actions_used_std"] = error_rows["actions_used_std"].fillna(0.0)
            error_rows["actions_used_sem"] = error_rows["actions_used_std"] / np.sqrt(
                np.clip(error_rows["count"].astype(float), 1.0, None)
            )
        else:
            error_rows = error_rows.loc[error_rows["direction"] == direction].copy()
        plot_df = plot_df.merge(error_rows, on=["dataset_key", "direction", "method"], how="left")
    else:
        plot_df["actions_used_std"] = 0.0
        plot_df["actions_used_sem"] = 0.0
    order = plot_df.groupby("dataset_name")["mean_actions_used"].mean().sort_values().index.tolist()
    methods = ["influence_pair", "rigging"]
    fig, ax = plt.subplots(figsize=(max(7.4, 1.15 * len(order)), 4.4))
    x = np.arange(len(order), dtype=float)
    width = 0.34
    for idx, method in enumerate(methods):
        color = POLICY_COLORS[method]
        sub = (
            plot_df.loc[plot_df["method"] == method, ["dataset_name", "mean_actions_used", "actions_used_std"]]
            .set_index("dataset_name")
            .reindex(order)
        )
        values = sub["mean_actions_used"].to_numpy(dtype=float)
        errors = sub["actions_used_std"].fillna(0.0).to_numpy(dtype=float)
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            yerr=errors,
            capsize=4,
            error_kw={"elinewidth": 1.0, "ecolor": "#3c3c3c"},
            width=width,
            color=color,
            edgecolor="#222222",
            linewidth=0.8,
            label=method.replace("_", " "),
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([_dataset_short_label(name) for name in order], rotation=20, ha="right")
    ax.set_ylabel("Mean actions used per run")
    ax.set_title(
        "Per-dataset action count comparison" if direction == "all" else f"Per-dataset {direction} action count comparison",
        pad=10,
    )
    _style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout(pad=0.75)
    _savefig_both(fig, output_path)
    plt.close(fig)


def plot_overall_actions(dataset_action_df: pd.DataFrame, *, output_path: Path):
    plot_df = dataset_action_df.loc[dataset_action_df["dataset_key"] == "all_datasets"].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    methods = plot_df["method"].tolist()
    values = plot_df["mean_actions_used"].to_numpy(dtype=float)
    colors = [POLICY_COLORS[str(method)] for method in methods]
    bars = ax.bar(np.arange(len(methods)), values, color=colors, width=0.58, edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([method.replace("_", " ") for method in methods])
    ax.set_ylabel("Mean actions used per run")
    ax.set_title("All-dataset action comparison")
    _style_axis(ax, grid_axis="y")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _savefig_both(fig, output_path)
    plt.close(fig)


def replot_existing_outputs(output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    dataset_action_path = output_dir / "dataset_action_summary.csv"
    dataset_action_both_success_path = output_dir / "dataset_action_summary_both_success.csv"
    all_runs_path = output_dir / "all_runs.csv"
    if not summary_path.exists() or not dataset_action_path.exists():
        raise FileNotFoundError(f"Missing summary CSVs in {output_dir}")

    summary_df = pd.read_csv(summary_path)
    dataset_action_df = pd.read_csv(dataset_action_path)
    dataset_action_both_success_df = (
        pd.read_csv(dataset_action_both_success_path) if dataset_action_both_success_path.exists() else pd.DataFrame()
    )
    trial_summary_df = pd.read_csv(output_dir / "trial_summary.csv") if (output_dir / "trial_summary.csv").exists() else pd.DataFrame()
    all_rows = pd.read_csv(all_runs_path) if all_runs_path.exists() else pd.DataFrame()

    if not all_rows.empty:
        config_path = output_dir / "run_config.csv"
        budget = 120
        if config_path.exists():
            config_df = pd.read_csv(config_path)
            if "budget" in config_df.columns and not config_df.empty:
                budget = int(config_df["budget"].iloc[0])
        plot_paired_metric(all_rows, metric="exposures_seen", budget=budget, output_path=output_dir / "paired_exposures.pdf")
        plot_paired_metric(all_rows, metric="actions_used", budget=budget, output_path=output_dir / "paired_actions.pdf")
        plot_paired_metric(all_rows, metric="rows_added", budget=budget, output_path=output_dir / "paired_rows_added.pdf")

    plot_advantage(summary_df, output_path=output_dir / "median_exposure_advantage.pdf")
    plot_advantage_vertical(
        summary_df,
        output_path=output_dir / "median_exposure_advantage_vertical.pdf",
        trial_summary_df=trial_summary_df,
    )
    plot_dataset_actions(dataset_action_df, output_path=output_dir / "dataset_action_comparison.pdf")
    plot_dataset_actions_vertical(
        dataset_action_df,
        output_path=output_dir / "dataset_action_comparison_vertical.pdf",
        all_rows=all_rows,
    )
    plot_dataset_actions_vertical(
        dataset_action_df,
        output_path=output_dir / "dataset_action_comparison_promote.pdf",
        all_rows=all_rows,
        direction="promote",
    )
    plot_dataset_actions_vertical(
        dataset_action_df,
        output_path=output_dir / "dataset_action_comparison_demote.pdf",
        all_rows=all_rows,
        direction="demote",
    )
    plot_overall_actions(dataset_action_df, output_path=output_dir / "overall_action_comparison.pdf")
    if not dataset_action_both_success_df.empty:
        plot_dataset_actions(dataset_action_both_success_df, output_path=output_dir / "dataset_action_comparison_both_success.pdf")
        plot_dataset_actions_vertical(
            dataset_action_both_success_df,
            output_path=output_dir / "dataset_action_comparison_vertical_both_success.pdf",
            all_rows=None,
        )
        plot_dataset_actions_vertical(
            dataset_action_both_success_df,
            output_path=output_dir / "dataset_action_comparison_promote_both_success.pdf",
            all_rows=None,
            direction="promote",
        )
        plot_dataset_actions_vertical(
            dataset_action_both_success_df,
            output_path=output_dir / "dataset_action_comparison_demote_both_success.pdf",
            all_rows=None,
            direction="demote",
        )
        plot_overall_actions(dataset_action_both_success_df, output_path=output_dir / "overall_action_comparison_both_success.pdf")


def build_config_frame(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "datasets": " ".join(map(str, args.datasets)),
                "tasks_per_dataset": int(args.tasks_per_dataset),
                "n_targets_per_direction": int(args.n_targets_per_direction),
                "trials": int(args.trials),
                "budget": int(args.budget),
                "rigging_mode": str(args.rigging_mode),
                "beta": float(args.beta),
                "seed_base": int(args.seed),
                "k_mode": "fixed" if args.k is not None else "stratified_random_three_regions",
                "fixed_k": int(args.k) if args.k is not None else np.nan,
            }
        ]
    )


def load_existing_checkpoint(
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], set[str]]:
    all_runs_path = output_dir / "all_runs.csv"
    tasks_path = output_dir / "tasks.csv"
    decision_path = output_dir / "decision_log.csv"
    progress_path = output_dir / "dataset_progress.csv"

    run_rows = pd.read_csv(all_runs_path).to_dict("records") if all_runs_path.exists() else []
    task_rows = pd.read_csv(tasks_path).to_dict("records") if tasks_path.exists() else []
    decision_rows = pd.read_csv(decision_path).to_dict("records") if decision_path.exists() else []
    progress_rows = pd.read_csv(progress_path).to_dict("records") if progress_path.exists() else []

    completed_dataset_keys = {
        str(row["dataset_key"])
        for row in progress_rows
        if str(row.get("status", "")) == "completed" and pd.notna(row.get("dataset_key"))
    }
    return run_rows, task_rows, decision_rows, progress_rows, completed_dataset_keys


def materialize_outputs(
    *,
    run_rows: list[dict[str, object]],
    task_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    budget: int,
) -> dict[str, pd.DataFrame]:
    all_rows = pd.DataFrame(run_rows).sort_values(["dataset_key", "direction", "initial_rank", "trial", "method"]).reset_index(drop=True)
    task_df = pd.DataFrame(task_rows).drop_duplicates().sort_values(["dataset_key", "direction", "initial_rank"]).reset_index(drop=True)
    decision_df = pd.DataFrame(decision_rows).sort_values(["dataset_key", "direction", "initial_rank", "trial", "exposure_index"]).reset_index(drop=True)
    summary_df = summarize_comparison(all_rows, budget=budget)
    trial_summary_df = summarize_trial_outcomes(all_rows)
    agreement_df = summarize_decision_agreement(decision_df)
    dataset_action_df = summarize_dataset_actions(all_rows, trial_summary_df)
    dataset_action_both_success_df = summarize_dataset_actions_both_success(all_rows, trial_summary_df)
    return {
        "all_rows": all_rows,
        "task_df": task_df,
        "decision_df": decision_df,
        "summary_df": summary_df,
        "trial_summary_df": trial_summary_df,
        "agreement_df": agreement_df,
        "dataset_action_df": dataset_action_df,
        "dataset_action_both_success_df": dataset_action_both_success_df,
    }


def build_aggregate_frame(
    *,
    summary_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    trial_summary_df: pd.DataFrame,
    trials: int,
    budget: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_tasks": int(len(summary_df)),
                "n_trials_per_task": int(trials),
                "budget": int(budget),
                "influence_mean_success_rate": float(summary_df["influence_success_rate"].mean()) if len(summary_df) else float("nan"),
                "rigging_mean_success_rate": float(summary_df["rigging_success_rate"].mean()) if len(summary_df) else float("nan"),
                "mean_median_exposure_advantage": float(summary_df["median_exposure_advantage_rigging_minus_influence"].mean()) if len(summary_df) else float("nan"),
                "mean_prob_rigging_exposures_ge_influence": float(summary_df["p_rigging_exposures_ge_influence"].mean()) if len(summary_df) else float("nan"),
                "mean_same_decision_rate_when_both_active": float(agreement_df["same_decision_rate_when_both_active"].mean()) if len(agreement_df) else float("nan"),
                "influence_successful_trials": int(trial_summary_df["influence_success"].sum()) if len(trial_summary_df) else 0,
                "rigging_successful_trials": int(trial_summary_df["rigging_success"].sum()) if len(trial_summary_df) else 0,
                "both_successful_trials": int((trial_summary_df["influence_success"] & trial_summary_df["rigging_success"]).sum()) if len(trial_summary_df) else 0,
                "influence_trial_wins": int((trial_summary_df["trial_winner"] == "influence_pair").sum()) if len(trial_summary_df) else 0,
                "rigging_trial_wins": int((trial_summary_df["trial_winner"] == "rigging").sum()) if len(trial_summary_df) else 0,
                "tied_trials": int((trial_summary_df["trial_winner"] == "tie").sum()) if len(trial_summary_df) else 0,
            }
        ]
    )


def write_outputs(
    *,
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    config_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    budget: int,
    trials: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = frames["all_rows"]
    task_df = frames["task_df"]
    decision_df = frames["decision_df"]
    summary_df = frames["summary_df"]
    trial_summary_df = frames["trial_summary_df"]
    agreement_df = frames["agreement_df"]
    dataset_action_df = frames["dataset_action_df"]
    dataset_action_both_success_df = frames["dataset_action_both_success_df"]
    target_summary_df = summarize_targets(all_rows, trial_summary_df, budget=budget)
    aggregate_df = build_aggregate_frame(
        summary_df=summary_df,
        agreement_df=agreement_df,
        trial_summary_df=trial_summary_df,
        trials=trials,
        budget=budget,
    )

    all_rows.to_csv(output_dir / "all_runs.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    task_df.to_csv(output_dir / "tasks.csv", index=False)
    decision_df.to_csv(output_dir / "decision_log.csv", index=False)
    trial_summary_df.to_csv(output_dir / "trial_summary.csv", index=False)
    agreement_df.to_csv(output_dir / "decision_agreement_summary.csv", index=False)
    dataset_action_df.to_csv(output_dir / "dataset_action_summary.csv", index=False)
    dataset_action_both_success_df.to_csv(output_dir / "dataset_action_summary_both_success.csv", index=False)
    target_summary_df.to_csv(output_dir / "target_summary.csv", index=False)
    config_df.to_csv(output_dir / "run_config.csv", index=False)
    progress_df.to_csv(output_dir / "dataset_progress.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate.csv", index=False)

    plot_paired_metric(all_rows, metric="exposures_seen", budget=budget, output_path=output_dir / "paired_exposures.pdf")
    plot_paired_metric(all_rows, metric="actions_used", budget=budget, output_path=output_dir / "paired_actions.pdf")
    plot_paired_metric(all_rows, metric="rows_added", budget=budget, output_path=output_dir / "paired_rows_added.pdf")
    plot_advantage(summary_df, output_path=output_dir / "median_exposure_advantage.pdf")
    plot_dataset_actions(dataset_action_df, output_path=output_dir / "dataset_action_comparison.pdf")
    plot_dataset_actions_vertical(
        dataset_action_df,
        output_path=output_dir / "dataset_action_comparison_vertical.pdf",
        all_rows=all_rows,
    )
    plot_dataset_actions_vertical(
        dataset_action_df,
        output_path=output_dir / "dataset_action_comparison_promote.pdf",
        all_rows=all_rows,
        direction="promote",
    )
    plot_dataset_actions_vertical(
        dataset_action_df,
        output_path=output_dir / "dataset_action_comparison_demote.pdf",
        all_rows=all_rows,
        direction="demote",
    )
    plot_overall_actions(dataset_action_df, output_path=output_dir / "overall_action_comparison.pdf")
    plot_dataset_actions(dataset_action_both_success_df, output_path=output_dir / "dataset_action_comparison_both_success.pdf")
    plot_dataset_actions_vertical(
        dataset_action_both_success_df,
        output_path=output_dir / "dataset_action_comparison_vertical_both_success.pdf",
        all_rows=None,
    )
    plot_dataset_actions_vertical(
        dataset_action_both_success_df,
        output_path=output_dir / "dataset_action_comparison_promote_both_success.pdf",
        all_rows=None,
        direction="promote",
    )
    plot_dataset_actions_vertical(
        dataset_action_both_success_df,
        output_path=output_dir / "dataset_action_comparison_demote_both_success.pdf",
        all_rows=None,
        direction="demote",
    )
    plot_overall_actions(dataset_action_both_success_df, output_path=output_dir / "overall_action_comparison_both_success.pdf")


def write_per_target_outputs(
    *,
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    config_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    budget: int,
    trials: int,
) -> None:
    task_df = frames["task_df"]
    for _, task in task_df.iterrows():
        mask = (
            (frames["all_rows"]["dataset_key"] == task["dataset_key"])
            & (frames["all_rows"]["direction"] == task["direction"])
            & (frames["all_rows"]["target_player"] == task["target_player"])
            & (frames["all_rows"]["k"] == task["k"])
        )
        decision_mask = (
            (frames["decision_df"]["dataset_key"] == task["dataset_key"])
            & (frames["decision_df"]["direction"] == task["direction"])
            & (frames["decision_df"]["target_player"] == task["target_player"])
            & (frames["decision_df"]["k"] == task["k"])
        )
        target_frames = materialize_outputs(
            run_rows=frames["all_rows"].loc[mask].to_dict("records"),
            task_rows=frames["task_df"].loc[
                (frames["task_df"]["dataset_key"] == task["dataset_key"])
                & (frames["task_df"]["direction"] == task["direction"])
                & (frames["task_df"]["target_player"] == task["target_player"])
                & (frames["task_df"]["k"] == task["k"])
            ].to_dict("records"),
            decision_rows=frames["decision_df"].loc[decision_mask].to_dict("records"),
            budget=budget,
        )
        target_dir = (
            output_dir
            / "per_target"
            / str(task["dataset_key"])
            / f"{task['direction']}__r{int(task['initial_rank'])}__k{int(task['k'])}__{_slugify_target(str(task['target_player']))}"
        )
        write_outputs(
            output_dir=target_dir,
            frames=target_frames,
            config_df=config_df,
            progress_df=progress_df.loc[progress_df["dataset_key"] == task["dataset_key"]].reset_index(drop=True),
            budget=budget,
            trials=trials,
        )


def main() -> int:
    args = parse_args()
    use_paper_rc()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_from_csv:
        replot_existing_outputs(args.output_dir)
        print(f"Rebuilt plots from existing CSV outputs in {args.output_dir}", flush=True)
        return 0

    run_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    progress_rows: list[dict[str, object]] = []
    config_df = build_config_frame(args)
    completed_dataset_keys: set[str] = set()

    if args.resume_existing:
        run_rows, task_rows, decision_rows, progress_rows, completed_dataset_keys = load_existing_checkpoint(args.output_dir)
        if completed_dataset_keys:
            print(
                "Resuming from existing checkpoint with completed datasets: "
                + ", ".join(sorted(completed_dataset_keys)),
                flush=True,
            )

    for dataset_index, dataset_key in enumerate(args.datasets, start=1):
        dataset_key = str(dataset_key)
        if dataset_key in completed_dataset_keys:
            print(f"Skipping already completed dataset {dataset_index}/{len(args.datasets)}: {dataset_key}", flush=True)
            continue

        try:
            built = build_named_dataset_model(dataset_key)
        except Exception as exc:
            if not args.skip_failed_datasets:
                raise
            progress_rows = [row for row in progress_rows if str(row.get("dataset_key", "")) != dataset_key]
            progress_rows.append(
                {
                    "dataset_index": dataset_index,
                    "dataset_key": dataset_key,
                    "dataset_name": dataset_key,
                    "status": "failed",
                    "n_tasks_completed": 0,
                    "n_datasets_total": int(len(args.datasets)),
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
            progress_df = pd.DataFrame(progress_rows)
            if run_rows:
                frames = materialize_outputs(
                    run_rows=run_rows,
                    task_rows=task_rows,
                    decision_rows=decision_rows,
                    budget=args.budget,
                )
                write_outputs(
                    output_dir=args.output_dir,
                    frames=frames,
                    config_df=config_df,
                    progress_df=progress_df,
                    budget=args.budget,
                    trials=args.trials,
                )
            else:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                config_df.to_csv(args.output_dir / "run_config.csv", index=False)
                progress_df.to_csv(args.output_dir / "dataset_progress.csv", index=False)
            print(
                f"Skipped dataset {dataset_index}/{len(args.datasets)} due to error: {dataset_key} ({type(exc).__name__}: {exc})",
                flush=True,
            )
            continue

        bt_model = built["bt_model"]
        dataset_name = str(built["dataset_name"])
        ranking = ranking_frame(bt_model)
        if args.k is not None:
            selected_ks = [int(args.k)]
        else:
            k_rng = np.random.default_rng(args.seed + dataset_index)
            selected_ks = choose_stratified_random_ks(
                ranking,
                n_targets_per_direction=args.n_targets_per_direction,
                rng=k_rng,
            )

        dataset_task_count = 0
        for k in selected_ks:
            dataset_tasks = select_tasks(
                boundary_targets(ranking, k=k, n_targets_per_direction=args.n_targets_per_direction),
                k=k,
                max_tasks=args.tasks_per_dataset,
            )
            dataset_task_count += len(dataset_tasks)
            for task in dataset_tasks:
                target_player = str(task["target_player"])
                direction = str(task["direction"])
                task_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset_name": dataset_name,
                        "k": k,
                        "direction": direction,
                        "target_player": target_player,
                        "initial_rank": int(task["initial_rank"]),
                    }
                )
                for trial in range(args.trials):
                    trial_seed = args.seed + trial
                    rigging_result, influence_result, decision_records = run_paired_trial(
                        bt_model,
                        dataset_key=str(dataset_key),
                        dataset_name=dataset_name,
                        target_player=target_player,
                        k=k,
                        direction=direction,
                        rigging_mode=args.rigging_mode,
                        beta=args.beta,
                        budget=args.budget,
                        seed=trial_seed,
                    )
                    run_rows.append(rigging_result.as_dict())
                    run_rows.append(influence_result.as_dict())
                    initial_rank = int(task["initial_rank"])
                    for record in decision_records:
                        row = record.as_dict()
                        row["initial_rank"] = initial_rank
                        decision_rows.append(row)

        progress_rows = [row for row in progress_rows if str(row.get("dataset_key", "")) != dataset_key]
        progress_rows.append(
            {
                "dataset_index": dataset_index,
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "status": "completed",
                "n_tasks_completed": int(dataset_task_count),
                "n_datasets_total": int(len(args.datasets)),
                "error_message": "",
            }
        )
        completed_dataset_keys.add(dataset_key)
        progress_df = pd.DataFrame(progress_rows)
        frames = materialize_outputs(run_rows=run_rows, task_rows=task_rows, decision_rows=decision_rows, budget=args.budget)
        write_outputs(
            output_dir=args.output_dir,
            frames=frames,
            config_df=config_df,
            progress_df=progress_df,
            budget=args.budget,
            trials=args.trials,
        )
        write_per_target_outputs(
            output_dir=args.output_dir,
            frames=frames,
            config_df=config_df,
            progress_df=progress_df,
            budget=args.budget,
            trials=args.trials,
        )
        dataset_output_dir = args.output_dir / "per_dataset" / str(dataset_key)
        dataset_mask_rows = frames["all_rows"]["dataset_key"] == dataset_key
        dataset_mask_tasks = frames["task_df"]["dataset_key"] == dataset_key
        dataset_mask_decisions = frames["decision_df"]["dataset_key"] == dataset_key
        dataset_frames = materialize_outputs(
            run_rows=frames["all_rows"].loc[dataset_mask_rows].to_dict("records"),
            task_rows=frames["task_df"].loc[dataset_mask_tasks].to_dict("records"),
            decision_rows=frames["decision_df"].loc[dataset_mask_decisions].to_dict("records"),
            budget=args.budget,
        )
        write_outputs(
            output_dir=dataset_output_dir,
            frames=dataset_frames,
            config_df=config_df,
            progress_df=progress_df.loc[progress_df["dataset_key"] == dataset_key].reset_index(drop=True),
            budget=args.budget,
            trials=args.trials,
        )
        write_per_target_outputs(
            output_dir=dataset_output_dir,
            frames=dataset_frames,
            config_df=config_df,
            progress_df=progress_df.loc[progress_df["dataset_key"] == dataset_key].reset_index(drop=True),
            budget=args.budget,
            trials=args.trials,
        )
        print(f"Checkpointed outputs after dataset {dataset_index}/{len(args.datasets)}: {dataset_key}")

    if run_rows:
        progress_df = pd.DataFrame(progress_rows)
        frames = materialize_outputs(run_rows=run_rows, task_rows=task_rows, decision_rows=decision_rows, budget=args.budget)
        write_outputs(
            output_dir=args.output_dir,
            frames=frames,
            config_df=config_df,
            progress_df=progress_df,
            budget=args.budget,
            trials=args.trials,
        )
        write_per_target_outputs(
            output_dir=args.output_dir,
            frames=frames,
            config_df=config_df,
            progress_df=progress_df,
            budget=args.budget,
            trials=args.trials,
        )

    print(f"Saved final outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
