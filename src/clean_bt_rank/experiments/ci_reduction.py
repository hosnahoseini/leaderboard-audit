from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..actions import build_add_candidates, compute_influence
from ..bt_model import BradleyTerryModel
from ..ci import compute_confidence_intervals
from ..objectives import PlayerUncertaintyObjective, compute_objective_action_influence
from .arena_active_sampling import ArenaActiveSamplingBaseline
from ._common import ExperimentState, apply_add_candidate, fit_model_from_state, make_state


@dataclass
class CIReductionResult:
    policy: str
    target_player: str
    ci_method: str
    selection_objective: str
    selection_mode: str
    budget: int
    history: pd.DataFrame
    selected_actions: pd.DataFrame

    def summary_dict(self) -> dict[str, object]:
        final = self.history.sort_values("step").iloc[-1]
        initial = self.history.sort_values("step").iloc[0]
        return {
            "policy": self.policy,
            "target_player": self.target_player,
            "ci_method": self.ci_method,
            "selection_mode": self.selection_mode,
            "budget": self.budget,
            "initial_ci_width": float(initial["ci_width"]),
            "final_ci_width": float(final["ci_width"]),
            "ci_width_reduction": float(initial["ci_width"] - final["ci_width"]),
            "initial_variance_proxy": float(initial["variance_proxy"]),
            "final_variance_proxy": float(final["variance_proxy"]),
            "variance_proxy_reduction": float(initial["variance_proxy"] - final["variance_proxy"]),
        }


@dataclass(frozen=True)
class PolicySpec:
    policy: str
    candidate_mode: str
    pair_mode: bool
    add_argmax_label: bool


POLICY_ALIASES: dict[str, PolicySpec] = {
    "influence_pairs": PolicySpec(
        policy="influence",
        candidate_mode="all_pairs",
        pair_mode=True,
        add_argmax_label=True,
    ),
    "influence_all_outcomes": PolicySpec(
        policy="influence",
        candidate_mode="all_outcomes",
        pair_mode=True,
        add_argmax_label=True,
    ),
    "influence_weighted_outcomes": PolicySpec(
        policy="influence",
        candidate_mode="all_outcomes_weighted",
        pair_mode=True,
        add_argmax_label=True,
    ),
    "influence_expected_pair": PolicySpec(
        policy="expected_pair",
        candidate_mode="all_pairs",
        pair_mode=True,
        add_argmax_label=True,
    ),
    "influence_v1": PolicySpec(
        policy="influence",
        candidate_mode="all_pairs",
        pair_mode=True,
        add_argmax_label=True,
    ),
    "influence_v2": PolicySpec(
        policy="influence",
        candidate_mode="all_outcomes",
        pair_mode=True,
        add_argmax_label=True,
    ),
    "influence_v3": PolicySpec(
        policy="influence",
        candidate_mode="all_outcomes_weighted",
        pair_mode=True,
        add_argmax_label=True,
    ),
    "influence_v4": PolicySpec(
        policy="expected_pair",
        candidate_mode="all_pairs",
        pair_mode=True,
        add_argmax_label=True,
    ),
}


def _resolve_policy_spec(policy: str, candidate_mode: str) -> PolicySpec:
    normalized_policy = policy.lower()
    if normalized_policy in POLICY_ALIASES:
        return POLICY_ALIASES[normalized_policy]
    return PolicySpec(
        policy=normalized_policy,
        candidate_mode=candidate_mode.lower(),
        pair_mode=candidate_mode.lower() == "all_pairs" or normalized_policy in {"expected_pair", "random_pair", "arena_active_pair"},
        add_argmax_label=False,
    )


def run_ci_reduction_experiment(
    bt_model: BradleyTerryModel,
    *,
    target_player: str,
    policy: str = "influence",
    budget: int = 10,
    influence_method: str = "1sn",
    ci_method: str = "gao_local",
    arena_active_mode: str = "target_only",
    random_seed: int = 0,
    candidate_mode: str = "all_pairs",
    outcome_mode: str = "deterministic",
) -> CIReductionResult:
    """
    Reduce a target player's uncertainty by iteratively adding rows.

    Policies:
    - `influence`: fixed initial influence ranking computed once on the fitted input model
    - `influence_pairs`: influence over unordered pairs with argmax-skill winner labels
    - `influence_all_outcomes`: influence over both outcomes per pair, but add argmax-skill winner labels
    - `influence_weighted_outcomes`: same as `influence_all_outcomes`, weighted by BT outcome probability
    - `influence_expected_pair`: pair score `p(win) * I(win) + p(loss) * I(loss)`, then add argmax-skill winner labels
    - `random`: sample uniformly from the candidate pool
    - `active_ranking`: highest logistic variance p(1-p)
    - `arena_active`: Chatbot Arena pair-variance / pair-count heuristic
    - `expected_pair`: active-learning pair rule using expected influence over both outcomes
    - `random_pair`: sample an unordered pair, then sample its realized outcome
    - `arena_active_pair`: Arena active pair rule, then sample its realized outcome

    candidate_mode controls the add-action space for influence and random:
    - `all_pairs`: one candidate per unordered pair (expected winner, outcome=1)
    - `all_outcomes`: one candidate per ordered pair (both directions, outcome=1)
    - `all_outcomes_weighted`: same as all_outcomes but influence weighted by win probability
    arena_active always uses all_pairs internally.
    outcome_mode controls how pair-based policies realize the match outcome:
    - `stochastic`: Bernoulli sample from the current BT pair win probability
    - `deterministic`: set the outcome to the BT argmax winner
    """
    bt_model._require_fit()
    requested_policy = policy.lower()
    policy_spec = _resolve_policy_spec(requested_policy, candidate_mode)
    policy = policy_spec.policy
    candidate_mode = policy_spec.candidate_mode
    valid_policies = {
        "influence",
        "random",
        "active_ranking",
        "arena_active",
        "expected_pair",
        "random_pair",
        "arena_active_pair",
    }
    if policy not in valid_policies:
        raise ValueError(f"policy must be one of {sorted(valid_policies)}.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if candidate_mode not in {"all_pairs", "all_outcomes", "all_outcomes_weighted"}:
        raise ValueError("candidate_mode must be 'all_pairs', 'all_outcomes', or 'all_outcomes_weighted'.")
    if policy in {"expected_pair", "random_pair", "arena_active_pair"} and candidate_mode != "all_pairs":
        raise ValueError(f"policy='{policy}' requires candidate_mode='all_pairs'.")
    outcome_mode = outcome_mode.lower()
    if outcome_mode not in {"stochastic", "deterministic"}:
        raise ValueError("outcome_mode must be 'stochastic' or 'deterministic'.")

    rng = np.random.default_rng(random_seed)
    objective = PlayerUncertaintyObjective(target_player)
    arena_active = ArenaActiveSamplingBaseline(mode=arena_active_mode)
    current_model = bt_model
    state = make_state(bt_model)
    history_rows = [_ci_history_row(current_model, target_player, objective, ci_method, step=0, policy=requested_policy)]
    selected_rows: list[dict[str, object]] = []
    used_keys: set[tuple[str, str]] = set()
    fixed_influence_report: pd.DataFrame | None = None

    if policy == "influence":
        fixed_influence_report = compute_influence(
            current_model,
            objective,
            action="add",
            method=influence_method,
            candidate_mode=candidate_mode,
        ).sort_values("influence", ascending=True, kind="stable").reset_index(drop=True)

    for step in range(1, budget + 1):
        if policy == "influence":
            assert fixed_influence_report is not None
            report = fixed_influence_report
            report = _filter_used_add_candidates(report, used_keys, pair_mode=policy_spec.pair_mode)
            if report.empty:
                break
            chosen = report.iloc[0].copy()
            chosen["selection_score"] = float(chosen["influence"])
        elif policy in {"random", "active_ranking"}:
            candidates = build_add_candidates(current_model, mode=candidate_mode)
            report = candidates.frame.copy()
            report["_candidate_x"] = list(np.asarray(candidates.X, dtype=float))
            report["influence"] = np.nan
            report = _filter_used_add_candidates(report, used_keys, pair_mode=policy_spec.pair_mode)
            if report.empty:
                break
            if policy == "active_ranking":
                report["selection_score"] = report["candidate_probability"] * (1.0 - report["candidate_probability"])
                chosen = report.sort_values("selection_score", ascending=False).iloc[0].copy()
            else:
                chosen = report.iloc[int(rng.integers(0, len(report)))].copy()
                chosen["selection_score"] = np.nan
        elif policy == "expected_pair":
            report = _expected_pair_influence_report(current_model, objective, influence_method=influence_method, used_keys=used_keys)
            if report.empty:
                break
            chosen = report.sort_values("expected_influence", ascending=True).iloc[0].copy()
            chosen["selection_score"] = float(chosen["expected_influence"])
            chosen = _realize_sampled_pair_outcome(chosen, rng, outcome_mode=outcome_mode)
        elif policy == "random_pair":
            report = _pair_candidate_report(current_model, used_keys=used_keys)
            if report.empty:
                break
            chosen = report.iloc[int(rng.integers(0, len(report)))].copy()
            chosen["selection_score"] = np.nan
            chosen = _realize_sampled_pair_outcome(chosen, rng, outcome_mode=outcome_mode)
        else:  # arena_active / arena_active_pair
            report = arena_active.score_candidates(current_model, target_player=target_player, used_keys=used_keys)
            if report.empty:
                break
            chosen = report.sort_values("arena_active_score", ascending=False).iloc[0].copy()
            chosen["selection_score"] = float(chosen["arena_active_score"])
            if policy == "arena_active_pair":
                chosen = _realize_sampled_pair_outcome(chosen, rng, outcome_mode=outcome_mode)

        if policy_spec.add_argmax_label:
            chosen = _canonicalize_to_pair_argmax(chosen)
        used_keys.add(_used_candidate_key(chosen, pair_mode=policy_spec.pair_mode))
        selected_rows.append(_selected_action_row(chosen, step=step, policy=requested_policy))
        state = apply_add_candidate(state, chosen)
        current_model = fit_model_from_state(bt_model, state)
        history_rows.append(_ci_history_row(current_model, target_player, objective, ci_method, step=step, policy=requested_policy))

    return CIReductionResult(
        policy=requested_policy,
        target_player=target_player,
        ci_method=ci_method,
        selection_objective=objective.name,
        selection_mode=arena_active_mode if policy in {"arena_active", "arena_active_pair"} else candidate_mode,
        budget=budget,
        history=pd.DataFrame(history_rows),
        selected_actions=pd.DataFrame(selected_rows),
    )


def run_ci_reduction_benchmark(
    bt_model: BradleyTerryModel,
    *,
    target_player: str,
    budget: int = 10,
    influence_method: str = "1sn",
    ci_method: str = "gao_local",
    arena_active_mode: str = "target_only",
    n_random_trials: int = 5,
    random_seed: int = 0,
    policies: list[str] | None = None,
    candidate_mode: str = "all_pairs",
    random_policy: str = "random",
    outcome_mode: str = "deterministic",
) -> tuple[dict[str, CIReductionResult | list[CIReductionResult]], pd.DataFrame, pd.DataFrame]:
    """
    Run the CI reduction benchmark.

    ``policies``: deterministic policies to run (default ``["influence", "arena_active"]``).
    ``candidate_mode``: add-action candidate space for influence and random
      (``"all_pairs"``, ``"all_outcomes"``, ``"all_outcomes_weighted"``).
      arena_active always uses ``all_pairs`` internally.
    ``random_policy``: stochastic baseline family to run repeatedly. Use
      ``"random"`` for legacy labeled-add runs or ``"random_pair"`` for the
      active-learning pair benchmark.
    """
    _DEFAULT_POLICIES = ["influence", "arena_active"]
    run_policies = [p.lower() for p in (policies if policies is not None else _DEFAULT_POLICIES)]
    random_policy = random_policy.lower()

    results: dict[str, CIReductionResult | list[CIReductionResult]] = {}
    for policy in run_policies:
        results[policy] = run_ci_reduction_experiment(
            bt_model,
            target_player=target_player,
            policy=policy,
            budget=budget,
            influence_method=influence_method,
            ci_method=ci_method,
            arena_active_mode=arena_active_mode,
            random_seed=random_seed,
            candidate_mode=candidate_mode,
            outcome_mode=outcome_mode,
        )

    random_results = [
        run_ci_reduction_experiment(
            bt_model,
            target_player=target_player,
            policy=random_policy,
            budget=budget,
            influence_method=influence_method,
            ci_method=ci_method,
            arena_active_mode=arena_active_mode,
            random_seed=random_seed + trial,
            candidate_mode=candidate_mode,
            outcome_mode=outcome_mode,
        )
        for trial in range(n_random_trials)
    ]
    results[random_policy] = random_results

    history = pd.concat(
        [
            *[_attach_trial(results[p].history, policy=p, trial=0) for p in run_policies],
            *[_attach_trial(result.history, policy=random_policy, trial=trial) for trial, result in enumerate(random_results)],
        ],
        ignore_index=True,
    )
    summary_rows = [
        *[results[p].summary_dict() for p in run_policies],
        *[result.summary_dict() | {"trial": trial} for trial, result in enumerate(random_results)],
    ]
    return results, pd.DataFrame(summary_rows), history


def _filter_used_add_candidates(
    report: pd.DataFrame,
    used_keys: set[tuple[str, str]],
    *,
    pair_mode: bool,
) -> pd.DataFrame:
    filtered = report.copy()
    candidate_keys = [_used_candidate_key(row, pair_mode=pair_mode) for _, row in filtered.iterrows()]
    keep = np.array([key not in used_keys for key in candidate_keys], dtype=bool)
    return filtered.loc[keep].reset_index(drop=True)


def _pair_candidate_report(
    bt_model: BradleyTerryModel,
    *,
    used_keys: set[tuple[str, str]],
) -> pd.DataFrame:
    candidates = build_add_candidates(bt_model, mode="all_pairs")
    report = candidates.frame.copy()
    report["_candidate_x"] = list(np.asarray(candidates.X, dtype=float))
    return _filter_used_add_candidates(report, used_keys, pair_mode=True)


def _expected_pair_influence_report(
    bt_model: BradleyTerryModel,
    objective: PlayerUncertaintyObjective,
    *,
    influence_method: str,
    used_keys: set[tuple[str, str]],
) -> pd.DataFrame:
    pair_report = _pair_candidate_report(bt_model, used_keys=used_keys)
    if pair_report.empty:
        return pair_report
    x_new = np.vstack(pair_report["_candidate_x"].to_list()).astype(float)
    influence_if_win = compute_objective_action_influence(
        bt_model,
        objective,
        action="add",
        method=influence_method,
        X_new=x_new,
        y_new=np.ones(len(pair_report), dtype=float),
    )
    influence_if_loss = _compute_zero_outcome_influence(bt_model, objective, x_new=x_new, method=influence_method)
    pair_report["influence_if_model_a_wins"] = influence_if_win
    pair_report["influence_if_model_b_wins"] = influence_if_loss
    prob = pair_report["candidate_probability"].to_numpy(dtype=float)
    pair_report["expected_influence"] = (
        prob * pair_report["influence_if_model_a_wins"].to_numpy(dtype=float)
        + (1.0 - prob) * pair_report["influence_if_model_b_wins"].to_numpy(dtype=float)
    )
    return pair_report


def _compute_zero_outcome_influence(
    bt_model: BradleyTerryModel,
    objective: PlayerUncertaintyObjective,
    *,
    x_new: np.ndarray,
    method: str,
) -> np.ndarray:
    return compute_objective_action_influence(
        bt_model,
        objective,
        action="add",
        method=method,
        X_new=x_new,
        y_new=np.zeros(x_new.shape[0], dtype=float),
    )


def _used_candidate_key(row: pd.Series, *, pair_mode: bool) -> tuple[str, str]:
    model_a = str(row["model_a"])
    model_b = str(row["model_b"])
    if not pair_mode:
        return model_a, model_b
    return tuple(sorted((model_a, model_b)))


def _realize_sampled_pair_outcome(
    chosen: pd.Series,
    rng: np.random.Generator,
    *,
    outcome_mode: str,
) -> pd.Series:
    realized = chosen.copy()
    prob = float(realized["candidate_probability"])
    if outcome_mode == "stochastic":
        outcome = float(rng.random() < prob)
    elif outcome_mode == "deterministic":
        outcome = float(prob >= 0.5)
    else:
        raise ValueError(f"Unsupported outcome_mode {outcome_mode!r}.")
    realized["outcome_mode"] = outcome_mode
    realized["outcome_probability"] = prob
    realized["realized_outcome"] = outcome
    realized["outcome"] = outcome
    return realized


def _canonicalize_to_pair_argmax(chosen: pd.Series) -> pd.Series:
    realized = chosen.copy()
    realized["selected_model_a"] = str(chosen["model_a"])
    realized["selected_model_b"] = str(chosen["model_b"])
    realized["selected_outcome"] = float(chosen.get("outcome", 1.0))
    prob = float(realized.get("candidate_probability", 0.5))
    forward_x = np.asarray(realized["_candidate_x"], dtype=float)
    if prob >= 0.5:
        realized["outcome_probability"] = prob
        realized["realized_outcome"] = 1.0
        realized["outcome_mode"] = "deterministic_argmax"
        realized["outcome"] = 1.0
        return realized

    realized["model_a"] = chosen["model_b"]
    realized["model_b"] = chosen["model_a"]
    realized["_candidate_x"] = -forward_x
    realized["candidate_probability"] = 1.0 - prob
    realized["outcome_probability"] = 1.0 - prob
    realized["realized_outcome"] = 1.0
    realized["outcome_mode"] = "deterministic_argmax"
    realized["outcome"] = 1.0
    return realized


def _ci_history_row(
    model: BradleyTerryModel,
    target_player: str,
    objective: PlayerUncertaintyObjective,
    ci_method: str,
    *,
    step: int,
    policy: str,
) -> dict[str, object]:
    ci_frame = compute_confidence_intervals(model, method=ci_method).to_frame(model.competitor_names_)
    target_row = ci_frame.loc[ci_frame["competitor"] == target_player].iloc[0]
    ci_width = float(target_row["ci_upper"] - target_row["ci_lower"])
    skill_range = float(ci_frame["rating"].max() - ci_frame["rating"].min())
    return {
        "step": step,
        "policy": policy,
        "target_player": target_player,
        "ci_method": ci_method,
        "ci_width": ci_width,
        "ci_width_normalized": ci_width / skill_range if skill_range > 0 else float("nan"),
        "skill_range": skill_range,
        "standard_error": float(target_row["standard_error"]),
        "variance_proxy": float(objective.value(model)),
        "rating": float(target_row["rating"]),
        "rank": int(ci_frame.index[ci_frame["competitor"] == target_player][0]) + 1,
    }


def _selected_action_row(chosen: pd.Series, *, step: int, policy: str) -> dict[str, object]:
    row = chosen.to_dict()
    row["step"] = step
    row["policy"] = policy
    return row


def _attach_trial(frame: pd.DataFrame, *, policy: str, trial: int) -> pd.DataFrame:
    out = frame.copy()
    out["policy"] = policy
    out["trial"] = trial
    return out
