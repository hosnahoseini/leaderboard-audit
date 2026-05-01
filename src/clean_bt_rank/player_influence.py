from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from .bt_model import BradleyTerryModel
from .datasets import BattleDataset
from .objectives import KendallTauObjective, Objective, compute_objective_action_influence
from .reporting import make_objective_influence_report


DEFAULT_PLAYER_FEATURES: tuple[str, ...] = (
    "skill",
    "degree",
    "influence",
    "bridge_var",
    "closeness",
    "surprise",
)


class _SubsetSmoothKendallObjective(Objective):
    def __init__(self, *, ranking: Iterable[int | str], temperature: float) -> None:
        self.ranking = list(ranking)
        self.temperature = float(temperature)

    @property
    def name(self) -> str:
        return "kendall_tau_subset"

    def value(self, bt_model: BradleyTerryModel) -> float:
        return smooth_kendall_tau_value_from_model(
            bt_model,
            ranking=self.ranking,
            temperature=self.temperature,
        )

    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        raise NotImplementedError("Subset player-drop objective is only used for value-based approximations.")


@dataclass(frozen=True)
class PlayerDropResult:
    player: str
    skill: float
    predicted_delta_sum: float
    predicted_delta_joint_newton: float
    predicted_delta_two_step: float
    actual_smooth_delta: float
    actual_rank_tau: float
    actual_rank_tau_delta: float
    baseline_smooth_value: float
    refit_smooth_value: float
    remaining_players: tuple[str, ...]


def ranking_from_model(bt_model: BradleyTerryModel) -> list[str]:
    bt_model._require_fit()
    order = np.argsort(-bt_model.full_skills())
    return [bt_model.competitor_names_[idx] for idx in order]


def kendall_tau_match_influence_report(
    bt_model: BradleyTerryModel,
    *,
    ranking: Iterable[int | str] | None = None,
    temperature: float = 0.5,
    method: str = "1sn",
) -> pd.DataFrame:
    objective = KendallTauObjective(
        ranking=list(ranking) if ranking is not None else ranking_from_model(bt_model),
        temperature=temperature,
    )
    influence = compute_objective_action_influence(
        bt_model,
        objective,
        action="drop",
        method=method,
    )
    return make_objective_influence_report(bt_model, objective, influence, influence_name="influence")


def compute_player_influence(
    bt_model: BradleyTerryModel,
    *,
    match_influence_report: pd.DataFrame | None = None,
    ranking: Iterable[int | str] | None = None,
    temperature: float = 0.5,
    method: str = "1sn",
) -> pd.DataFrame:
    report = _ensure_match_influence_report(
        bt_model,
        match_influence_report=match_influence_report,
        ranking=ranking,
        temperature=temperature,
        method=method,
    )
    incidence = build_player_incidence_frame(bt_model, match_influence_report=report)

    aggregated = (
        incidence.groupby(["player", "player_index"], as_index=False)
        .agg(
            player_influence_signed=("influence", "sum"),
            player_influence_abs=("influence_abs", "sum"),
            fitted_row_count=("row_index", "size"),
        )
        .sort_values("player_influence_abs", ascending=False)
        .reset_index(drop=True)
    )
    reference_ranking = list(ranking) if ranking is not None else ranking_from_model(bt_model)
    fit_frame = _fit_frame_with_indices(bt_model)
    model_a = fit_frame["model_a"].to_numpy()
    model_b = fit_frame["model_b"].to_numpy()

    def _player_drop_delta(player_name: str, approximation: str) -> float:
        player_ranking = [name for name in reference_ranking if name != player_name]
        objective = _SubsetSmoothKendallObjective(ranking=player_ranking, temperature=temperature)
        drop_indices = np.flatnonzero((model_a == player_name) | (model_b == player_name))
        if approximation == "joint_newton":
            return compute_group_drop_joint_newton_delta(
                bt_model,
                objective=objective,
                drop_indices=drop_indices,
            )
        if approximation == "two_step":
            return compute_group_drop_two_step_delta(
                bt_model,
                objective=objective,
                drop_indices=drop_indices,
            )
        raise ValueError(f"Unknown approximation {approximation!r}.")

    aggregated["player_influence_joint_newton"] = aggregated["player"].map(
        lambda player_name: _player_drop_delta(player_name, "joint_newton")
    )
    aggregated["player_influence_joint_newton_abs"] = aggregated["player_influence_joint_newton"].abs()
    aggregated["player_influence_two_step"] = aggregated["player"].map(
        lambda player_name: _player_drop_delta(player_name, "two_step")
    )
    aggregated["player_influence_two_step_abs"] = aggregated["player_influence_two_step"].abs()
    return aggregated.sort_values("player_influence_joint_newton_abs", ascending=False).reset_index(drop=True)


def compute_player_statistics(
    bt_model: BradleyTerryModel,
    *,
    player_influence: pd.DataFrame | None = None,
    match_influence_report: pd.DataFrame | None = None,
    ranking: Iterable[int | str] | None = None,
    temperature: float = 0.5,
    method: str = "1sn",
    surprise_metric: str = "mae",
) -> pd.DataFrame:
    if surprise_metric.lower() != "mae":
        raise ValueError("Only surprise_metric='mae' is currently supported.")

    report = _ensure_match_influence_report(
        bt_model,
        match_influence_report=match_influence_report,
        ranking=ranking,
        temperature=temperature,
        method=method,
    )
    influence_df = (
        compute_player_influence(
            bt_model,
            match_influence_report=report,
            ranking=ranking,
            temperature=temperature,
            method=method,
        )
        if player_influence is None
        else player_influence.copy()
    )

    skills = pd.DataFrame(
        {
            "player": bt_model.competitor_names_,
            "player_index": np.arange(bt_model.n_players_),
            "skill": bt_model.full_skills(),
            "rating": bt_model.scaled_skills(),
        }
    )

    incidence = build_player_incidence_frame(bt_model, match_influence_report=report)
    degree_col = "match_id" if "match_id" in incidence.columns else "row_index"
    grouped = incidence.groupby(["player", "player_index"], as_index=False)
    stats = grouped.agg(
        degree=(degree_col, "nunique"),
        bridge_var=("opponent_skill", "var"),
        closeness=("skill_gap_abs", "mean"),
        surprise=("prediction_error_abs", "mean"),
    )
    stats["bridge_var"] = stats["bridge_var"].fillna(0.0)

    result = (
        skills.merge(influence_df, on=["player", "player_index"], how="left")
        .merge(stats, on=["player", "player_index"], how="left")
        .sort_values("player_influence_joint_newton_abs", ascending=False)
        .reset_index(drop=True)
    )
    result["influence"] = result["player_influence_abs"]
    return result


def compute_group_drop_joint_newton_delta(
    bt_model: BradleyTerryModel,
    *,
    objective: Objective,
    drop_indices: np.ndarray | list[int],
    ridge: float = 1e-8,
) -> float:
    """
    Approximate a grouped drop with the same Newton-style update used in
    ``player_influence_script.py``.

    The update is:

    1. apply the grouped first-order deletion step at the full-data optimum,
    2. take one Newton correction on the kept-data score equation.

    In the older notebook this appeared under the name "two-step"; here we
    expose it as ``joint_newton`` to match the reference script.
    """

    bt_model._require_fit()
    drop_indices = np.asarray(drop_indices, dtype=int)
    if drop_indices.size == 0:
        return 0.0

    assert bt_model.X is not None and bt_model.y is not None
    assert bt_model.beta_hat_ is not None and bt_model.residuals_ is not None
    assert bt_model.hessian_reg_ is not None

    x_full = np.asarray(bt_model.X, dtype=float)
    y_full = np.asarray(bt_model.y, dtype=float)
    theta0 = np.asarray(bt_model.beta_hat_, dtype=float)
    residuals = np.asarray(bt_model.residuals_, dtype=float)

    score_rows = x_full * residuals[:, None]
    g_drop = np.sum(score_rows[drop_indices], axis=0)
    inv_h0 = bt_model._solve(np.asarray(bt_model.hessian_reg_, dtype=float), np.eye(theta0.shape[0], dtype=float))
    theta1 = theta0 - inv_h0 @ g_drop

    keep_mask = np.ones(x_full.shape[0], dtype=bool)
    keep_mask[drop_indices] = False
    x_keep = x_full[keep_mask]
    y_keep = y_full[keep_mask]

    if x_keep.shape[0] == 0:
        theta_joint = np.zeros_like(theta0)
    else:
        p1 = _sigmoid(x_keep @ theta1)
        score1 = x_keep.T @ (y_keep - p1)
        w1 = p1 * (1.0 - p1)
        h1 = x_keep.T @ (x_keep * w1[:, None]) + ridge * np.eye(x_keep.shape[1], dtype=float)
        theta_joint = theta1 + bt_model._solve(h1, score1)

    approx_model = _model_with_free_theta(bt_model, theta_joint)
    return float(objective.value(approx_model) - objective.value(bt_model))


def compute_group_drop_two_step_delta(
    bt_model: BradleyTerryModel,
    *,
    objective: Objective,
    drop_indices: np.ndarray | list[int],
    ridge: float = 1e-8,
) -> float:
    return compute_group_drop_joint_newton_delta(
        bt_model,
        objective=objective,
        drop_indices=drop_indices,
        ridge=ridge,
    )


def compute_player_influence_correlations(
    player_stats: pd.DataFrame,
    *,
    feature_columns: Iterable[str] = DEFAULT_PLAYER_FEATURES,
    target_columns: Iterable[str] = (
        "player_influence_signed",
        "player_influence_abs",
        "player_influence_joint_newton",
        "player_influence_joint_newton_abs",
        "player_influence_two_step",
        "player_influence_two_step_abs",
    ),
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for target in target_columns:
        for feature in feature_columns:
            valid = player_stats[[target, feature]].dropna()
            pearson_r, pearson_p = _safe_corr(valid[target], valid[feature], kind="pearson")
            spearman_rho, spearman_p = _safe_corr(valid[target], valid[feature], kind="spearman")
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "n": int(len(valid)),
                    "pearson_r": pearson_r,
                    "pearson_pvalue": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_pvalue": spearman_p,
                }
            )
    return pd.DataFrame(rows)


def run_extreme_skill_ablation(
    bt_model: BradleyTerryModel,
    player_stats: pd.DataFrame,
    *,
    temperature: float = 0.5,
) -> pd.DataFrame:
    stats = player_stats.sort_values("skill", ascending=False).reset_index(drop=True)
    selected = [stats.iloc[0]["player"], stats.iloc[-1]["player"]]
    results = [
        drop_player_and_measure_effect(
            bt_model,
            player=player,
            player_stats=player_stats,
            temperature=temperature,
        )
        for player in selected
    ]
    return pd.DataFrame([result.__dict__ for result in results])


def drop_player_and_measure_effect(
    bt_model: BradleyTerryModel,
    *,
    player: int | str,
    player_stats: pd.DataFrame,
    temperature: float = 0.5,
) -> PlayerDropResult:
    bt_model._require_fit()
    player_idx = bt_model.resolve_player(player)
    player_name = bt_model.competitor_names_[player_idx]
    fit_frame = _fit_frame_with_indices(bt_model)
    filtered = fit_frame.loc[
        (fit_frame["model_a"] != player_name) & (fit_frame["model_b"] != player_name)
    ].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"Removing {player_name!r} leaves no matches to refit.")

    refit_model = _refit_from_fitted_frame(bt_model, filtered)
    remaining_players = tuple(ranking_from_model(bt_model))
    remaining_players = tuple(name for name in remaining_players if name in refit_model.competitor_names_)

    baseline_smooth = smooth_kendall_tau_value_from_model(
        bt_model,
        ranking=remaining_players,
        temperature=temperature,
    )
    refit_smooth = smooth_kendall_tau_value_from_model(
        refit_model,
        ranking=remaining_players,
        temperature=temperature,
    )
    refit_ranking = ranking_from_model(refit_model)
    exact_tau = exact_kendall_tau(remaining_players, refit_ranking)

    predicted_sum = float(
        player_stats.loc[player_stats["player"] == player_name, "player_influence_signed"].iloc[0]
    )
    predicted_joint_newton = float(
        player_stats.loc[player_stats["player"] == player_name, "player_influence_joint_newton"].iloc[0]
    )
    predicted_two_step = float(
        player_stats.loc[player_stats["player"] == player_name, "player_influence_two_step"].iloc[0]
    )
    skill = float(player_stats.loc[player_stats["player"] == player_name, "skill"].iloc[0])
    return PlayerDropResult(
        player=player_name,
        skill=skill,
        predicted_delta_sum=predicted_sum,
        predicted_delta_joint_newton=predicted_joint_newton,
        predicted_delta_two_step=predicted_two_step,
        actual_smooth_delta=float(refit_smooth - baseline_smooth),
        actual_rank_tau=float(exact_tau),
        actual_rank_tau_delta=float(exact_tau - 1.0),
        baseline_smooth_value=float(baseline_smooth),
        refit_smooth_value=float(refit_smooth),
        remaining_players=remaining_players,
    )


def build_player_incidence_frame(
    bt_model: BradleyTerryModel,
    *,
    match_influence_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    bt_model._require_fit()
    frame = _fit_frame_with_indices(bt_model)
    report = match_influence_report.copy() if match_influence_report is not None else frame[["row_index"]].copy()
    if "row_index" not in report.columns:
        report = report.reset_index(names="row_index")
    keep_cols = ["row_index"] + [col for col in ["influence", "influence_abs"] if col in report.columns]
    report = report.loc[:, keep_cols]
    merged = frame.merge(report, on="row_index", how="left")

    rows = []
    for side, opponent_side in (("a", "b"), ("b", "a")):
        player_index_col = f"player_{side}_index"
        opponent_index_col = f"player_{opponent_side}_index"
        player_names = merged["model_a"] if side == "a" else merged["model_b"]
        opponent_names = merged["model_b"] if side == "a" else merged["model_a"]
        player_prob = merged["fitted_probability"] if side == "a" else 1.0 - merged["fitted_probability"]
        player_outcome = merged["outcome"] if side == "a" else 1.0 - merged["outcome"]
        rows.append(
            pd.DataFrame(
                {
                    "row_index": merged["row_index"],
                    "match_id": merged["match_id"] if "match_id" in merged.columns else merged["row_index"],
                    "player": player_names,
                    "player_index": merged[player_index_col],
                    "opponent": opponent_names,
                    "opponent_index": merged[opponent_index_col],
                    "player_skill": merged["player_a_skill"] if side == "a" else merged["player_b_skill"],
                    "opponent_skill": merged["player_b_skill"] if side == "a" else merged["player_a_skill"],
                    "fitted_probability": player_prob,
                    "player_outcome": player_outcome,
                    "prediction_error_abs": np.abs(player_outcome - player_prob),
                    "skill_gap_abs": np.abs(
                        (merged["player_a_skill"] if side == "a" else merged["player_b_skill"])
                        - (merged["player_b_skill"] if side == "a" else merged["player_a_skill"])
                    ),
                    "influence": merged.get("influence", np.nan),
                    "influence_abs": merged.get("influence_abs", np.nan),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def smooth_kendall_tau_value_from_model(
    bt_model: BradleyTerryModel,
    *,
    ranking: Iterable[int | str],
    temperature: float = 0.5,
) -> float:
    ranking_names = _resolve_ranking_names(bt_model, ranking)
    skills = dict(zip(bt_model.competitor_names_, bt_model.full_skills()))
    skill_vector = np.asarray([skills[name] for name in ranking_names], dtype=float)
    sign_matrix = _ranking_sign_matrix(ranking_names)
    return _smooth_kendall_from_skills(skill_vector, sign_matrix, temperature=temperature)


def exact_kendall_tau(reference_ranking: Iterable[str], candidate_ranking: Iterable[str]) -> float:
    reference = list(reference_ranking)
    candidate = list(candidate_ranking)
    if set(reference) != set(candidate):
        raise ValueError("reference_ranking and candidate_ranking must contain the same players.")
    reference_order = {player: rank for rank, player in enumerate(reference)}
    candidate_order = {player: rank for rank, player in enumerate(candidate)}
    players = reference
    x = np.asarray([reference_order[player] for player in players], dtype=float)
    y = np.asarray([candidate_order[player] for player in players], dtype=float)
    return float(kendalltau(x, y).statistic)


def _ensure_match_influence_report(
    bt_model: BradleyTerryModel,
    *,
    match_influence_report: pd.DataFrame | None,
    ranking: Iterable[int | str] | None,
    temperature: float,
    method: str,
) -> pd.DataFrame:
    if match_influence_report is not None:
        return match_influence_report.copy()
    return kendall_tau_match_influence_report(
        bt_model,
        ranking=ranking,
        temperature=temperature,
        method=method,
    )


def _fit_frame_with_indices(bt_model: BradleyTerryModel) -> pd.DataFrame:
    bt_model._require_fit()
    if bt_model.match_frame_ is None:
        raise ValueError("Player-level analysis requires bt_model.match_frame_.")

    frame = bt_model.match_frame_.copy().reset_index(drop=True)
    frame["row_index"] = np.arange(len(frame))
    name_to_index = {name: idx for idx, name in enumerate(bt_model.competitor_names_)}
    frame["player_a_index"] = frame["model_a"].map(name_to_index)
    frame["player_b_index"] = frame["model_b"].map(name_to_index)
    frame["player_a_skill"] = frame["player_a_index"].map(lambda idx: bt_model.full_skills()[idx])
    frame["player_b_skill"] = frame["player_b_index"].map(lambda idx: bt_model.full_skills()[idx])
    frame["fitted_probability"] = np.asarray(bt_model.probabilities_, dtype=float)
    frame["outcome"] = np.asarray(bt_model.y, dtype=float)
    return frame


def _model_with_free_theta(bt_model: BradleyTerryModel, free_theta: np.ndarray) -> BradleyTerryModel:
    approx = BradleyTerryModel(
        bt_model.X,
        bt_model.y,
        competitor_names=bt_model.competitor_names_,
        **bt_model._config_kwargs,
    )
    approx.match_frame_ = None if bt_model.match_frame_ is None else bt_model.match_frame_.copy()
    approx.beta_hat_ = np.asarray(free_theta, dtype=float)
    approx.full_beta_hat_ = approx.expand_free_vector(approx.beta_hat_)
    approx.reported_skills_ = approx.full_beta_hat_.copy()
    return approx


def _refit_from_fitted_frame(bt_model: BradleyTerryModel, frame: pd.DataFrame) -> BradleyTerryModel:
    present = pd.Index(pd.concat([frame["model_a"], frame["model_b"]], ignore_index=True).unique()).tolist()
    competitors = [name for name in bt_model.competitor_names_ if name in set(present)]
    index = {name: idx for idx, name in enumerate(competitors)}
    pairs = np.column_stack(
        [
            frame["model_a"].map(index).to_numpy(dtype=int),
            frame["model_b"].map(index).to_numpy(dtype=int),
        ]
    )
    outcomes = frame["outcome"].to_numpy(dtype=float)
    dataset = BattleDataset(
        competitors=competitors,
        pairs=pairs,
        outcomes=outcomes,
        frame=frame.loc[:, [col for col in frame.columns if col in {"model_a", "model_b", "winner", "match_id", "match_copy", "outcome"}]].copy(),
    )
    config = dict(bt_model._config_kwargs)
    reference_name = bt_model.competitor_names_[bt_model.reference_player]
    config["reference_player"] = competitors.index(reference_name) if reference_name in competitors else 0

    anchor_player = config.get("anchor_player")
    if anchor_player is not None:
        anchor_name = (
            bt_model.competitor_names_[anchor_player]
            if isinstance(anchor_player, int)
            else str(anchor_player)
        )
        config["anchor_player"] = anchor_name if anchor_name in competitors else None

    model = BradleyTerryModel(
        dataset.design_matrix(),
        dataset.outcomes,
        competitor_names=dataset.competitors,
        **config,
    )
    model.match_frame_ = dataset.frame.copy()
    model.fit()
    return model


def _resolve_ranking_names(bt_model: BradleyTerryModel, ranking: Iterable[int | str]) -> list[str]:
    names = []
    for player in ranking:
        if isinstance(player, str):
            names.append(player)
        else:
            names.append(bt_model.competitor_names_[bt_model.resolve_player(player)])
    return names


def _ranking_sign_matrix(ranking_names: Iterable[str]) -> np.ndarray:
    ranking_names = list(ranking_names)
    n_players = len(ranking_names)
    order = {name: idx for idx, name in enumerate(ranking_names)}
    sign = np.zeros((n_players, n_players), dtype=float)
    for a_idx, a_name in enumerate(ranking_names):
        for b_idx, b_name in enumerate(ranking_names):
            if a_idx == b_idx:
                continue
            sign[a_idx, b_idx] = 1.0 if order[a_name] < order[b_name] else -1.0
    return sign


def _smooth_kendall_from_skills(skills: np.ndarray, sign_matrix: np.ndarray, *, temperature: float) -> float:
    skills = np.asarray(skills, dtype=float)
    n_players = skills.shape[0]
    normalizer = 2.0 / (n_players * (n_players - 1))
    total = 0.0
    for a_idx in range(n_players):
        for b_idx in range(a_idx + 1, n_players):
            total += sign_matrix[a_idx, b_idx] * np.tanh((skills[a_idx] - skills[b_idx]) / temperature)
    return float(normalizer * total)


def _safe_corr(x: pd.Series, y: pd.Series, *, kind: str) -> tuple[float, float]:
    if len(x) < 2 or np.isclose(np.nanstd(x.to_numpy(dtype=float)), 0.0) or np.isclose(np.nanstd(y.to_numpy(dtype=float)), 0.0):
        return float("nan"), float("nan")
    if kind == "pearson":
        return tuple(float(value) for value in pearsonr(x, y))
    if kind == "spearman":
        return tuple(float(value) for value in spearmanr(x, y))
    raise ValueError(f"Unknown correlation kind: {kind}")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
