import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from clean_bt_rank import (
    BTParameterInfluence,
    BattleDataset,
    BradleyTerryModel,
    CIBoundaryObjective,
    CIStrictGapObjective,
    GlobalCIWidthObjective,
    KendallTauObjective,
    ObjectiveInfluence,
    PlayerUncertaintyObjective,
    SkillGapObjective,
    TraceUncertaintyObjective,
    compute_player_influence,
    compute_player_influence_correlations,
    compute_player_statistics,
    drop_player_and_measure_effect,
    generate_synthetic_dataset,
    kendall_tau_match_influence_report,
    plot_ratings,
)
from clean_bt_rank.ci_aware_actions_needed import predicted_gap_target_met
from clean_bt_rank.experiments._common import apply_add_candidate, apply_drop_row, apply_flip_row, make_state
from clean_bt_rank.iterative_actions import (
    _select_top_alpha_matches,
    apply_action_on_top_alpha_influential_matches,
    compute_all_action_influences,
    refit_model_with_action,
)


def test_arena_workflow() -> None:
    rows = []
    rows.extend([{"model_a": "alpha", "model_b": "beta", "winner": "model_a"} for _ in range(30)])
    rows.extend([{"model_a": "alpha", "model_b": "gamma", "winner": "model_a"} for _ in range(25)])
    rows.extend([{"model_a": "beta", "model_b": "gamma", "winner": "model_a"} for _ in range(20)])
    rows.extend([{"model_a": "beta", "model_b": "alpha", "winner": "model_b"} for _ in range(8)])
    rows.extend([{"model_a": "alpha", "model_b": "beta", "winner": "tie"} for _ in range(2)])
    arena_df = pd.DataFrame(rows)

    dataset = BattleDataset.from_dataframe(arena_df)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    summary = model.summary(ci_method="sandwich")

    ratings = dict(zip(summary["competitor"], summary["rating"]))
    assert ratings["alpha"] > ratings["beta"] > ratings["gamma"]
    assert np.all(summary["ci_upper"] >= summary["ci_lower"])

    param_infl = BTParameterInfluence(model)
    gap_obj = SkillGapObjective("alpha", "beta")
    obj_infl = ObjectiveInfluence(model, param_infl)
    gap_if = obj_infl.compute_match_influence(gap_obj, method="if")
    gap_1sn = obj_infl.compute_match_influence(gap_obj, method="1sn")
    assert gap_if.shape == (dataset.n_matches,)
    assert gap_1sn.shape == (dataset.n_matches,)

    fig, ax = plot_ratings(summary, top_n=3, title="test")
    assert fig is not None
    assert ax is not None


def test_ties_resolve_to_first_model_by_default() -> None:
    df = pd.DataFrame([{"model_a": "alpha", "model_b": "beta", "winner": "tie"}])
    dataset = BattleDataset.from_dataframe(df)
    assert dataset.n_matches == 2
    assert dataset.outcomes.tolist() == [1.0, 0.0]


def test_weighted_symmetric_ties() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "beta", "winner": "model_b"},
            {"model_a": "alpha", "model_b": "beta", "winner": "tie"},
        ]
    )

    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)

    assert dataset.n_matches == 6
    assert dataset.outcomes.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    assert dataset.frame["model_a"].tolist() == ["alpha", "alpha", "alpha", "beta", "beta", "beta"]
    assert dataset.frame["model_b"].tolist() == ["beta", "beta", "beta", "alpha", "alpha", "alpha"]

    model = BradleyTerryModel.from_dataset(dataset).fit()
    summary = model.summary(ci_method="sandwich")
    assert np.all(summary["ci_upper"] >= summary["ci_lower"])


def test_model_rejects_non_binary_outcomes() -> None:
    x = np.array([[1.0], [-1.0]])
    y = np.array([1.0, 0.5])
    try:
        BradleyTerryModel(x, y)
    except ValueError as exc:
        assert "binary outcomes" in str(exc)
    else:
        raise AssertionError("Expected BradleyTerryModel to reject non-binary outcomes.")


def test_constraint_options() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "alpha", "winner": "model_b"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df)

    reference_model = BradleyTerryModel.from_dataset(dataset).fit()
    reference_summary = reference_model.summary(ci_method="sandwich")
    reference_ratings = reference_summary.set_index("competitor")["rating"]
    assert np.isclose(reference_ratings.loc[dataset.competitors[0]], 1000.0)

    anchored_model = BradleyTerryModel.from_dataset(
        dataset,
        anchor_player=dataset.competitors[0],
        anchor_rating=0.0,
    ).fit()
    anchored_summary = anchored_model.summary(ci_method="sandwich")
    anchored_ratings = anchored_summary.set_index("competitor")["rating"]
    assert np.isclose(anchored_ratings.loc[dataset.competitors[0]], 0.0)

    ref_gap = SkillGapObjective("alpha", "beta").value(reference_model)
    anchored_gap = SkillGapObjective("alpha", "beta").value(anchored_model)
    assert np.isclose(ref_gap, anchored_gap)


def test_top_alpha_selection_expands_forward_reverse_pairs() -> None:
    report = pd.DataFrame(
        {
            "match_id": [0, 0, 1, 1],
            "match_copy": ["forward", "reverse", "forward", "reverse"],
            "row_uid": [0, 1, 2, 3],
            "influence": [10.0, 1.0, 9.0, 8.0],
        }
    )

    selected = _select_top_alpha_matches(report, alpha=1)

    assert selected["match_id"].tolist() == [0, 0]
    assert selected["match_copy"].tolist() == ["forward", "reverse"]


def test_apply_action_drop_uses_full_forward_reverse_pair() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    objective = SkillGapObjective("alpha", "beta")
    report = model.match_frame_.copy()
    report["row_uid"] = np.arange(len(report))
    report["influence"] = [100.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    result = apply_action_on_top_alpha_influential_matches(
        model,
        objective,
        report,
        alpha=1,
        action="drop",
    )
    selected = result["selected_matches"]

    assert selected["match_id"].tolist() == [0, 0]
    assert set(selected["match_copy"]) == {"forward", "reverse"}
    assert result["n_applied"] == 1


def test_add_candidates_are_paired_and_count_as_one_action() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    objective = SkillGapObjective("alpha", "beta")
    report = compute_all_action_influences(model, objective, action="add", candidate_mode="all_outcomes")

    selected = _select_top_alpha_matches(report, alpha=1, group_by_match=True)

    assert len(selected) == 2
    assert selected["match_id"].nunique() == 1
    assert set(selected["match_copy"]) == {"forward", "reverse"}

    result = apply_action_on_top_alpha_influential_matches(
        model,
        objective,
        report,
        alpha=1,
        action="add",
        recompute_mode="approximate",
        group_by_match=True,
    )

    assert result["n_applied"] == 1
    assert len(result["selected_matches"]) == 2


def test_experiment_state_add_appends_forward_reverse_rows() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    state = make_state(model)

    chosen = pd.Series(
        {
            "model_a": "alpha",
            "model_b": "gamma",
            "outcome": 1.0,
            "_candidate_x": np.array([1.0, -1.0]),
        }
    )
    updated = apply_add_candidate(state, chosen)

    assert updated.X.shape[0] == state.X.shape[0] + 2
    assert updated.y[-2:].tolist() == [1.0, 0.0]
    assert updated.frame.iloc[-2:]["match_copy"].tolist() == ["forward", "reverse"]
    assert updated.frame.iloc[-2:]["match_id"].nunique() == 1


def test_experiment_state_drop_removes_full_forward_reverse_pair() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    state = make_state(model)

    updated = apply_drop_row(state, int(state.frame.iloc[0]["row_uid"]))

    assert updated.X.shape[0] == state.X.shape[0] - 2
    assert updated.frame["match_id"].nunique() == state.frame["match_id"].nunique() - 1
    assert 0 not in updated.frame["match_id"].tolist()


def test_experiment_state_flip_flips_full_forward_reverse_pair() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    state = make_state(model)

    pair_mask = state.frame["match_id"] == int(state.frame.iloc[0]["match_id"])
    original_pair_outcomes = state.y[pair_mask.to_numpy()].tolist()

    updated = apply_flip_row(state, int(state.frame.iloc[0]["row_uid"]))

    assert updated.y[pair_mask.to_numpy()].tolist() == [1.0 - value for value in original_pair_outcomes]


def test_refit_model_with_action_drop_removes_full_forward_reverse_pair() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "tie"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    frame = model.match_frame_.copy()
    selected = frame.loc[frame["match_id"] == int(frame.iloc[0]["match_id"])].copy()

    updated = refit_model_with_action(model, selected, "drop")

    assert updated.X.shape[0] == model.X.shape[0] - 2
    assert updated.match_frame_["match_id"].nunique() == model.match_frame_["match_id"].nunique() - 1


def test_refit_model_with_action_flip_flips_full_forward_reverse_pair() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "tie"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    frame = model.match_frame_.copy()
    selected = frame.loc[frame["match_id"] == int(frame.iloc[0]["match_id"])].copy()
    mask = frame["match_id"] == int(selected.iloc[0]["match_id"])

    updated = refit_model_with_action(model, selected, "flip")

    assert updated.y[mask.to_numpy()].tolist() == [1.0 - value for value in model.y[mask.to_numpy()].tolist()]


def test_refit_model_with_action_add_appends_forward_reverse_pair_once_for_logical_match() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "tie"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    objective = SkillGapObjective("alpha", "beta")
    report = compute_all_action_influences(model, objective, action="add", candidate_mode="all_outcomes")
    selected = _select_top_alpha_matches(report, alpha=1, group_by_match=True)

    updated = refit_model_with_action(model, selected, "add")

    assert updated.X.shape[0] == model.X.shape[0] + 2
    assert updated.match_frame_.iloc[-2:]["match_copy"].tolist() == ["forward", "reverse"]
    assert updated.match_frame_.iloc[-2:]["match_id"].nunique() == 1


def test_predicted_gap_target_met_matches_gap_crossing_direction() -> None:
    assert not predicted_gap_target_met(1.0, 0.1)
    assert predicted_gap_target_met(1.0, 0.0)
    assert predicted_gap_target_met(1.0, -0.1)
    assert not predicted_gap_target_met(-1.0, -0.1)
    assert predicted_gap_target_met(-1.0, 0.0)
    assert predicted_gap_target_met(-1.0, 0.1)


def test_ci_strict_gap_objective_matches_best_vs_worst_formula() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "alpha", "winner": "model_b"},
        ]
    )
    dataset = BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    model = BradleyTerryModel.from_dataset(dataset).fit()

    objective = CIStrictGapObjective("alpha", "beta", ci_method="gao_local", alpha=0.05)
    summary = model.summary(ci_method="gao_local", alpha=0.05).set_index("competitor")
    expected = float(summary.loc["alpha", "rating"] + 1.959963984540054 * summary.loc["alpha", "standard_error"])
    expected -= float(summary.loc["beta", "rating"] - 1.959963984540054 * summary.loc["beta", "standard_error"])
    expected /= model.alpha

    assert np.isclose(objective.value(model), expected)


def test_synthetic_workflow() -> None:
    dataset, truth = generate_synthetic_dataset(n_models=6, n_matches=2500, seed=3, tie_probability=0.03)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    summary = model.summary(ci_method="bootstrap", n_bootstrap=20, seed=1)

    estimated = summary.set_index("competitor").loc[truth["competitor"], "rating"].to_numpy()
    corr = np.corrcoef(estimated, truth["true_skill"].to_numpy())[0, 1]
    assert corr > 0.8
    assert np.all(summary["ci_upper"] >= summary["ci_lower"])


def _finite_difference_gradient(model: BradleyTerryModel, objective, eps: float = 1e-6) -> np.ndarray:
    beta = model.beta_hat_.copy()
    grad = np.zeros_like(beta)
    for idx in range(len(beta)):
        step = np.zeros_like(beta)
        step[idx] = eps

        plus = BradleyTerryModel(
            model.X,
            model.y,
            competitor_names=model.competitor_names_,
            reference_player=model.reference_player,
            scale=model.scale,
            base=model.base,
            init_rating=model.init_rating,
            anchor_player=model.anchor_player,
            anchor_rating=model.anchor_rating,
            hessian_ridge=model.hessian_ridge,
        )
        minus = BradleyTerryModel(
            model.X,
            model.y,
            competitor_names=model.competitor_names_,
            reference_player=model.reference_player,
            scale=model.scale,
            base=model.base,
            init_rating=model.init_rating,
            anchor_player=model.anchor_player,
            anchor_rating=model.anchor_rating,
            hessian_ridge=model.hessian_ridge,
        )

        plus.beta_hat_ = beta + step
        plus.full_beta_hat_ = plus.expand_free_vector(plus.beta_hat_)
        plus.reported_skills_ = plus.full_beta_hat_.copy()

        minus.beta_hat_ = beta - step
        minus.full_beta_hat_ = minus.expand_free_vector(minus.beta_hat_)
        minus.reported_skills_ = minus.full_beta_hat_.copy()

        grad[idx] = (objective.value(plus) - objective.value(minus)) / (2.0 * eps)
    return grad


def test_new_objective_gradients_and_influence() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "delta", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "delta", "winner": "model_a"},
            {"model_a": "gamma", "model_b": "delta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "beta", "winner": "tie"},
            {"model_a": "gamma", "model_b": "delta", "winner": "tie"},
        ]
    )
    model = BradleyTerryModel.from_dataset(
        BattleDataset.from_dataframe(df, weighted_symmetric_ties=True)
    ).fit()

    objectives = [
        PlayerUncertaintyObjective("alpha"),
        TraceUncertaintyObjective(),
        GlobalCIWidthObjective(ci_method="gao_local"),
        KendallTauObjective(ranking=["alpha", "beta", "gamma", "delta"], temperature=0.5),
    ]
    for objective in objectives:
        analytic = objective.gradient_free(model)
        numeric = _finite_difference_gradient(model, objective)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)

    param_infl = BTParameterInfluence(model)
    obj_infl = ObjectiveInfluence(model, param_infl)
    kendall_if = obj_infl.compute_match_influence(objectives[-1], method="if")
    trace_1sn = obj_infl.compute_match_influence(objectives[1], method="1sn")
    global_ci_if = obj_infl.compute_match_influence(objectives[2], method="if")
    assert kendall_if.shape == (model.X.shape[0],)
    assert trace_1sn.shape == (model.X.shape[0],)
    assert global_ci_if.shape == (model.X.shape[0],)


def test_ci_objectives_explicit_weight_terms_are_finite() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "gamma", "model_b": "delta", "winner": "model_a"},
            {"model_a": "beta", "model_b": "delta", "winner": "model_b"},
            {"model_a": "alpha", "model_b": "delta", "winner": "model_a"},
        ]
    )
    model = BradleyTerryModel.from_dataset(BattleDataset.from_dataframe(df)).fit()

    player_ci = PlayerUncertaintyObjective("alpha")
    trace_ci = TraceUncertaintyObjective()
    global_ci = GlobalCIWidthObjective(ci_method="gao_local")
    boundary = CIBoundaryObjective(model, k=1, ci_method="gao_local", freeze_se=False)

    for objective in [player_ci, trace_ci, global_ci, boundary]:
        drop_term = objective.explicit_weight_influence(model, action="drop")
        add_term = objective.explicit_weight_influence(model, action="add", X_new=model.X, y_new=model.y)
        flip_term = objective.explicit_weight_influence(model, action="flip")
        assert drop_term.shape == (model.X.shape[0],)
        assert add_term.shape == (model.X.shape[0],)
        assert flip_term.shape == (model.X.shape[0],)
        assert np.all(np.isfinite(drop_term))
        assert np.all(np.isfinite(add_term))
        assert np.all(np.isfinite(flip_term))

    param_infl = BTParameterInfluence(model)
    obj_infl = ObjectiveInfluence(model, param_infl)
    boundary_if = obj_infl.compute_match_influence(boundary, method="if")
    trace_if = obj_infl.compute_match_influence(trace_ci, method="if")
    global_ci_if = obj_infl.compute_match_influence(global_ci, method="if")
    assert boundary_if.shape == (model.X.shape[0],)
    assert trace_if.shape == (model.X.shape[0],)
    assert global_ci_if.shape == (model.X.shape[0],)
    assert np.all(np.isfinite(boundary_if))
    assert np.all(np.isfinite(trace_if))
    assert np.all(np.isfinite(global_ci_if))


def test_player_level_kendall_influence_pipeline() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "gamma", "model_b": "delta", "winner": "model_a"},
            {"model_a": "beta", "model_b": "delta", "winner": "model_b"},
            {"model_a": "alpha", "model_b": "delta", "winner": "tie"},
        ]
    )
    model = BradleyTerryModel.from_dataset(BattleDataset.from_dataframe(df)).fit()

    match_report = kendall_tau_match_influence_report(model, temperature=0.5, method="1sn")
    player_influence = compute_player_influence(model, match_influence_report=match_report)
    player_stats = compute_player_statistics(
        model,
        player_influence=player_influence,
        match_influence_report=match_report,
        temperature=0.5,
    )
    corr = compute_player_influence_correlations(player_stats)

    assert set(player_influence["player"]) == {"alpha", "beta", "gamma", "delta"}
    assert np.all(player_influence["player_influence_abs"] >= 0.0)
    assert set(player_stats.columns) >= {
        "player",
        "skill",
        "degree",
        "influence",
        "bridge_var",
        "closeness",
        "surprise",
        "player_influence_signed",
        "player_influence_abs",
        "player_influence_joint_newton",
        "player_influence_joint_newton_abs",
        "player_influence_two_step",
        "player_influence_two_step_abs",
    }
    assert np.all(player_stats["degree"] > 0)
    assert np.all(player_stats["surprise"] >= 0.0)
    assert np.all(player_influence["player_influence_joint_newton_abs"] >= 0.0)

    influence_lookup = player_stats.set_index("player")["influence"]
    np.testing.assert_allclose(
        influence_lookup.loc[player_influence["player"]].to_numpy(),
        player_influence["player_influence_abs"].to_numpy(),
    )

    identical_row = corr[
        (corr["target"] == "player_influence_abs") & (corr["feature"] == "influence")
    ].iloc[0]
    assert np.isclose(float(identical_row["pearson_r"]), 1.0)
    assert np.isclose(float(identical_row["spearman_rho"]), 1.0)


def test_player_drop_effect_returns_finite_outputs() -> None:
    df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "gamma", "model_b": "delta", "winner": "model_a"},
            {"model_a": "beta", "model_b": "delta", "winner": "model_b"},
            {"model_a": "alpha", "model_b": "delta", "winner": "model_a"},
            {"model_a": "beta", "model_b": "alpha", "winner": "model_b"},
        ]
    )
    model = BradleyTerryModel.from_dataset(BattleDataset.from_dataframe(df)).fit()
    player_stats = compute_player_statistics(model, temperature=0.5)

    result = drop_player_and_measure_effect(
        model,
        player=player_stats.sort_values("skill", ascending=False).iloc[0]["player"],
        player_stats=player_stats,
        temperature=0.5,
    )

    assert np.isfinite(result.predicted_delta_sum)
    assert np.isfinite(result.predicted_delta_joint_newton)
    assert np.isfinite(result.predicted_delta_two_step)
    assert np.isfinite(result.actual_smooth_delta)
    assert np.isfinite(result.actual_rank_tau)
    assert result.actual_rank_tau <= 1.0
    assert len(result.remaining_players) == model.n_players_ - 1
