from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from clean_bt_rank import (
    BradleyTerryModel,
    CIBoundaryObjective,
    CIStrictGapObjective,
    GlobalCIWidthObjective,
    KendallTauObjective,
    PlayerUncertaintyObjective,
    SkillGapObjective,
    TraceUncertaintyObjective,
    generate_synthetic_dataset,
)
from clean_bt_rank.actions import compute_influence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small synthetic Bradley-Terry influence experiment and compare "
            "influence approximations against exact single-action refits."
        )
    )
    parser.add_argument("--n-players", type=int, default=5)
    parser.add_argument("--n-matches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skill-scale", type=float, default=0.8)
    parser.add_argument("--tie-probability", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("new_result") / "synthetic_influence_loo_5p_1000m_seed0",
    )
    return parser.parse_args()


def ranking_frame(model: BradleyTerryModel) -> pd.DataFrame:
    skills = model.full_skills()
    frame = pd.DataFrame(
        {
            "competitor": model.competitor_names_,
            "skill": skills,
            "rank": np.arange(1, len(skills) + 1),
        }
    ).sort_values("skill", ascending=False, kind="stable")
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame.reset_index(drop=True)


def build_objectives(model: BradleyTerryModel) -> tuple[list[tuple[str, object]], pd.DataFrame]:
    ranking = ranking_frame(model)
    ordered_players = ranking["competitor"].tolist()
    top1 = ordered_players[0]
    top2 = ordered_players[1]

    objectives: list[tuple[str, object]] = [
        ("skill_gap_top1_vs_top2", SkillGapObjective(top1, top2)),
        ("player_uncertainty_top1", PlayerUncertaintyObjective(top1)),
        ("trace_uncertainty", TraceUncertaintyObjective()),
        ("global_ci_width", GlobalCIWidthObjective(ci_method="gao_local")),
        ("kendall_tau_fitted_ranking_T0.5", KendallTauObjective(ranking=ordered_players, temperature=0.5)),
        ("ci_boundary_top1_top2", CIBoundaryObjective(model, k=1, ci_method="gao_local")),
        ("ci_strict_gap_top2_vs_top1", CIStrictGapObjective(top2, top1, ci_method="gao_local")),
    ]

    catalog = pd.DataFrame(
        [
            {
                "objective_name": name,
                "objective_class": objective.__class__.__name__,
                "base_value": float(objective.value(model)),
            }
            for name, objective in objectives
        ]
    )
    return objectives, catalog


def summarize_ci_methods(model: BradleyTerryModel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ci_frames: list[pd.DataFrame] = []
    for ci_method in ("gao_local", "sandwich"):
        frame = model.summary(ci_method=ci_method).copy()
        frame["ci_width"] = frame["ci_upper"] - frame["ci_lower"]
        frame["rank"] = np.arange(1, len(frame) + 1)
        ci_frames.append(frame)

    gao = ci_frames[0].rename(
        columns={
            "rating": "gao_rating",
            "standard_error": "gao_standard_error",
            "ci_lower": "gao_ci_lower",
            "ci_upper": "gao_ci_upper",
            "ci_width": "gao_ci_width",
            "rank": "gao_rank",
        }
    )
    sandwich = ci_frames[1].rename(
        columns={
            "rating": "sandwich_rating",
            "standard_error": "sandwich_standard_error",
            "ci_lower": "sandwich_ci_lower",
            "ci_upper": "sandwich_ci_upper",
            "ci_width": "sandwich_ci_width",
            "rank": "sandwich_rank",
        }
    )

    comparison = gao.merge(
        sandwich,
        on="competitor",
        how="inner",
        suffixes=("_gao", "_sandwich"),
    )

    summary_rows: list[dict[str, object]] = []
    metric_pairs = [
        ("rating", "gao_rating", "sandwich_rating"),
        ("standard_error", "gao_standard_error", "sandwich_standard_error"),
        ("ci_lower", "gao_ci_lower", "sandwich_ci_lower"),
        ("ci_upper", "gao_ci_upper", "sandwich_ci_upper"),
        ("ci_width", "gao_ci_width", "sandwich_ci_width"),
        ("rank", "gao_rank", "sandwich_rank"),
    ]
    for metric_name, gao_col, sandwich_col in metric_pairs:
        summary_rows.append(
            {
                "metric": metric_name,
                "n_players": int(len(comparison)),
                "pearson": corr_or_nan(comparison[gao_col], comparison[sandwich_col], "pearson"),
                "spearman": corr_or_nan(comparison[gao_col], comparison[sandwich_col], "spearman"),
                "kendall": corr_or_nan(comparison[gao_col], comparison[sandwich_col], "kendall"),
                "mean_abs_diff": float((comparison[gao_col] - comparison[sandwich_col]).abs().mean()),
            }
        )

    ci_summary = pd.DataFrame(summary_rows)
    ci_summaries = pd.concat(ci_frames, ignore_index=True)
    return ci_summaries, comparison, ci_summary


def _refit_with_arrays(
    base_model: BradleyTerryModel,
    X: np.ndarray,
    y: np.ndarray,
) -> BradleyTerryModel:
    updated = BradleyTerryModel(
        X,
        y,
        competitor_names=base_model.competitor_names_,
        **base_model._config_kwargs,
    )

    beta = np.zeros(updated.n_params_, dtype=float)
    if base_model.beta_hat_ is not None and len(base_model.beta_hat_) == updated.n_params_:
        beta = np.asarray(base_model.beta_hat_, dtype=float).copy()

    ridge = float(updated.hessian_ridge)
    for _ in range(40):
        logits = X @ beta
        probabilities = expit(logits)
        residuals = y - probabilities
        weights = probabilities * (1.0 - probabilities)
        gradient = X.T @ residuals - ridge * beta
        hessian = X.T @ (weights[:, None] * X) + ridge * np.eye(updated.n_params_)
        step = updated._solve(hessian, gradient)
        beta_next = beta + step
        if np.max(np.abs(step)) < 1e-10:
            beta = beta_next
            break
        beta = beta_next

    probabilities = expit(X @ beta)
    residuals = y - probabilities
    weights = probabilities * (1.0 - probabilities)
    hessian_reg = X.T @ (weights[:, None] * X) + ridge * np.eye(updated.n_params_)
    solve_h_xt = updated._solve(hessian_reg, X.T).T
    leverage = weights * np.einsum("ij,ij->i", X, solve_h_xt)

    updated.beta_hat_ = beta
    updated.full_beta_hat_ = updated._expand(beta)
    updated.reported_skills_ = updated.full_beta_hat_.copy()
    updated.probabilities_ = probabilities
    updated.residuals_ = residuals
    updated.hessian_reg_ = hessian_reg
    updated.solve_h_xt_ = solve_h_xt
    updated.leverage_ = leverage
    updated.covariance_free_ = None
    updated.reported_covariance_ = None
    return updated


def exact_drop_deltas(
    model: BradleyTerryModel,
    objectives: list[tuple[str, object]],
) -> pd.DataFrame:
    assert model.X is not None and model.y is not None
    rows: list[dict[str, object]] = []
    base_values = {name: float(obj.value(model)) for name, obj in objectives}

    for row_id in range(model.X.shape[0]):
        keep_mask = np.ones(model.X.shape[0], dtype=bool)
        keep_mask[row_id] = False
        updated = _refit_with_arrays(model, model.X[keep_mask], model.y[keep_mask])
        payload: dict[str, object] = {"candidate_id": row_id}
        for name, objective in objectives:
            payload[f"exact_delta__{name}"] = float(objective.value(updated) - base_values[name])
        rows.append(payload)

    return pd.DataFrame(rows)


def corr_or_nan(series_a: pd.Series, series_b: pd.Series, method: str) -> float:
    if series_a.nunique(dropna=False) <= 1 or series_b.nunique(dropna=False) <= 1:
        return float("nan")
    return float(series_a.corr(series_b, method=method))


def build_action_tables(
    model: BradleyTerryModel,
    objectives: list[tuple[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    add_reference = compute_influence(
        model,
        objectives[0][1],
        action="add",
        method="1sn",
        candidate_mode="all_outcomes",
    ).reset_index(drop=True)

    exact_drop = exact_drop_deltas(model, objectives)

    all_influence_details: list[pd.DataFrame] = []
    loo_details: list[pd.DataFrame] = []
    loo_summary_rows: list[dict[str, object]] = []

    for action in ("drop", "flip", "add"):
        candidate_mode = "all_outcomes" if action == "add" else "all_pairs"
        for method in ("if", "1sn"):
            for objective_name, objective in objectives:
                influence_report = compute_influence(
                    model,
                    objective,
                    action=action,
                    method=method,
                    candidate_mode=candidate_mode,
                ).reset_index(drop=True)
                influence_report = influence_report.copy()
                influence_report["candidate_id"] = np.arange(len(influence_report))
                influence_report["action"] = action
                influence_report["method"] = method
                influence_report["objective_name"] = objective_name
                ordered_cols = [
                    "action",
                    "method",
                    "objective_name",
                    "candidate_id",
                    "influence",
                ]
                remaining_cols = [col for col in influence_report.columns if col not in ordered_cols]
                all_influence_details.append(influence_report[ordered_cols + remaining_cols])

                if action != "drop":
                    continue

                merged = influence_report.merge(exact_drop, on="candidate_id", how="left")
                exact_col = f"exact_delta__{objective_name}"
                merged["exact_delta"] = merged[exact_col].astype(float)
                merged["error"] = merged["influence"] - merged["exact_delta"]
                merged["abs_error"] = merged["error"].abs()
                detail_cols = [
                    "action",
                    "method",
                    "objective_name",
                    "candidate_id",
                    "influence",
                    "exact_delta",
                    "error",
                    "abs_error",
                ]
                keep_cols = detail_cols + [col for col in merged.columns if col not in detail_cols and not col.startswith("exact_delta__")]
                loo_details.append(merged[keep_cols])

                loo_summary_rows.append(
                    {
                        "action": action,
                        "method": method,
                        "objective_name": objective_name,
                        "n_candidates": int(len(merged)),
                        "pearson": corr_or_nan(merged["influence"], merged["exact_delta"], "pearson"),
                        "spearman": corr_or_nan(merged["influence"], merged["exact_delta"], "spearman"),
                        "kendall": corr_or_nan(merged["influence"], merged["exact_delta"], "kendall"),
                        "mae": float(merged["abs_error"].mean()),
                        "rmse": float(np.sqrt(np.mean(np.square(merged["error"])))),
                        "sign_agreement": float(
                            np.mean(np.sign(merged["influence"].to_numpy()) == np.sign(merged["exact_delta"].to_numpy()))
                        ),
                    }
                )

    influence_details = pd.concat(all_influence_details, ignore_index=True)
    loo_detail_df = pd.concat(loo_details, ignore_index=True)
    loo_summary = pd.DataFrame(loo_summary_rows).sort_values(["objective_name", "method"]).reset_index(drop=True)
    return influence_details, loo_detail_df, loo_summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, truth = generate_synthetic_dataset(
        n_models=args.n_players,
        n_matches=args.n_matches,
        seed=args.seed,
        skill_scale=args.skill_scale,
        tie_probability=args.tie_probability,
    )
    model = BradleyTerryModel.from_dataset(dataset).fit()

    ranking = ranking_frame(model)
    objectives, objective_catalog = build_objectives(model)
    influence_details, loo_details, loo_summary = build_action_tables(model, objectives)
    ci_summaries, ci_comparison, ci_summary = summarize_ci_methods(model)

    dataset.frame.to_csv(output_dir / "synthetic_matches.csv", index=False)
    truth.to_csv(output_dir / "synthetic_truth.csv", index=False)
    ranking.to_csv(output_dir / "fitted_ranking.csv", index=False)
    objective_catalog.to_csv(output_dir / "objective_catalog.csv", index=False)
    ci_summaries.to_csv(output_dir / "ci_summaries.csv", index=False)
    ci_comparison.to_csv(output_dir / "ci_method_comparison_details.csv", index=False)
    ci_summary.to_csv(output_dir / "ci_method_correlation_summary.csv", index=False)
    influence_details.to_csv(output_dir / "all_action_influence_details.csv", index=False)
    loo_details.to_csv(output_dir / "drop_loo_vs_influence_details.csv", index=False)
    loo_summary.to_csv(output_dir / "drop_loo_vs_influence_summary.csv", index=False)

    metadata = {
        "n_players": int(args.n_players),
        "n_matches": int(args.n_matches),
        "seed": int(args.seed),
        "skill_scale": float(args.skill_scale),
        "tie_probability": float(args.tie_probability),
        "output_dir": str(output_dir),
        "actions": ["drop", "flip", "add"],
        "methods": ["if", "1sn"],
        "objectives": objective_catalog["objective_name"].tolist(),
        "ci_methods_compared": ["gao_local", "sandwich"],
        "add_candidate_mode": "all_outcomes",
        "drop_baseline": "exact leave-one-out refit delta",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved results to {output_dir}")
    print("\nGao vs sandwich CI correlation summary")
    print(ci_summary.to_string(index=False))
    print("\nInfluence vs exact leave-one-out summary")
    print(loo_summary.to_string(index=False))


if __name__ == "__main__":
    main()
