from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clean_bt_rank import (
    BTParameterInfluence,
    BattleDataset,
    BradleyTerryModel,
    ObjectiveInfluence,
    SkillGapObjective,
    generate_synthetic_dataset,
    make_objective_influence_report,
    plot_ratings,
    top_absolute,
)


def run_arena_demo() -> None:
    arena_df = pd.DataFrame(
        [
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "beta", "winner": "model_a"},
            {"model_a": "alpha", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "beta", "model_b": "gamma", "winner": "model_a"},
            {"model_a": "gamma", "model_b": "alpha", "winner": "model_b"},
            {"model_a": "beta", "model_b": "alpha", "winner": "tie"},
        ]
    )

    dataset = BattleDataset.from_dataframe(arena_df)
    model = BradleyTerryModel.from_dataset(dataset).fit()

    sandwich = model.summary(ci_method="sandwich")
    bootstrap = model.summary(ci_method="bootstrap", n_bootstrap=100, seed=0)
    param_infl = BTParameterInfluence(model)
    gap_obj = SkillGapObjective("alpha", "beta")
    obj_infl = ObjectiveInfluence(model, param_infl)
    gap_if = obj_infl.compute_match_influence(gap_obj, method="if")
    gap_1sn = obj_infl.compute_match_influence(gap_obj, method="1sn")
    report = make_objective_influence_report(model, gap_obj, gap_1sn, influence_name="gap_1sn")

    print("\nArena summary with sandwich CIs")
    print(sandwich.to_string(index=False))
    print("\nArena summary with bootstrap CIs")
    print(bootstrap.to_string(index=False))
    print("\nGap objective value beta_alpha - beta_beta")
    print(gap_obj.value(model))
    print("\nTop absolute IF influence on alpha - beta")
    print(top_absolute(make_objective_influence_report(model, gap_obj, gap_if, influence_name="gap_if"), "gap_if", k=5).to_string(index=False))
    print("\nTop absolute 1sN influence on alpha - beta")
    print(top_absolute(report, "gap_1sn", k=5).to_string(index=False))

    fig, _ = plot_ratings(sandwich, title="Arena Demo")
    fig.savefig("arena_demo.png", dpi=160)


def run_synthetic_demo() -> None:
    dataset, truth = generate_synthetic_dataset(n_models=8, n_matches=3000, seed=7, tie_probability=0.05)
    model = BradleyTerryModel.from_dataset(dataset).fit()
    summary = model.summary(ci_method="sandwich")

    print("\nSynthetic truth")
    print(truth.to_string(index=False))
    print("\nSynthetic fit")
    print(summary.to_string(index=False))

    fig, _ = plot_ratings(summary, top_n=8, title="Synthetic Demo")
    fig.savefig("synthetic_demo.png", dpi=160)


if __name__ == "__main__":
    run_arena_demo()
    run_synthetic_demo()
