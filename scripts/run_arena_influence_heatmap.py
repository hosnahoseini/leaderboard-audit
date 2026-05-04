#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
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
import seaborn as sns

from clean_bt_rank import (
    BradleyTerryModel,
    CIBoundaryObjective,
    KendallTauObjective,
    PlayerUncertaintyObjective,
    SkillGapObjective,
    TraceUncertaintyObjective,
    build_action_candidate_report,
    load_named_battle_data,
    top_absolute,
    use_paper_rc,
)
from clean_bt_rank.ci_aware_actions_needed import build_named_dataset_model

ACTION_SPECS = {
    "drop": {"action": "drop", "candidate_mode": None},
    "add_all_outcomes": {"action": "add", "candidate_mode": "all_outcomes"},
    "add_all_pairs": {"action": "add", "candidate_mode": "all_pairs"},
    "add_weighted": {"action": "add", "candidate_mode": "weighted"},
    "flip": {"action": "flip", "candidate_mode": None},
}
COVARIATES = ["match_count", "bridge_variance", "closeness_log_gap", "surprise"]
METRIC_LABELS = {
    "match_count": "Match count",
    "bridge_variance": "Bridge var.",
    "closeness_log_gap": "Closeness\n(ln-gap)",
    "surprise": "Surprise",
}
EPS = 1e-12
TOP_K = 20
use_paper_rc()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Arena55k notebook-style influence heatmaps.")
    parser.add_argument("--dataset", default="arena55k", help="Dataset key. Default: arena55k.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "new_result" / "arena_ablation_reset",
        help="Directory for CSV/PNG outputs.",
    )
    return parser.parse_args()


def build_default_objectives(bt_model: BradleyTerryModel) -> dict[str, object]:
    summary = bt_model.summary(ci_method="gao_local").reset_index(drop=True)
    ranking = summary["competitor"].tolist()
    top_model = str(ranking[0])
    second_model = str(ranking[1])
    return {
        f"skill_gap__{top_model}__vs__{second_model}": SkillGapObjective(top_model, second_model),
        f"ci_boundary__top1_top2__{top_model}__vs__{second_model}": CIBoundaryObjective(
            bt_model, k=1, ci_method="gao_local", alpha=0.05, freeze_se=False
        ),
        f"player_uncertainty__{top_model}": PlayerUncertaintyObjective(top_model),
        "trace_uncertainty": TraceUncertaintyObjective(),
        "kendall_tau__fitted_ranking__T0.5": KendallTauObjective(ranking=ranking, temperature=0.5),
    }


def focus_players_from_objective_key(objective_key: str) -> set[str]:
    parts = str(objective_key).split("__")
    if objective_key.startswith("skill_gap__") and len(parts) >= 4:
        return {parts[1], parts[3]}
    if objective_key.startswith("player_uncertainty__") and len(parts) >= 2:
        return {parts[1]}
    if objective_key.startswith("ci_boundary__") and "vs" in parts:
        idx = parts.index("vs")
        if idx - 1 >= 0 and idx + 1 < len(parts):
            return {parts[idx - 1], parts[idx + 1]}
    return set()


def make_covariate_augmenter(bt_model: BradleyTerryModel):
    skill_by_player = dict(zip(bt_model.competitor_names_, bt_model.reported_skills()))
    pair_freq = (
        bt_model.match_frame_
        .assign(pair_key=lambda d: d.apply(lambda r: tuple(sorted((str(r["model_a"]), str(r["model_b"])))), axis=1))
        .groupby("pair_key")
        .size()
        .to_dict()
    )

    def augment(report: pd.DataFrame) -> pd.DataFrame:
        enriched = report.copy()
        enriched["model_a"] = enriched["model_a"].astype(str)
        enriched["model_b"] = enriched["model_b"].astype(str)
        enriched["skill_a"] = enriched["model_a"].map(skill_by_player)
        enriched["skill_b"] = enriched["model_b"].map(skill_by_player)
        enriched["skill_gap_abs"] = (enriched["skill_a"] - enriched["skill_b"]).abs()
        enriched["pair_key"] = enriched.apply(lambda r: tuple(sorted((r["model_a"], r["model_b"]))), axis=1)
        enriched["match_count"] = enriched["pair_key"].map(pair_freq).fillna(0).astype(int)
        enriched["bridge_variance"] = ((enriched["skill_a"] - enriched["skill_b"]) ** 2) / 4.0
        enriched["closeness_log_gap"] = np.log(enriched["skill_gap_abs"].clip(lower=EPS))
        observed = enriched["outcome"] if "outcome" in enriched.columns else enriched["winner"]
        win_prob = 1.0 / (1.0 + np.exp(-(enriched["skill_a"] - enriched["skill_b"])))
        enriched["surprise"] = (observed.astype(float) - win_prob).abs()
        return enriched

    return augment


def short_row_label(objective_key: str, action: str) -> str:
    action_short = {
        "drop": "drop",
        "add_all_outcomes": "add-out",
        "add_all_pairs": "add-pair",
        "add_weighted": "add-w",
        "flip": "flip",
    }
    core = str(objective_key).replace("__", " · ")
    if len(core) > 34:
        core = core[:33] + "…"
    return f"{action_short.get(action, action)} | {core}"


def save_heatmap(frame: pd.DataFrame, *, value_cols: list[str], title: str, cbar_label: str, output_path: Path) -> None:
    fig_height = max(4.8, 0.32 * max(1, len(frame)))
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    heat = frame.set_index("row_label")[value_cols]
    sns.heatmap(
        heat,
        cmap="coolwarm",
        center=0.0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    built = build_named_dataset_model(str(args.dataset))
    bt_model = built["bt_model"]
    objectives = build_default_objectives(bt_model)
    augment = make_covariate_augmenter(bt_model)

    corr_rows: list[dict[str, object]] = []
    focus_rows: list[dict[str, object]] = []

    for objective_key, objective in objectives.items():
        focus_players = focus_players_from_objective_key(objective_key)
        for action_key, config in ACTION_SPECS.items():
            report = build_action_candidate_report(
                bt_model,
                objective,
                config["action"],
                influence_method="1sn",
                candidate_mode=config["candidate_mode"],
            )
            enriched = augment(report)
            enriched.to_csv(output_dir / f"{built['dataset_key']}__{action_key}__{objective_key}__1sn.csv", index=False)

            row = {"objective_key": objective_key, "action": action_key}
            for cov in COVARIATES:
                row[cov] = float(enriched["influence_abs"].corr(enriched[cov], method="spearman"))
            corr_rows.append(row)

            top_rep = top_absolute(enriched, "influence", k=TOP_K).copy()
            contains_focus = (
                top_rep.apply(
                    lambda r: (str(r["model_a"]) in focus_players) or (str(r["model_b"]) in focus_players),
                    axis=1,
                )
                if focus_players and len(top_rep)
                else pd.Series(dtype=bool)
            )
            focus_rows.append(
                {
                    "objective_key": objective_key,
                    "action": action_key,
                    "focus_players": ", ".join(sorted(focus_players)),
                    "top_k": TOP_K,
                    "focus_hit_rate": float(contains_focus.mean()) if len(contains_focus) else np.nan,
                    "focus_match_count": int(contains_focus.sum()) if len(contains_focus) else 0,
                }
            )

    corr_df = pd.DataFrame(corr_rows).sort_values(["objective_key", "action"]).reset_index(drop=True)
    corr_df["row_label"] = [short_row_label(ok, ac) for ok, ac in zip(corr_df["objective_key"], corr_df["action"])]
    corr_df = corr_df.rename(columns=METRIC_LABELS)
    corr_df.to_csv(output_dir / f"{built['dataset_key']}__influence_covariate_correlation__1sn.csv", index=False)
    save_heatmap(
        corr_df,
        value_cols=list(METRIC_LABELS.values()),
        title=f"{built['dataset_name']}: influence-covariate correlations",
        cbar_label="Spearman rho",
        output_path=output_dir / f"{built['dataset_key']}__influence_covariate_correlation__1sn.png",
    )

    focus_df = pd.DataFrame(focus_rows).sort_values(["objective_key", "action"]).reset_index(drop=True)
    focus_df["row_label"] = [short_row_label(ok, ac) for ok, ac in zip(focus_df["objective_key"], focus_df["action"])]
    focus_df.to_csv(output_dir / f"{built['dataset_key']}__influence_focus_hit_at_20__1sn.csv", index=False)
    save_heatmap(
        focus_df,
        value_cols=["focus_hit_rate"],
        title=f"{built['dataset_name']}: top-{TOP_K} focus hit rate",
        cbar_label="Hit rate",
        output_path=output_dir / f"{built['dataset_key']}__influence_focus_hit_at_20__1sn.png",
    )
    print(f"Saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
