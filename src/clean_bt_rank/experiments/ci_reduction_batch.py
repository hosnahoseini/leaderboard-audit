from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..bt_model import BradleyTerryModel
from ..datasets import BattleDataset, load_named_battle_data
from .ci_reduction import run_ci_reduction_benchmark


DEFAULT_TARGET_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

METHOD_LABELS: dict[str, str] = {
    "influence": "Ours",
    "influence_pairs": "Ours V1",
    "influence_all_outcomes": "Ours V2",
    "influence_weighted_outcomes": "Ours V3",
    "influence_expected_pair": "Ours V4",
    "influence_v1": "Ours V1",
    "influence_v2": "Ours V2",
    "influence_v3": "Ours V3",
    "influence_v4": "Ours V4",
    "expected_pair": "Ours Active",
    "arena_active": "Arena Active",
    "arena_active_pair": "Arena Active Pair",
    "random": "Random",
    "random_pair": "Random Pair",
}

METHOD_COLORS: dict[str, str] = {
    "influence": "#264653",
    "influence_pairs": "#264653",
    "influence_all_outcomes": "#3D5A80",
    "influence_weighted_outcomes": "#6D597A",
    "influence_expected_pair": "#1D3557",
    "influence_v1": "#264653",
    "influence_v2": "#3D5A80",
    "influence_v3": "#6D597A",
    "influence_v4": "#1D3557",
    "expected_pair": "#1D3557",
    "arena_active": "#2A9D8F",
    "arena_active_pair": "#3A9B92",
    "random": "#E9C46A",
    "random_pair": "#F4A261",
}


@dataclass
class DatasetBenchmarkResult:
    dataset_key: str
    dataset_name: str
    leaderboard: pd.DataFrame
    target_table: pd.DataFrame
    task_metrics: pd.DataFrame
    task_curves: pd.DataFrame


@dataclass
class BatchBenchmarkResult:
    task_metrics: pd.DataFrame
    task_curves: pd.DataFrame
    target_table: pd.DataFrame
    summary_table: pd.DataFrame
    dataset_improvement_table: pd.DataFrame
    output_dir: Path | None = None


def fit_named_dataset_model(dataset_key: str) -> tuple[dict[str, object], BattleDataset, BradleyTerryModel]:
    loaded = load_named_battle_data(dataset_key)
    dataset = BattleDataset.from_dataframe(loaded.battle_frame.copy())
    bt_model = BradleyTerryModel.from_dataset(dataset, **loaded.fit_kwargs).fit()
    leaderboard = bt_model.summary(ci_method="sandwich").copy()
    leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)
    built = {
        "dataset_key": loaded.key,
        "dataset_name": loaded.display_name,
        "loaded": loaded,
        "leaderboard": leaderboard,
    }
    return built, dataset, bt_model


def choose_ci_targets(
    leaderboard: pd.DataFrame,
    battle_frame: pd.DataFrame,
    *,
    quantiles: Sequence[float] = DEFAULT_TARGET_QUANTILES,
    min_matches: int = 20,
) -> pd.DataFrame:
    counts = (
        pd.concat([battle_frame["model_a"], battle_frame["model_b"]], ignore_index=True)
        .astype(str)
        .value_counts()
        .rename_axis("competitor")
        .reset_index(name="n_matches")
    )
    merged = leaderboard.merge(counts, on="competitor", how="left")
    merged["n_matches"] = merged["n_matches"].fillna(0).astype(int)
    eligible = merged.loc[merged["n_matches"] >= int(min_matches)].copy()
    if eligible.empty:
        eligible = merged.copy()

    picked_rows: list[dict[str, object]] = []
    used: set[str] = set()
    n = len(eligible)
    for q in quantiles:
        pos = int(np.clip(round(float(q) * (n - 1)), 0, max(n - 1, 0)))
        row = eligible.iloc[pos]
        if str(row["competitor"]) not in used:
            picked_rows.append({"row": row, "target_quantile": float(q)})
            used.add(str(row["competitor"]))

    if not picked_rows:
        picked_rows.append({"row": eligible.iloc[len(eligible) // 2], "target_quantile": 0.5})

    target_table = pd.DataFrame([entry["row"] for entry in picked_rows]).reset_index(drop=True)
    target_table["target_quantile"] = [float(entry["target_quantile"]) for entry in picked_rows]
    return target_table[
        [
            "competitor",
            "rank",
            "rating",
            "ci_lower",
            "ci_upper",
            "standard_error",
            "n_matches",
            "target_quantile",
        ]
    ].rename(columns={"competitor": "target_player"})


def compute_normalized_ci_auc(history: pd.DataFrame, *, budget: int) -> float:
    ordered = history.sort_values("step")
    steps = ordered["step"].to_numpy(dtype=int)
    widths = ordered["ci_width"].to_numpy(dtype=float)
    if steps.size == 0:
        return float("nan")
    all_steps = np.arange(0, int(budget) + 1)
    padded = np.full(all_steps.shape, widths[-1], dtype=float)
    padded[steps] = widths
    padded = pd.Series(padded).ffill().to_numpy(dtype=float)
    initial = float(padded[0])
    if initial <= 0.0:
        return float("nan")
    return float(np.mean(padded / initial))


def summarize_task_metrics(
    history: pd.DataFrame,
    *,
    dataset_key: str,
    dataset_name: str,
    target_player: str,
    budget: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (policy, trial), group in history.groupby(["policy", "trial"], dropna=False):
        ordered = group.sort_values("step").reset_index(drop=True)
        initial_ci = float(ordered.iloc[0]["ci_width"])
        final_ci = float(ordered.iloc[-1]["ci_width"])
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "target_player": target_player,
                "policy": str(policy),
                "trial": int(trial),
                "budget": int(budget),
                "initial_ci_width": initial_ci,
                "final_ci_width": final_ci,
                "final_ci_ratio": final_ci / initial_ci if initial_ci > 0 else float("nan"),
                "ci_reduction": initial_ci - final_ci,
                "ci_reduction_fraction": (initial_ci - final_ci) / initial_ci if initial_ci > 0 else float("nan"),
                "nauc_ci_ratio": compute_normalized_ci_auc(ordered, budget=budget),
                "steps_completed": int(ordered["step"].max()),
            }
        )
    return pd.DataFrame(rows)


def run_named_dataset_ci_benchmark(
    dataset_key: str,
    *,
    budget: int = 12,
    target_quantiles: Sequence[float] = DEFAULT_TARGET_QUANTILES,
    min_matches: int = 20,
    n_random_trials: int = 30,
    ci_method: str = "gao_local",
    influence_method: str = "1sn",
    candidate_mode: str = "all_pairs",
    random_seed: int = 0,
    policies: Sequence[str] | None = None,
    random_policy: str = "random",
    outcome_mode: str = "deterministic",
) -> DatasetBenchmarkResult:
    built, dataset, bt_model = fit_named_dataset_model(dataset_key)
    target_table = choose_ci_targets(
        built["leaderboard"],
        built["loaded"].battle_frame,  # type: ignore[index]
        quantiles=target_quantiles,
        min_matches=min_matches,
    )
    target_table.insert(0, "dataset_key", built["dataset_key"])
    target_table.insert(1, "dataset_name", built["dataset_name"])

    all_metrics: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    for target_idx, target_row in target_table.reset_index(drop=True).iterrows():
        target_player = str(target_row["target_player"])
        _, _, history = run_ci_reduction_benchmark(
            bt_model,
            target_player=target_player,
            budget=budget,
            influence_method=influence_method,
            ci_method=ci_method,
            n_random_trials=n_random_trials,
            random_seed=random_seed + 1000 * target_idx,
            policies=list(policies) if policies is not None else ["influence", "arena_active"],
            candidate_mode=candidate_mode,
            random_policy=random_policy,
            outcome_mode=outcome_mode,
        )
        history = history.copy()
        history["dataset_key"] = built["dataset_key"]
        history["dataset_name"] = built["dataset_name"]
        history["target_player"] = target_player
        history["target_rank"] = int(target_row["rank"])
        history["target_quantile"] = float(target_row["target_quantile"])
        baseline_skill_range = float(history.sort_values("step").iloc[0]["skill_range"])
        if baseline_skill_range > 0:
            history["ci_width_ratio"] = history["ci_width"] / float(history.sort_values("step").iloc[0]["ci_width"])
        else:
            history["ci_width_ratio"] = np.nan
        all_curves.append(history)
        all_metrics.append(
            summarize_task_metrics(
                history,
                dataset_key=str(built["dataset_key"]),
                dataset_name=str(built["dataset_name"]),
                target_player=target_player,
                budget=budget,
            ).assign(
                target_rank=int(target_row["rank"]),
                target_quantile=float(target_row["target_quantile"]),
                target_match_count=int(target_row["n_matches"]),
            )
        )

    return DatasetBenchmarkResult(
        dataset_key=str(built["dataset_key"]),
        dataset_name=str(built["dataset_name"]),
        leaderboard=built["leaderboard"],  # type: ignore[index]
        target_table=target_table,
        task_metrics=pd.concat(all_metrics, ignore_index=True),
        task_curves=pd.concat(all_curves, ignore_index=True),
    )


def _pairwise_win_rate(task_metrics: pd.DataFrame, winner_policy: str, loser_policy: str) -> float:
    rows: list[float] = []
    keys = ["dataset_key", "target_player"]
    for _, group in task_metrics.groupby(keys):
        winner_vals = group.loc[group["policy"] == winner_policy, "nauc_ci_ratio"]
        loser_vals = group.loc[group["policy"] == loser_policy, "nauc_ci_ratio"]
        if winner_vals.empty or loser_vals.empty:
            continue
        rows.extend(float(w < l) for w in winner_vals for l in loser_vals)
    return float(np.mean(rows)) if rows else float("nan")


def build_summary_table(task_metrics: pd.DataFrame, *, primary_policy: str = "influence") -> pd.DataFrame:
    method_rows: list[dict[str, object]] = []
    policy_order = [
        "influence",
        "influence_pairs",
        "influence_all_outcomes",
        "influence_weighted_outcomes",
        "influence_expected_pair",
        "influence_v1",
        "influence_v2",
        "influence_v3",
        "influence_v4",
        "expected_pair",
        "arena_active",
        "arena_active_pair",
        "random",
        "random_pair",
    ]
    policies = [policy for policy in policy_order if policy in set(task_metrics["policy"].astype(str))]
    n_tasks = int(task_metrics[["dataset_key", "target_player"]].drop_duplicates().shape[0])
    for policy in policies:
        subset = task_metrics.loc[task_metrics["policy"] == policy].copy()
        row = {
            "method": METHOD_LABELS.get(policy, policy),
            "policy": policy,
            "mean_nauc_ci_ratio": float(subset["nauc_ci_ratio"].mean()),
            "median_final_ci_ratio": float(subset["final_ci_ratio"].median()),
            "mean_ci_reduction_fraction": float(subset["ci_reduction_fraction"].mean()),
            "win_rate_vs_primary": float("nan") if policy == primary_policy else _pairwise_win_rate(task_metrics, policy, primary_policy),
            "primary_policy": primary_policy,
            "n_tasks": n_tasks,
        }
        method_rows.append(row)
    return pd.DataFrame(method_rows)


def build_dataset_improvement_table(task_metrics: pd.DataFrame, *, primary_policy: str = "influence") -> pd.DataFrame:
    base_keys = ["dataset_key", "dataset_name", "target_player"]
    influence = (
        task_metrics.loc[task_metrics["policy"] == primary_policy, base_keys + ["nauc_ci_ratio"]]
        .rename(columns={"nauc_ci_ratio": "primary_nauc"})
        .drop_duplicates()
    )
    baseline_policies = [policy for policy in sorted(task_metrics["policy"].astype(str).unique()) if policy != primary_policy]
    rows: list[pd.DataFrame] = []
    for baseline in baseline_policies:
        base = (
            task_metrics.loc[task_metrics["policy"] == baseline, base_keys + ["nauc_ci_ratio"]]
            .groupby(base_keys, as_index=False)["nauc_ci_ratio"]
            .mean()
            .rename(columns={"nauc_ci_ratio": "baseline_nauc"})
        )
        merged = influence.merge(base, on=base_keys, how="inner")
        merged["baseline_policy"] = baseline
        merged["primary_policy"] = primary_policy
        merged["improvement_pct"] = 100.0 * (merged["baseline_nauc"] - merged["primary_nauc"]) / merged["baseline_nauc"]
        rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset_name",
                "baseline_policy",
                "primary_policy",
                "mean_improvement_pct",
                "median_improvement_pct",
                "n_targets",
            ]
        )

    task_level = pd.concat(rows, ignore_index=True)
    dataset_level = (
        task_level.groupby(["dataset_key", "dataset_name", "baseline_policy", "primary_policy"], as_index=False)
        .agg(
            mean_improvement_pct=("improvement_pct", "mean"),
            median_improvement_pct=("improvement_pct", "median"),
            n_targets=("target_player", "nunique"),
        )
    )
    overall = (
        task_level.groupby(["baseline_policy", "primary_policy"], as_index=False)
        .agg(
            mean_improvement_pct=("improvement_pct", "mean"),
            median_improvement_pct=("improvement_pct", "median"),
            n_targets=("target_player", "nunique"),
        )
    )
    overall["dataset_key"] = "all"
    overall["dataset_name"] = "All datasets"
    return pd.concat([dataset_level, overall], ignore_index=True)


def plot_dataset_improvements(dataset_improvement_table: pd.DataFrame, output_path: Path) -> Path:
    frame = dataset_improvement_table.copy()
    if frame.empty:
        return output_path
    order = frame["dataset_name"].drop_duplicates().tolist()
    if "All datasets" in order:
        order = [name for name in order if name != "All datasets"] + ["All datasets"]

    y_positions = np.arange(len(order))
    baselines = frame["baseline_policy"].drop_duplicates().tolist()
    offsets = np.linspace(-0.18, 0.18, num=max(len(baselines), 1))
    offset_map = {baseline: float(offset) for baseline, offset in zip(baselines, offsets)}
    primary_policy = str(frame["primary_policy"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.6 * len(order) + 0.8)))
    ax.axvline(0.0, color="#666666", lw=1.0, ls="--", alpha=0.7)

    for baseline in baselines:
        subset = frame.loc[frame["baseline_policy"] == baseline].copy()
        subset["y"] = subset["dataset_name"].map({name: idx for idx, name in enumerate(order)}).astype(float)
        subset["y"] = subset["y"] + offset_map[baseline]
        ax.scatter(
            subset["mean_improvement_pct"],
            subset["y"],
            s=60,
            color=METHOD_COLORS.get(baseline, "#666666"),
            label=f"{METHOD_LABELS.get(primary_policy, primary_policy)} vs {METHOD_LABELS.get(baseline, baseline)}",
            alpha=0.95,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(order)
    ax.set_xlabel("nAUC improvement of ours over baseline (%)")
    ax.set_title("CI reduction advantage across datasets")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_ci_reduction_batch(
    dataset_keys: Iterable[str],
    *,
    budget: int = 12,
    target_quantiles: Sequence[float] = DEFAULT_TARGET_QUANTILES,
    min_matches: int = 20,
    n_random_trials: int = 30,
    ci_method: str = "gao_local",
    influence_method: str = "1sn",
    candidate_mode: str = "all_pairs",
    random_seed: int = 0,
    output_dir: str | Path | None = None,
    policies: Sequence[str] | None = None,
    random_policy: str = "random",
    primary_policy: str = "influence",
    outcome_mode: str = "deterministic",
) -> BatchBenchmarkResult:
    dataset_keys = list(dataset_keys)
    saved_output_dir: Path | None = None
    dataset_results: list[DatasetBenchmarkResult] = []
    if output_dir is not None:
        saved_output_dir = Path(output_dir)
        saved_output_dir.mkdir(parents=True, exist_ok=True)
        (saved_output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "dataset_keys": list(dataset_keys),
                    "budget": budget,
                    "target_quantiles": list(target_quantiles),
                    "min_matches": min_matches,
                    "n_random_trials": n_random_trials,
                    "ci_method": ci_method,
                    "influence_method": influence_method,
                    "candidate_mode": candidate_mode,
                    "random_seed": random_seed,
                    "policies": list(policies) if policies is not None else None,
                    "random_policy": random_policy,
                    "primary_policy": primary_policy,
                    "outcome_mode": outcome_mode,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    for idx, dataset_key in enumerate(dataset_keys):
        result = run_named_dataset_ci_benchmark(
            dataset_key,
            budget=budget,
            target_quantiles=target_quantiles,
            min_matches=min_matches,
            n_random_trials=n_random_trials,
            ci_method=ci_method,
            influence_method=influence_method,
            candidate_mode=candidate_mode,
            random_seed=random_seed + 10_000 * idx,
            policies=policies,
            random_policy=random_policy,
            outcome_mode=outcome_mode,
        )
        dataset_results.append(result)

        if saved_output_dir is not None:
            task_metrics = pd.concat([r.task_metrics for r in dataset_results], ignore_index=True)
            task_curves = pd.concat([r.task_curves for r in dataset_results], ignore_index=True)
            target_table = pd.concat([r.target_table for r in dataset_results], ignore_index=True)
            summary_table = build_summary_table(task_metrics, primary_policy=primary_policy)
            dataset_improvement_table = build_dataset_improvement_table(task_metrics, primary_policy=primary_policy)

            task_metrics.to_csv(saved_output_dir / "task_metrics.csv", index=False)
            task_curves.to_csv(saved_output_dir / "task_curves.csv", index=False)
            target_table.to_csv(saved_output_dir / "target_table.csv", index=False)
            summary_table.to_csv(saved_output_dir / "summary_table.csv", index=False)
            dataset_improvement_table.to_csv(saved_output_dir / "dataset_improvement_table.csv", index=False)
            plot_dataset_improvements(dataset_improvement_table, saved_output_dir / "dataset_improvement_forest.png")
            (saved_output_dir / "progress.txt").write_text(
                "\n".join(
                    [
                        f"completed: {len(dataset_results)}/{len(dataset_keys)}",
                        "datasets:",
                        *[f"- {r.dataset_key}" for r in dataset_results],
                    ]
                )
                + "\n"
            )

    task_metrics = pd.concat([result.task_metrics for result in dataset_results], ignore_index=True)
    task_curves = pd.concat([result.task_curves for result in dataset_results], ignore_index=True)
    target_table = pd.concat([result.target_table for result in dataset_results], ignore_index=True)
    summary_table = build_summary_table(task_metrics, primary_policy=primary_policy)
    dataset_improvement_table = build_dataset_improvement_table(task_metrics, primary_policy=primary_policy)

    if output_dir is not None:
        task_metrics.to_csv(saved_output_dir / "task_metrics.csv", index=False)
        task_curves.to_csv(saved_output_dir / "task_curves.csv", index=False)
        target_table.to_csv(saved_output_dir / "target_table.csv", index=False)
        summary_table.to_csv(saved_output_dir / "summary_table.csv", index=False)
        dataset_improvement_table.to_csv(saved_output_dir / "dataset_improvement_table.csv", index=False)
        plot_dataset_improvements(dataset_improvement_table, saved_output_dir / "dataset_improvement_forest.png")

    return BatchBenchmarkResult(
        task_metrics=task_metrics,
        task_curves=task_curves,
        target_table=target_table,
        summary_table=summary_table,
        dataset_improvement_table=dataset_improvement_table,
        output_dir=saved_output_dir,
    )
