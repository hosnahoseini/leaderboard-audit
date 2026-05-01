from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT, ROOT / "IsRankingRobust"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from clean_bt_rank import (
    ACTION_LABEL_MAP,
    KendallTauObjective,
    available_hf_battle_datasets,
    plot_variant_order_curves,
    ranking_from_model,
    run_objective_curve_steps,
    use_paper_rc,
)
from clean_bt_rank.ci_aware_actions_needed import build_named_dataset_model
from clean_bt_rank.iterative_actions import compute_all_action_influences
from clean_bt_rank.objectives import TraceUncertaintyObjective


VARIANTS = [
    {"variant": "drop", "action": "drop", "candidate_mode": None},
    {"variant": "flip", "action": "flip", "candidate_mode": None},
    {"variant": "add_pairs", "action": "add", "candidate_mode": "all_pairs"},
    {"variant": "add_outcomes", "action": "add", "candidate_mode": "all_outcomes"},
    {"variant": "add_weighted", "action": "add", "candidate_mode": "weighted"},
]

use_paper_rc()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Kendall-tau and trace-uncertainty proxy curves versus number of actions."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset keys to run. Default: all datasets supported by the CI-aware loader.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "notebooks" / "artifacts" / "tau_ci_curve_analysis",
        help="Directory for CSV/PDF/PNG outputs.",
    )
    parser.add_argument(
        "--curve-steps",
        type=int,
        default=25,
        help="Maximum number of actions to evaluate on each curve. Default: 25.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Base seed for randomized-order baselines. Default: 7.",
    )
    parser.add_argument(
        "--tau-temperature",
        type=float,
        default=0.5,
        help="Temperature for KendallTauObjective. Default: 0.5.",
    )
    parser.add_argument(
        "--replot-from-history",
        action="store_true",
        help="Rebuild figures from existing *_curve_history.csv files in --output-dir without recomputing curves.",
    )
    return parser.parse_args()


def make_selection_report(report: pd.DataFrame, *, ascending: bool, random_seed: int | None = None) -> pd.DataFrame:
    work = report.copy()
    if random_seed is None:
        signed = work["influence"].astype(float)
        work["influence"] = -signed if ascending else signed
    else:
        rng = np.random.default_rng(random_seed)
        work["influence"] = rng.random(len(work))
    return work


def run_order_curve(bt_model, objective, action: str, selection_report: pd.DataFrame, *, steps: int) -> pd.DataFrame:
    return run_objective_curve_steps(
        bt_model,
        objective,
        action,
        selection_report,
        steps=steps,
        recompute_mode="refit",
    )


def savefig_both(fig: plt.Figure, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)


def choose_legend_location(curve_map: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> str:
    values: list[np.ndarray] = []
    for greedy_curve, random_curve in curve_map.values():
        values.append(greedy_curve["plot_value"].to_numpy(dtype=float))
        values.append(random_curve["plot_value"].to_numpy(dtype=float))
    all_values = np.concatenate(values) if values else np.array([0.0], dtype=float)
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size == 0:
        return "best"
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if np.isclose(y_min, y_max):
        return "best"
    normalized_median = float((np.median(finite_values) - y_min) / (y_max - y_min))
    return "lower right" if normalized_median >= 0.58 else "upper right"


def choose_y_limits(
    curve_map: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    objective_key: str,
) -> tuple[float, float] | None:
    values: list[np.ndarray] = []
    for greedy_curve, random_curve in curve_map.values():
        values.append(greedy_curve["plot_value"].to_numpy(dtype=float))
        values.append(random_curve["plot_value"].to_numpy(dtype=float))

    all_values = np.concatenate(values) if values else np.array([], dtype=float)
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size == 0:
        return None

    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    baseline = 1.0 if objective_key == "kendall_tau" else None
    if baseline is not None:
        y_min = min(y_min, baseline)
        y_max = max(y_max, baseline)

    span = y_max - y_min
    pad = max(span * 0.08, 1e-6) if not np.isclose(span, 0.0) else max(abs(y_min) * 0.05, 1e-6)
    return (y_min - pad, y_max + pad)


def prepare_trace_relative_curve(curve: pd.DataFrame) -> pd.DataFrame:
    out = curve.copy()
    baseline = out["initial_value"].astype(float).iloc[0]
    denom = abs(float(baseline))
    if np.isclose(denom, 0.0):
        out["plot_value"] = np.nan
    else:
        out["plot_value"] = 100.0 * out["delta_from_initial"].astype(float) / denom
    return out


def percent_formatter(value: float, _pos: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:,.0f}%"
    if abs_value >= 10:
        return f"{value:.0f}%"
    if abs_value >= 1:
        return f"{value:.1f}%"
    return f"{value:.2f}%"


def augment_curve(
    curve: pd.DataFrame,
    *,
    dataset_key: str,
    dataset_name: str,
    objective_key: str,
    variant: str,
    action: str,
    candidate_mode: str | None,
    order_mode: str,
    initial_value: float,
) -> pd.DataFrame:
    out = curve.copy()
    out.insert(0, "dataset_key", dataset_key)
    out.insert(1, "dataset_name", dataset_name)
    out.insert(2, "objective_key", objective_key)
    out.insert(3, "variant", variant)
    out.insert(4, "action", action)
    out.insert(5, "candidate_mode", candidate_mode)
    out.insert(6, "order_mode", order_mode)
    out["variant_label"] = ACTION_LABEL_MAP.get(variant, variant)
    out["initial_value"] = float(initial_value)
    out["delta_from_initial"] = out["objective_value"] - float(initial_value)
    if objective_key == "kendall_tau":
        denom = float(initial_value)
        out["normalized_value"] = np.nan if np.isclose(denom, 0.0) else out["objective_value"] / denom
    else:
        out["normalized_value"] = np.nan
    return out


def summarize_curve_pair(
    greedy_curve: pd.DataFrame,
    random_curve: pd.DataFrame,
    *,
    dataset_key: str,
    dataset_name: str,
    objective_key: str,
    variant: str,
    action: str,
    candidate_mode: str | None,
) -> dict[str, object]:
    row = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "objective_key": objective_key,
        "variant": variant,
        "action": action,
        "candidate_mode": candidate_mode,
        "initial_value": float(greedy_curve["objective_value"].iloc[0]),
        "greedy_final_value": float(greedy_curve["objective_value"].iloc[-1]),
        "random_final_value": float(random_curve["objective_value"].iloc[-1]),
        "greedy_steps_completed": int(greedy_curve["step"].iloc[-1]),
        "random_steps_completed": int(random_curve["step"].iloc[-1]),
    }
    if objective_key == "kendall_tau":
        row["initial_normalized_value"] = float(greedy_curve["normalized_value"].iloc[0])
        row["greedy_final_normalized_value"] = float(greedy_curve["normalized_value"].iloc[-1])
        row["random_final_normalized_value"] = float(random_curve["normalized_value"].iloc[-1])
    return row


def plot_curve_panel(
    curve_map: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    title: str,
    ylabel: str,
    path_stem: Path,
    objective_key: str,
) -> None:
    fig, ax = plot_variant_order_curves(
        curve_map,
        title=title,
        ylabel=ylabel,
        baseline=1.0 if objective_key == "kendall_tau" else 0.0,
    )
    if objective_key == "trace_uncertainty":
        ax.set_yscale("symlog", linthresh=1.0, base=10)
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    else:
        y_limits = choose_y_limits(curve_map, objective_key=objective_key)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.legend(
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handlelength=2.5,
        loc=choose_legend_location(curve_map),
    )
    savefig_both(fig, path_stem)
    plt.close(fig)


def build_curve_map_from_history(history_df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    curve_map: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for variant, variant_df in history_df.groupby("variant", sort=False):
        greedy_curve = variant_df[variant_df["order_mode"] == "greedy"].sort_values("step").copy()
        random_curve = variant_df[variant_df["order_mode"] == "random"].sort_values("step").copy()
        if greedy_curve.empty or random_curve.empty:
            continue
        curve_map[str(variant)] = (greedy_curve, random_curve)
    return curve_map


def replot_from_history(output_dir: Path) -> None:
    history_paths = sorted(output_dir.glob("*_curve_history.csv"))
    if not history_paths:
        raise FileNotFoundError(f"No *_curve_history.csv files found in {output_dir}")

    for history_path in history_paths:
        if history_path.name == "all_datasets_curve_history.csv":
            continue
        history_df = pd.read_csv(history_path)
        if history_df.empty:
            continue
        dataset_key = str(history_df["dataset_key"].iloc[0])
        dataset_name = str(history_df["dataset_name"].iloc[0])

        tau_df = history_df[history_df["objective_key"] == "kendall_tau"].copy()
        if not tau_df.empty:
            tau_curve_map = build_curve_map_from_history(tau_df)
            plot_curve_panel(
                tau_curve_map,
                title=f"{dataset_name}: Kendall tau vs actions",
                ylabel="Normalized Kendall-tau surrogate",
                path_stem=output_dir / f"{dataset_key}_kendall_tau_curves",
                objective_key="kendall_tau",
            )

        trace_df = history_df[history_df["objective_key"] == "trace_uncertainty"].copy()
        if not trace_df.empty:
            trace_df = prepare_trace_relative_curve(trace_df)
            trace_curve_map = build_curve_map_from_history(trace_df)
            plot_curve_panel(
                trace_curve_map,
                title=f"{dataset_name}: trace uncertainty change vs actions",
                ylabel="Trace uncertainty change vs start (%)",
                path_stem=output_dir / f"{dataset_key}_trace_uncertainty_curves",
                objective_key="trace_uncertainty",
            )


def run_dataset(
    built: dict[str, object],
    *,
    output_dir: Path,
    curve_steps: int,
    random_seed: int,
    tau_temperature: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    bt_model = built["bt_model"]
    ranking_order = ranking_from_model(bt_model)
    tau_obj = KendallTauObjective(ranking=ranking_order, temperature=tau_temperature)
    ci_obj = TraceUncertaintyObjective()

    objective_specs = [
        ("kendall_tau", tau_obj, "Normalized Kendall-tau surrogate", True),
        ("trace_uncertainty", ci_obj, "Trace uncertainty proxy", False),
    ]

    curve_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for objective_key, objective, ylabel, normalize_for_plot in objective_specs:
        curve_map: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for offset, spec in enumerate(VARIANTS):
            base_report = compute_all_action_influences(
                bt_model,
                objective,
                spec["action"],
                influence_method="1sn",
                candidate_mode=spec["candidate_mode"],
            )
            greedy_report = make_selection_report(base_report, ascending=True)
            random_report = make_selection_report(
                base_report,
                ascending=True,
                random_seed=random_seed + (100 if objective_key == "trace_uncertainty" else 0) + offset,
            )
            greedy_curve = run_order_curve(bt_model, objective, spec["action"], greedy_report, steps=curve_steps)
            random_curve = run_order_curve(bt_model, objective, spec["action"], random_report, steps=curve_steps)

            greedy_curve = augment_curve(
                greedy_curve,
                dataset_key=built["dataset_key"],
                dataset_name=built["dataset_name"],
                objective_key=objective_key,
                variant=spec["variant"],
                action=spec["action"],
                candidate_mode=spec["candidate_mode"],
                order_mode="greedy",
                initial_value=float(greedy_curve["objective_value"].iloc[0]),
            )
            random_curve = augment_curve(
                random_curve,
                dataset_key=built["dataset_key"],
                dataset_name=built["dataset_name"],
                objective_key=objective_key,
                variant=spec["variant"],
                action=spec["action"],
                candidate_mode=spec["candidate_mode"],
                order_mode="random",
                initial_value=float(random_curve["objective_value"].iloc[0]),
            )
            if objective_key == "trace_uncertainty":
                greedy_curve = prepare_trace_relative_curve(greedy_curve)
                random_curve = prepare_trace_relative_curve(random_curve)
            else:
                greedy_curve["plot_value"] = (
                    greedy_curve["normalized_value"] if normalize_for_plot else greedy_curve["objective_value"]
                )
                random_curve["plot_value"] = (
                    random_curve["normalized_value"] if normalize_for_plot else random_curve["objective_value"]
                )
            curve_map[spec["variant"]] = (greedy_curve, random_curve)
            curve_tables.extend([greedy_curve, random_curve])
            summary_rows.append(
                summarize_curve_pair(
                    greedy_curve,
                    random_curve,
                    dataset_key=built["dataset_key"],
                    dataset_name=built["dataset_name"],
                    objective_key=objective_key,
                    variant=spec["variant"],
                    action=spec["action"],
                    candidate_mode=spec["candidate_mode"],
                )
            )

        plot_title = (
            f"{built['dataset_name']}: Kendall tau vs actions"
            if objective_key == "kendall_tau"
            else f"{built['dataset_name']}: trace uncertainty proxy vs actions"
        )
        path_stem = output_dir / f"{built['dataset_key']}_{objective_key}_curves"
        plot_curve_panel(
            curve_map,
            title=plot_title,
            ylabel=ylabel,
            path_stem=path_stem,
            objective_key=objective_key,
        )

    curves_df = pd.concat(curve_tables, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    meta = {
        "dataset_key": built["dataset_key"],
        "dataset_name": built["dataset_name"],
        "n_original_matches": int(len(built["raw"])),
        "n_fitted_rows": int(built["dataset"].n_matches),
        "n_models": int(len(ranking_order)),
        "top1": ranking_order[0],
        "top2": ranking_order[1],
        "tau_temperature": float(tau_temperature),
        "curve_steps_requested": int(curve_steps),
    }
    return curves_df, summary_df, meta


def load_dataset_builds(dataset_keys: list[str]) -> list[dict[str, object]]:
    builds: list[dict[str, object]] = []
    for dataset_key in dataset_keys:
        print(f"Loading {dataset_key} ...", flush=True)
        builds.append(build_named_dataset_model(dataset_key))
    builds.sort(key=lambda built: (int(built["dataset"].n_matches), str(built["dataset_key"])))
    return builds


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_from_history:
        replot_from_history(output_dir)
        print(f"Rebuilt plots from existing CSV history in {output_dir}", flush=True)
        return

    dataset_keys = sorted(available_hf_battle_datasets()) if args.datasets is None else list(args.datasets)
    dataset_builds = load_dataset_builds(dataset_keys)

    all_curve_rows: list[pd.DataFrame] = []
    all_summary_rows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []

    for built in dataset_builds:
        print(
            f"Running {built['dataset_key']} ({int(built['dataset'].n_matches)} fitted rows) ...",
            flush=True,
        )
        curves_df, summary_df, meta = run_dataset(
            built,
            output_dir=output_dir,
            curve_steps=args.curve_steps,
            random_seed=args.random_seed,
            tau_temperature=args.tau_temperature,
        )
        curves_df.to_csv(output_dir / f"{meta['dataset_key']}_curve_history.csv", index=False)
        summary_df.to_csv(output_dir / f"{meta['dataset_key']}_curve_summary.csv", index=False)
        metadata_rows.append(meta)
        all_curve_rows.append(curves_df)
        all_summary_rows.append(summary_df.assign(**meta))

    metadata_df = pd.DataFrame(metadata_rows)
    all_curves_df = pd.concat(all_curve_rows, ignore_index=True) if all_curve_rows else pd.DataFrame()
    all_summary_df = pd.concat(all_summary_rows, ignore_index=True) if all_summary_rows else pd.DataFrame()

    metadata_df.to_csv(output_dir / "all_datasets_curve_metadata.csv", index=False)
    all_curves_df.to_csv(output_dir / "all_datasets_curve_history.csv", index=False)
    all_summary_df.to_csv(output_dir / "all_datasets_curve_summary.csv", index=False)
    print(f"Saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
