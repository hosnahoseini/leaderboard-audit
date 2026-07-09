from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT / "src", ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_ACTIONS = ["drop", "flip", "add_pairs", "add_outcomes", "add_weighted"]
BREAK_MISSING_TOKENS = {"robust", "none", "nan", "na", "n/a", "missing", ""}
LOGGER = logging.getLogger("compute_dataset_robustness_scores")


@dataclass
class DatasetAccumulator:
    dataset: str
    n_matches: int | None = None
    topk_counts: dict[str, Any] = field(default_factory=dict)
    trace_curves: dict[str, pd.Series] = field(default_factory=dict)
    tau_curves: dict[str, pd.Series] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


DATASET_DISPLAY_NAMES = {
    "arena55k": "Arena 55k",
    "llm_judge_arena": "Arena LLM-J",
    "mt_bench_human": "MT-Bench",
    "nba_elo_top50": "NBA Top-50",
    "tennis_top10_atp": "ATP Top-10",
    "vision_arena": "Vision Arena",
    "webdev_arena": "WebDev Arena",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset-level robustness scores from existing experiment outputs.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=ROOT / "new_result",
        help="Root directory containing robustness result files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "new_result" / "dataset_robustness_scores",
        help="Directory for aggregated CSV/LaTeX outputs.",
    )
    parser.add_argument(
        "--budget_frac",
        type=float,
        default=0.05,
        help="Budget fraction used to define B_max = budget_frac * N.",
    )
    parser.add_argument(
        "--tau_exclude_initial",
        dest="tau_exclude_initial",
        action="store_true",
        default=True,
        help="Exclude b=0 from tau averaging when step-0 points are present.",
    )
    parser.add_argument(
        "--include_tau_initial",
        dest="tau_exclude_initial",
        action="store_false",
        help="Include b=0 in tau averaging.",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=list(DEFAULT_ACTIONS),
        help="Action variants to include.",
    )
    parser.add_argument(
        "--print_markdown",
        action="store_true",
        help="Print a markdown table to stdout.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def canonicalize_action(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    mapping = {
        "drop": "drop",
        "flip": "flip",
        "add_pairs": "add_pairs",
        "all_pairs": "add_pairs",
        "add_all_pairs": "add_pairs",
        "add_outcomes": "add_outcomes",
        "all_outcomes": "add_outcomes",
        "add_all_outcomes": "add_outcomes",
        "add_weighted": "add_weighted",
        "weighted": "add_weighted",
    }
    return mapping.get(text, text)


def canonicalize_dataset(value: Any) -> str:
    return str(value).strip()


def load_table(path: Path) -> pd.DataFrame | list[dict[str, Any]] | dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError(f"Unsupported file type: {path}")


def parse_action_count(value: Any, B_max: float) -> float:
    if value is None:
        return float(B_max)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in BREAK_MISSING_TOKENS:
            return float(B_max)
        try:
            value = float(text)
        except ValueError:
            return float(B_max)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(B_max)
    if not np.isfinite(numeric):
        return float(B_max)
    return max(0.0, numeric)


def compute_break_score(action_to_count: dict[str, Any], B_max: float) -> tuple[float, str | None, float]:
    if not action_to_count:
        return (np.nan, None, np.nan)
    parsed = {action: parse_action_count(value, B_max) for action, value in action_to_count.items()}
    worst_action, min_actions = min(parsed.items(), key=lambda item: (item[1], item[0]))
    score = min(1.0, float(min_actions) / float(B_max)) if B_max > 0 else np.nan
    return (score, worst_action, float(min_actions))


def compute_tau_score(
    action_to_tau_curve: dict[str, pd.Series],
    exclude_initial: bool = True,
) -> tuple[float, str | None, float]:
    if not action_to_tau_curve:
        return (np.nan, None, np.nan)
    action_means: dict[str, float] = {}
    for action, curve in action_to_tau_curve.items():
        if curve is None or len(curve) == 0:
            continue
        series = pd.Series(curve, dtype=float).dropna()
        if series.empty:
            continue
        if exclude_initial and len(series) > 1:
            min_step = series.index.min()
            series = series.loc[series.index != min_step]
        if series.empty:
            continue
        action_means[action] = float(series.mean())
    if not action_means:
        return (np.nan, None, np.nan)
    worst_action, tau_auc = min(action_means.items(), key=lambda item: (item[1], item[0]))
    return (float(tau_auc), worst_action, float(tau_auc))


def compute_trace_score(action_to_trace_curve: dict[str, pd.Series]) -> tuple[float, str | None, float]:
    if not action_to_trace_curve:
        return (np.nan, None, np.nan)
    action_scores: dict[str, float] = {}
    action_finals: dict[str, float] = {}
    for action, curve in action_to_trace_curve.items():
        if curve is None or len(curve) == 0:
            continue
        series = pd.Series(curve, dtype=float).dropna().sort_index()
        if series.empty:
            continue
        initial_value = float(series.iloc[0])
        final_value = float(series.iloc[-1])
        if not np.isfinite(initial_value) or initial_value <= 0.0 or not np.isfinite(final_value):
            continue
        action_scores[action] = final_value / initial_value
        action_finals[action] = final_value
    if not action_scores:
        return (np.nan, None, np.nan)
    worst_action, score = min(action_scores.items(), key=lambda item: (item[1], item[0]))
    return (float(score), worst_action, float(action_finals[worst_action]))


def compute_all_score(R_topk: float, R_trace: float, R_tau: float) -> float:
    values = np.array([R_topk, R_trace, R_tau], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(finite.mean())


def format_latex_table(df: pd.DataFrame) -> str:
    display = df.rename(
        columns={
            "dataset_display": "Dataset",
            "R_topk": r"$\boldsymbol{R_{\mathrm{Top}\text{-}1}}$",
            "R_CITrace": r"$\boldsymbol{R_{\mathrm{CITrace}}}$",
            "R_tau": r"$\boldsymbol{R_{\tau}}$",
            "R_all": r"$\boldsymbol{R_{\mathrm{all}}}$",
        }
    )[["Dataset", r"$\boldsymbol{R_{\mathrm{Top}\text{-}1}}$", r"$\boldsymbol{R_{\mathrm{CITrace}}}$", r"$\boldsymbol{R_{\tau}}$", r"$\boldsymbol{R_{\mathrm{all}}}$"]].copy()
    for column in [r"$\boldsymbol{R_{\mathrm{Top}\text{-}1}}$", r"$\boldsymbol{R_{\mathrm{CITrace}}}$", r"$\boldsymbol{R_{\tau}}$", r"$\boldsymbol{R_{\mathrm{all}}}$"]:
        display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    return display.to_latex(index=False, escape=False)



def _find_files(results_dir: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(results_dir.rglob(pattern))
    return sorted({path.resolve() for path in found})


def _find_files_multi(results_dirs: list[Path], patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        found.extend(_find_files(results_dir, patterns))
    return sorted({path.resolve() for path in found})


def _ensure_accumulator(store: dict[str, DatasetAccumulator], dataset: str) -> DatasetAccumulator:
    if dataset not in store:
        store[dataset] = DatasetAccumulator(dataset=dataset)
    return store[dataset]


def _update_n_matches(acc: DatasetAccumulator, value: Any, source: Path) -> None:
    if value is None:
        return
    try:
        n_matches = int(value)
    except (TypeError, ValueError):
        return
    if n_matches <= 0:
        return
    if acc.n_matches is None:
        acc.n_matches = n_matches
        return
    if acc.n_matches != n_matches:
        message = f"Conflicting n_matches for {acc.dataset}: kept {acc.n_matches}, saw {n_matches} in {source.name}"
        LOGGER.warning(message)
        acc.warnings.append(message)


def ingest_topk_breaks(results_dir: Path, actions: set[str], store: dict[str, DatasetAccumulator]) -> None:
    files = _find_files(results_dir, ["all_datasets_nonci_actions_needed.csv", "*_nonci_actions_needed.csv", "*top1_actions_needed.csv"])
    seen: dict[tuple[str, str], tuple[int, Any]] = {}
    for path in files:
        try:
            data = load_table(path)
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", path, exc)
            continue
        if not isinstance(data, pd.DataFrame) or data.empty:
            continue
        if "variant" not in data.columns or "n_actions" not in data.columns:
            continue
        path_rank = 0 if "nonci_actions_needed" in path.name else 1
        for _, row in data.iterrows():
            dataset = row.get("dataset_key")
            if pd.isna(dataset) and "dataset_key" not in data.columns:
                dataset = path.name.split("_top1_actions_needed")[0]
            if pd.isna(dataset) and "dataset_name" in data.columns:
                dataset = row.get("dataset_name")
            if pd.isna(dataset):
                continue
            action = canonicalize_action(row.get("variant") or row.get("action") or row.get("candidate_mode"))
            if action not in actions:
                continue
            acc = _ensure_accumulator(store, canonicalize_dataset(dataset))
            _update_n_matches(acc, row.get("n_original_matches"), path)
            value = row.get("n_actions")
            if "met" in data.columns and not bool(row.get("met")):
                value = None
            key = (acc.dataset, action)
            current = seen.get(key)
            if current is None or path_rank < current[0]:
                seen[key] = (path_rank, value)
                acc.topk_counts[action] = value


def ingest_tau_curves(results_dir: Path, actions: set[str], store: dict[str, DatasetAccumulator]) -> None:
    files = _find_files(results_dir, ["all_datasets_curve_history.csv", "*_curve_history.csv"])
    for path in files:
        try:
            data = load_table(path)
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", path, exc)
            continue
        if not isinstance(data, pd.DataFrame) or data.empty:
            continue
        if "objective_key" not in data.columns or "variant" not in data.columns:
            continue
        curve_df = data.loc[
            data["objective_key"].astype(str).isin(["kendall_tau", "trace_uncertainty"])
        ].copy()
        if curve_df.empty:
            continue
        if "order_mode" in curve_df.columns:
            greedy = curve_df.loc[curve_df["order_mode"].astype(str) == "greedy"].copy()
            if not greedy.empty:
                curve_df = greedy
        grouped = curve_df.groupby(["dataset_key", "variant", "objective_key"], dropna=True)
        for (dataset, variant, objective_key), group in grouped:
            action = canonicalize_action(variant)
            if action not in actions:
                continue
            acc = _ensure_accumulator(store, canonicalize_dataset(dataset))
            if "n_original_matches" in group.columns:
                _update_n_matches(acc, group["n_original_matches"].dropna().iloc[0] if not group["n_original_matches"].dropna().empty else None, path)
            value_col = "normalized_value" if objective_key == "kendall_tau" and "normalized_value" in group.columns else "objective_value"
            series = (
                group.sort_values("step")
                .set_index("step")[value_col]
                .astype(float)
            )
            if objective_key == "kendall_tau":
                acc.tau_curves[action] = series
            else:
                acc.trace_curves[action] = series


def ingest_n_matches_metadata(results_dir: Path, store: dict[str, DatasetAccumulator]) -> None:
    files = _find_files(results_dir, ["all_datasets_curve_summary.csv", "all_datasets_k_selection.csv"])
    for path in files:
        try:
            data = load_table(path)
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", path, exc)
            continue
        if not isinstance(data, pd.DataFrame) or data.empty or "dataset_key" not in data.columns:
            continue
        for _, row in data.iterrows():
            acc = _ensure_accumulator(store, canonicalize_dataset(row["dataset_key"]))
            _update_n_matches(acc, row.get("n_original_matches"), path)


def build_scores(
    store: dict[str, DatasetAccumulator],
    *,
    budget_frac: float,
    tau_exclude_initial: bool,
    actions: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    action_set = set(actions)
    for dataset in sorted(store):
        acc = store[dataset]
        notes = list(acc.warnings)
        if acc.n_matches is None:
            warning = f"Missing n_matches for dataset {dataset}; scores set to NaN."
            LOGGER.warning(warning)
            notes.append(warning)
            rows.append(
                {
                    "dataset": dataset,
                    "n_matches": np.nan,
                    "budget_frac": budget_frac,
                    "B_max": np.nan,
                    "R_topk": np.nan,
                    "R_CITrace": np.nan,
                    "R_tau": np.nan,
                    "R_all": np.nan,
                    "topk_worst_action": np.nan,
                    "topk_min_actions": np.nan,
                    "trace_worst_action": np.nan,
                    "trace_final_value": np.nan,
                    "tau_worst_action": np.nan,
                    "tau_auc": np.nan,
                    "notes": " | ".join(notes),
                }
            )
            continue
        B_max = max(1.0, float(budget_frac) * float(acc.n_matches))

        topk_available = {a: v for a, v in acc.topk_counts.items() if a in action_set}
        trace_available = {a: v for a, v in acc.trace_curves.items() if a in action_set}
        tau_available = {a: v for a, v in acc.tau_curves.items() if a in action_set}

        missing_topk_actions = [a for a in actions if a not in topk_available]
        if missing_topk_actions:
            message = f"{dataset}: missing top-k actions {missing_topk_actions}"
            LOGGER.warning(message)
            notes.append(message)
        missing_trace_actions = [a for a in actions if a not in trace_available]
        if missing_trace_actions:
            message = f"{dataset}: missing CI-trace curves for actions {missing_trace_actions}"
            LOGGER.warning(message)
            notes.append(message)
        missing_tau_actions = [a for a in actions if a not in tau_available]
        if missing_tau_actions:
            message = f"{dataset}: missing Kendall curves for actions {missing_tau_actions}"
            LOGGER.warning(message)
            notes.append(message)

        if not topk_available:
            message = f"{dataset}: no available actions for R_topk"
            LOGGER.warning(message)
            notes.append(message)
        if not trace_available:
            message = f"{dataset}: no available curves for R_CITrace"
            LOGGER.warning(message)
            notes.append(message)
        if not tau_available:
            message = f"{dataset}: no available Kendall curves for R_tau"
            LOGGER.warning(message)
            notes.append(message)

        R_topk, topk_worst_action, topk_min_actions = compute_break_score(topk_available, B_max)
        R_trace, trace_worst_action, trace_final_value = compute_trace_score(trace_available)
        R_tau, tau_worst_action, tau_auc = compute_tau_score(tau_available, exclude_initial=tau_exclude_initial)
        R_all = compute_all_score(R_topk, R_trace, R_tau)

        n_components = int(np.isfinite([R_topk, R_trace, R_tau]).sum())
        if 0 < n_components < 3:
            message = f"{dataset}: R_all averaged over {n_components} non-NaN components"
            LOGGER.warning(message)
            notes.append(message)

        rows.append(
            {
                "dataset": dataset,
                "dataset_display": DATASET_DISPLAY_NAMES.get(dataset, dataset),
                "n_matches": int(acc.n_matches),
                "budget_frac": float(budget_frac),
                "B_max": float(B_max),
                "R_topk": R_topk,
                "R_CITrace": R_trace,
                "R_tau": R_tau,
                "R_all": R_all,
                "topk_worst_action": topk_worst_action,
                "topk_min_actions": topk_min_actions,
                "trace_worst_action": trace_worst_action,
                "trace_final_value": trace_final_value,
                "tau_worst_action": tau_worst_action,
                "tau_auc": tau_auc,
                "notes": " | ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    display = df[["dataset_display", "R_topk", "R_CITrace", "R_tau", "R_all"]].copy()
    for col in ["R_topk", "R_CITrace", "R_tau", "R_all"]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    headers = list(display.columns)
    rows = [[str(row[col]) for col in headers] for _, row in display.iterrows()]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        cells = [cell.ljust(widths[idx]) for idx, cell in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    parts = [fmt(headers), separator]
    parts.extend(fmt(row) for row in rows)
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    setup_logging()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actions = [canonicalize_action(action) for action in args.actions]
    action_set = {action for action in actions if action is not None}

    store: dict[str, DatasetAccumulator] = {}
    ingest_topk_breaks(results_dir, action_set, store)
    ingest_tau_curves(results_dir, action_set, store)
    ingest_n_matches_metadata(results_dir, store)

    scores_df = build_scores(
        store,
        budget_frac=float(args.budget_frac),
        tau_exclude_initial=bool(args.tau_exclude_initial),
        actions=[action for action in actions if action is not None],
    )

    csv_path = output_dir / "dataset_robustness_scores.csv"
    tex_path = output_dir / "dataset_robustness_scores.tex"
    snippet_path = output_dir / "dataset_robustness_scores_snippet.tex"
    scores_df.to_csv(csv_path, index=False)
    tex_path.write_text(format_latex_table(scores_df), encoding="utf-8")
    snippet_path.write_text(format_subsection_snippet(scores_df), encoding="utf-8")

    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", tex_path)
    LOGGER.info("Wrote %s", snippet_path)

    if args.print_markdown:
        print(markdown_table(scores_df))


if __name__ == "__main__":
    main()
