from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clean_bt_rank.experiments.ci_reduction_batch import (
    build_dataset_improvement_table,
    build_summary_table,
    plot_dataset_improvements,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-dataset CI reduction influence-variant outputs into one pooled result."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("final_result/ci_reduction_influence_variants"),
        help="Directory containing one subdirectory per dataset job.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for merged outputs. Defaults to <input-root>/merged.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any dataset subdirectory is missing task_metrics.csv.",
    )
    return parser.parse_args()


def _load_finished_dataset_dirs(input_root: Path, *, output_dir: Path | None = None) -> tuple[list[Path], list[str]]:
    finished: list[Path] = []
    missing: list[str] = []
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if output_dir is not None and dataset_dir.resolve() == output_dir.resolve():
            continue
        if (dataset_dir / "task_metrics.csv").exists():
            finished.append(dataset_dir)
        else:
            missing.append(dataset_dir.name)
    return finished, missing


def _concat_csvs(dataset_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames = [pd.read_csv(dataset_dir / filename) for dataset_dir in dataset_dirs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _infer_primary_policy(dataset_dirs: list[Path]) -> str:
    for dataset_dir in dataset_dirs:
        config_path = dataset_dir / "run_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        primary_policy = config.get("primary_policy")
        if primary_policy:
            return str(primary_policy)
    return "influence_v4"


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or (input_root / "merged")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    finished_dirs, missing_datasets = _load_finished_dataset_dirs(input_root, output_dir=output_dir)
    if args.require_all and missing_datasets:
        raise SystemExit(
            "Missing completed outputs for datasets: " + ", ".join(missing_datasets)
        )
    if not finished_dirs:
        raise SystemExit(f"No completed dataset outputs found in {input_root}.")

    primary_policy = _infer_primary_policy(finished_dirs)
    task_metrics = _concat_csvs(finished_dirs, "task_metrics.csv")
    task_curves = _concat_csvs(finished_dirs, "task_curves.csv")
    target_table = _concat_csvs(finished_dirs, "target_table.csv")
    summary_table = build_summary_table(task_metrics, primary_policy=primary_policy)
    dataset_improvement_table = build_dataset_improvement_table(task_metrics, primary_policy=primary_policy)

    task_metrics.to_csv(output_dir / "task_metrics.csv", index=False)
    task_curves.to_csv(output_dir / "task_curves.csv", index=False)
    target_table.to_csv(output_dir / "target_table.csv", index=False)
    summary_table.to_csv(output_dir / "summary_table.csv", index=False)
    dataset_improvement_table.to_csv(output_dir / "dataset_improvement_table.csv", index=False)
    plot_dataset_improvements(dataset_improvement_table, output_dir / "dataset_improvement_forest.png")

    merge_status = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "finished_datasets": [path.name for path in finished_dirs],
        "missing_datasets": missing_datasets,
        "primary_policy": primary_policy,
    }
    (output_dir / "merge_status.json").write_text(json.dumps(merge_status, indent=2) + "\n")

    print("Merged datasets:")
    for dataset_name in merge_status["finished_datasets"]:
        print(f" - {dataset_name}")
    if missing_datasets:
        print("Missing datasets:")
        for dataset_name in missing_datasets:
            print(f" - {dataset_name}")
    print("\n=== Pooled CI Reduction Summary ===")
    print(summary_table.to_string(index=False))
    print(f"\nSaved merged artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
