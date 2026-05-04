from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clean_bt_rank import available_hf_battle_datasets, load_named_battle_data, run_ci_reduction_batch


def resolve_dataset_order(dataset_keys: list[str], order_mode: str) -> tuple[list[str], list[tuple[str, int]]]:
    keys = [str(key) for key in dataset_keys]
    if order_mode == "alpha":
        ordered = sorted(keys)
        return ordered, []

    size_rows: list[tuple[str, int]] = []
    for key in keys:
        loaded = load_named_battle_data(key)
        size_rows.append((key, int(len(loaded.battle_frame))))

    reverse = order_mode == "size_desc"
    size_rows.sort(key=lambda item: (item[1], item[0]), reverse=reverse)
    ordered = [key for key, _ in size_rows]
    return ordered, size_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fair CI-reduction comparison: influence(all_pairs) vs arena_active(all_pairs-like)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=tuple(sorted(available_hf_battle_datasets())),
        help="Dataset keys to evaluate.",
    )
    parser.add_argument(
        "--dataset-order",
        choices=("alpha", "size_asc", "size_desc"),
        default="size_asc",
        help="How to order datasets before running the comparison.",
    )
    parser.add_argument("--budget", type=int, default=12, help="Added-match budget per target.")
    parser.add_argument(
        "--target-quantiles",
        type=float,
        nargs="+",
        default=(0.1, 0.5, 0.9),
        help="Rank quantiles used to choose targets within each dataset.",
    )
    parser.add_argument("--min-matches", type=int, default=20, help="Minimum matches for target eligibility.")
    parser.add_argument("--ci-method", default="gao_local", help="CI backend for evaluation.")
    parser.add_argument("--influence-method", default="1sn", help="Influence approximation method.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/artifacts/ci_reduction_fair_all_pairs_vs_arena_active"),
        help="Directory for CSV outputs and the pooled figure.",
    )
    parser.add_argument("--random-seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the dataset order, then exit without running the comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.txt").write_text("resolving dataset order...\n")

    datasets, size_rows = resolve_dataset_order(list(args.datasets), args.dataset_order)
    print("Resolved dataset order:", flush=True)
    for idx, dataset_key in enumerate(datasets, start=1):
        size_suffix = ""
        for key, n_rows in size_rows:
            if key == dataset_key:
                size_suffix = f" ({n_rows} battle rows)"
                break
        print(f"{idx:>2}. {dataset_key}{size_suffix}", flush=True)

    if args.dry_run:
        (args.output_dir / "status.txt").write_text("dry run complete\n")
        return

    result = run_ci_reduction_batch(
        datasets,
        budget=args.budget,
        target_quantiles=args.target_quantiles,
        min_matches=args.min_matches,
        n_random_trials=0,
        ci_method=args.ci_method,
        influence_method=args.influence_method,
        candidate_mode="all_pairs",
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        policies=("influence", "arena_active"),
        primary_policy="influence",
    )
    (args.output_dir / "status.txt").write_text("batch complete\n")
    print("\n=== Fair Comparison Summary ===", flush=True)
    print(result.summary_table.to_string(index=False), flush=True)
    print(f"\nSaved artifacts to: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
