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


DEFAULT_INFLUENCE_VARIANT_POLICIES: tuple[str, ...] = (
    "influence_v1",
    "influence_v2",
    "influence_v3",
    "influence_v4",
)


def resolve_dataset_order(dataset_keys: list[str], order_mode: str) -> tuple[list[str], list[tuple[str, int]]]:
    keys = [str(key) for key in dataset_keys]
    if order_mode == "provided":
        return keys, []
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
    parser = argparse.ArgumentParser(description="Run multi-dataset CI reduction benchmark.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=tuple(sorted(available_hf_battle_datasets())),
        help="Dataset keys to evaluate.",
    )
    parser.add_argument(
        "--dataset-order",
        choices=("provided", "alpha", "size_asc", "size_desc"),
        default="alpha",
        help="How to order datasets before running the batch.",
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
    parser.add_argument("--n-random-trials", type=int, default=30, help="Number of random trials per task.")
    parser.add_argument("--ci-method", default="gao_local", help="CI backend for evaluation.")
    parser.add_argument("--influence-method", default="1sn", help="Influence approximation method.")
    parser.add_argument("--candidate-mode", default="all_pairs", help="Candidate mode for add actions.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=("influence", "arena_active"),
        help="Deterministic policies to run once per task.",
    )
    parser.add_argument(
        "--random-policy",
        default="random",
        help="Stochastic policy family to repeat across random trials.",
    )
    parser.add_argument(
        "--outcome-mode",
        choices=("stochastic", "deterministic"),
        default="deterministic",
        help="How pair-selection policies realize the added match label.",
    )
    parser.add_argument(
        "--primary-policy",
        default="influence",
        help="Policy treated as the main method in summary tables.",
    )
    parser.add_argument(
        "--compare-influence-variants",
        action="store_true",
        help="Shortcut for comparing influence variants against Arena Active and random.",
    )
    parser.add_argument(
        "--influence-variant-policies",
        nargs="+",
        default=DEFAULT_INFLUENCE_VARIANT_POLICIES,
        help="Influence-style policies to use with --compare-influence-variants.",
    )
    parser.add_argument(
        "--arena-policy",
        default="arena_active_pair",
        help="Arena Active policy to use with --compare-influence-variants.",
    )
    parser.add_argument(
        "--variant-random-policy",
        default="random_pair",
        help="Random baseline to use with --compare-influence-variants.",
    )
    parser.add_argument(
        "--variant-primary-policy",
        default="influence_v4",
        help="Primary policy to use with --compare-influence-variants.",
    )
    parser.add_argument("--random-seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/artifacts/ci_reduction_batch"),
        help="Directory for CSV outputs and the pooled figure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the dataset order, then exit without running the benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.txt").write_text("resolving dataset order...\n")
    policies = list(args.policies)
    random_policy = args.random_policy
    primary_policy = args.primary_policy
    candidate_mode = args.candidate_mode
    if args.compare_influence_variants:
        policies = [*args.influence_variant_policies, args.arena_policy]
        random_policy = args.variant_random_policy
        primary_policy = args.variant_primary_policy
        candidate_mode = "all_pairs"

    datasets, size_rows = resolve_dataset_order(list(args.datasets), args.dataset_order)
    print("Resolved dataset order:", flush=True)
    for idx, dataset_key in enumerate(datasets, start=1):
        size_suffix = ""
        for key, n_rows in size_rows:
            if key == dataset_key:
                size_suffix = f" ({n_rows} battle rows)"
                break
        print(f"{idx:>2}. {dataset_key}{size_suffix}", flush=True)

    if args.compare_influence_variants:
        print("\nUsing influence-variant comparison preset:", flush=True)
        print(f"  policies: {policies}", flush=True)
        print(f"  random_policy: {random_policy}", flush=True)
        print(f"  primary_policy: {primary_policy}", flush=True)
        print(f"  candidate_mode: {candidate_mode}", flush=True)

    if args.dry_run:
        (args.output_dir / "status.txt").write_text("dry run complete\n")
        return

    result = run_ci_reduction_batch(
        datasets,
        budget=args.budget,
        target_quantiles=args.target_quantiles,
        min_matches=args.min_matches,
        n_random_trials=args.n_random_trials,
        ci_method=args.ci_method,
        influence_method=args.influence_method,
        candidate_mode=candidate_mode,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        policies=policies,
        random_policy=random_policy,
        primary_policy=primary_policy,
        outcome_mode=args.outcome_mode,
    )
    (args.output_dir / "status.txt").write_text("batch complete\n")
    print("\n=== CI Reduction Summary ===", flush=True)
    print(result.summary_table.to_string(index=False), flush=True)
    print(f"\nSaved artifacts to: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
