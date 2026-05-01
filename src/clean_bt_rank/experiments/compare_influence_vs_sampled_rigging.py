"""Compare greedy influence vote additions to sampled-pair rigging exposure.

Run from repo root (editable install or PYTHONPATH=src)::

    python -m clean_bt_rank.experiments.compare_influence_vs_sampled_rigging
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clean_bt_rank import (
    BattleDataset,
    BradleyTerryModel,
    SkillGapObjective,
    compute_objective_action_influence,
    load_named_battle_data,
)


def load_base_arena_state(dataset_key: str) -> tuple[pd.DataFrame, dict[str, object], list[str], pd.DataFrame]:
    loaded = load_named_battle_data(dataset_key)
    raw = loaded.battle_frame.copy().reset_index(drop=True)
    fit_kwargs = loaded.fit_kwargs
    dataset = BattleDataset.from_dataframe(raw)
    model = BradleyTerryModel.from_dataset(dataset, **fit_kwargs).fit()
    ranking = ranking_frame_from_model(model)
    return raw, fit_kwargs, dataset.competitors, ranking


def build_raw_vote_candidates(
    competitors: list[str],
    *,
    reference_player: int = 0,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    n_players = len(competitors)
    n_params = n_players - 1
    candidates: list[dict[str, object]] = []
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []

    for i in range(n_players):
        for j in range(i + 1, n_players):
            model_a = competitors[i]
            model_b = competitors[j]

            row_ab = np.zeros(n_params, dtype=float)
            if i != reference_player:
                row_ab[i - (1 if i > reference_player else 0)] = 1.0
            if j != reference_player:
                row_ab[j - (1 if j > reference_player else 0)] = -1.0
            row_ba = -row_ab

            start = len(x_rows)
            x_rows.extend([row_ab, row_ba])
            y_rows.extend([1.0, 0.0])
            candidates.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "winner": "model_a",
                    "unordered_pair": tuple(sorted((model_a, model_b))),
                    "start": start,
                }
            )

            start = len(x_rows)
            x_rows.extend([row_ab, row_ba])
            y_rows.extend([0.0, 1.0])
            candidates.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "winner": "model_b",
                    "unordered_pair": tuple(sorted((model_a, model_b))),
                    "start": start,
                }
            )

            start = len(x_rows)
            x_rows.extend([row_ab, row_ba])
            y_rows.extend([1.0, 1.0])
            candidates.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "winner": "tie",
                    "unordered_pair": tuple(sorted((model_a, model_b))),
                    "start": start,
                }
            )

    return candidates, np.vstack(x_rows), np.asarray(y_rows, dtype=float)


def ranking_frame_from_model(model: BradleyTerryModel) -> pd.DataFrame:
    ratings = model.scaled_skills()
    order = np.argsort(-ratings)
    ranked = pd.DataFrame(
        {
            "competitor": [model.competitor_names_[idx] for idx in order],
            "rating": ratings[order],
        }
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def greedy_influence_promotion(
    target_player: str,
    *,
    dataset_key: str = "arena55k",
    k: int = 15,
    max_steps: int = 20,
    method: str = "1sn",
    raw: pd.DataFrame | None = None,
    fit_kwargs: dict[str, object] | None = None,
    competitors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw is None or fit_kwargs is None or competitors is None:
        loaded = load_named_battle_data(dataset_key)
        raw = loaded.battle_frame.copy().reset_index(drop=True)
        fit_kwargs = loaded.fit_kwargs
        competitors = BattleDataset.from_dataframe(raw).competitors
    else:
        raw = raw.copy().reset_index(drop=True)
    candidates, x_new, y_new = build_raw_vote_candidates(competitors)

    history_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for step in range(max_steps + 1):
        dataset = BattleDataset.from_dataframe(raw, competitors=competitors)
        model = BradleyTerryModel.from_dataset(dataset, **fit_kwargs).fit()
        ranking = ranking_frame_from_model(model)

        target_rank = int(ranking.loc[ranking["competitor"] == target_player, "rank"].iloc[0])
        boundary_player = str(ranking.loc[ranking["rank"] == k, "competitor"].iloc[0])
        boundary_gap = float(model.reported_gap(target_player, boundary_player))
        history_rows.append(
            {
                "step": step,
                "target_player": target_player,
                "target_rank": target_rank,
                "boundary_player": boundary_player,
                "boundary_gap": boundary_gap,
            }
        )
        if target_rank <= k:
            break

        objective = SkillGapObjective(target_player, boundary_player)
        influence_rows = compute_objective_action_influence(
            model,
            objective,
            action="add",
            method=method,
            X_new=x_new,
            y_new=y_new,
        )
        raw_vote_influence = np.asarray(
            [float(influence_rows[row["start"]] + influence_rows[row["start"] + 1]) for row in candidates],
            dtype=float,
        )
        best_idx = int(np.argmax(raw_vote_influence))
        chosen = dict(candidates[best_idx])
        chosen["influence"] = float(raw_vote_influence[best_idx])
        chosen["step"] = step + 1
        selected_rows.append(chosen)

        raw = pd.concat(
            [raw, pd.DataFrame([{"model_a": chosen["model_a"], "model_b": chosen["model_b"], "winner": chosen["winner"]}])],
            ignore_index=True,
        )

    return pd.DataFrame(history_rows), pd.DataFrame(selected_rows)


def expected_uniform_pair_opportunities(unordered_pair_counts: Counter[tuple[str, str]], n_players: int) -> float:
    pair_types = tuple(count for _, count in sorted(unordered_pair_counts.items()))
    total_pairs = n_players * (n_players - 1) // 2

    @lru_cache(maxsize=None)
    def solve(state: tuple[int, ...]) -> float:
        if not any(state):
            return 0.0
        active = [idx for idx, count in enumerate(state) if count > 0]
        active_count = len(active)
        next_expectation = 0.0
        for idx in active:
            next_state = list(state)
            next_state[idx] -= 1
            next_expectation += solve(tuple(next_state))
        return float(total_pairs / active_count) + next_expectation / active_count

    return solve(pair_types)


def benchmark_targets(
    targets: list[str],
    *,
    dataset_key: str,
    k: int,
    max_steps: int,
    method: str,
) -> pd.DataFrame:
    raw, fit_kwargs, competitors, base_ranking = load_base_arena_state(dataset_key)
    rank_lookup = dict(zip(base_ranking["competitor"], base_ranking["rank"]))
    n_players = len(competitors)
    rows: list[dict[str, object]] = []

    for target in targets:
        history, selected = greedy_influence_promotion(
            target,
            dataset_key=dataset_key,
            k=k,
            max_steps=max_steps,
            method=method,
            raw=raw,
            fit_kwargs=fit_kwargs,
            competitors=competitors,
        )
        initial_rank = int(rank_lookup[target])
        final_rank = int(history.iloc[-1]["target_rank"])
        added_votes = int(len(selected))
        success = bool(final_rank <= k)
        if added_votes > 0:
            unordered_counts = Counter(tuple(pair) for pair in selected["unordered_pair"])
            expected_opps = float(expected_uniform_pair_opportunities(unordered_counts, n_players))
        else:
            expected_opps = 0.0
        rows.append(
            {
                "target": target,
                "initial_rank": initial_rank,
                "goal_k": k,
                "success": success,
                "final_rank": final_rank,
                "influence_added_votes": added_votes,
                "expected_uniform_sampled_pair_opportunities": expected_opps,
                "opportunity_to_vote_multiplier": (expected_opps / added_votes) if added_votes else 0.0,
                "final_boundary_gap": float(history.iloc[-1]["boundary_gap"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["initial_rank", "target"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare greedy influence vote additions to sampled-pair rigging exposure.")
    parser.add_argument("--dataset", default="arena55k")
    parser.add_argument("--target", default="gpt-3.5-turbo-0314")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--method", default="1sn")
    parser.add_argument("--target-rank-start", type=int, default=None)
    parser.add_argument("--target-rank-end", type=int, default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    if args.targets is not None or args.target_rank_start is not None or args.target_rank_end is not None:
        raw, fit_kwargs, competitors, base_ranking = load_base_arena_state(args.dataset)
        del raw, fit_kwargs, competitors
        if args.targets is not None:
            targets = list(args.targets)
        else:
            start = args.target_rank_start if args.target_rank_start is not None else args.k + 1
            end = args.target_rank_end if args.target_rank_end is not None else start
            target_frame = base_ranking.loc[base_ranking["rank"].between(start, end)].copy()
            targets = target_frame["competitor"].astype(str).tolist()

        result = benchmark_targets(
            targets,
            dataset_key=args.dataset,
            k=args.k,
            max_steps=args.max_steps,
            method=args.method,
        )
        print(result.to_string(index=False))
        if args.csv_out:
            out_path = Path(args.csv_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out_path, index=False)
            print()
            print(f"Saved CSV to {out_path}")
        return

    history, selected = greedy_influence_promotion(
        args.target,
        dataset_key=args.dataset,
        k=args.k,
        max_steps=args.max_steps,
        method=args.method,
    )

    print("Promotion history")
    print(history.to_string(index=False))
    print()
    print("Selected raw votes")
    if selected.empty:
        print("(none)")
        return
    print(selected[["step", "model_a", "model_b", "winner", "influence"]].to_string(index=False))
    print()

    final_rank = int(history.iloc[-1]["target_rank"])
    n_added = int(len(selected))
    print(f"Added raw votes: {n_added}")
    print(f"Final rank: {final_rank}")

    unordered_counts = Counter(tuple(pair) for pair in selected["unordered_pair"])
    n_players = load_named_battle_data(args.dataset).battle_frame.pipe(BattleDataset.from_dataframe).n_competitors
    expected_opps = expected_uniform_pair_opportunities(unordered_counts, n_players)
    print(f"Expected sampled pair opportunities to encounter the same helpful unordered pairs under uniform sampling: {expected_opps:.1f}")
    if n_added > 0:
        print(f"Opportunity-to-added-vote multiplier: {expected_opps / n_added:.1f}x")


if __name__ == "__main__":
    main()
