from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASELINE_ROOT = ROOT / "IsRankingRobust"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from clean_bt_rank import BradleyTerryModel  # noqa: E402
from clean_bt_rank.iterative_actions import gap_based_objective_search_across_player_pairs  # noqa: E402
from package.RankAMIP.data_script import simulate_bt_design_matrix  # noqa: E402
from package.RankAMIP.logistic import isRankingRobust  # noqa: E402


def _free_index_to_name(names: list[str], free_idx: int | None) -> str:
    if free_idx is None:
        return names[0]
    return names[free_idx + 1]


def _baseline_find_closest_matchups(player_scores: np.ndarray, k: int) -> list[tuple[int | None, int | None, float]]:
    p = player_scores.shape[0] + 1
    full_score = np.concatenate((np.array([0.0]), np.asarray(player_scores, dtype=float)))
    asort = np.argsort(full_score)[::-1]

    matchups: list[tuple[int | None, int | None, float]] = []
    for i in range(k):
        for j in range(p - k):
            diff = np.abs(full_score[asort[i]] - full_score[asort[j + k]]).item()
            tm1 = asort[i].item() - 1
            tm2 = asort[j + k].item() - 1
            if tm1 == -1:
                matchups.append((int(tm2), None, float(diff)))
            elif tm2 == -1:
                matchups.append((int(tm1), None, float(diff)))
            else:
                matchups.append((int(tm1), int(tm2), float(diff)))

    return sorted(matchups, key=lambda item: item[2])


def test_gap_drop_search_matches_isrankingrobust_for_k1() -> None:
    X, y = simulate_bt_design_matrix(10, 200, 7)
    names = [f"p{i}" for i in range(X.shape[1] + 1)]
    bt = BradleyTerryModel(X, y, competitor_names=names, reference_player=0, hessian_ridge=0.0).fit()

    raw_pairs = _baseline_find_closest_matchups(bt.beta_hat_, 1)
    player_pairs = [(_free_index_to_name(names, a), _free_index_to_name(names, b)) for a, b, _ in raw_pairs]

    for alpha in range(1, 6):
        baseline_player_a, baseline_player_b, baseline_initial, baseline_final, baseline_rows = isRankingRobust(1, alpha, X, y)
        result = gap_based_objective_search_across_player_pairs(
            bt,
            player_pairs,
            "drop",
            start_alpha=alpha,
            max_alpha=alpha,
            recompute_mode="refit",
            influence_method="1sn",
        )
        print(result["result"]["selected_matches"])
        print(baseline_rows)
        if baseline_player_a == -1:
            assert result["met"] is False
            assert result["player_pair"] is None
            continue

        expected_pair = (
            _free_index_to_name(names, int(baseline_player_a)),
            _free_index_to_name(names, None if baseline_player_b is None else int(baseline_player_b)),
        )
        assert result["met"] is True
        assert result["player_pair"] == expected_pair
        np.testing.assert_allclose(result["result"]["initial_value"], baseline_initial, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(result["result"]["final_value"], baseline_final, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(
            result["result"]["selected_matches"]["row_uid"].to_numpy(dtype=int),
            np.asarray(baseline_rows, dtype=int),
        )
        
if __name__ == "__main__":
    test_gap_drop_search_matches_isrankingrobust_for_k1()   
    