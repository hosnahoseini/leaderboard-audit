from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BASELINE_ROOT = Path("/Users/hoyarhos/Desktop/IF_framework/IsRankingRobust")
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from clean_bt_rank import (  # noqa: E402
    BTParameterInfluence,
    BattleDataset,
    BradleyTerryModel,
    ObjectiveInfluence,
    SkillGapObjective,
    ranking_from_model,
)
from clean_bt_rank.iterative_actions import compute_all_action_influences  # noqa: E402
from package.RankAMIP.data_script import make_BT_design_matrix  # noqa: E402
from package.RankAMIP.logistic import LogisticAMIP  # noqa: E402


def make_bt_design_matrix_wtd(raw: pd.DataFrame):
    """
    Match the trusted baseline exactly.

    Each original match contributes:
    - one forward row
    - one reverse row

    Ties become y=1 in both rows, exactly as in the baseline script.
    """
    winner_fwd = raw["winner_model_a"].copy().astype(float)
    winner_rev = (1 - raw["winner_model_a"]).astype(float)

    tie_mask = raw["winner_tie"] == 1
    winner_fwd[tie_mask] = 1.0
    winner_rev[tie_mask] = 1.0

    fwd = pd.DataFrame(
        {
            "model_a": raw["model_a"].values,
            "model_b": raw["model_b"].values,
            "winner": winner_fwd.values.astype(int),
        }
    )
    rev = pd.DataFrame(
        {
            "model_a": raw["model_b"].values,
            "model_b": raw["model_a"].values,
            "winner": winner_rev.values.astype(int),
        }
    )

    combined = pd.concat([fwd, rev], ignore_index=True)
    X, y, player_to_id = make_BT_design_matrix(combined)
    return X, y, player_to_id


def load_arena55k_raw_dataframe() -> pd.DataFrame:
    ds = load_dataset("lmarena-ai/arena-human-preference-55k")
    df = ds["train"].to_pandas()
    return df[["model_a", "model_b", "winner_model_a", "winner_tie"]].copy()


def find_closest_matchups(player_scores: np.ndarray, k: int) -> list[tuple[int | None, int | None, float]]:
    """
    Baseline matchup selector from IsRankingRobust.

    Returns tuples of free-parameter indices, where ``None`` denotes the
    reference player inserted at the front of the full score vector.
    """
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

    return sorted(matchups, key=lambda x: x[2])


def notebook_top_k_membership_player_pairs(
    bt_model: BradleyTerryModel,
    k: int,
) -> list[tuple[str, str, float]]:
    ranking = ranking_from_model(bt_model)
    k = max(1, min(int(k), len(ranking) - 1))
    pairs = [
        (player_a, player_b, abs(float(bt_model.reported_gap(player_a, player_b))))
        for player_a in ranking[:k]
        for player_b in ranking[k:]
    ]
    return sorted(pairs, key=lambda item: item[2])


def baseline_matchups_as_player_pairs(
    bt_model: BradleyTerryModel,
    k: int,
) -> list[tuple[str, str, float]]:
    reference_name = bt_model.competitor_names_[bt_model.reference_player]

    def free_index_to_name(free_idx: int | None) -> str:
        if free_idx is None:
            return reference_name
        full_idx = free_idx if free_idx < bt_model.reference_player else free_idx + 1
        return bt_model.competitor_names_[full_idx]

    raw_pairs = find_closest_matchups(bt_model.beta_hat_, k)
    return [(free_index_to_name(a), free_index_to_name(b), diff) for a, b, diff in raw_pairs]


def verify_notebook_matchup_selection(bt: BradleyTerryModel) -> None:
    for k in (1, 5):
        notebook_pairs = notebook_top_k_membership_player_pairs(bt, k)
        baseline_pairs = baseline_matchups_as_player_pairs(bt, k)
        if len(notebook_pairs) != len(baseline_pairs):
            raise AssertionError(
                f"Mismatch in number of candidate pairs for k={k}: "
                f"{len(notebook_pairs)} != {len(baseline_pairs)}"
            )
        print(notebook_pairs)
        print(baseline_pairs)
        for idx, (notebook_row, baseline_row) in enumerate(zip(notebook_pairs, baseline_pairs, strict=True)):
            np.testing.assert_equal(notebook_row[0], baseline_row[0])
            np.testing.assert_equal(notebook_row[1], baseline_row[1])
            np.testing.assert_allclose(notebook_row[2], baseline_row[2], rtol=0.0, atol=1e-12)

        print(
            f"Notebook top-k matchup selection matches baseline find_closest_matchups for k={k} "
            f"({len(notebook_pairs)} pairs)."
        )


def verify_gap_action_report_against_baseline(
    bt: BradleyTerryModel,
    baseline: LogisticAMIP,
    player_to_id: dict[str, int],
    *,
    player_a: str,
    player_b: str,
    influence_method: str = "1sn",
    action: str = "drop",
    candidate_mode: str | None = None,
) -> None:
    if action != "drop":
        raise ValueError("IsRankingRobust only provides a direct baseline for drop-style match influence.")

    objective = SkillGapObjective(player_a=player_a, player_b=player_b)
    report = compute_all_action_influences(
        bt,
        objective,
        action,
        influence_method=influence_method,
        candidate_mode=candidate_mode,
    )

    baseline_method = "1sN" if influence_method.lower() == "1sn" else "IF"
    id_a = int(player_to_id[player_a])
    id_b = int(player_to_id[player_b])

    if id_a == 0 and id_b == 0:
        baseline_rows = np.zeros(bt.X.shape[0], dtype=float)
    elif id_b == 0:
        baseline_rows = baseline.compute_player_influence(id_a - 1, None, method=baseline_method)
    elif id_a == 0:
        baseline_rows = -baseline.compute_player_influence(id_b - 1, None, method=baseline_method)
    else:
        baseline_rows = baseline.compute_player_influence(id_a - 1, id_b - 1, method=baseline_method)

    np.testing.assert_allclose(
        report["influence"].to_numpy(dtype=float),
        baseline_rows,
        rtol=1e-12,
        atol=1e-12,
    )

    if {"model_a", "model_b", "match_copy"}.issubset(report.columns):
        print(
            f"compute_all_action_influences(..., action='drop') matches IsRankingRobust for "
            f"{player_a} vs {player_b} with method={influence_method}."
        )


def verify_topk_gap_action_reports_against_baseline(
    bt: BradleyTerryModel,
    baseline: LogisticAMIP,
    player_to_id: dict[str, int],
) -> None:
    for k in (1, 5):
        notebook_pairs = notebook_top_k_membership_player_pairs(bt, k)
        for player_a, player_b, _ in notebook_pairs:
            verify_gap_action_report_against_baseline(
                bt,
                baseline,
                player_to_id,
                player_a=player_a,
                player_b=player_b,
                influence_method="1sn",
                action="drop",
            )
        print(
            f"All notebook top-k SkillGapObjective drop reports match IsRankingRobust for k={k} "
            f"({len(notebook_pairs)} player pairs, method=1sn)."
        )


def verify_against_baseline() -> None:
    raw = load_arena55k_raw_dataframe()
    X, y, player_to_id = make_bt_design_matrix_wtd(raw)

    dataset = BattleDataset.from_dataframe(
        raw.rename(columns={"winner_model_a": "winner"})
        .assign(
            winner=lambda frame: np.where(
                frame["winner_tie"] == 1,
                "tie",
                np.where(frame["winner"] == 1, "model_a", "model_b"),
            )
        )[["model_a", "model_b", "winner"]],
        competitors=[name for name, _ in sorted(player_to_id.items(), key=lambda item: item[1])],
        weighted_symmetric_ties=True,
    )
    print(dataset.design_matrix())
    print(X)
    np.testing.assert_allclose(dataset.design_matrix(), X, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dataset.outcomes, y, rtol=0.0, atol=0.0)

    player_a = "gpt-4-0613"
    player_b = "gpt-4-0314"
    if player_a not in player_to_id or player_b not in player_to_id:
        raise ValueError("Expected Arena 55k GPT-4 players were not found in player_to_id.")
    dim_a = player_to_id[player_a] - 1
    dim_b = player_to_id[player_b] - 1

    baseline = LogisticAMIP(X, y, fit_intercept=False, penalty=None)
    baseline_if_rows = baseline.compute_player_influence(dim_a, dim_b, method="IF")
    baseline_1sn_rows = baseline.compute_player_influence(dim_a, dim_b, method="1sN")

    bt = BradleyTerryModel(
        dataset.design_matrix(),
        dataset.outcomes,
        competitor_names=dataset.competitors,
        reference_player=0,
        hessian_ridge=0.0,
    ).fit()
    param_infl = BTParameterInfluence(bt)
    gap_obj = SkillGapObjective(player_a=player_a, player_b=player_b)
    obj_infl = ObjectiveInfluence(bt, param_infl)

    modular_if_rows = obj_infl.compute_match_influence(gap_obj, method="if")
    modular_1sn_rows = obj_infl.compute_match_influence(gap_obj, method="1sn")
    print(modular_if_rows)
    print(modular_1sn_rows)
    np.testing.assert_allclose(bt.beta_hat_, baseline.model.coef_[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bt.probabilities_, baseline.pos_p_hats, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(modular_if_rows, baseline_if_rows, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(modular_1sn_rows, baseline_1sn_rows, rtol=1e-12, atol=1e-12)

    n_original = len(raw)
    baseline_if_matches = baseline_if_rows[:n_original] + baseline_if_rows[n_original:]
    baseline_1sn_matches = baseline_1sn_rows[:n_original] + baseline_1sn_rows[n_original:]
    modular_if_matches = modular_if_rows[:n_original] + modular_if_rows[n_original:]
    modular_1sn_matches = modular_1sn_rows[:n_original] + modular_1sn_rows[n_original:]

    np.testing.assert_allclose(modular_if_matches, baseline_if_matches, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(modular_1sn_matches, baseline_1sn_matches, rtol=1e-12, atol=1e-12)
    verify_notebook_matchup_selection(bt)
    verify_gap_action_report_against_baseline(
        bt,
        baseline,
        player_to_id,
        player_a=player_a,
        player_b=player_b,
        influence_method="if",
        action="drop",
    )
    verify_gap_action_report_against_baseline(
        bt,
        baseline,
        player_to_id,
        player_a=player_a,
        player_b=player_b,
        influence_method="1sn",
        action="drop",
    )
    verify_topk_gap_action_reports_against_baseline(bt, baseline, player_to_id)

    print("Baseline verification passed.")
    print(f"Players compared: {player_a} vs {player_b}")
    print(f"Reference player: {[name for name, idx in player_to_id.items() if idx == 0][0]}")
    print("IF and 1sN match elementwise at row level and aggregated match level.")


if __name__ == "__main__":
    verify_against_baseline()
