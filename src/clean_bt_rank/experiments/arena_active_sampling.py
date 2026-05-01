from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..actions import build_add_candidates
from ..bt_model import BradleyTerryModel
from ..ci import compute_skill_covariance


ArenaActiveMode = str


@dataclass
class ArenaActiveSamplingBaseline:
    """
    Chatbot Arena-style active sampling baseline for pair selection.

    We score candidate unordered pairs using the paper-style expected CI
    reduction rule

        sqrt(var_a / N_a) - sqrt(var_a / (N_a + 1))

    where `var_a` is a pair-level variance proxy and `N_a` is the number of
    previously observed samples of that pair.

    In this codebase, `var_a` is approximated by the variance of the BT skill
    gap `theta_u - theta_v` under the current sandwich covariance estimate:

        Var(theta_u - theta_v) = Sigma_uu + Sigma_vv - 2 Sigma_uv

    Supported modes:
    - `target_only`: only score pairs that involve the target player
    - `global`: score all pairs

    Outcome rule:
    - candidate rows are generated with `all_pairs`, so the added outcome is the
      current BT expected winner for that unordered pair.
    """

    mode: ArenaActiveMode = "target_only"
    covariance_method: str = "sandwich"
    candidate_mode: str = "all_pairs"

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in {"target_only", "global"}:
            raise ValueError("mode must be 'target_only' or 'global'.")
        if self.candidate_mode != "all_pairs":
            raise ValueError("ArenaActiveSamplingBaseline currently expects candidate_mode='all_pairs'.")

    def score_candidates(
        self,
        bt_model: BradleyTerryModel,
        *,
        target_player: str,
        used_keys: set[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        bt_model._require_fit()
        used_keys = set() if used_keys is None else used_keys

        candidates = build_add_candidates(bt_model, mode=self.candidate_mode)
        report = candidates.frame.copy()
        report["_candidate_x"] = list(np.asarray(candidates.X, dtype=float))

        covariance = compute_skill_covariance(bt_model, method=self.covariance_method)
        pair_counts = bt_model.observed_pair_counts() / 2.0
        report["pair_variance"] = [
            _pair_gap_variance(covariance, int(a_idx), int(b_idx))
            for a_idx, b_idx in zip(report["player_a_index"], report["player_b_index"])
        ]
        report["pair_count"] = [
            float(pair_counts[int(a_idx), int(b_idx)])
            for a_idx, b_idx in zip(report["player_a_index"], report["player_b_index"])
        ]
        report["arena_active_score"] = [
            _arena_active_score(float(var), float(count))
            for var, count in zip(report["pair_variance"], report["pair_count"])
        ]

        if self.mode == "target_only":
            target_mask = (report["model_a"].astype(str) == target_player) | (report["model_b"].astype(str) == target_player)
            report = report.loc[target_mask].copy()

        candidate_keys = list(zip(report["model_a"].astype(str), report["model_b"].astype(str)))
        keep = np.array([key not in used_keys for key in candidate_keys], dtype=bool)
        return report.loc[keep].reset_index(drop=True)

    def select_candidate(
        self,
        bt_model: BradleyTerryModel,
        *,
        target_player: str,
        used_keys: set[tuple[str, str]] | None = None,
    ) -> pd.Series:
        report = self.score_candidates(bt_model, target_player=target_player, used_keys=used_keys)
        if report.empty:
            raise ValueError("No available candidates for Arena active sampling baseline.")
        return report.sort_values("arena_active_score", ascending=False).iloc[0].copy()


def select_pair_chatbot_arena_baseline(
    bt_model: BradleyTerryModel,
    *,
    target_player: str,
    mode: str = "target_only",
    used_keys: set[tuple[str, str]] | None = None,
) -> pd.Series:
    baseline = ArenaActiveSamplingBaseline(mode=mode)
    return baseline.select_candidate(bt_model, target_player=target_player, used_keys=used_keys)


def _pair_gap_variance(covariance: np.ndarray, player_a_index: int, player_b_index: int) -> float:
    var = (
        covariance[player_a_index, player_a_index]
        + covariance[player_b_index, player_b_index]
        - 2.0 * covariance[player_a_index, player_b_index]
    )
    return float(max(var, 0.0))


def _arena_active_score(pair_variance: float, pair_count: float) -> float:
    if pair_variance <= 0.0:
        return 0.0
    if pair_count <= 0.0:
        # For unseen pairs the paper-style formula is singular at N=0. We use
        # the first finite one-step reduction obtained by treating the pair as
        # going from 1 effective sample to 2 effective samples.
        return float(np.sqrt(pair_variance / 1.0) - np.sqrt(pair_variance / 2.0))
    return float(np.sqrt(pair_variance / pair_count) - np.sqrt(pair_variance / (pair_count + 1.0)))
