from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .datasets import BattleDataset


class BradleyTerryModel:
    def __init__(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        competitor_names: Iterable[str] | None = None,
        reference_player: int = 0,
        scale: float = 400.0,
        base: float = 10.0,
        init_rating: float = 1000.0,
        anchor_player: int | str | None = None,
        anchor_rating: float | None = 1114,
        hessian_ridge: float = 0.0,
    ) -> None:
        if hessian_ridge < 0.0:
            raise ValueError("hessian_ridge must be non-negative.")

        self.scale = scale
        self.base = base
        self.init_rating = init_rating
        self.anchor_player = anchor_player
        self.anchor_rating = anchor_rating
        self.hessian_ridge = hessian_ridge
        self.alpha = scale / np.log(base)
        self.reference_player = int(reference_player)

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.match_frame_: pd.DataFrame | None = None
        self.competitor_names_: list[str] | None = None
        self.n_players_: int | None = None
        self.n_params_: int | None = None

        self.beta_hat_: np.ndarray | None = None
        self.full_beta_hat_: np.ndarray | None = None
        self.reported_skills_: np.ndarray | None = None
        self.probabilities_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.hessian_reg_: np.ndarray | None = None
        self.solve_h_xt_: np.ndarray | None = None
        self.leverage_: np.ndarray | None = None
        self.covariance_free_: np.ndarray | None = None
        self.reported_covariance_: np.ndarray | None = None

        if X is not None and y is not None:
            self._set_arrays(X, y, competitor_names=competitor_names, reference_player=reference_player)
        elif X is not None or y is not None:
            raise ValueError("Provide both X and y, or neither.")

    @classmethod
    def from_dataset(cls, dataset: BattleDataset, **kwargs) -> "BradleyTerryModel":
        model = cls(
            dataset.design_matrix(),
            dataset.outcomes,
            competitor_names=dataset.competitors,
            reference_player=0,
            **kwargs,
        )
        model.match_frame_ = dataset.frame.copy()
        return model

    def fit(self) -> "BradleyTerryModel":
        self._require_data()
        assert self.X is not None and self.y is not None and self.n_params_ is not None

        estimator = LogisticRegression(fit_intercept=False, penalty=None)
        estimator.fit(self.X, self.y)
        beta_hat = np.asarray(estimator.coef_[0], dtype=float)
        probabilities = np.asarray(estimator.predict_proba(self.X)[:, 1], dtype=float)
        
        residuals = self.y - probabilities
        weights = probabilities * (1.0 - probabilities)
        hessian_reg = self.X.T @ (weights[:, None] * self.X) + self.hessian_ridge * np.eye(self.n_params_)
        solve_h_xt = self._solve(hessian_reg, self.X.T).T
        leverage = weights * np.einsum("ij,ij->i", self.X, solve_h_xt)

        self.beta_hat_ = beta_hat
        self.full_beta_hat_ = self._expand(beta_hat)
        self.reported_skills_ = self.full_beta_hat_.copy()
        self.probabilities_ = probabilities
        self.residuals_ = residuals
        self.hessian_reg_ = hessian_reg
        self.solve_h_xt_ = solve_h_xt
        self.leverage_ = leverage
        self.covariance_free_ = None
        self.reported_covariance_ = None
        return self

    def summary(
        self,
        ci_method: str = "sandwich",
        alpha: float = 0.05,
        n_bootstrap: int = 200,
        seed: int = 0,
    ) -> pd.DataFrame:
        from .ci import compute_confidence_intervals

        result = compute_confidence_intervals(
            self,
            method=ci_method,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        return result.to_frame(self.competitor_names_)

    def resolve_player(self, player: int | str) -> int:
        self._require_data()
        if isinstance(player, str):
            if player not in self.competitor_names_:
                raise ValueError(f"Unknown player name {player!r}.")
            return self.competitor_names_.index(player)
        idx = int(player)
        if not 0 <= idx < self.n_players_:
            raise ValueError(f"Player index must be in [0, {self.n_players_}).")
        return idx

    def expand_free_vector(self, free_vector: np.ndarray) -> np.ndarray:
        return self._expand(free_vector)

    def project_full_gradient(self, full_gradient: np.ndarray) -> np.ndarray:
        self._require_data()
        full_gradient = np.asarray(full_gradient, dtype=float)
        if full_gradient.shape != (self.n_players_,):
            raise ValueError("full_gradient must have shape (n_players,).")
        return full_gradient[np.arange(self.n_players_) != self.reference_player]

    def reported_gap(self, player_a: int | str, player_b: int | str, scaled: bool = False) -> float:
        self._require_fit()
        a_idx = self.resolve_player(player_a)
        b_idx = self.resolve_player(player_b)
        gap = float(self.reported_skills_[a_idx] - self.reported_skills_[b_idx])
        return float(self.scale_skills(np.array([gap]))[0] - self.init_rating) if scaled else gap

    @property
    def _config_kwargs(self) -> dict:
        return {
            "reference_player": self.reference_player,
            "scale": self.scale,
            "base": self.base,
            "init_rating": self.init_rating,
            "anchor_player": self.anchor_player,
            "anchor_rating": self.anchor_rating,
            "hessian_ridge": self.hessian_ridge,
        }

    def full_skills(self) -> np.ndarray:
        self._require_fit()
        return np.asarray(self.full_beta_hat_, dtype=float)

    def reported_skills(self) -> np.ndarray:
        self._require_fit()
        return np.asarray(self.reported_skills_, dtype=float)

    def scale_skills(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        scaled = self.init_rating + self.alpha * values
        return scaled + self.anchor_shift()

    def scaled_skills(self) -> np.ndarray:
        return self.scale_skills(self.reported_skills())

    def anchor_shift(self) -> float:
        if self.anchor_player is None or self.anchor_rating is None:
            return 0.0
        self._require_fit()
        anchor_idx = self.resolve_player(self.anchor_player)
        anchor_base = self.init_rating + self.alpha * float(self.reported_skills_[anchor_idx])
        return float(self.anchor_rating) - anchor_base

    def observed_pair_counts(self) -> np.ndarray:
        self._require_data()
        counts = np.zeros((self.n_players_, self.n_players_), dtype=float)
        for row in self.X:
            full = self._expand(row)
            pos = np.flatnonzero(full > 0.0)
            neg = np.flatnonzero(full < 0.0)
            if len(pos) == 1 and len(neg) == 1:
                i, j = int(pos[0]), int(neg[0])
            elif len(pos) == 1:
                i, j = int(pos[0]), int(self.reference_player)
            elif len(neg) == 1:
                i, j = int(self.reference_player), int(neg[0])
            else:
                raise ValueError("Each BT design row must compare exactly two players.")
            counts[i, j] += 1.0
            counts[j, i] += 1.0
        return counts

    def _set_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        competitor_names: Iterable[str] | None,
        reference_player: int,
    ) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be 2D, y must be 1D, and they must have the same number of rows.")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("BradleyTerryModel expects binary outcomes in {0, 1}.")

        n_players = X.shape[1] + 1
        reference_player = int(reference_player)
        if not 0 <= reference_player < n_players:
            raise ValueError(f"reference_player must be in [0, {n_players}).")

        names = [f"player_{i}" for i in range(n_players)] if competitor_names is None else list(competitor_names)
        if len(names) != n_players:
            raise ValueError("Length of competitor_names must equal the number of players.")

        self.X = X
        self.y = y
        self.reference_player = reference_player
        self.competitor_names_ = names
        self.n_players_ = n_players
        self.n_params_ = X.shape[1]

    def _expand(self, values: np.ndarray) -> np.ndarray:
        self._require_data()
        arr = np.asarray(values, dtype=float)
        mask = np.arange(self.n_players_) != self.reference_player
        if arr.ndim == 1:
            if arr.shape[0] != self.n_params_:
                raise ValueError("Wrong free-vector length.")
            full = np.zeros(self.n_players_, dtype=float)
            full[mask] = arr
            return full
        if arr.ndim == 2:
            if arr.shape[0] != self.n_params_:
                raise ValueError("Wrong free-matrix leading dimension.")
            full = np.zeros((self.n_players_, arr.shape[1]), dtype=float)
            full[mask, :] = arr
            return full
        raise ValueError("Expected a 1D or 2D free-parameter array.")

    def _solve(self, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix) @ rhs

    def _require_data(self) -> None:
        if self.X is None or self.y is None or self.n_players_ is None or self.n_params_ is None:
            raise RuntimeError("Model data has not been set.")

    def _require_fit(self) -> None:
        if self.beta_hat_ is None or self.full_beta_hat_ is None or self.reported_skills_ is None:
            raise RuntimeError("Fit the model before calling this method.")
