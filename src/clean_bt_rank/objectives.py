from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from scipy.stats import norm

from .bt_model import BradleyTerryModel
from .ci import compute_standard_errors
from .parameter_influence import BTParameterInfluence


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


class Objective(ABC):
    """Base interface for scalar objectives of the fitted BT parameters."""

    @abstractmethod
    def value(self, bt_model: BradleyTerryModel) -> float:
        """Return the scalar objective value at the fitted BT parameters."""

    @abstractmethod
    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        """
        Return the objective gradient with respect to the free BT parameters.

        Shape: (p,)
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable objective name."""

    def explicit_weight_influence(
        self,
        bt_model: BradleyTerryModel,
        *,
        action: str = "drop",
        X_new: np.ndarray | None = None,
        y_new: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return the explicit data-weight term df/dw for a chosen action.

        This is the second term in

            d/dw_n f(theta_hat(w), w)
              = grad_theta f^T d theta_hat / dw_n + partial f / partial w_n.

        The default implementation is zero, corresponding to objectives that
        depend on the data only through the fitted parameters.
        """
        action = action.lower()
        if action == "drop":
            assert bt_model.X is not None
            return np.zeros(bt_model.X.shape[0], dtype=float)
        if action == "add":
            if X_new is None:
                raise ValueError("action='add' requires X_new.")
            return np.zeros(np.asarray(X_new).shape[0], dtype=float)
        if action == "flip":
            assert bt_model.X is not None
            return np.zeros(bt_model.X.shape[0], dtype=float)
        raise ValueError("action must be 'drop', 'add', or 'flip'.")


def _bt_variance_derivative(diff: np.ndarray) -> np.ndarray:
    prob = _sigmoid(diff)
    var = prob * (1.0 - prob)
    return var * (1.0 - 2.0 * prob)


def _rho_squared(bt_model: BradleyTerryModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_skills = bt_model.full_skills()
    pair_counts = bt_model.observed_pair_counts()
    diff = full_skills[:, None] - full_skills[None, :]
    prob = _sigmoid(diff)
    var = prob * (1.0 - prob)
    rho_sq = np.sum(pair_counts * var, axis=1)
    return rho_sq, pair_counts, diff


def _row_pairs_from_design(bt_model: BradleyTerryModel, x_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_rows = np.asarray(x_rows, dtype=float)
    if x_rows.ndim == 1:
        x_rows = x_rows[None, :]
    player_a = np.zeros(x_rows.shape[0], dtype=int)
    player_b = np.zeros(x_rows.shape[0], dtype=int)
    for row_idx, row in enumerate(x_rows):
        full = bt_model._expand(row)
        pos = np.flatnonzero(full > 0.0)
        neg = np.flatnonzero(full < 0.0)
        if len(pos) == 1 and len(neg) == 1:
            player_a[row_idx] = int(pos[0])
            player_b[row_idx] = int(neg[0])
        elif len(pos) == 1:
            player_a[row_idx] = int(pos[0])
            player_b[row_idx] = int(bt_model.reference_player)
        elif len(neg) == 1:
            player_a[row_idx] = int(bt_model.reference_player)
            player_b[row_idx] = int(neg[0])
        else:
            raise ValueError("Each BT design row must compare exactly two players.")
    return player_a, player_b


def _row_pair_variance(bt_model: BradleyTerryModel, x_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    player_a, player_b = _row_pairs_from_design(bt_model, x_rows)
    skills = bt_model.full_skills()
    diff = skills[player_a] - skills[player_b]
    prob = _sigmoid(diff)
    var = prob * (1.0 - prob)
    return player_a, player_b, var


class SkillGapObjective(Objective):
    """
    Objective f(beta) = skill_a - skill_b.

    ``player_a`` and ``player_b`` are expressed in the full player index space
    used by the dataset / competitor list. If either player is the reference
    player, its skill is fixed to 0 and contributes no free-parameter entry.
    """

    def __init__(self, player_a: int | str, player_b: int | str) -> None:
        if player_a == player_b:
            raise ValueError("player_a and player_b must be different.")
        self.player_a = player_a
        self.player_b = player_b

    @property
    def name(self) -> str:
        return "skill_gap"

    def value(self, bt_model: BradleyTerryModel) -> float:
        bt_model._require_fit()
        a_idx = bt_model.resolve_player(self.player_a)
        b_idx = bt_model.resolve_player(self.player_b)
        assert bt_model.full_beta_hat_ is not None
        return float(bt_model.full_beta_hat_[a_idx] - bt_model.full_beta_hat_[b_idx])

    def gradient_full(self, bt_model: BradleyTerryModel) -> np.ndarray:
        a_idx = bt_model.resolve_player(self.player_a)
        b_idx = bt_model.resolve_player(self.player_b)
        grad = np.zeros(bt_model.n_players_, dtype=float)
        grad[a_idx] += 1.0
        grad[b_idx] -= 1.0
        return grad

    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return bt_model.project_full_gradient(self.gradient_full(bt_model))


class PlayerUncertaintyObjective(Objective):
    """Proxy variance objective for one player: f(theta) = 1 / rho_i(theta)^2."""

    def __init__(self, player: int | str) -> None:
        self.player = player

    @property
    def name(self) -> str:
        return "player_uncertainty"

    def value(self, bt_model: BradleyTerryModel) -> float:
        player_idx = bt_model.resolve_player(self.player)
        rho_sq, _, _ = _rho_squared(bt_model)
        return float(np.divide(1.0, rho_sq[player_idx], out=np.array(np.inf), where=rho_sq[player_idx] > 0.0))

    def gradient_full(self, bt_model: BradleyTerryModel) -> np.ndarray:
        player_idx = bt_model.resolve_player(self.player)
        rho_sq, pair_counts, diff = _rho_squared(bt_model)
        drho = np.zeros(bt_model.n_players_, dtype=float)
        deriv = _bt_variance_derivative(diff[player_idx])

        for other_idx in range(bt_model.n_players_):
            if other_idx == player_idx or pair_counts[player_idx, other_idx] == 0.0:
                continue
            term = pair_counts[player_idx, other_idx] * deriv[other_idx]
            drho[player_idx] += term
            drho[other_idx] -= term

        scale = -1.0 / np.clip(rho_sq[player_idx], 1e-12, None) ** 2
        return scale * drho

    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return bt_model.project_full_gradient(self.gradient_full(bt_model))

    def explicit_weight_influence(
        self,
        bt_model: BradleyTerryModel,
        *,
        action: str = "drop",
        X_new: np.ndarray | None = None,
        y_new: np.ndarray | None = None,
    ) -> np.ndarray:
        action = action.lower()
        if action == "flip":
            assert bt_model.X is not None
            return np.zeros(bt_model.X.shape[0], dtype=float)

        x_rows = np.asarray(bt_model.X if action == "drop" else X_new, dtype=float)
        player_idx = bt_model.resolve_player(self.player)
        rho_sq, _, _ = _rho_squared(bt_model)
        safe_rho_sq = np.clip(rho_sq[player_idx], 1e-12, None)
        player_a, player_b, var = _row_pair_variance(bt_model, x_rows)
        derivative = -var * ((player_a == player_idx) | (player_b == player_idx)) / (safe_rho_sq ** 2)
        sign = -1.0 if action == "drop" else 1.0
        return sign * derivative


class TraceUncertaintyObjective(Objective):
    """Global uncertainty proxy: sum_i 1 / rho_i(theta)^2."""

    @property
    def name(self) -> str:
        return "trace_uncertainty"

    def value(self, bt_model: BradleyTerryModel) -> float:
        rho_sq, _, _ = _rho_squared(bt_model)
        return float(np.sum(np.divide(1.0, rho_sq, out=np.full_like(rho_sq, np.inf), where=rho_sq > 0.0)))

    def gradient_full(self, bt_model: BradleyTerryModel) -> np.ndarray:
        rho_sq, pair_counts, diff = _rho_squared(bt_model)
        n_players = bt_model.n_players_
        grad = np.zeros(n_players, dtype=float)

        for player_idx in range(n_players):
            deriv = _bt_variance_derivative(diff[player_idx])
            drho = np.zeros(n_players, dtype=float)
            for other_idx in range(n_players):
                if other_idx == player_idx or pair_counts[player_idx, other_idx] == 0.0:
                    continue
                term = pair_counts[player_idx, other_idx] * deriv[other_idx]
                drho[player_idx] += term
                drho[other_idx] -= term
            grad += (-1.0 / np.clip(rho_sq[player_idx], 1e-12, None) ** 2) * drho
        return grad

    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return bt_model.project_full_gradient(self.gradient_full(bt_model))

    def explicit_weight_influence(
        self,
        bt_model: BradleyTerryModel,
        *,
        action: str = "drop",
        X_new: np.ndarray | None = None,
        y_new: np.ndarray | None = None,
    ) -> np.ndarray:
        action = action.lower()
        if action == "flip":
            assert bt_model.X is not None
            return np.zeros(bt_model.X.shape[0], dtype=float)

        x_rows = np.asarray(bt_model.X if action == "drop" else X_new, dtype=float)
        rho_sq, _, _ = _rho_squared(bt_model)
        safe_rho_sq = np.clip(rho_sq, 1e-12, None)
        player_a, player_b, var = _row_pair_variance(bt_model, x_rows)
        derivative = -var * (
            1.0 / (safe_rho_sq[player_a] ** 2) + 1.0 / (safe_rho_sq[player_b] ** 2)
        )
        sign = -1.0 if action == "drop" else 1.0
        return sign * derivative


class GlobalCIWidthObjective(Objective):
    """Sum of coordinate-wise CI widths across all players."""

    def __init__(
        self,
        *,
        ci_method: str = "gao_local",
        alpha: float = 0.05,
        **ci_kwargs,
    ) -> None:
        self.ci_method = ci_method
        self.alpha = float(alpha)
        self.ci_kwargs = dict(ci_kwargs)
        self.z_value = float(norm.ppf(1.0 - self.alpha / 2.0))

    @property
    def name(self) -> str:
        return "global_ci_width"

    def value(self, bt_model: BradleyTerryModel) -> float:
        se = compute_standard_errors(bt_model, method=self.ci_method, **self.ci_kwargs).standard_error
        return float(2.0 * self.z_value * np.sum(se))

    def gradient_full(self, bt_model: BradleyTerryModel) -> np.ndarray:
        se_result = compute_standard_errors(bt_model, method=self.ci_method, **self.ci_kwargs)
        if se_result.gradient is None:
            raise RuntimeError(
                f"CI method {self.ci_method!r} has no SE gradient. "
                "GlobalCIWidthObjective requires a CI backend with SE gradients."
            )
        return 2.0 * self.z_value * np.sum(se_result.gradient, axis=0)

    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return bt_model.project_full_gradient(self.gradient_full(bt_model))

    def explicit_weight_influence(
        self,
        bt_model: BradleyTerryModel,
        *,
        action: str = "drop",
        X_new: np.ndarray | None = None,
        y_new: np.ndarray | None = None,
    ) -> np.ndarray:
        action = action.lower()
        if self.ci_method not in {"gao_local", "local_asymptotic"}:
            return super().explicit_weight_influence(bt_model, action=action, X_new=X_new, y_new=y_new)
        if action == "flip":
            assert bt_model.X is not None
            return np.zeros(bt_model.X.shape[0], dtype=float)

        x_rows = np.asarray(bt_model.X if action == "drop" else X_new, dtype=float)
        rho_sq, _, _ = _rho_squared(bt_model)
        safe_rho_sq = np.clip(rho_sq, 1e-12, None)
        player_a, player_b, var = _row_pair_variance(bt_model, x_rows)
        derivative = -self.z_value * var * (
            1.0 / (safe_rho_sq[player_a] ** 1.5) + 1.0 / (safe_rho_sq[player_b] ** 1.5)
        )
        sign = -1.0 if action == "drop" else 1.0
        return sign * derivative


class KendallTauObjective(Objective):
    """
    Smooth Kendall's tau surrogate against a fixed reference ranking.

    The objective is

        (2 / (P (P - 1))) * sum_{a < b} s_ab * tanh((theta_a - theta_b) / T)

    where s_ab in {-1, +1} is induced by the reference ranking.
    """

    def __init__(
        self,
        *,
        ranking: Iterable[int | str] | None = None,
        sign_matrix: np.ndarray | None = None,
        temperature: float = 1.0,
    ) -> None:
        if (ranking is None) == (sign_matrix is None):
            raise ValueError("Provide exactly one of ranking or sign_matrix.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.ranking = list(ranking) if ranking is not None else None
        self.sign_matrix = None if sign_matrix is None else np.asarray(sign_matrix, dtype=float)
        self.temperature = float(temperature)

    @property
    def name(self) -> str:
        return "kendall_tau"

    def value(self, bt_model: BradleyTerryModel) -> float:
        sign_matrix = self._resolve_sign_matrix(bt_model)
        skills = bt_model.full_skills()
        n_players = bt_model.n_players_
        normalizer = 2.0 / (n_players * (n_players - 1))

        total = 0.0
        for a_idx in range(n_players):
            for b_idx in range(a_idx + 1, n_players):
                total += sign_matrix[a_idx, b_idx] * np.tanh((skills[a_idx] - skills[b_idx]) / self.temperature)
        return float(normalizer * total)

    def gradient_full(self, bt_model: BradleyTerryModel) -> np.ndarray:
        sign_matrix = self._resolve_sign_matrix(bt_model)
        skills = bt_model.full_skills()
        n_players = bt_model.n_players_
        normalizer = 2.0 / (n_players * (n_players - 1))
        grad = np.zeros(n_players, dtype=float)

        for a_idx in range(n_players):
            for b_idx in range(a_idx + 1, n_players):
                diff = (skills[a_idx] - skills[b_idx]) / self.temperature
                weight = sign_matrix[a_idx, b_idx] * (1.0 - np.tanh(diff) ** 2) / self.temperature
                grad[a_idx] += weight
                grad[b_idx] -= weight
        return normalizer * grad

    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return bt_model.project_full_gradient(self.gradient_full(bt_model))

    def _resolve_sign_matrix(self, bt_model: BradleyTerryModel) -> np.ndarray:
        n_players = bt_model.n_players_
        if self.sign_matrix is not None:
            if self.sign_matrix.shape != (n_players, n_players):
                raise ValueError("sign_matrix must have shape (n_players, n_players).")
            sign_matrix = np.asarray(self.sign_matrix, dtype=float)
            if not np.allclose(np.diag(sign_matrix), 0.0):
                raise ValueError("sign_matrix must have zeros on the diagonal.")
            if not np.allclose(sign_matrix, -sign_matrix.T):
                raise ValueError("sign_matrix must be antisymmetric.")
            if not np.all(np.isin(sign_matrix, [-1.0, 0.0, 1.0])):
                raise ValueError("sign_matrix entries must lie in {-1, 0, 1}.")
            return sign_matrix

        assert self.ranking is not None
        if len(self.ranking) != n_players:
            raise ValueError("ranking must list every player exactly once.")
        ranking_idx = [bt_model.resolve_player(player) for player in self.ranking]
        if len(set(ranking_idx)) != n_players:
            raise ValueError("ranking must contain each player exactly once.")

        order = np.empty(n_players, dtype=int)
        for rank, player_idx in enumerate(ranking_idx):
            order[player_idx] = rank

        sign_matrix = np.zeros((n_players, n_players), dtype=float)
        for a_idx in range(n_players):
            for b_idx in range(a_idx + 1, n_players):
                sign = 1.0 if order[a_idx] < order[b_idx] else -1.0
                sign_matrix[a_idx, b_idx] = sign
                sign_matrix[b_idx, a_idx] = -sign
        return sign_matrix


class CIBoundaryObjective(Objective):
    """
    CI-aware top-k boundary margin with a frozen rank-k / rank-(k+1) pair.

    This objective always uses the full gradient path and does not use the
    frozen-standard-error approximation.
    """

    def __init__(
        self,
        bt_model: BradleyTerryModel,
        k: int,
        *,
        ci_method: str = "gao_local",
        alpha: float = 0.05,
        freeze_ranking: bool = True,
        freeze_se: bool | None = None,
        **ci_kwargs,
    ) -> None:
        bt_model._require_fit()
        if not freeze_ranking:
            raise ValueError("CIBoundaryObjective requires freeze_ranking=True.")
        if not 1 <= int(k) < bt_model.n_players_:
            raise ValueError(f"k must be in [1, {bt_model.n_players_ - 1}].")

        self.k = int(k)
        self.ci_method = ci_method
        self.alpha = float(alpha)
        self.freeze_ranking = True
        self.freeze_se = freeze_se
        self.ci_kwargs = dict(ci_kwargs)
        self.z_value = float(norm.ppf(1.0 - self.alpha / 2.0))

        ranking = np.argsort(-bt_model.full_skills())
        self.player_a_idx = int(ranking[self.k - 1])
        self.player_b_idx = int(ranking[self.k])
        self.player_a_name = bt_model.competitor_names_[self.player_a_idx]
        self.player_b_name = bt_model.competitor_names_[self.player_b_idx]

    @property
    def name(self) -> str:
        return "ci_boundary"

    def value(self, bt_model: BradleyTerryModel) -> float:
        skills = bt_model.full_skills()
        se = self._standard_errors(bt_model)
        a = self.player_a_idx
        b = self.player_b_idx
        return float((skills[a] - self.z_value * se[a]) - (skills[b] + self.z_value * se[b]))

    def gradient_full(self, bt_model: BradleyTerryModel) -> np.ndarray:
        grad = np.zeros(bt_model.n_players_, dtype=float)
        grad[self.player_a_idx] += 1.0
        grad[self.player_b_idx] -= 1.0

        se_result = compute_standard_errors(bt_model, method=self.ci_method, **self.ci_kwargs)
        if se_result.gradient is None:
            raise RuntimeError(
                f"CI method {self.ci_method!r} has no SE gradient. "
                "CIBoundaryObjective requires a CI backend with SE gradients."
            )
        grad -= self.z_value * se_result.gradient[self.player_a_idx]
        grad -= self.z_value * se_result.gradient[self.player_b_idx]
        return grad

    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return bt_model.project_full_gradient(self.gradient_full(bt_model))

    def explicit_weight_influence(
        self,
        bt_model: BradleyTerryModel,
        *,
        action: str = "drop",
        X_new: np.ndarray | None = None,
        y_new: np.ndarray | None = None,
    ) -> np.ndarray:
        action = action.lower()
        if self.ci_method not in {"gao_local", "local_asymptotic"}:
            return super().explicit_weight_influence(bt_model, action=action, X_new=X_new, y_new=y_new)
        if action == "flip":
            assert bt_model.X is not None
            return np.zeros(bt_model.X.shape[0], dtype=float)

        x_rows = np.asarray(bt_model.X if action == "drop" else X_new, dtype=float)
        rho_sq, _, _ = _rho_squared(bt_model)
        safe_rho_sq = np.clip(rho_sq, 1e-12, None)
        player_a, player_b, var = _row_pair_variance(bt_model, x_rows)
        se_deriv_a = -0.5 * var * (
            ((player_a == self.player_a_idx) | (player_b == self.player_a_idx)) / (safe_rho_sq[self.player_a_idx] ** 1.5)
        )
        se_deriv_b = -0.5 * var * (
            ((player_a == self.player_b_idx) | (player_b == self.player_b_idx)) / (safe_rho_sq[self.player_b_idx] ** 1.5)
        )
        derivative = -self.z_value * se_deriv_a - self.z_value * se_deriv_b
        sign = -1.0 if action == "drop" else 1.0
        return sign * derivative

    def _standard_errors(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return compute_standard_errors(bt_model, method=self.ci_method, **self.ci_kwargs).standard_error


class CIStrictGapObjective(Objective):
    """
    Strict CI-aware pairwise gap between an insider and outsider.

    The objective is

        (skill_inside + z * se_inside) - (skill_outside - z * se_outside)

    Negative values certify that the outsider remains above the insider even
    under the outsider's worst CI realization and the insider's best CI
    realization.
    """

    def __init__(
        self,
        player_inside: int | str,
        player_outside: int | str,
        *,
        ci_method: str = "gao_local",
        alpha: float = 0.05,
        freeze_se: bool | None = None,
        **ci_kwargs,
    ) -> None:
        if player_inside == player_outside:
            raise ValueError("player_inside and player_outside must be different.")
        self.player_inside = player_inside
        self.player_outside = player_outside
        self.ci_method = ci_method
        self.alpha = float(alpha)
        self.freeze_se = freeze_se
        self.ci_kwargs = dict(ci_kwargs)
        self.z_value = float(norm.ppf(1.0 - self.alpha / 2.0))

    @property
    def name(self) -> str:
        return "ci_strict_gap"

    def value(self, bt_model: BradleyTerryModel) -> float:
        bt_model._require_fit()
        skills = bt_model.full_skills()
        se = self._standard_errors(bt_model)
        inside = bt_model.resolve_player(self.player_inside)
        outside = bt_model.resolve_player(self.player_outside)
        return float((skills[inside] + self.z_value * se[inside]) - (skills[outside] - self.z_value * se[outside]))

    def gradient_full(self, bt_model: BradleyTerryModel) -> np.ndarray:
        inside = bt_model.resolve_player(self.player_inside)
        outside = bt_model.resolve_player(self.player_outside)
        grad = np.zeros(bt_model.n_players_, dtype=float)
        grad[inside] += 1.0
        grad[outside] -= 1.0

        se_result = compute_standard_errors(bt_model, method=self.ci_method, **self.ci_kwargs)
        if se_result.gradient is None:
            raise RuntimeError(
                f"CI method {self.ci_method!r} has no SE gradient. "
                "CIStrictGapObjective requires a CI backend with SE gradients."
            )
        grad += self.z_value * se_result.gradient[inside]
        grad += self.z_value * se_result.gradient[outside]
        return grad

    def gradient_free(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return bt_model.project_full_gradient(self.gradient_full(bt_model))

    def explicit_weight_influence(
        self,
        bt_model: BradleyTerryModel,
        *,
        action: str = "drop",
        X_new: np.ndarray | None = None,
        y_new: np.ndarray | None = None,
    ) -> np.ndarray:
        action = action.lower()
        if self.ci_method not in {"gao_local", "local_asymptotic"}:
            return super().explicit_weight_influence(bt_model, action=action, X_new=X_new, y_new=y_new)
        if action == "flip":
            assert bt_model.X is not None
            return np.zeros(bt_model.X.shape[0], dtype=float)

        x_rows = np.asarray(bt_model.X if action == "drop" else X_new, dtype=float)
        rho_sq, _, _ = _rho_squared(bt_model)
        safe_rho_sq = np.clip(rho_sq, 1e-12, None)
        inside = bt_model.resolve_player(self.player_inside)
        outside = bt_model.resolve_player(self.player_outside)
        player_a, player_b, var = _row_pair_variance(bt_model, x_rows)
        se_deriv_inside = -0.5 * var * (
            ((player_a == inside) | (player_b == inside)) / (safe_rho_sq[inside] ** 1.5)
        )
        se_deriv_outside = -0.5 * var * (
            ((player_a == outside) | (player_b == outside)) / (safe_rho_sq[outside] ** 1.5)
        )
        derivative = self.z_value * se_deriv_inside + self.z_value * se_deriv_outside
        sign = -1.0 if action == "drop" else 1.0
        return sign * derivative

    def _standard_errors(self, bt_model: BradleyTerryModel) -> np.ndarray:
        return compute_standard_errors(bt_model, method=self.ci_method, **self.ci_kwargs).standard_error


class ObjectiveInfluence:
    """Compute objective influence for drop/add actions."""

    def __init__(self, bt_model: BradleyTerryModel, parameter_influence: BTParameterInfluence) -> None:
        if parameter_influence.bt_model is not bt_model:
            raise ValueError("parameter_influence must be built from the same BT model.")
        self.bt_model = bt_model
        self.parameter_influence = parameter_influence

    def compute(
        self,
        objective: Objective,
        *,
        action: str = "drop",
        method: str = "if",
        X_new: np.ndarray | None = None,
        y_new: np.ndarray | None = None,
    ) -> np.ndarray:
        action = action.lower()
        if action == "drop":
            return self._project_objective(
                self.parameter_influence.compute_action(action="drop", method=method),
                objective,
                action=action,
            )
        if action == "add":
            if X_new is None or y_new is None:
                raise ValueError("action='add' requires X_new and y_new.")
            return self._project_objective(
                self.parameter_influence.compute_action(action="add", method=method, X=X_new, y=y_new),
                objective,
                action=action,
                X_new=X_new,
                y_new=y_new,
            )
        raise ValueError("action must be 'drop' or 'add'.")

    def compute_match_influence(
        self,
        objective: Objective,
        *,
        method: str = "if",
    ) -> np.ndarray:
        """Backward-compatible alias for per-match drop influence."""
        return self.compute(objective, action="drop", method=method)

    def _project_objective(
        self,
        delta_beta: np.ndarray,
        objective: Objective,
        *,
        action: str,
        X_new: np.ndarray | None = None,
        y_new: np.ndarray | None = None,
    ) -> np.ndarray:
        grad = objective.gradient_free(self.bt_model)
        theta_term = np.asarray(delta_beta, dtype=float) @ np.asarray(grad, dtype=float)
        explicit_term = objective.explicit_weight_influence(
            self.bt_model,
            action=action,
            X_new=X_new,
            y_new=y_new,
        )
        return theta_term + np.asarray(explicit_term, dtype=float)


def compute_objective_action_influence(
    bt_model: BradleyTerryModel,
    objective: Objective,
    *,
    action: str = "drop",
    method: str = "if",
    parameter_influence: BTParameterInfluence | None = None,
    X_new: np.ndarray | None = None,
    y_new: np.ndarray | None = None,
) -> np.ndarray:
    """
    Minimal public helper: objective + action -> influence vector.
    """
    engine = ObjectiveInfluence(bt_model, parameter_influence or BTParameterInfluence(bt_model))
    return engine.compute(objective, action=action, method=method, X_new=X_new, y_new=y_new)
