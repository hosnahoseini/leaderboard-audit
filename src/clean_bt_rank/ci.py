from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from .bt_model import BradleyTerryModel


class CIResult(NamedTuple):
    method: str
    alpha: float
    estimate: np.ndarray
    standard_error: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def to_frame(self, competitors: list[str]) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "competitor": competitors,
                    "rating": self.estimate,
                    "standard_error": self.standard_error,
                    "ci_lower": self.lower,
                    "ci_upper": self.upper,
                    "ci_method": self.method,
                }
            )
            .sort_values("rating", ascending=False, ignore_index=True)
        )


class CIStandardErrorResult(NamedTuple):
    method: str
    standard_error: np.ndarray
    gradient: np.ndarray | None


class CIBackend(ABC):
    method_name: str

    @abstractmethod
    def compute(self, model: BradleyTerryModel, alpha: float, **kwargs) -> CIResult:
        pass


class SandwichCI(CIBackend):
    method_name = "sandwich"

    def compute(self, model: BradleyTerryModel, alpha: float, **kwargs) -> CIResult:
        _, reported_cov = _compute_sandwich_covariance(model)
        se = np.sqrt(np.clip(np.diag(reported_cov), 0.0, None))
        z = norm.ppf(1.0 - alpha / 2.0)
        estimate = model.scaled_skills()
        lower = model.scale_skills(model.reported_skills() - z * se)
        upper = model.scale_skills(model.reported_skills() + z * se)
        return CIResult(self.method_name, alpha, estimate, model.alpha * se, lower, upper)


class BootstrapCI(CIBackend):
    method_name = "bootstrap"

    def compute(self, model: BradleyTerryModel, alpha: float, **kwargs) -> CIResult:
        n_bootstrap = int(kwargs.get("n_bootstrap", 200))
        seed = int(kwargs.get("seed", 0))
        samples = _bootstrap_rating_samples(model, n_bootstrap=n_bootstrap, seed=seed)
        estimate = model.scaled_skills()
        lower = np.quantile(samples, alpha / 2.0, axis=0)
        upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
        se = np.std(samples, axis=0, ddof=1)
        return CIResult(self.method_name, alpha, estimate, se, lower, upper)


class GaoLocalCI(CIBackend):
    """
    Coordinate-wise local asymptotic CI following Gao, Shen, and Zhang.

    We use the sample analogue

        rho_i(theta_hat)^2 = sum_j N_ij * p_ij * (1 - p_ij)

    where N_ij is the observed number of comparisons between i and j in the
    fitted dataset and p_ij = sigmoid(theta_i - theta_j).

    Then

        se_i = 1 / rho_i(theta_hat)
        CI_i = theta_hat_i ± z_{1-alpha/2} * se_i

    The CI is returned in the model's reported parameterisation and then scaled
    to the external rating scale.
    """

    method_name = "gao_local"

    def compute(self, model: BradleyTerryModel, alpha: float, **kwargs) -> CIResult:
        full_skills = model.full_skills()
        pair_counts = model.observed_pair_counts()
        n_players = pair_counts.shape[0]

        rho_sq = np.zeros(n_players, dtype=float)
        for i in range(n_players):
            total = 0.0
            for j in range(n_players):
                if i == j or pair_counts[i, j] == 0.0:
                    continue
                diff = full_skills[i] - full_skills[j]
                p_ij = 1.0 / (1.0 + np.exp(-diff))
                v_ij = p_ij * (1.0 - p_ij)
                total += pair_counts[i, j] * v_ij
            rho_sq[i] = total

        rho = np.sqrt(np.clip(rho_sq, 0.0, None))
        se_natural = np.divide(1.0, rho, out=np.full_like(rho, np.inf), where=rho > 0.0)
        z = norm.ppf(1.0 - alpha / 2.0)
        reported = model.reported_skills()
        lower = model.scale_skills(reported - z * se_natural)
        upper = model.scale_skills(reported + z * se_natural)
        return CIResult(
            self.method_name,
            alpha,
            model.scaled_skills(),
            model.alpha * se_natural,
            lower,
            upper,
        )


CI_BACKENDS: dict[str, CIBackend] = {
    "sandwich": SandwichCI(),
    "bootstrap": BootstrapCI(),
    "gao_local": GaoLocalCI(),
    "local_asymptotic": GaoLocalCI(),
}


def compute_confidence_intervals(
    model: BradleyTerryModel,
    method: str = "sandwich",
    alpha: float = 0.05,
    **kwargs,
) -> CIResult:
    model._require_fit()
    try:
        backend = CI_BACKENDS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown ci_method {method!r}.") from exc
    return backend.compute(model, alpha=alpha, **kwargs)


def compute_standard_errors(
    model: BradleyTerryModel,
    method: str = "sandwich",
    **kwargs,
) -> CIStandardErrorResult:
    """
    Return natural-scale standard errors and, when available, their gradients.

    The returned `standard_error` is on the BT skill scale, not the displayed
    rating scale. If `gradient` is not `None`, it has shape
    `(n_players, n_players)` where row `i` is the full-gradient of `se_i`
    with respect to the full BT skill vector.
    """
    model._require_fit()
    if method == "sandwich":
        _, reported_cov = _compute_sandwich_covariance(model)
        se = np.sqrt(np.clip(np.diag(reported_cov), 0.0, None))
        return CIStandardErrorResult(method=method, standard_error=se, gradient=None)

    if method == "bootstrap":
        n_bootstrap = int(kwargs.get("n_bootstrap", 200))
        seed = int(kwargs.get("seed", 0))
        samples = _bootstrap_skill_samples(model, n_bootstrap=n_bootstrap, seed=seed)
        se = np.std(samples, axis=0, ddof=1)
        return CIStandardErrorResult(method=method, standard_error=se, gradient=None)

    if method in {"gao_local", "local_asymptotic"}:
        return _gao_local_standard_errors(model, method="gao_local")

    raise ValueError(f"Unknown ci_method {method!r}.")


def compute_skill_covariance(
    model: BradleyTerryModel,
    method: str = "sandwich",
    **kwargs,
) -> np.ndarray:
    """
    Return a full natural-scale covariance matrix for the BT skill vector.

    Currently only the sandwich backend provides a covariance matrix in the
    codebase. Other CI methods return intervals or standard errors but not a
    full covariance estimate.
    """
    model._require_fit()
    method = method.lower()
    if method == "sandwich":
        _, reported_cov = _compute_sandwich_covariance(model)
        return np.asarray(reported_cov, dtype=float)
    raise ValueError(f"CI method {method!r} does not provide a full covariance matrix.")


def _bootstrap_rating_samples(
    model: BradleyTerryModel,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> np.ndarray:
    return model.scale_skills(_bootstrap_skill_samples(model, n_bootstrap=n_bootstrap, seed=seed))


def _bootstrap_skill_samples(
    model: BradleyTerryModel,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> np.ndarray:
    model._require_fit()
    from .bt_model import BradleyTerryModel

    rng = np.random.default_rng(seed)
    n_matches = model.X.shape[0]
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_matches, n_matches)
        boot = BradleyTerryModel(
            model.X[idx],
            model.y[idx],
            competitor_names=model.competitor_names_,
            **model._config_kwargs,
        ).fit()
        samples.append(boot.reported_skills())
    return np.vstack(samples)


def _gao_local_standard_errors(
    model: BradleyTerryModel,
    *,
    method: str,
) -> CIStandardErrorResult:
    full_skills = model.full_skills()
    pair_counts = model.observed_pair_counts()
    diff = full_skills[:, None] - full_skills[None, :]
    prob = 1.0 / (1.0 + np.exp(-diff))
    var = prob * (1.0 - prob)
    rho_sq = np.sum(pair_counts * var, axis=1)
    se = np.divide(1.0, np.sqrt(np.clip(rho_sq, 0.0, None)), out=np.full_like(rho_sq, np.inf), where=rho_sq > 0.0)

    n_players = model.n_players_
    deriv = var * (1.0 - 2.0 * prob)
    grad = np.zeros((n_players, n_players), dtype=float)
    safe_rho_sq = np.clip(rho_sq, 1e-12, None)

    for player_idx in range(n_players):
        drho_sq = np.zeros(n_players, dtype=float)
        for other_idx in range(n_players):
            if other_idx == player_idx or pair_counts[player_idx, other_idx] == 0.0:
                continue
            term = pair_counts[player_idx, other_idx] * deriv[player_idx, other_idx]
            drho_sq[player_idx] += term
            drho_sq[other_idx] -= term
        grad[player_idx] = -0.5 * safe_rho_sq[player_idx] ** (-1.5) * drho_sq

    return CIStandardErrorResult(method=method, standard_error=se, gradient=grad)


def _compute_sandwich_covariance(
    model: BradleyTerryModel,
) -> tuple[np.ndarray, np.ndarray]:
    model._require_fit()
    if model.covariance_free_ is not None and model.reported_covariance_ is not None:
        return model.covariance_free_, model.reported_covariance_

    score = (model.y - model.probabilities_)[:, None] * model.X
    meat = score.T @ score
    solved_meat = model._solve(model.hessian_reg_, meat)
    cov_free = model._solve(model.hessian_reg_, solved_meat.T).T

    expand = model.expand_free_vector(np.eye(model.n_params_))
    cov_full = expand @ cov_free @ expand.T

    reported_cov = cov_full
    model.covariance_free_ = cov_free
    model.reported_covariance_ = reported_cov
    return cov_free, reported_cov
