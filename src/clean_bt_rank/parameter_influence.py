from __future__ import annotations

import numpy as np
from scipy.special import expit

from .bt_model import BradleyTerryModel


class BTParameterInfluence:
    """
    Parameter-level deletion approximations for Bradley-Terry.

    For match i with score contribution s_i(beta) = x_i (y_i - p_i), this class
    returns estimated parameter changes after deleting that match:

    - IF:  Delta beta_i ≈ - H^{-1} s_i
    - 1sN: Delta beta_i ≈ - H^{-1} s_i / (1 - h_i)

    where h_i = v_i x_i^T H^{-1} x_i and v_i = p_i (1 - p_i).
    """

    def __init__(self, bt_model: BradleyTerryModel) -> None:
        bt_model._require_fit()
        self.bt_model = bt_model

    def score_contributions(self) -> np.ndarray:
        """Return score contributions with shape (n_matches, p)."""
        bt = self.bt_model
        assert bt.X is not None and bt.residuals_ is not None
        return bt.X * bt.residuals_[:, None]

    def candidate_parameter_change_if(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        dim: int | None = None,
    ) -> np.ndarray:
        """
        Return first-order parameter changes after adding hypothetical matches.

        Each row in ``X_new`` and ``y_  new`` is treated as a new weighted sample
        added to the current fit.
        """
        solve_h_x, residuals, _ = self._candidate_cache(X_new, y_new)
        delta = residuals[:, None] * solve_h_x
        return self._maybe_select_dim(delta, dim)

    def candidate_parameter_change_1sn(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        dim: int | None = None,
    ) -> np.ndarray:
        """
        Return one-step Newton parameter changes after adding hypothetical matches.

        This uses the additive analogue of the case-deletion correction:

            Delta beta_add^1sN ≈ Delta beta_add^IF / (1 + h_new)
        """
        solve_h_x, residuals, leverage = self._candidate_cache(X_new, y_new)
        delta_if = residuals[:, None] * solve_h_x
        delta_1sn = delta_if / np.clip(1.0 + leverage, 1e-12, None)[:, None]
        return self._maybe_select_dim(delta_1sn, dim)

    def compute_action(
        self,
        *,
        action: str = "drop",
        method: str = "if",
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        dim: int | None = None,
    ) -> np.ndarray:
        action = action.lower()
        if action == "drop":
            bt = self.bt_model
            assert bt.residuals_ is not None and bt.solve_h_xt_ is not None and bt.leverage_ is not None
            return self._compute_from_components(
                residuals=np.asarray(bt.residuals_, dtype=float),
                solve_h_x=np.asarray(bt.solve_h_xt_, dtype=float),
                leverage=np.asarray(bt.leverage_, dtype=float),
                sign=-1.0,
                method=method,
                dim=dim,
            )
        if action == "add":
            if X is None or y is None:
                raise ValueError("action='add' requires X and y.")
            solve_h_x, residuals, leverage = self._candidate_cache(X, y)
            return self._compute_from_components(
                residuals=residuals,
                solve_h_x=solve_h_x,
                leverage=leverage,
                sign=1.0,
                method=method,
                dim=dim,
            )
        raise ValueError("action must be 'drop' or 'add'.")

    def _compute_from_components(
        self,
        *,
        residuals: np.ndarray,
        solve_h_x: np.ndarray,
        leverage: np.ndarray,
        sign: float,
        method: str,
        dim: int | None,
    ) -> np.ndarray:
        method = method.lower()
        delta_if = sign * (np.asarray(residuals, dtype=float)[:, None] * np.asarray(solve_h_x, dtype=float))
        if method == "if":
            return self._maybe_select_dim(delta_if, dim)
        if method == "1sn":
            denom = np.clip(1.0 + sign * np.asarray(leverage, dtype=float), 1e-12, None)[:, None]
            return self._maybe_select_dim(delta_if / denom, dim)
        raise ValueError("method must be 'if' or '1sn'.")

    def _maybe_select_dim(self, values: np.ndarray, dim: int | None) -> np.ndarray:
        if dim is None:
            return values
        dim = int(dim)
        if not 0 <= dim < values.shape[1]:
            raise ValueError(f"dim must be in [0, {values.shape[1]}).")
        return values[:, dim]

    def _candidate_cache(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bt = self.bt_model
        bt._require_fit()
        X_new = np.asarray(X_new, dtype=float)
        y_new = np.asarray(y_new, dtype=float)
        if X_new.ndim != 2:
            raise ValueError("X_new must be a 2D array.")
        if y_new.ndim != 1 or y_new.shape[0] != X_new.shape[0]:
            raise ValueError("y_new must be a 1D array with one entry per candidate row.")
        if X_new.shape[1] != bt.n_params_:
            raise ValueError("X_new has the wrong number of columns.")
        if not np.all(np.isin(y_new, [0.0, 1.0])):
            raise ValueError("Candidate outcomes must be binary in {0, 1}.")

        logits = X_new @ bt.beta_hat_
        probabilities = expit(logits)
        residuals = y_new - probabilities
        solve_h_x = bt._solve(bt.hessian_reg_, X_new.T).T
        variances = probabilities * (1.0 - probabilities)
        leverage = variances * np.einsum("ij,ij->i", X_new, solve_h_x)
        return solve_h_x, residuals, leverage
