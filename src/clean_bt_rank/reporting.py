from __future__ import annotations

import numpy as np
import pandas as pd

from .bt_model import BradleyTerryModel
from .objectives import Objective


def make_objective_influence_report(
    bt_model: BradleyTerryModel,
    objective: Objective,
    influence: pd.Series | list[float] | tuple[float, ...] | np.ndarray,
    *,
    influence_name: str = "objective_influence",
) -> pd.DataFrame:
    bt_model._require_fit()
    influence_array = np.asarray(influence, dtype=float)
    if influence_array.ndim != 1:
        raise ValueError("influence must be a 1D array-like object.")
    if bt_model.X is not None and influence_array.shape[0] != bt_model.X.shape[0]:
        raise ValueError("influence length must match the number of matches.")

    if bt_model.match_frame_ is not None:
        report = bt_model.match_frame_.copy()
    else:
        report = pd.DataFrame({"match_id": range(influence_array.shape[0])})

    report[influence_name] = influence_array
    report[f"{influence_name}_abs"] = report[influence_name].abs()
    report["fitted_probability"] = bt_model.probabilities_
    report["residual"] = bt_model.residuals_
    report["leverage"] = bt_model.leverage_
    report["objective_value"] = objective.value(bt_model)
    return report


def top_positive(report: pd.DataFrame, column: str, k: int = 10) -> pd.DataFrame:
    return report.sort_values(column, ascending=False).head(k).reset_index(drop=True)


def top_negative(report: pd.DataFrame, column: str, k: int = 10) -> pd.DataFrame:
    return report.sort_values(column, ascending=True).head(k).reset_index(drop=True)


def top_absolute(report: pd.DataFrame, column: str, k: int = 10) -> pd.DataFrame:
    abs_col = f"{column}_abs"
    work = report.copy()
    if abs_col not in work.columns:
        work[abs_col] = work[column].abs()
    return work.sort_values(abs_col, ascending=False).head(k).reset_index(drop=True)
