from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..bt_model import BradleyTerryModel


@dataclass
class ExperimentState:
    X: np.ndarray
    y: np.ndarray
    frame: pd.DataFrame
    next_row_uid: int


def make_state(bt_model: BradleyTerryModel) -> ExperimentState:
    bt_model._require_fit()
    if bt_model.match_frame_ is not None:
        frame = bt_model.match_frame_.copy().reset_index(drop=True)
    else:
        frame = pd.DataFrame({"match_id": np.arange(bt_model.X.shape[0])})
    if "row_uid" not in frame.columns:
        frame["row_uid"] = np.arange(len(frame))
    return ExperimentState(
        X=np.asarray(bt_model.X, dtype=float).copy(),
        y=np.asarray(bt_model.y, dtype=float).copy(),
        frame=frame,
        next_row_uid=int(frame["row_uid"].max()) + 1 if len(frame) else 0,
    )


def fit_model_from_state(base_model: BradleyTerryModel, state: ExperimentState) -> BradleyTerryModel:
    config = dict(base_model._config_kwargs)
    model = BradleyTerryModel(
        state.X,
        state.y,
        competitor_names=base_model.competitor_names_,
        reference_player=base_model.reference_player,
        scale=config["scale"],
        base=config["base"],
        init_rating=config["init_rating"],
        anchor_player=config["anchor_player"],
        anchor_rating=config["anchor_rating"],
        hessian_ridge=config["hessian_ridge"],
    ).fit()
    model.match_frame_ = state.frame.copy().reset_index(drop=True)
    return model


def apply_add_candidate(state: ExperimentState, chosen: pd.Series) -> ExperimentState:
    new_frame_row = pd.DataFrame(
        [
            {
                "model_a": chosen["model_a"],
                "model_b": chosen["model_b"],
                "winner": float(chosen.get("outcome", 1.0)),
                "outcome": float(chosen.get("outcome", 1.0)),
                "match_id": int(chosen.get("match_id", -1)),
                "match_copy": "added",
                "row_uid": state.next_row_uid,
            }
        ]
    )
    return ExperimentState(
        X=np.vstack([state.X, np.asarray(chosen["_candidate_x"], dtype=float)]),
        y=np.append(state.y, float(chosen.get("outcome", 1.0))),
        frame=pd.concat([state.frame, new_frame_row], ignore_index=True),
        next_row_uid=state.next_row_uid + 1,
    )


def apply_drop_row(state: ExperimentState, row_uid: int) -> ExperimentState:
    row_pos = row_position(state.frame, row_uid)
    mask = np.ones(len(state.frame), dtype=bool)
    mask[row_pos] = False
    return ExperimentState(
        X=state.X[mask],
        y=state.y[mask],
        frame=state.frame.loc[mask].reset_index(drop=True),
        next_row_uid=state.next_row_uid,
    )


def apply_flip_row(state: ExperimentState, row_uid: int) -> ExperimentState:
    row_pos = row_position(state.frame, row_uid)
    new_y = state.y.copy()
    new_y[row_pos] = 1.0 - new_y[row_pos]
    frame = state.frame.copy()
    if "winner" in frame.columns:
        frame.loc[row_pos, "winner"] = new_y[row_pos]
    if "outcome" in frame.columns:
        frame.loc[row_pos, "outcome"] = new_y[row_pos]
    return ExperimentState(X=state.X.copy(), y=new_y, frame=frame, next_row_uid=state.next_row_uid)


def row_position(frame: pd.DataFrame, row_uid: int) -> int:
    matches = np.flatnonzero(frame["row_uid"].to_numpy(dtype=int) == int(row_uid))
    if matches.size != 1:
        raise ValueError(f"Could not find a unique row with row_uid={row_uid}.")
    return int(matches[0])


def ranking_frame(bt_model: BradleyTerryModel, *, ci_method: str = "sandwich") -> pd.DataFrame:
    summary = bt_model.summary(ci_method=ci_method).copy()
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def is_target_favored(row: pd.Series, target_player: str) -> bool:
    if str(row["model_a"]) == target_player:
        return float(row.get("outcome", row.get("winner", 1.0))) >= 0.5
    if str(row["model_b"]) == target_player:
        return float(row.get("outcome", row.get("winner", 1.0))) <= 0.5
    return False

