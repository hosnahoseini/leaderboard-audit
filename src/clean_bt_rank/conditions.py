from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ObjectiveCondition:
    is_met: Callable[[float], bool]
    distance: Callable[[float], float]
    description: str


def make_threshold_condition(target: float, relation: str) -> ObjectiveCondition:
    if relation == "<=":
        return ObjectiveCondition(
            is_met=lambda value, t=target: value <= t,
            distance=lambda value, t=target: max(0.0, value - t),
            description=f"value <= {target:.6f}",
        )
    if relation == ">=":
        return ObjectiveCondition(
            is_met=lambda value, t=target: value >= t,
            distance=lambda value, t=target: max(0.0, t - value),
            description=f"value >= {target:.6f}",
        )
    raise ValueError("relation must be '<=' or '>='.")
