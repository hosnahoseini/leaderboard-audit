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

from clean_bt_rank import BattleDataset, BradleyTerryModel, plot_ci_comparison  # noqa: E402


ARENA_ANCHOR_PLAYER = "mixtral-8x7b-instruct-v0.1"
ARENA_ANCHOR_RATING = 1114.0


def load_arena55k_dataframe() -> pd.DataFrame:
    ds = load_dataset("lmarena-ai/arena-human-preference-55k")
    df = ds["train"].to_pandas()

    winner = np.where(
        df["winner_tie"] == 1,
        "tie",
        np.where(df["winner_model_a"] == 1, "model_a", "model_b"),
    )
    return pd.DataFrame(
        {
            "model_a": df["model_a"].values,
            "model_b": df["model_b"].values,
            "winner": winner,
        }
    )


def run_arena55k_ci_comparison() -> Path:
    arena_df = load_arena55k_dataframe()
    dataset = BattleDataset.from_dataframe(arena_df)
    model = BradleyTerryModel.from_dataset(
        dataset,
        anchor_player=ARENA_ANCHOR_PLAYER,
        anchor_rating=ARENA_ANCHOR_RATING,
    ).fit()

    summaries = {
        "sandwich": model.summary(ci_method="sandwich"),
        "gao_local": model.summary(ci_method="gao_local"),
        "bootstrap": model.summary(ci_method="bootstrap", n_bootstrap=5, seed=0),
    }

    for name, summary in summaries.items():
        assert len(summary) == dataset.n_competitors, f"{name} summary has wrong length"
        assert np.all(summary["ci_upper"] >= summary["ci_lower"]), f"{name} has invalid intervals"
        assert "standard_error" in summary.columns, f"{name} missing standard_error column"

    out_dir = ROOT / "tests" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "arena55k_ci_comparison.png"
    fig, _ = plot_ci_comparison(
        summaries,
        top_n=12,
        title=f"Arena 55k CI Comparison (anchor: {ARENA_ANCHOR_PLAYER} = {ARENA_ANCHOR_RATING:.0f})",
    )
    fig.savefig(out_path, dpi=160)
    return out_path


if __name__ == "__main__":
    out_path = run_arena55k_ci_comparison()
    print(f"Saved plot to: {out_path}")
