from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_OUTCOME_MAP: dict[str, float] = {
    "model_a": 1.0,
    "model_b": 0.0,
    "tie": 0.5,
    "both_bad": 0.5,
}

WINNER_LABEL_NORMALIZATION: dict[str, str] = {
    "tie (bothbad)": "both_bad",
    "tie(bothbad)": "both_bad",
    "bothbad": "both_bad",
    "both_bad": "both_bad",
}


@dataclass(frozen=True)
class HFBattleDatasetSpec:
    key: str
    hf_dataset: str | None
    split: str | None
    display_name: str
    outcome_mode: str = "winner"
    source_kind: str = "hf"
    anchor_player: str | None = None
    anchor_rating: float | None = None


@dataclass
class LoadedBattleData:
    key: str
    display_name: str
    hf_dataset: str
    split: str
    source_frame: pd.DataFrame
    battle_frame: pd.DataFrame
    fit_kwargs: dict[str, object]

# all dataset names
ALL_DATASET_NAMES = [
    "arena55k",
    "webdev_arena",
    "mt_bench_human",
    "llm_judge_arena",
    "vision_arena",
    "tennis_top10_atp",
    "nba_elo_top50",
]

HF_BATTLE_DATASETS: dict[str, HFBattleDatasetSpec] = {
    "arena55k": HFBattleDatasetSpec(
        key="arena55k",
        hf_dataset="lmarena-ai/arena-human-preference-55k",
        split="train",
        display_name="Chatbot Arena 55k",
        outcome_mode="winner_binary_tie",
        anchor_player="mixtral-8x7b-instruct-v0.1",
        anchor_rating=1114.0,
    ),
    "webdev_arena": HFBattleDatasetSpec(
        key="webdev_arena",
        hf_dataset="lmarena-ai/webdev-arena-preference-10k",
        split="test",
        display_name="WebDev Arena",
        outcome_mode="winner",
    ),
    "mt_bench_human": HFBattleDatasetSpec(
        key="mt_bench_human",
        hf_dataset="lmsys/mt_bench_human_judgments",
        split="human",
        display_name="MT-Bench Human Judgments",
        outcome_mode="winner",
    ),
    "llm_judge_arena": HFBattleDatasetSpec(
        key="llm_judge_arena",
        hf_dataset="potsawee/chatbot-arena-llm-judges",
        split="train",
        display_name="Chatbot Arena LLM Judges",
        outcome_mode="winner_binary_tie",
    ),
    "vision_arena": HFBattleDatasetSpec(
        key="vision_arena",
        hf_dataset="lmarena-ai/VisionArena-Battle",
        split="train",
        display_name="Vision Arena",
        outcome_mode="winner",
    ),
    "tennis_top10_atp": HFBattleDatasetSpec(
        key="tennis_top10_atp",
        hf_dataset=None,
        split=None,
        display_name="ATP Top-10 Matchups (2020-2024)",
        outcome_mode="winner",
        source_kind="tennis_top10_atp",
    ),
    "nba_elo_top50": HFBattleDatasetSpec(
        key="nba_elo_top50",
        hf_dataset=None,
        split=None,
        display_name="NBA Elo Top-50 Teams",
        outcome_mode="winner",
        source_kind="nba_elo_top50",
    ),
}

WRITABLE_HF_CACHE_ROOT = Path(os.environ.get("CLEAN_BT_RANK_HF_CACHE", "/tmp/clean_bt_rank_hf_cache"))


def _map_outcome(value: object, outcome_map: dict | Callable[[object], float] | None) -> float:
    if callable(outcome_map):
        out = float(outcome_map(value))
    elif outcome_map is not None:
        out = float(outcome_map.get(value, np.nan))
    elif isinstance(value, (int, float, np.floating)):
        out = float(value)
    else:
        out = float(DEFAULT_OUTCOME_MAP.get(value, np.nan))  # type: ignore[arg-type]

    if np.isnan(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"Outcome {value!r} could not be mapped to a number in [0, 1].")
    return out


def _build_weighted_symmetric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    outcomes = frame["winner"].map(lambda value: _map_outcome(value, None)).to_numpy(dtype=float)
    if not np.all(np.isin(outcomes, [0.0, 0.5, 1.0])):
        raise ValueError("Weighted symmetric tie handling requires outcomes in {0, 0.5, 1.0}.")

    forward = frame.copy()
    reverse = frame.copy()
    reverse["model_a"] = frame["model_b"].to_numpy()
    reverse["model_b"] = frame["model_a"].to_numpy()

    forward_outcome = outcomes.copy()
    reverse_outcome = 1.0 - outcomes
    tie_mask = outcomes == 0.5
    forward_outcome[tie_mask] = 1.0
    reverse_outcome[tie_mask] = 1.0

    forward["winner"] = forward_outcome
    reverse["winner"] = reverse_outcome
    forward["match_copy"] = "forward"
    reverse["match_copy"] = "reverse"
    return pd.concat([forward, reverse], ignore_index=True)


def available_hf_battle_datasets() -> dict[str, HFBattleDatasetSpec]:
    return dict(HF_BATTLE_DATASETS)


def get_hf_battle_dataset_spec(dataset_key: str) -> HFBattleDatasetSpec:
    key = str(dataset_key).strip().lower()
    if key not in HF_BATTLE_DATASETS:
        available = ", ".join(sorted(HF_BATTLE_DATASETS))
        raise ValueError(f"Unknown dataset_key {dataset_key!r}. Expected one of: {available}.")
    return HF_BATTLE_DATASETS[key]


def standardize_battle_dataframe(
    frame: pd.DataFrame,
    *,
    model_a_col: str = "model_a",
    model_b_col: str = "model_b",
    winner_col: str | None = None,
    winner_model_a_col: str | None = None,
    winner_tie_col: str | None = None,
) -> pd.DataFrame:
    required = [model_a_col, model_b_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = frame[[model_a_col, model_b_col]].copy()
    out.columns = ["model_a", "model_b"]

    if winner_col is not None and winner_col in frame.columns:
        out["winner"] = (
            frame[winner_col]
            .astype(str)
            .str.strip()
            .map(lambda value: WINNER_LABEL_NORMALIZATION.get(value, value))
            .astype(object)
        )
        return out.reset_index(drop=True)

    if (
        winner_model_a_col is not None
        and winner_tie_col is not None
        and winner_model_a_col in frame.columns
        and winner_tie_col in frame.columns
    ):
        winner_model_a = frame[winner_model_a_col].to_numpy(dtype=int)
        winner_tie = frame[winner_tie_col].to_numpy(dtype=int)
        out["winner"] = np.where(
            winner_tie == 1,
            "tie",
            np.where(winner_model_a == 1, "model_a", "model_b"),
        )
        return out.reset_index(drop=True)

    raise ValueError(
        "Could not standardize dataframe. Provide either a winner column or both "
        "winner_model_a and winner_tie columns."
    )


def default_bt_fit_kwargs_for_dataset(dataset_key: str) -> dict[str, object]:
    spec = get_hf_battle_dataset_spec(dataset_key)
    fit_kwargs: dict[str, object] = {"hessian_ridge": 0.0}
    if spec.anchor_player is not None:
        fit_kwargs["anchor_player"] = spec.anchor_player
    if spec.anchor_rating is not None:
        fit_kwargs["anchor_rating"] = spec.anchor_rating
    return fit_kwargs


def _hf_cache_names(dataset_name: str) -> tuple[str, str]:
    org, repo = dataset_name.split("/", 1)
    return f"{org}___{repo}", f"datasets--{org}--{repo}"


def _prepare_writable_hf_cache(dataset_name: str) -> str:
    cache_root = WRITABLE_HF_CACHE_ROOT
    datasets_root = cache_root / "datasets"
    hub_root = cache_root / "hub"
    datasets_root.mkdir(parents=True, exist_ok=True)
    hub_root.mkdir(parents=True, exist_ok=True)

    home_root = Path.home() / ".cache" / "huggingface"
    dataset_dir_name, hub_dir_name = _hf_cache_names(dataset_name)

    source_dataset_dir = home_root / "datasets" / dataset_dir_name
    target_dataset_dir = datasets_root / dataset_dir_name
    if source_dataset_dir.exists() and not target_dataset_dir.exists():
        shutil.copytree(source_dataset_dir, target_dataset_dir)

    source_hub_dir = home_root / "hub" / hub_dir_name
    target_hub_dir = hub_root / hub_dir_name
    if source_hub_dir.exists() and not target_hub_dir.exists():
        shutil.copytree(source_hub_dir, target_hub_dir)

    if target_dataset_dir.exists() and target_hub_dir.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    return str(cache_root)


def _load_cached_hf_frame(dataset_name: str, split: str) -> pd.DataFrame | None:
    cache_root = Path(_prepare_writable_hf_cache(dataset_name))
    dataset_dir_name, _ = _hf_cache_names(dataset_name)
    dataset_root = cache_root / "datasets" / dataset_dir_name
    if not dataset_root.exists():
        return None

    arrow_paths = sorted(dataset_root.glob(f"**/*-{split}.arrow"))
    if not arrow_paths:
        arrow_paths = sorted(dataset_root.glob("**/*.arrow"))
    if not arrow_paths:
        return None

    from datasets import Dataset

    return Dataset.from_file(str(arrow_paths[0])).to_pandas()


def _load_tennis_top10_atp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    years = [2020, 2021, 2022, 2023, 2024]
    matches = pd.concat(
        [
            pd.read_csv(
                f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_{year}.csv"
            )
            for year in years
        ],
        ignore_index=True,
    )
    rankings = pd.read_csv(
        "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_rankings_current.csv"
    )

    top_player_ids = rankings["player"].head(10)
    top_matchups = matches[
        matches["winner_id"].isin(top_player_ids) & matches["loser_id"].isin(top_player_ids)
    ].copy()

    winner_counts = top_matchups["winner_id"].value_counts()
    loser_counts = top_matchups["loser_id"].value_counts()
    game_counts = winner_counts.add(loser_counts, fill_value=0)
    valid_players = game_counts[game_counts > 20].index
    filtered = top_matchups[
        top_matchups["winner_id"].isin(valid_players) & top_matchups["loser_id"].isin(valid_players)
    ].copy()

    battle_frame = filtered[["winner_name", "loser_name"]].rename(
        columns={"winner_name": "model_a", "loser_name": "model_b"}
    )
    battle_frame["winner"] = "model_a"
    return filtered.reset_index(drop=True), battle_frame.reset_index(drop=True)


def _load_nba_elo_top50_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv")
    source["team_won"] = (source["game_result"] == "W").astype(int)

    rawbt = source[["team_id", "opp_id", "team_won"]].copy()
    team_games = pd.concat([rawbt["team_id"], rawbt["opp_id"]], ignore_index=True).value_counts()
    top_teams = team_games.nlargest(50).index
    filtered = rawbt[rawbt["team_id"].isin(top_teams) & rawbt["opp_id"].isin(top_teams)].copy()

    battle_frame = filtered.rename(columns={"team_id": "model_a", "opp_id": "model_b"})
    battle_frame["winner"] = np.where(battle_frame["team_won"].to_numpy(dtype=int) == 1, "model_a", "model_b")
    battle_frame = battle_frame[["model_a", "model_b", "winner"]]
    return filtered.reset_index(drop=True), battle_frame.reset_index(drop=True)


def load_named_battle_data(dataset_key: str) -> LoadedBattleData:
    spec = get_hf_battle_dataset_spec(dataset_key)
    if spec.source_kind == "hf":
        from datasets import DownloadConfig, load_dataset

        source_frame = _load_cached_hf_frame(spec.hf_dataset, spec.split)
        if source_frame is None:
            cache_dir = _prepare_writable_hf_cache(spec.hf_dataset)
            source_frame = load_dataset(
                spec.hf_dataset,
                split=spec.split,
                cache_dir=cache_dir,
                download_config=DownloadConfig(local_files_only=True),
            ).to_pandas()
        if spec.outcome_mode == "winner":
            battle_frame = standardize_battle_dataframe(source_frame, winner_col="winner")
        elif spec.outcome_mode == "winner_binary_tie":
            battle_frame = standardize_battle_dataframe(
                source_frame,
                winner_model_a_col="winner_model_a",
                winner_tie_col="winner_tie",
            )
        else:
            raise ValueError(f"Unsupported outcome_mode {spec.outcome_mode!r}.")
    elif spec.source_kind == "tennis_top10_atp":
        source_frame, battle_frame = _load_tennis_top10_atp_data()
    elif spec.source_kind == "nba_elo_top50":
        source_frame, battle_frame = _load_nba_elo_top50_data()
    else:
        raise ValueError(f"Unsupported source_kind {spec.source_kind!r}.")

    return LoadedBattleData(
        key=spec.key,
        display_name=spec.display_name,
        hf_dataset=spec.hf_dataset or spec.source_kind,
        split=spec.split or "",
        source_frame=source_frame,
        battle_frame=battle_frame,
        fit_kwargs=default_bt_fit_kwargs_for_dataset(spec.key),
    )


@dataclass
class BattleDataset:
    competitors: list[str]
    pairs: np.ndarray
    outcomes: np.ndarray
    frame: pd.DataFrame

    @property
    def competitor_to_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.competitors)}

    @property
    def n_competitors(self) -> int:
        return len(self.competitors)

    @property
    def n_matches(self) -> int:
        return int(len(self.outcomes))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        model_a_col: str = "model_a",
        model_b_col: str = "model_b",
        outcome_col: str = "winner",
        outcome_map: dict | Callable[[object], float] | None = None,
        competitors: list[str] | None = None,
        *,
        weighted_symmetric_ties: bool = False,
    ) -> "BattleDataset":
        required = [model_a_col, model_b_col, outcome_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        frame = df.loc[:, required].copy().reset_index(drop=True)
        frame.columns = ["model_a", "model_b", "winner"]
        frame["match_id"] = np.arange(len(frame))
        frame["winner"] = frame["winner"].map(lambda value: _map_outcome(value, outcome_map))
        if not weighted_symmetric_ties:
            mapped = frame["winner"].to_numpy(dtype=float)
            mapped[mapped == 0.5] = 1.0
            frame["winner"] = mapped
        frame = _build_weighted_symmetric_frame(frame)
        frame["outcome"] = frame["winner"].astype(float)

        if competitors is None:
            competitors = pd.Index(pd.concat([frame["model_a"], frame["model_b"]], ignore_index=True).unique()).tolist()
        else:
            competitors = list(competitors)
            seen = set(pd.concat([frame["model_a"], frame["model_b"]], ignore_index=True).unique())
            unknown = sorted(seen - set(competitors))
            if unknown:
                raise ValueError(f"Found competitors not present in the supplied competitor list: {unknown}")

        index = {name: idx for idx, name in enumerate(competitors)}
        pairs = np.column_stack(
            [
                frame["model_a"].map(index).to_numpy(dtype=int),
                frame["model_b"].map(index).to_numpy(dtype=int),
            ]
        )
        outcomes = frame["outcome"].to_numpy(dtype=float)
        return cls(competitors=competitors, pairs=pairs, outcomes=outcomes, frame=frame)

    def design_matrix(self) -> np.ndarray:
        x = np.zeros((self.n_matches, self.n_competitors - 1), dtype=float)
        rows = np.arange(self.n_matches)
        a_idx = self.pairs[:, 0]
        b_idx = self.pairs[:, 1]

        a_mask = a_idx > 0
        b_mask = b_idx > 0
        x[rows[a_mask], a_idx[a_mask] - 1] = 1.0
        x[rows[b_mask], b_idx[b_mask] - 1] = -1.0
        return x


def generate_synthetic_dataframe(
    n_models: int = 8,
    n_matches: int = 2000,
    seed: int = 0,
    skill_scale: float = 1.0,
    tie_probability: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_models < 2:
        raise ValueError("n_models must be at least 2.")
    if not 0.0 <= tie_probability < 1.0:
        raise ValueError("tie_probability must be in [0, 1).")

    rng = np.random.default_rng(seed)
    names = [f"model_{i}" for i in range(n_models)]
    true_skills = rng.normal(0.0, skill_scale, size=n_models)
    true_skills = true_skills - true_skills.mean()

    matches = []
    for match_id in range(n_matches):
        a = int(rng.integers(0, n_models))
        b = (a + int(rng.integers(1, n_models))) % n_models
        prob_a = 1.0 / (1.0 + np.exp(-(true_skills[a] - true_skills[b])))

        if rng.random() < tie_probability:
            winner = "tie"
        else:
            winner = "model_a" if rng.random() < prob_a else "model_b"

        matches.append({"match_id": match_id, "model_a": names[a], "model_b": names[b], "winner": winner})

    frame = pd.DataFrame(matches)
    truth = (
        pd.DataFrame({"competitor": names, "true_skill": true_skills})
        .sort_values("true_skill", ascending=False)
        .reset_index(drop=True)
    )
    return frame, truth


def generate_synthetic_dataset(
    n_models: int = 8,
    n_matches: int = 2000,
    seed: int = 0,
    skill_scale: float = 1.0,
    tie_probability: float = 0.0,
) -> tuple[BattleDataset, pd.DataFrame]:
    frame, truth = generate_synthetic_dataframe(
        n_models=n_models,
        n_matches=n_matches,
        seed=seed,
        skill_scale=skill_scale,
        tie_probability=tie_probability,
    )
    return BattleDataset.from_dataframe(frame), truth
