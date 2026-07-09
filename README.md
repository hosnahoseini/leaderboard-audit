# Leaderboard Audit

[![Tests](https://github.com/hosnahoseini/leaderboard-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/hosnahoseini/leaderboard-audit/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/website-companion%20site-2c6e8f.svg)](https://hosnahoseini.github.io/leaderboard-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

How stable is a Bradley–Terry leaderboard? This repository fits BT models to
pairwise preference data, estimates the **influence** of individual matches on the
resulting ranking, and measures how many data edits — dropping, flipping, or
adding matches — are needed to change a ranking outcome.

It contains the code behind *A Unified Perturbation Framework for Leaderboard
Stability and Manipulation* — Hosna Oyarhoseini, Jimmy Lin, Amir-Hossein Karimi —
presented as an **oral at the CTB Workshop @ ICML 2026** — together with the
scripts that regenerate the published figures.

**[Explore the interactive companion site →](https://hosnahoseini.github.io/leaderboard-audit/)**
It walks through the perturbation framework chapter by chapter, with a live
sandbox for dropping, flipping, and adding matches on a real leaderboard.

## Features

- Fit Bradley–Terry models from pairwise comparisons.
- Compute ratings, confidence intervals, and ranking summaries.
- Estimate per-match influence for `drop`, `add`, and `flip` actions.
- Run iterative robustness and manipulation experiments across seven datasets.
- Compare against the `IsRankingRobust` baseline (ICML 2025).

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/hosnahoseini/leaderboard-audit.git
cd leaderboard-audit
python -m pip install -e ".[dev]"
```

## Quickstart

```python
import pandas as pd
from clean_bt_rank import BattleDataset, BradleyTerryModel, ranking_from_model

df = pd.DataFrame({
    "model_a": ["gpt", "gpt", "claude", "claude", "llama", "gpt"],
    "model_b": ["claude", "llama", "llama", "gpt", "gpt", "llama"],
    "winner":  ["model_a", "model_a", "model_a", "model_b", "model_b", "model_a"],
})

data = BattleDataset.from_dataframe(df)
model = BradleyTerryModel.from_dataset(data).fit()

print(ranking_from_model(model))   # ['gpt', 'claude', 'llama']
print(model.summary())             # ratings with sandwich confidence intervals
```

## Tests

```bash
pytest                 # offline suite
pytest -m network      # additionally runs tests that download from Hugging Face
```

## Datasets

Corpora are downloaded on first use and cached under `$CLEAN_BT_RANK_HF_CACHE`
(default `/tmp/clean_bt_rank_hf_cache`).

| Key | Source |
| --- | --- |
| `arena55k` | `lmarena-ai/arena-human-preference-55k` (Hugging Face) |
| `webdev_arena` | `lmarena-ai/webdev-arena-preference-10k` (Hugging Face) |
| `mt_bench_human` | `lmsys/mt_bench_human_judgments` (Hugging Face) |
| `llm_judge_arena` | Hugging Face |
| `vision_arena` | Hugging Face |
| `nba_elo_top50` | `fivethirtyeight/data` (GitHub) |
| `tennis_top10_atp` | `JeffSackmann/tennis_atp` (GitHub) — **see note below** |

> [!WARNING]
> **The `tennis_top10_atp` source is no longer available.** The upstream repository
> `JeffSackmann/tennis_atp` now returns HTTP 404, so `load_named_battle_data("tennis_top10_atp")`
> cannot fetch its inputs. Tennis figures can still be regenerated from the archived
> curve histories in `figure_data/`, but tennis experiments cannot be rerun from raw
> data without supplying a mirror of `atp_matches_2020..2024.csv` and
> `atp_rankings_current.csv`.

## Reproducing the thesis figures

Small CSVs backing each published figure are committed under `figure_data/`
(2.4 MB total), so most figures replot in seconds without rerunning the full
compute sweep. Scripts write to gitignored output directories by default; pass
`--output-dir` to control where results land.

### Replot directly from archived data

```bash
# Feature-vs-influence summary heatmap (main text)
python scripts/render_feature_summary_heatmap.py \
  --dataset all_datasets \
  --input-csv figure_data/player_kendall_influence_plots/player_influence_joint_newton_abs_per_dataset_feature_summary_plot_data.csv \
  --player-stats-dir figure_data/player_kendall_influence_plots \
  --output-dir out \
  --output-stem all_datasets__player_influence_joint_newton_abs_feature_summary_neurips_new

# Influence-vs-covariate correlation heatmap (appendix)
python scripts/plot_arena_influence_covariate_heatmap.py \
  --dataset arena55k --input-dir figure_data/arena_ablation_reset --output-dir out

# Trace-uncertainty curves, all 7 datasets (main text + appendix)
mkdir -p out/tau_recompute && cp figure_data/tau_ci_curve_analysis_recompute/*.csv out/tau_recompute/
python scripts/run_tau_ci_curve_analysis_recompute.py --replot-from-history --output-dir out/tau_recompute
```

`--replot-from-history` is a flag, not a path: it rebuilds figures from the
per-dataset `*_curve_history.csv` files already present in `--output-dir`.

### Rerun from raw data

These download corpora and refit models; expect minutes to hours per dataset.

```bash
# Kendall-tau curves, all 7 datasets
python scripts/run_tau_ci_curve_analysis.py --curve-steps 30 --output-dir out/tau

# Top-k actions-needed ablation (appendix)
python scripts/run_arena_ablation_reset.py --dataset arena55k --output-dir out/ablation

# Ranking before/after CI-aware vs non-CI drops (main text)
python scripts/run_ci_vs_nonci_robustness_actions_needed.py --datasets arena55k --output-dir out/ci_vs_nonci

# Cross-dataset action comparison (main text)
python scripts/run_topk_vs_rigging_comparison.py --output-dir out/topk_vs_rigging
```

### Figure index

| Published figure | Script | Replottable from `figure_data/` |
| --- | --- | --- |
| `all_datasets__…_feature_summary_neurips_new` | `render_feature_summary_heatmap.py` | yes |
| `arena55k__influence_covariate_correlation_heatmap__1sn` | `plot_arena_influence_covariate_heatmap.py` | yes |
| `<dataset>_trace_uncertainty_curves` (×7) | `run_tau_ci_curve_analysis_recompute.py` | yes |
| `<dataset>_kendall_tau_curves` (×7) | `run_tau_ci_curve_analysis.py` | no — needs `--curve-steps 30` |
| `arena55k_topk_actions_needed_curve` | `run_arena_ablation_reset.py` | no |
| `arena55k_k22_{ci_aware,nonci}_drop_ranking_before_after` | `run_ci_vs_nonci_robustness_actions_needed.py` | no |
| `dataset_action_comparison_vertical` | `run_topk_vs_rigging_comparison.py` | no |

Two details worth knowing when comparing against the thesis:

- **Kendall-tau vs. trace-uncertainty curves come from different runs.** The
  published Kendall-tau panels were produced by `run_tau_ci_curve_analysis.py`
  (influence computed once, 30 action steps); the trace-uncertainty panels were
  produced by `run_tau_ci_curve_analysis_recompute.py` (influence recomputed after
  each action, 25 steps). The figure titles record which is which. Only the
  25-step histories were archived, so the Kendall-tau curves require a full rerun.
- **`skill` is summarized but not plotted** in the feature-summary heatmap. It is
  the fitted BT parameter rather than a graph covariate, and its Q4–Q1 mean *z*
  (≈ −2.6) would dominate the shared color scale. Pass `--features` to override.

## Repository layout

```
src/clean_bt_rank/     Python package (BT fitting, influence, objectives, actions)
  experiments/         reusable experiment drivers
scripts/               experiment and figure entrypoints
figure_data/           small CSVs backing the published figures
tests/                 regression and workflow tests
IsRankingRobust/       vendored ICML 2025 baseline (MIT, see below)
docs/                  companion site, published at hosnahoseini.github.io/leaderboard-audit
```

## Baseline

`IsRankingRobust/` vendors the reference implementation from
**"Dropping Just a Handful of Preferences Can Change Top Large Language Model
Rankings"** (Huang et al., ICML 2025 Workshop on Models of Human Feedback for AI
Alignment), used as the comparison baseline in `tests/test_isrankingrobust_parity.py`.
It is redistributed under its own MIT license (© 2025 Jenny Huang); see
`IsRankingRobust/LICENSE`. Its upstream `unit_tests/` contain interactive
`breakpoint()` calls and are excluded from collection via `norecursedirs`.

## License

MIT — see [LICENSE](LICENSE).
