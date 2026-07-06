# Leaderboard Audit

Tools for analyzing leaderboard stability with Bradley-Terry models, confidence
intervals, and first-order influence estimates.

The project supports pairwise preference data, Arena-style datasets, and
experiments that estimate how many data edits are needed to change ranking
outcomes.

## Features

- Fit Bradley-Terry models from pairwise comparisons.
- Compute ratings, confidence intervals, and ranking summaries.
- Estimate match influence for `drop`, `add`, and `flip` actions.
- Run iterative robustness and manipulation experiments.
- Serve an interactive static companion site with GitHub Pages.

## Repository Layout

- `src/clean_bt_rank/`: Python package.
- `scripts/`: experiment and batch-run entrypoints.
- `tests/`: regression and workflow tests.
- `docs/`: static website artifact for GitHub Pages.
- `.github/workflows/pages.yml`: GitHub Pages deployment workflow.

## Installation

```bash
python -m pip install -e .
```

## Tests

```bash
pytest
```

## Website

The static site is published from `docs/index.html`.

After pushing to GitHub, enable Pages with:

```text
Settings -> Pages -> Source: GitHub Actions
```

The workflow deploys automatically on pushes to `master` or `main`.

## License

Add the project license before public release.
