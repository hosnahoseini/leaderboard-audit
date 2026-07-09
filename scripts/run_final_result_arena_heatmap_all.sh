#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLEAN_BT_RANK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYRUN="$ROOT/scripts/slurm_run_python_cleanbt.sh"
OUT_ROOT="$ROOT/final_result/arena_influence_heatmap"

datasets=(
  tennis_top10_atp
  mt_bench_human
  webdev_arena
  vision_arena
  llm_judge_arena
  arena55k
  nba_elo_top50
)

mkdir -p "$OUT_ROOT"

for dataset in "${datasets[@]}"; do
  dataset_out="$OUT_ROOT/$dataset"
  mkdir -p "$dataset_out"
  "$PYRUN" "$ROOT/scripts/run_arena_influence_heatmap.py" \
    --dataset "$dataset" \
    --output-dir "$dataset_out"
done
