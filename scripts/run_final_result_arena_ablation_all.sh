#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLEAN_BT_RANK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYRUN="$ROOT/scripts/slurm_run_python_cleanbt.sh"
OUT_ROOT="$ROOT/final_result/arena_ablation_reset"

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

  "$PYRUN" "$ROOT/scripts/run_arena_ablation_reset.py" \
    --dataset "$dataset" \
    --output-dir "$dataset_out"

  "$PYRUN" "$ROOT/scripts/plot_arena_ablation_actions_vs_k.py" \
    --input "$dataset_out/${dataset}_topk_actions_needed_summary.csv" \
    --output-dir "$dataset_out" \
    --stem "${dataset}_topk_actions_needed_curve_neurips"
done
