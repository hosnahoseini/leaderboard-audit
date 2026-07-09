#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLEAN_BT_RANK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$ROOT/final_result/logs"
PYRUN="$ROOT/scripts/slurm_run_python_cleanbt.sh"
OUT_ROOT="$ROOT/final_result/ci_reduction_influence_variants"

DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  DATASETS=(
    tennis_top10_atp
    mt_bench_human
    webdev_arena
    vision_arena
    llm_judge_arena
    arena55k
    nba_elo_top50
  )
fi

mkdir -p "$LOG_DIR" "$OUT_ROOT"

submit() {
  local dataset="$1"
  local job_name="ci_var_${dataset}"
  local out_dir="$OUT_ROOT/$dataset"
  mkdir -p "$out_dir"
  sbatch --parsable \
    -J "$job_name" \
    -D "$ROOT" \
    -p JIMMY \
    -c 4 \
    --mem 32G \
    -t 12:00:00 \
    -o "$LOG_DIR/${job_name}_%j.log" \
    --wrap "$PYRUN $ROOT/scripts/run_ci_reduction_batch.py --datasets $dataset --dataset-order provided --compare-influence-variants --output-dir $out_dir"
}

for dataset in "${DATASETS[@]}"; do
  echo "$dataset $(submit "$dataset")"
done
