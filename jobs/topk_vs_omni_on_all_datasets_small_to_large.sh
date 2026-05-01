#!/bin/bash
#SBATCH --job-name=topk_omni_on_all
#SBATCH --partition=ALL
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/topk_omni_on_all_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/topk_omni_on_all_%j.log

set -euo pipefail

export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_topk_omni_on_all_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_topk_omni_on_all_${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

source ~/venvs/cleanbt/bin/activate
cd /u501/hoyarhos/clean_bt_rank

python scripts/run_topk_vs_rigging_comparison.py \
  --datasets tennis_top10_atp mt_bench_human webdev_arena vision_arena llm_judge_arena arena55k nba_elo_top50 \
  --n-targets-per-direction 1 \
  --tasks-per-dataset 0 \
  --trials 5 \
  --budget 120 \
  --rigging-mode omni_on \
  --seed 0 \
  --resume-existing \
  --skip-failed-datasets \
  --output-dir /u501/hoyarhos/clean_bt_rank/notebooks/artifacts/topk_vs_omni_on_all_datasets_small_to_large_3k_2tasks_5trials
