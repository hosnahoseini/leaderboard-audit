#!/bin/bash
#SBATCH --job-name=topk_rig_ckpt
#SBATCH --partition=ALL
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/topk_rig_ckpt_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/topk_rig_ckpt_%j.log

set -euo pipefail

export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_topk_rig_ckpt_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_topk_rig_ckpt_${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

source ~/venvs/cleanbt/bin/activate
cd /u501/hoyarhos/clean_bt_rank

python scripts/run_topk_vs_rigging_comparison.py \
  --datasets arena55k llm_judge_arena mt_bench_human nba_elo_top50 tennis_top10_atp vision_arena webdev_arena \
  --n-targets-per-direction 3 \
  --tasks-per-dataset 0 \
  --trials 5 \
  --budget 120 \
  --rigging-mode omni_bt_diff \
  --seed 0 \
  --resume-existing \
  --skip-failed-datasets \
  --output-dir /u501/hoyarhos/clean_bt_rank/notebooks/artifacts/topk_vs_rigging_all_datasets_3k_6tasks_5trials_checkpointed
