#!/bin/bash
#SBATCH --job-name=topk_rig_ten
#SBATCH --partition=ALL
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/topk_rig_ten_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/topk_rig_ten_%j.log

set -euo pipefail

export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_topk_rig_tennis_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_topk_rig_tennis_${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

source ~/venvs/cleanbt/bin/activate
cd /u501/hoyarhos/clean_bt_rank

python scripts/run_topk_vs_rigging_comparison.py \
  --datasets tennis_top10_atp \
  --n-targets-per-direction 3 \
  --tasks-per-dataset 0 \
  --trials 5 \
  --budget 120 \
  --rigging-mode omni_bt_diff \
  --seed 0 \
  --output-dir /u501/hoyarhos/clean_bt_rank/notebooks/artifacts/topk_vs_rigging_tennis_only_3k_6tasks_5trials
