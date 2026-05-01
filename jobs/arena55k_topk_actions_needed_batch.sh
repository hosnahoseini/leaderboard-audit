#!/bin/bash
#SBATCH --job-name=cb_topk_act55k
#SBATCH --partition=JIMMY
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/arena55k_topk_actions_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/arena55k_topk_actions_%j.log

set -euo pipefail

export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_arena55k_topk_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg_arena55k_topk_${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

source ~/venvs/cleanbt/bin/activate
cd /u501/hoyarhos/clean_bt_rank

python scripts/run_arena55k_topk_robustness_actions_needed.py "$@"
python scripts/plot_arena55k_topk_actions_needed.py
