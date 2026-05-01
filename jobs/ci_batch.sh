#!/bin/bash
#SBATCH --job-name=cleanbt_ci
#SBATCH --partition=JIMMY
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/ci_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/ci_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_ci_$SLURM_JOB_ID
mkdir -p "$MPLCONFIGDIR"
source ~/venvs/cleanbt/bin/activate
cd ~/clean_bt_rank
python scripts/run_ci_aware_robustness_actions_needed.py "$@"
