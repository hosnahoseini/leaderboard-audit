#!/bin/bash
#SBATCH --job-name=cleanbt_ci_vs_nonci
#SBATCH --partition=JIMMY
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/ci_vs_nonci_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/ci_vs_nonci_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_ci_vs_nonci_$SLURM_JOB_ID
mkdir -p "$MPLCONFIGDIR"
source ~/venvs/cleanbt/bin/activate
cd ~/clean_bt_rank
python scripts/run_ci_vs_nonci_robustness_actions_needed.py \
  --dataset-order size_asc \
  --max-action-fraction 0.10 \
  --output-dir /u501/hoyarhos/clean_bt_rank/notebooks/artifacts/ci_vs_nonci_actions_needed_20260424
