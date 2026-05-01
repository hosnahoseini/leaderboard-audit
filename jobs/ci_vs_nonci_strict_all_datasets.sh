#!/bin/bash
#SBATCH --job-name=cb_ci_vs_nonci_strict
#SBATCH --partition=JIMMY
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/ci_vs_nonci_strict_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/ci_vs_nonci_strict_%j.log
set -euo pipefail

export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_ci_vs_nonci_strict_$SLURM_JOB_ID
mkdir -p "$MPLCONFIGDIR"

source ~/venvs/cleanbt/bin/activate
cd ~/clean_bt_rank

python scripts/run_ci_vs_nonci_robustness_actions_needed.py \
  --dataset-order size_asc \
  --ci-objective strict \
  --k-selection-mode meaningful \
  --output-dir /u501/hoyarhos/clean_bt_rank/notebooks/artifacts/ci_vs_nonci_strict_all_datasets_20260501
