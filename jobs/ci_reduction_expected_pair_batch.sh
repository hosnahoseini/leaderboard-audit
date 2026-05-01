#!/bin/bash
#SBATCH --job-name=cleanbt_ci_exp_pair
#SBATCH --partition=JIMMY
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/ci_reduction_expected_pair_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/ci_reduction_expected_pair_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_ci_reduction_expected_pair_$SLURM_JOB_ID
mkdir -p "$MPLCONFIGDIR"
source ~/venvs/cleanbt/bin/activate
cd ~/clean_bt_rank
python scripts/run_ci_reduction_batch.py \
  --dataset-order size_asc \
  --candidate-mode all_pairs \
  --policies expected_pair arena_active_pair \
  --random-policy random_pair \
  --primary-policy expected_pair \
  "$@"
