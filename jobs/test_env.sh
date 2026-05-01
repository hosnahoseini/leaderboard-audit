#!/bin/bash
#SBATCH --job-name=cleanbt_test
#SBATCH --partition=JIMMY
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/u501/hoyarhos/clean_bt_rank/logs/test_%j.log
#SBATCH --error=/u501/hoyarhos/clean_bt_rank/logs/test_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR=/tmp/mpl_test_$SLURM_JOB_ID
mkdir -p "$MPLCONFIGDIR"
source ~/venvs/cleanbt/bin/activate
cd ~/clean_bt_rank
python - <<"PY"
import numpy, scipy, sklearn, datasets
print("imports-ok")
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("datasets", datasets.__version__)
PY
