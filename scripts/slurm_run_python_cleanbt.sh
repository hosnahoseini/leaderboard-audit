#!/usr/bin/env bash
# Thin wrapper used by the SLURM submit scripts to run a Python entrypoint with
# the thread/cache settings the batch jobs expect.
#
#   CLEAN_BT_RANK_ROOT    repository root      (default: parent of this script)
#   CLEAN_BT_RANK_PYTHON  interpreter to use   (default: whatever `python` resolves to)
#
# HF_*_OFFLINE are set because compute nodes have no outbound network: warm the
# Hugging Face cache from a login node first, then submit.
set -euo pipefail

ROOT="${CLEAN_BT_RANK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cleanbt_run}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec "${CLEAN_BT_RANK_PYTHON:-python}" "$@"
