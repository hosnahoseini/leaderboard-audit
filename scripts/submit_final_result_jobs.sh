#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLEAN_BT_RANK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$ROOT/final_result/logs"
PYRUN="$ROOT/scripts/slurm_run_python_cleanbt.sh"
DATASETS=(
  tennis_top10_atp
  mt_bench_human
  webdev_arena
  vision_arena
  llm_judge_arena
  arena55k
  nba_elo_top50
)

mkdir -p "$LOG_DIR"

dataset_args="${DATASETS[*]}"

submit() {
  local name="$1"
  local time_limit="$2"
  shift 2
  sbatch --parsable -J "$name" -D "$ROOT" -c 1 -t "$time_limit" -o "$LOG_DIR/${name}_%j.log" "$@"
}

echo "arena_ablation_all $(submit final_ablation_all 24:00:00 "$ROOT/scripts/run_final_result_arena_ablation_all.sh")"
echo "arena_heatmap_all $(submit final_heatmap_all 24:00:00 "$ROOT/scripts/run_final_result_arena_heatmap_all.sh")"
echo "top1 $(submit final_top1 24:00:00 --wrap "$PYRUN $ROOT/scripts/run_arena_robustness_actions_needed.py --datasets $dataset_args --dataset-order provided --output-dir $ROOT/final_result/top1_robustness_actions_needed")"
echo "ci_reduction $(submit final_ci_reduction 24:00:00 --wrap "$PYRUN $ROOT/scripts/run_ci_reduction_batch.py --datasets $dataset_args --dataset-order provided --output-dir $ROOT/final_result/ci_reduction_batch")"
echo "ci_vs_nonci $(submit final_ci_vs_nonci 24:00:00 --wrap "$PYRUN $ROOT/scripts/run_ci_vs_nonci_robustness_actions_needed.py --datasets $dataset_args --dataset-order provided --output-dir $ROOT/final_result/ci_vs_nonci_actions_needed")"
echo "fair_ci $(submit final_fair_ci 24:00:00 --wrap "$PYRUN $ROOT/scripts/run_fair_ci_reduction_comparison.py --datasets $dataset_args --dataset-order provided --output-dir $ROOT/final_result/fair_ci_reduction_comparison")"
echo "player_kendall $(submit final_player_kendall 12:00:00 --wrap "$PYRUN $ROOT/scripts/run_player_kendall_influence_plots.py --datasets $dataset_args --output-dir $ROOT/final_result/player_kendall_influence_plots")"
echo "tau_ci $(submit final_tau_ci 24:00:00 --wrap "$PYRUN $ROOT/scripts/run_tau_ci_curve_analysis.py --datasets $dataset_args --output-dir $ROOT/final_result/tau_ci_curve_analysis")"
echo "topk_vs_rigging $(submit final_topk_vs_rigging 48:00:00 --wrap "$PYRUN $ROOT/scripts/run_topk_vs_rigging_comparison.py --datasets $dataset_args --dataset-order provided --output-dir $ROOT/final_result/topk_vs_rigging_comparison")"
