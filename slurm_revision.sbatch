#!/usr/bin/env bash
#SBATCH --job-name=ids-revision
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --array=0-5
# NOTE: no --gres=gpu. This is a CPU workload; see run_on_cluster.sh for why
# the GPU is deliberately not requested. If your cluster's partition policy
# forces a GPU allocation, add the -account/-partition flags your site
# requires, but leave CUDA_VISIBLE_DEVICES empty as set below.
#
# Submit with:   sbatch slurm_revision.sbatch
# One array task per (dataset, experiment) pair:
#   0 nsl_kdd ablation        3 nsl_kdd multisplit
#   1 cse_cic_ids2018 ablation 4 cse_cic_ids2018 multisplit
#   2 unsw_nb15 ablation       5 unsw_nb15 multisplit
#
# Each task is idempotent: re-submitting after a timeout resumes from the
# completed (config, seed) cells already written to results/.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

# shellcheck disable=SC1091
source .venv/bin/activate

DATASETS=(nsl_kdd cse_cic_ids2018 unsw_nb15 nsl_kdd cse_cic_ids2018 unsw_nb15)
EXPERIMENTS=(ablation ablation ablation multisplit multisplit multisplit)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
EXP="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "task=$SLURM_ARRAY_TASK_ID dataset=$DS experiment=$EXP host=$(hostname) threads=$OMP_NUM_THREADS"
bash run_on_cluster.sh "$EXP" "$DS"
