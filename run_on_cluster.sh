#!/usr/bin/env bash
# Revision experiments for the IEEE Access resubmission, cluster edition.
#
# IMPORTANT: this is a CPU workload. Do not pass any CUDA/GPU flags. XGBoost's
# GPU path uses different histogram algorithms and would not reproduce the
# submitted CPU results, and LightGBM's GPU build does not support the
# parameters being ablated. A GPU node is useful here only for its cores/RAM.
#
# THREAD COUNT CHANGES LIGHTGBM RESULTS. Measured on NSL-KDD, LightGBM 4.7.0,
# config lgb_no_subsampling: 0.6962 with 32 threads vs 0.6887 with 1 thread,
# bit-identical within each thread count. OMP_NUM_THREADS is therefore pinned
# below and must be reported in the paper alongside the CPU model.
#
# Usage:
#   bash run_on_cluster.sh setup                 # env + deps, once
#   bash run_on_cluster.sh data-nslkdd           # download NSL-KDD (~22 MB)
#   bash run_on_cluster.sh data-cic              # download + prepare CSE-CIC-IDS2018 (~6.4 GB)
#   bash run_on_cluster.sh reanalyses            # needs the ORIGINAL matrix CSVs in results/
#   bash run_on_cluster.sh ablation  <dataset>   # 16 configs x 40 seeds
#   bash run_on_cluster.sh multisplit <dataset>  # 6 partitions x 40 seeds x 2 archs
#   bash run_on_cluster.sh validate              # bootstrap decomposition validation
#   bash run_on_cluster.sh repro     <dataset>   # LightGBM thread/run-to-run diagnostic
#   bash run_on_cluster.sh all-datasets          # ablation + multisplit for all three, sequentially
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""          # make it impossible to silently use the GPU

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
LOGDIR="$REPO_DIR/logs"; mkdir -p "$LOGDIR"
PY="${PY:-python}"
stamp() { date +%Y%m%dT%H%M%S; }

banner() {
  echo "=============================================================="
  echo "$1"
  echo "  host=$(hostname)  OMP_NUM_THREADS=$OMP_NUM_THREADS  $(date)"
  echo "=============================================================="
}

case "${1:-help}" in

setup)
  banner "Environment setup"
  $PY -m venv .venv 2>/dev/null || true
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  # Versions pinned to the machine that produced the submitted results, so the
  # cluster reproduces the paper rather than a newer library's behaviour.
  pip install "numpy>=2,<3" "pandas>=2.2" "scikit-learn==1.8.0" "scipy>=1.13" \
              "statsmodels>=0.14" "xgboost==3.3.0" "lightgbm==4.6.0" psutil
  $PY - <<'PYEOF'
import numpy, pandas, sklearn, scipy, statsmodels, xgboost, lightgbm, os
print("numpy", numpy.__version__, "| pandas", pandas.__version__,
      "| sklearn", sklearn.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__,
      "| xgboost", xgboost.__version__, "| lightgbm", lightgbm.__version__)
print("cpus visible:", os.cpu_count(), "| OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PYEOF
  echo "Setup done. Activate later with: source $REPO_DIR/.venv/bin/activate"
  ;;

data-nslkdd)
  banner "NSL-KDD download"
  mkdir -p data
  curl -fL --retry 3 -o "data/KDDTrain+.txt" \
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
  curl -fL --retry 3 -o "data/KDDTest+.txt" \
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
  echo "Row counts (must be 125973 and 22544):"
  wc -l "data/KDDTrain+.txt" "data/KDDTest+.txt"
  ;;

data-cic)
  banner "CSE-CIC-IDS2018 download + prepare (~6.4 GB, slow)"
  command -v aws >/dev/null || pip install awscli
  mkdir -p data/cicids2018_raw
  aws s3 cp --no-sign-request --region us-east-1 \
    "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
    data/cicids2018_raw/ --recursive
  $PY prepare_cicids2018.py 2>&1 | tee "$LOGDIR/prepare_cic_$(stamp).log"
  echo "Confirm the printed row counts match config.DATASETS['cse_cic_ids2018'] (200000 / 39999)."
  ;;

data-unsw)
  banner "UNSW-NB15"
  echo "Place these two official files in data/ (no public mirror is used here):"
  echo "  data/UNSW_NB15_training-set.csv   (175341 rows)"
  echo "  data/UNSW_NB15_testing-set.csv    ( 82332 rows)"
  echo "Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
  ls -la data/UNSW_NB15_*.csv 2>/dev/null || echo "  (not present yet)"
  ;;

reanalyses)
  banner "Pure re-analyses of the ORIGINAL 720-run outputs"
  echo "REQUIRES in results/: <dataset>_matrix_summary.csv and <dataset>_matrix_per_instance.csv"
  echo "from the submitted runs. Do NOT regenerate these; copy them from the machine that"
  echo "produced the paper, or the supplementary tables will not match the published ones."
  for DS in nsl_kdd cse_cic_ids2018 unsw_nb15; do
    [ -f "results/${DS}_matrix_summary.csv" ] || { echo "  skip $DS (no summary CSV)"; continue; }
    $PY analyze_seed_adequacy_random_subsets.py --dataset "$DS" \
        2>&1 | tee "$LOGDIR/seedadequacy_${DS}_$(stamp).log"
    [ -f "results/${DS}_matrix_per_instance.csv" ] && \
      $PY analyze_overlap_split.py --dataset "$DS" \
        2>&1 | tee "$LOGDIR/overlap_${DS}_$(stamp).log"
  done
  $PY export_levene_bh_tables.py 2>&1 | tee "$LOGDIR/levene_$(stamp).log"
  ;;

ablation)
  DS="${2:?usage: run_on_cluster.sh ablation <dataset>}"
  banner "Ablation: $DS (16 configs x 40 seeds)"
  $PY run_ablation.py --dataset "$DS" --configs all --n-seeds 40 \
      2>&1 | tee "$LOGDIR/ablation_${DS}_$(stamp).log"
  $PY run_ablation.py --dataset "$DS" --analyze --n-bootstrap 1000 \
      2>&1 | tee "$LOGDIR/ablation_${DS}_analyze_$(stamp).log"
  ;;

multisplit)
  DS="${2:?usage: run_on_cluster.sh multisplit <dataset>}"
  banner "Multi-split: $DS (official + 5 re-partitions, 40 seeds, lightgbm+xgboost)"
  $PY run_multisplit.py --dataset "$DS" --n-partitions 5 --n-seeds 40 \
      2>&1 | tee "$LOGDIR/multisplit_${DS}_$(stamp).log"
  $PY run_multisplit.py --dataset "$DS" --analyze --n-bootstrap 1000 \
      2>&1 | tee "$LOGDIR/multisplit_${DS}_analyze_$(stamp).log"
  ;;

validate)
  banner "Bootstrap decomposition validation (no data needed)"
  $PY validate_bootstrap_decomposition.py --all --n-simulations 60 --n-bootstrap 300 \
      2>&1 | tee "$LOGDIR/bootstrap_validation_$(stamp).log"
  ;;

repro)
  DS="${2:-nsl_kdd}"
  banner "LightGBM run-to-run / thread-count diagnostic: $DS"
  $PY check_lightgbm_reproducibility.py --dataset "$DS" \
      --configs baseline,no_subsampling,l2_reg1,baseline_deterministic_flag \
      --threads -1,1 --repeats 6 --seeds 0,1,2,3,4,5,6,7 \
      2>&1 | tee "$LOGDIR/repro_${DS}_$(stamp).log"
  ;;

all-datasets)
  for DS in nsl_kdd cse_cic_ids2018 unsw_nb15; do
    bash "$0" ablation "$DS"
    bash "$0" multisplit "$DS"
  done
  bash "$0" repro nsl_kdd
  ;;

*)
  sed -n '1,30p' "$0"
  ;;
esac
