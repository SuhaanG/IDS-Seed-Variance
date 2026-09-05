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
#   bash run_on_cluster.sh prep-cic              # prepare only, raw files already downloaded
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
unset CUDA_VISIBLE_DEVICES  # was: export CUDA_VISIBLE_DEVICES="" -- xgboost 3.3.0 CUDA build raises cudaErrorNoDevice on fit when all GPUs are hidden; estimators default to device=cpu (verified), GPU stays unused

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
LOGDIR="$REPO_DIR/logs"; mkdir -p "$LOGDIR"

# Activate a virtualenv ourselves rather than trusting the caller's shell.
# `nohup`, `sbatch` and `bash -c` subprocesses frequently lose an interactively
# activated environment, which shows up as "ModuleNotFoundError: numpy".
if [ -z "${VIRTUAL_ENV:-}" ] && [ "${1:-}" != "setup" ]; then
  for CAND in "$REPO_DIR/venv" "$REPO_DIR/.venv"; do
    if [ -f "$CAND/bin/activate" ]; then
      # shellcheck disable=SC1091
      source "$CAND/bin/activate"
      echo "activated virtualenv: $CAND"
      break
    fi
  done
fi
PY="${PY:-python}"
stamp() { date +%Y%m%dT%H%M%S; }

preflight() {
  # Fail loudly and immediately with a useful message, rather than after a
  # banner and a bare traceback.
  "$PY" - <<'PYEOF' || { echo "PREFLIGHT FAILED: see above. Run 'bash run_on_cluster.sh doctor'." >&2; exit 1; }
import sys, importlib, os
missing = []
for m in ["numpy", "pandas", "sklearn", "scipy", "statsmodels", "xgboost", "lightgbm"]:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
print(f"python {sys.version.split()[0]} at {sys.executable}")
print("threads:", os.environ.get("OMP_NUM_THREADS"), "| cuda hidden:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
if missing:
    print("MISSING MODULES:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
import lightgbm, xgboost, sklearn
print(f"lightgbm {lightgbm.__version__} | xgboost {xgboost.__version__} | sklearn {sklearn.__version__}")
PYEOF
}

banner() {
  echo "=============================================================="
  echo "$1"
  echo "  host=$(hostname)  OMP_NUM_THREADS=$OMP_NUM_THREADS  $(date)"
  echo "=============================================================="
}

acquire_lock() {
  # Refuse to run two instances of the same job concurrently. The runners are
  # idempotent across sequential restarts, but two SIMULTANEOUS processes each
  # read the completed-cell list at startup, so both would rerun the same
  # seeds, append duplicate summary rows, and race on the per-seed .npy files.
  local name="$1"
  local lockfile="$LOGDIR/.lock_${name}"
  exec 9>"$lockfile"
  if ! flock -n 9; then
    echo "ERROR: another '$name' run already holds $lockfile." >&2
    echo "       Refusing to start a second one; it would corrupt results/." >&2
    echo "       Check with: ps -ef | grep -E 'run_ablation|run_multisplit'" >&2
    exit 1
  fi
  echo "lock acquired: $lockfile (pid $$)"
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
  acquire_lock "data_cic"
  command -v aws >/dev/null || pip install awscli
  mkdir -p data/cicids2018_raw
  aws s3 cp --no-sign-request --region us-east-1 \
    "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
    data/cicids2018_raw/ --recursive
  $PY prepare_cicids2018.py 2>&1 | tee "$LOGDIR/prepare_cic_$(stamp).log"
  echo "Confirm the printed row counts match config.DATASETS['cse_cic_ids2018'] (200000 / 39999)."
  ;;

prep-cic)
  banner "CSE-CIC-IDS2018 prepare only (raw files already downloaded)"
  acquire_lock "data_cic"
  preflight
  n_raw=$(ls data/cicids2018_raw/*.csv 2>/dev/null | wc -l)
  echo "raw daily files present: $n_raw (expected 10)"
  [ "$n_raw" -eq 10 ] || { echo "ERROR: expected 10 raw CSVs, found $n_raw. Run 'data-cic' to download." >&2; exit 1; }
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
  acquire_lock "reanalyses"
  preflight
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

doctor)
  banner "Environment diagnostic"
  echo "which python: $(command -v "$PY" || echo NOT FOUND)"
  echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-<none>}"
  echo "nproc: $(nproc)"
  preflight
  echo "data/:"; ls -la data/ 2>/dev/null | tail -n +2
  echo "results/ (row counts of any matrix CSVs):"
  for f in results/*_matrix_summary.csv results/*_matrix_per_instance.csv; do
    [ -f "$f" ] && echo "  $(wc -l < "$f") lines  $f"
  done
  ;;

ablation)
  DS="${2:?usage: run_on_cluster.sh ablation <dataset>}"
  banner "Ablation: $DS (16 configs x 40 seeds)"
  acquire_lock "ablation_$DS"
  preflight
  $PY run_ablation.py --dataset "$DS" --configs all --n-seeds 40 \
      2>&1 | tee "$LOGDIR/ablation_${DS}_$(stamp).log"
  $PY run_ablation.py --dataset "$DS" --analyze --n-bootstrap 1000 \
      2>&1 | tee "$LOGDIR/ablation_${DS}_analyze_$(stamp).log"
  ;;

multisplit)
  DS="${2:?usage: run_on_cluster.sh multisplit <dataset>}"
  banner "Multi-split: $DS (official + 5 re-partitions, 40 seeds, lightgbm+xgboost)"
  acquire_lock "multisplit_$DS"
  preflight
  $PY run_multisplit.py --dataset "$DS" --n-partitions 5 --n-seeds 40 \
      2>&1 | tee "$LOGDIR/multisplit_${DS}_$(stamp).log"
  $PY run_multisplit.py --dataset "$DS" --analyze --n-bootstrap 1000 \
      2>&1 | tee "$LOGDIR/multisplit_${DS}_analyze_$(stamp).log"
  ;;

validate)
  banner "Bootstrap decomposition validation (no data needed)"
  preflight
  $PY validate_bootstrap_decomposition.py --all --n-simulations 60 --n-bootstrap 300 \
      2>&1 | tee "$LOGDIR/bootstrap_validation_$(stamp).log"
  ;;

repro)
  DS="${2:-nsl_kdd}"
  banner "LightGBM run-to-run / thread-count diagnostic: $DS"
  acquire_lock "repro_$DS"
  preflight
  $PY check_lightgbm_reproducibility.py --dataset "$DS" \
      --configs baseline,no_subsampling,l2_reg1,baseline_deterministic_flag \
      --threads=-1,1 --repeats 6 --seeds 0,1,2,3,4,5,6,7 \
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
