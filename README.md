# IDS Seed Variance Study

Tests whether machine learning-based intrusion detection models that show
stable aggregate accuracy across random seeds actually show unstable
per-attack-category detection underneath, and whether this differs across
architectures (deep learning vs. ensemble methods) and datasets.

## Current scope

- **Datasets implemented:** NSL-KDD (4-category taxonomy: DoS, Probe,
  R2L, U2R) and CSE-CIC-IDS2018 (7-category taxonomy: BruteForce, DoS,
  Web_Attacks, Infiltration, Botnet, DDoS, matching the ACAFS paper's
  taxonomy for direct comparison).
- **Architectures implemented:** DNN (PyTorch), Random Forest
  (scikit-learn), XGBoost. All three enabled by default in
  `config.ARCHITECTURES`.
- **Seed counts:** 10 for the pilot, 40 for the full run
  (`config.PILOT_SEEDS` / `config.FULL_SEEDS`).

## What was specifically caught and fixed during development

Documenting this here deliberately, since it's evidence the pipeline was
tested against failure modes, not just the happy path.

1. **XGBoost's `random_state` initially had no effect on training.**
   With default `subsample=1.0` and `colsample_bytree=1.0`, every boosting
   round deterministically used all rows and all features, so the seed
   had nothing to actually randomize. Caught by an explicit
   same-seed-vs-different-seed determinism test. Fixed via
   `XGB_SUBSAMPLE=0.8` and `XGB_COLSAMPLE_BYTREE=0.8` in `config.py`.

2. **Duplicate seed rows from an interrupted-and-restarted run.** Fixed
   by making `run_pilot.py` and `run_matrix.py` idempotent (they check
   which seeds are already complete and only run what's missing), plus a
   duplicate-run check in `stats_analysis.py` as a second line of defense.

3. **The bootstrap variance decomposition was validated against
   synthetic data with a known ground-truth variance.** Recovered
   variance was within 1.6% of the true value when a genuine effect
   was simulated, and correctly near-zero in a simulated null case.

## Setup

```bash
pip install -r requirements.txt
# if the environment is externally managed:
pip install -r requirements.txt --break-system-packages
```

### NSL-KDD

Download `KDDTrain+.txt` and `KDDTest+.txt` from the official NSL-KDD
source and place them in `data/`. Row counts are validated automatically
on load (125973 train / 22544 test).

### CSE-CIC-IDS2018

This dataset does not ship as a single clean file, it's distributed as
multiple daily CSVs, hosted on AWS S3, and requires a one-time
preparation step before use.

1. Install the AWS CLI if not already present: `pip install awscli`
2. Download the pre-processed traffic data (no AWS account needed):

```bash
aws s3 cp --no-sign-request --region us-east-1 \
  s3://cse-cic-ids2018/Processed%20Traffic%20Data%20for%20ML%20Algorithms/ \
  data/cicids2018_raw/ --recursive
```

This is a multi-gigabyte download; expect it to take a while.

3. Before running the prep script, verify the actual downloaded column
   names and unique label values match what's assumed in
   `config.CIC_IDS2018_ATTACK_MAP`. This dataset is documented to have
   inconsistent label spacing/capitalization across files, don't assume
   the map is complete until confirmed against your real files.

4. Run the one-time preparation script:

```bash
python prepare_cicids2018.py
```

This cleans column names, drops identifier/leakage-risk columns
(Timestamp, any IP columns), handles the dataset's known Infinity/NaN
issue in rate columns, maps raw labels to the 7-category taxonomy, and
writes a stratified subsample (natural class proportions preserved, no
SMOTE/balancing, a deliberate choice disclosed in the Methods section)
to `data/CSECICIDS2018_train.csv` and `data/CSECICIDS2018_test.csv`.

**If this script raises an error listing unmapped label values**, add
them to `config.CIC_IDS2018_ATTACK_MAP` and rerun, it collects all
unmapped labels at once rather than failing on the first one.

**After it completes successfully**, update
`config.DATASETS["cse_cic_ids2018"]["expected_train_rows"]` and
`"expected_test_rows"` with the exact numbers it prints, so `data.py`'s
row-count validation checks against a real, confirmed number instead of
`None`.

## Running the pilot (Stage 1: 10 seeds, DNN only, NSL-KDD only)

```bash
python run_pilot.py
python stats_analysis.py
```

Resumable: rerunning after an interruption skips completed seeds.

## Running the full matrix (Stage 2: after the pilot is reviewed)

```bash
python run_matrix.py --seeds pilot   # sanity check with 10 seeds first, across BOTH datasets
python run_matrix.py --seeds full    # 40 seeds, only after reviewing the above
```

`run_matrix.py` automatically loops over every dataset registered in
`config.DATASETS` and every architecture in `config.ARCHITECTURES`, no
code changes needed when a new dataset is added, only a new registry
entry and loader.

Analyze a specific combination:

```python
from stats_analysis import run_full_analysis, compare_architectures, load_results, load_per_instance

run_full_analysis(
    "results/cse_cic_ids2018_matrix_summary.csv",
    "results/cse_cic_ids2018_matrix_per_instance.csv",
    dataset="cse_cic_ids2018", architecture="random_forest",
)

summary_df = load_results("results/nsl_kdd_matrix_summary.csv")
per_instance_df = load_per_instance("results/nsl_kdd_matrix_per_instance.csv")
compare_architectures(summary_df, per_instance_df, "r2l", "nsl_kdd",
                       ["dnn", "random_forest", "xgboost"])
```

## Adding a new dataset

1. Add an entry to `config.DATASETS` following the `nsl_kdd` or
   `cse_cic_ids2018` pattern (paths, attack map, categories, expected
   row counts).
2. Add a loader function in `data.py`, and register it in `_LOADERS`,
   including the same row-count validation, category-completeness
   check, NaN check, and train/test overlap audit used by the existing
   loaders, don't skip these for a new dataset just because they passed
   for the others.
3. If the raw data needs cleaning/subsampling before use (like
   CSE-CIC-IDS2018), write a one-time preparation script following
   `prepare_cicids2018.py`'s pattern.

## Files

- `config.py` — all pre-registered parameters, dataset/architecture
  registries, results schema version.
- `data.py` — dataset loading with row-count validation,
  category-completeness checking, NaN/Inf detection, train/test overlap
  audit, and a real dispatch across both implemented datasets.
- `prepare_cicids2018.py` — one-time raw-to-clean preparation for
  CSE-CIC-IDS2018 (multi-file loading, column cleanup, Infinity/NaN
  handling, label mapping with all-unmapped-labels-at-once error
  reporting, stratified natural-proportion subsampling).
- `model.py` — all three architectures behind a consistent
  `fit`/`predict` interface, plus full-determinism seeding. Includes the
  documented XGBoost subsample fix.
- `train.py` — single `(architecture, seed)` training and evaluation,
  crash-safe incremental CSV writing, schema-version stamping, loud
  failure behavior with full context on error.
- `run_pilot.py` — Stage 1 runner, NSL-KDD/DNN only by design, resumable.
- `run_matrix.py` — Stage 2 general runner, loops over every enabled
  dataset/architecture/seed combination automatically, resumable.
- `stats_analysis.py` — real bootstrap variance decomposition (validated
  against synthetic ground truth), Benjamini-Hochberg-corrected Levene's
  tests, effect sizes, Clopper-Pearson intervals for small classes,
  seed-adequacy analysis, cross-architecture comparison, duplicate-run
  and schema-consistency checks on load.
- `environment_capture.py` — records exact software versions, hardware,
  git commit hash (and working-tree-dirty flag), active hyperparameters,
  and data file checksums for every run.

## Before running the full-scale experiments for the paper

1. Commit all code changes (`git status` clean) so the git commit hash in
   each environment log actually reflects the code that produced the
   results.
2. If running the DNN on a GPU node, check `gpu_info.cuda_available` in
   the environment log, don't assume the GPU is being used.
3. Re-run the pilot's analysis and confirm the same pattern holds before
   committing to the full 40-seed matrix across all three architectures
   and both datasets.
4. Confirm `config.CIC_IDS2018_ATTACK_MAP` covers every unique label
   actually present in the real CSE-CIC-IDS2018 files, `prepare_cicids2018.py`
   will refuse to proceed if it doesn't, but double-check the mapping
   makes semantic sense, not just that it's complete.

## Revision materials (IEEE Access resubmission, September 2026)

Everything added for the revised manuscript is documented in
[REVISION_EXPERIMENTS.md](REVISION_EXPERIMENTS.md), whose final section maps every
table and figure of the revised paper to the command that produces it and the file
it writes. Entry points: `run_on_cluster.sh` (runs every revision experiment with a
pinned thread count), `run_ablation.py`, `run_multisplit.py`,
`run_second_subsample.py`, `rerun_logistic_regression.py`,
`check_lightgbm_reproducibility.py`, `analyze_overlap_split.py`,
`analyze_seed_adequacy_random_subsets.py`, `export_levene_bh_tables.py`, and
`figures/build_figure1.py` / `figures/build_figure3.py`. The independent second
CSE-CIC-IDS2018 subsample is in `data/` in gzipped form, with its SHA-256 checksums
recorded in `config.py`; the result files behind every new table are in `results/`.
