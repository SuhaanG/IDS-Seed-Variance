# IEEE Access revision: experiments runbook

Decision on Access-2026-36031 (2026-09-02): reject with one resubmission. Five
reviewers plus a structured Review_Report.pdf. This file maps every concern
that needs NEW COMPUTATION or NEW ANALYSIS to a script, a command, the machine
it should run on, and where the output goes in the paper.

Nothing here modifies the code that produced the submitted 720 runs
(`config.py`, `data.py`, `model.py`, `train.py`, `run_matrix.py`,
`stats_analysis.py` are untouched). All new scripts import from them and
write to new files under `results/`. Every SNR reported by the new scripts is
computed by the unmodified `stats_analysis.true_variance_decomposition`.

Smoke test (no real data needed, numbers are meaningless):

    python run_ablation.py --dataset synthetic --configs core --n-seeds 5 && python run_ablation.py --dataset synthetic --analyze
    python run_multisplit.py --dataset synthetic --n-partitions 2 --n-seeds 4 && python run_multisplit.py --dataset synthetic --analyze

## Disclosure the ablation was built to confirm

In the submitted LightGBM configuration `subsample=0.8` was **inactive**:
LightGBM only applies row subsampling when `bagging_freq` (`subsample_freq` in
the sklearn API) is > 0, and it was left at its default of 0. Only
`colsample_bytree=0.8` was active, which is why the same-seed-vs-different-seed
test still passed. XGBoost's `subsample=0.8` *was* active. So "matched
hyperparameters" in Section II-E / IV needs correcting to "matched settings,
of which LightGBM's row subsampling was inactive by library default". The
`lgb_rowsub_only_freq0` ablation demonstrates this empirically (if the seed
becomes inert when feature subsampling is switched off, row subsampling was
never on). Two other defaults differ between the libraries and are candidate
mechanisms the grid tests: L2 leaf regularization (XGBoost `lambda=1`,
LightGBM `0`) and minimum leaf hessian (XGBoost `min_child_weight=1`,
LightGBM `1e-3`).

## Map: reviewer concern -> script

| # | Concern (who) | Script / command | Machine | Est. runtime | Output | Goes to |
|---|---|---|---|---|---|---|
| M1 | Single fixed train/test partition (Report item 1 MAJOR, R4, R5) | `python run_multisplit.py --dataset nsl_kdd` then `--analyze`; repeat for `cse_cic_ids2018`, `unsw_nb15` | Gopal (all data); NSL-KDD can also run here once downloaded | NSL-KDD ~25 min (6 partitions x 20 seeds x 2 archs); CIC/UNSW ~40 min each | `results/multisplit_<ds>_analysis.csv`, `_crosspartition.csv` | new Results subsection "Robustness to the train/test partition" + Limitations rewrite |
| M2 | LightGBM mechanism untested (Report item 2 MAJOR, R1 #2, R2 #1, R4, R5) | `python run_ablation.py --dataset nsl_kdd --configs all` then `--analyze`; at least `--configs core` on the other two datasets | Gopal; NSL-KDD also here | NSL-KDD `all` ~1 h (11 LGB configs ~2 s/fit, 5 XGB configs ~10 s/fit, 43 fits each); `core` ~20 min | `results/ablation_<ds>_analysis.csv`, `_determinism.json` | new Results subsection "Ablation of the LightGBM instability" + hyperparameter table + Discussion rewrite |
| M3 | Bootstrap decomposition validation must be IN the paper (R1 #1, R2 #2, R5, Report item 5) | `python validate_bootstrap_decomposition.py --all` (DONE 2026-09-02 on skuiu's machine, ~12 min; reruns are seeded and reproduce) | anywhere (no data) | ~12 min | console + `results/bootstrap_validation_small_variance.csv` (gitignored by `results/*.csv`, regenerate or copy into the paper repo) | new Methods subsection: the two original tests reproduced exactly (-1.6% bias; 0.0000046 null) plus the sweep. NOTE the sweep CONTRADICTS the reviewer's stated direction: clip-at-zero bias is UPWARD in expectation (null floor ~0.1 x noise variance, apparent SNR ~0.1); the real cost is per-cell, 15-55% of cells with true SNR < 0.3 are reported as exactly 0, so "0.000000" = "not detectable", not "absent". Estimator is within a few % of unbiased for true SNR >= ~0.5. State both facts and correct the reviewer politely with the numbers. |
| Mod3 | Overlap vs non-overlap instability (Report item 3, R1 #5, R5) | `python analyze_overlap_split.py --dataset <ds>` | Gopal only (needs `results/<ds>_matrix_per_instance.csv`) | minutes to ~1 h (UNSW largest) | `results/overlap_split_<ds>_analysis.csv` | Results III-E data-quality section + Limitations |
| Mod6 | Levene/BH promised in Methods, absent from Results (Report item 6, R5) | `python export_levene_bh_tables.py` | Gopal only (needs summary CSVs) | seconds | `results/levene_bh_all.csv`, `results/levene_bh_table.tex` | supplementary table; one sentence in Results pointing to it |
| R2 #3 | Sequential-prefix seed adequacy has ordering bias | `python analyze_seed_adequacy_random_subsets.py --dataset <ds>` | Gopal only (needs summary CSVs) | seconds | `results/seed_adequacy_random_<ds>.csv`, `_rule_<ds>.csv/.tex` | replaces/extends Fig. 3 analysis; supplies the adaptive stopping rule R1 #4 asks for |
| R1 #4 | "40 seeds" presented as sufficient | same script, `seed_adequacy_rule_<ds>.tex` ("seeds required for +/- 1 pp / 0.5 pp") | Gopal | seconds | as above | Discussion + Conclusion rewrite: adaptive criterion, not a fixed number |
| R1 #6 | Justify / re-draw the CIC-IDS2018 subsample | see "Second subsample" below | Gopal only (needs the 16M-row raw files) | prep ~15 min + LightGBM/XGBoost 40 seeds ~30 min | new summary CSV | one paragraph + one table row in Results |
| R1 #3, R4 | Hyperparameter justification table | no compute: text table from `config.py` + the ablation's sensitivity columns | - | - | - | Methods II-E table |

Text-only items (no scripts): convexity test downgraded to illustrative
(Report item 4), template placeholders removed (R4), typos (Report item 9),
N/A-vs-Degen footnote (item 7), grayscale figures (item 8), soften universal
claims (item 10), references cleanup (R1, R5: add the LightGBM and XGBoost
papers, replace/justify arXiv-only [8]/[10], drop [9]).

## Second CIC-IDS2018 subsample (R1 #6)

`prepare_cicids2018.py` hard-codes `CIC_IDS2018_SUBSAMPLE_RNG_SEED = 777_777`
and the output paths. To draw an independent subsample without touching the
original script or `config.py`, run it with a temporary override:

    python -c "import config, prepare_cicids2018 as p; \
      config.CIC_IDS2018_SUBSAMPLE_RNG_SEED = 777_778; \
      p.OUTPUT_TRAIN_PATH = 'data/CSECICIDS2018_train_seedB.csv'; \
      p.OUTPUT_TEST_PATH  = 'data/CSECICIDS2018_test_seedB.csv'; \
      p.main()"

Then register it as a second dataset entry in a local copy of the config
(e.g. `cse_cic_ids2018_seedB` pointing at the two new files, same categories,
`expected_*_rows` from the script's printout) and run
`run_ablation.py --dataset cse_cic_ids2018_seedB --configs lgb_baseline,xgb_baseline`
to compare against the original subsample. Report the two side by side.

## Interpreting the ablation

- `lgb_rowsub_only_freq0` seed inert (`seed_inert = True`, `different_seed_identical`): confirms row subsampling was inactive in the submitted runs. State it.
- Instability collapses under `lgb_l2_reg1` / `lgb_min_child_weight1` / `lgb_xgb_matched_reg` but NOT under `xgb_lossguide`: the driver is LightGBM's weaker default leaf regularization, not leaf-wise growth. Rewrite the Discussion hypothesis accordingly.
- `xgb_lossguide` (or `xgb_lightgbm_like`) becomes unstable: supports the growth-policy hypothesis from the other direction; say so, and note which of lambda / min_child_weight was also required.
- Instability persists under every LightGBM variant and XGBoost stays stable under every variant: the effect is implementation-level, not explained by these parameters; keep the mechanism explicitly open and say the ablation ruled these candidates out.
- `lgb_no_subsampling` seed inert: the seed only enters LightGBM through subsampling; the instability is sensitivity to *which features* are sampled, which is a real property of the trained model, not a bug in seeding.

## Interpreting the multi-split check

Partition 0 reproduces the official split. If LightGBM's per-partition CV stays
an order of magnitude above XGBoost's on partitions 1..5 as well, the headline
finding is not partition-specific. `std_of_partition_means` versus
`mean_within_partition_seed_std` gives the size of partition variance relative
to seed variance, which is the number Review_Report item 1 asks for. Remember
the NSL-KDD caveat: re-partitions remove the official distribution shift, so
only within-partition instability is comparable, not absolute accuracy.
