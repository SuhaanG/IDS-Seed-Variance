"""
Repeated-run reproducibility check for LightGBM (revision diagnostic).

Motivation: in the NSL-KDD ablation, the configuration with NO subsampling
(subsample=1.0, colsample_bytree=1.0) was seed-inert in the two-run
determinism check (seeds 0 and 1 gave identical predictions), yet its 40
"seeds" still spanned ~27 accuracy points. If the seed cannot change the
model, that spread must come from run-to-run nondeterminism (e.g. thread
scheduling in multi-threaded histogram construction) amplified by a
numerically unstable training trajectory. That would mean part of what
the paper calls "seed-driven" instability is not controlled by the seed
at all, which the revision must characterize correctly.

This script separates the two sources directly:
  (A) fit the SAME seed R times per configuration -> number of distinct
      prediction vectors and the accuracy spread (pure run-to-run
      nondeterminism);
  (B) fit S different seeds once each -> spread attributable to the seed;
each under n_jobs=-1 (as in the paper) and n_jobs=1 (single thread), plus
LightGBM's documented reproducibility switch (deterministic=True,
force_row_wise=True).

Usage:
    python check_lightgbm_reproducibility.py --dataset nsl_kdd
    python check_lightgbm_reproducibility.py --dataset nsl_kdd --configs baseline,no_subsampling --repeats 4 --seeds 0,1,2,3
"""

import os
import argparse
import hashlib
import warnings

import numpy as np
import pandas as pd

import config
from revision_common import (
    build_estimator, fit_predict_estimator, dataset_categories, load_official_split,
    labels_to_idx, df_to_markdown,
)

CONFIGS = {
    "baseline": {},
    "no_subsampling": dict(subsample=1.0, colsample_bytree=1.0),
    "l2_reg1": dict(reg_lambda=1.0),
    "baseline_deterministic_flag": dict(deterministic=True, force_row_wise=True),
}


def pred_hash(preds):
    return hashlib.sha1(np.ascontiguousarray(preds).tobytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nsl_kdd")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--threads", default="-1,1", help="Comma list of n_jobs settings to test.")
    ap.add_argument("--repeats", type=int, default=6, help="Same-seed repeats (seed 0).")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7", help="Different seeds, one fit each.")
    args = ap.parse_args()

    dataset = args.dataset
    categories = dataset_categories(dataset)
    X_train, y_train, X_test, y_test, _ = load_official_split(dataset)
    y_train_idx = labels_to_idx(y_train, categories)
    y_test_idx = labels_to_idx(y_test, categories)
    num_classes = len(categories)

    out_csv = os.path.join(config.RESULTS_DIR, f"lgb_reproducibility_{dataset}.csv")
    rows = []

    def record(cfg, n_jobs, mode, seed, rep, preds, t):
        rows.append({"config": cfg, "n_jobs": n_jobs, "mode": mode, "seed": seed, "repeat": rep,
                     "accuracy": float(np.mean(preds == y_test_idx)), "pred_hash": pred_hash(preds),
                     "train_time_sec": round(t, 1)})
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    for cfg in [c.strip() for c in args.configs.split(",")]:
        for n_jobs in [int(t) for t in args.threads.split(",")]:
            overrides = dict(CONFIGS[cfg]); overrides["n_jobs"] = n_jobs
            print(f"\n[{cfg} | n_jobs={n_jobs}] same seed x{args.repeats}:")
            for rep in range(args.repeats):
                est = build_estimator("lightgbm", num_classes, 0, overrides)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    preds, t = fit_predict_estimator(est, X_train, y_train_idx, X_test, 0)
                record(cfg, n_jobs, "same_seed", 0, rep, preds, t)
                print(f"    rep {rep}: acc={rows[-1]['accuracy']:.4f} hash={rows[-1]['pred_hash']} ({t:.1f}s)")
            print(f"[{cfg} | n_jobs={n_jobs}] different seeds:")
            for seed in [int(s) for s in args.seeds.split(",")]:
                est = build_estimator("lightgbm", num_classes, seed, overrides)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    preds, t = fit_predict_estimator(est, X_train, y_train_idx, X_test, seed)
                record(cfg, n_jobs, "different_seeds", seed, 0, preds, t)
                print(f"    seed {seed}: acc={rows[-1]['accuracy']:.4f} hash={rows[-1]['pred_hash']} ({t:.1f}s)")

    df = pd.DataFrame(rows)
    summary = (df.groupby(["config", "n_jobs", "mode"])
                 .agg(n_fits=("accuracy", "size"), distinct_prediction_vectors=("pred_hash", "nunique"),
                      acc_min=("accuracy", "min"), acc_max=("accuracy", "max"), acc_std=("accuracy", lambda s: s.std(ddof=1)))
                 .reset_index())
    summary["acc_range_pp"] = ((summary["acc_max"] - summary["acc_min"]) * 100).round(1)
    print("\n=== Reproducibility summary ===")
    print(df_to_markdown(summary))
    summary.to_csv(os.path.join(config.RESULTS_DIR, f"lgb_reproducibility_{dataset}_summary.csv"), index=False)
    print(f"\nSaved: {out_csv}")
    print("Reading: same_seed rows with distinct_prediction_vectors > 1 = run-to-run nondeterminism the seed does not control; "
          "compare n_jobs=-1 vs 1 and the deterministic flag to see whether threading is the source.")


if __name__ == "__main__":
    main()
