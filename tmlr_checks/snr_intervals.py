"""
Seed-bootstrap intervals for the signal-to-noise ratios in the paper's tables.

The paper's estimator (stats_analysis.true_variance_decomposition) subtracts the
mean within-seed sampling variance from the across-seed variance of the 40
per-seed estimates. The within-seed term is a bootstrap variance of a
proportion (recall within a category, or accuracy), whose expectation is
p(1 - p) / n. That closed form makes it cheap to ask how stable each ratio is
across seeds: resample the 40 seeds with replacement, recompute

    SNR* = max(0, var(x*, ddof=1) - mean(p*(1 - p*) / n)) / mean(p*(1 - p*) / n)

and take percentiles. The point estimate from the closed form is reported next
to each interval so it can be checked against the bootstrap value in the paper.

Usage:
    python tmlr_checks/snr_intervals.py            # writes results/snr_intervals.csv
"""

import os
import numpy as np
import pandas as pd

RES = "results"
B = 10_000
SEED = 20261026

# (table, dataset key, summary file, architectures to take from it; None = every row)
SOURCES = [
    ("NSL-KDD", "nsl_kdd", "nsl_kdd_matrix_summary.csv", ["dnn", "random_forest", "xgboost", "shallow_mlp"]),
    ("NSL-KDD", "nsl_kdd", "nsl_kdd_lgb_det.csv", None),
    ("NSL-KDD", "nsl_kdd", "nsl_kdd_logreg_clean.csv", None),
    ("CSE-CIC-IDS2018", "cse_cic_ids2018", "cse_cic_ids2018_matrix_summary.csv", ["dnn", "random_forest", "xgboost", "shallow_mlp"]),
    ("CSE-CIC-IDS2018", "cse_cic_ids2018", "cse_cic_ids2018_lgb_det.csv", None),
    ("CSE-CIC-IDS2018", "cse_cic_ids2018", "cse_cic_ids2018_logreg_clean.csv", None),
    ("UNSW-NB15", "unsw_nb15", "unsw_nb15_matrix_summary.csv", ["dnn", "random_forest", "xgboost", "shallow_mlp"]),
    ("UNSW-NB15", "unsw_nb15", "unsw_nb15_lgb_det.csv", None),
    ("UNSW-NB15", "unsw_nb15", "unsw_nb15_logreg_clean.csv", None),
    ("CSE-CIC-IDS2018 draw 2", "cse_cic_ids2018_draw2", "cse_cic_ids2018_draw2_matrix_summary.csv",
     ["dnn", "random_forest", "xgboost", "logistic_regression", "shallow_mlp"]),
    ("CSE-CIC-IDS2018 draw 2", "cse_cic_ids2018_draw2", "cse_cic_ids2018_draw2_lgb_det.csv", None),
]


def snr(x, noise):
    m = noise.mean()
    v = x.var(ddof=1)
    if m <= 0:
        return np.nan if v < 1e-12 else np.inf
    return max(0.0, v - m) / m


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for table, ds, fname, archs in SOURCES:
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            print(f"skipping {fname}: not present")
            continue
        df = pd.read_csv(path)
        if archs is not None:
            df = df[df["architecture"].isin(archs)]
        cats = [c[:-len("_recall")] for c in df.columns if c.endswith("_recall")]
        for arch, g in df.groupby("architecture"):
            g = g.drop_duplicates("seed").sort_values("seed")
            if len(g) != 40:
                raise SystemExit(f"{fname} {arch}: {len(g)} seeds, expected 40")
            n_total = g[[f"{c}_support" for c in cats]].sum(axis=1).to_numpy(float)
            cells = [("aggregate_accuracy", g["aggregate_accuracy"].to_numpy(float), n_total)]
            cells += [(c, g[f"{c}_recall"].to_numpy(float), g[f"{c}_support"].to_numpy(float)) for c in cats]
            idx = rng.integers(0, 40, size=(B, 40))
            for cell, x, n in cells:
                ok = ~np.isnan(x)
                noise = np.where(ok, x * (1 - x) / n, np.nan)
                point = snr(x[ok], noise[ok])
                xs, ns = x[idx], noise[idx]
                m = np.nanmean(ns, axis=1)
                v = np.nanvar(xs, axis=1, ddof=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    boot = np.where(m > 0, np.maximum(0.0, v - m) / m, np.nan)
                lo, hi = (np.nanpercentile(boot, [5, 95]) if np.isfinite(boot).any() else (np.nan, np.nan))
                rows.append({"table": table, "dataset": ds, "architecture": arch, "cell": cell,
                             "n_seeds": int(ok.sum()), "snr_closed_form": point,
                             "lo90": lo, "hi90": hi,
                             "excludes_1": bool(lo > 1) if np.isfinite(lo) else False})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "snr_intervals.csv"), index=False)
    print(f"wrote {len(out)} cells to results/snr_intervals.csv")


if __name__ == "__main__":
    main()
