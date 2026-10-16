"""
Figure 2: raw aggregate accuracy of each of the 40 seeds for LightGBM and
XGBoost, one panel per dataset (NSL-KDD, CSE-CIC-IDS2018, UNSW-NB15).

STYLE
-----
Matches the submitted figure: jittered points per seed, a horizontal line at
the mean of each architecture, LightGBM in red and XGBoost in green, one panel
per dataset with its own y-axis.

PROVENANCE OF THE PLOTTED VALUES (round-3 revision)
---------------------------------------------------
LightGBM: the deterministic re-runs (deterministic=True, force_row_wise=True),
run set C, results/<dataset>_lgb_det.csv. These replace the multithreaded runs
of the submitted figure, which mixed seed-driven and run-to-run variation.
XGBoost: NSL-KDD from run set B (results/nsl_kdd_matrix_summary.csv);
CSE-CIC-IDS2018 and UNSW-NB15 from run set C (results/<dataset>_matrix_summary.csv,
written by run_second_subsample.py --architectures xgboost). The original
XGBoost per-seed values for those two datasets were not retained.
The jitter uses a fixed RNG seed, so the figure is reproducible.

Usage:
    python figures/build_figure2.py
Writes figures/fig2_raw_seeds.pdf and a .png preview.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

PANELS = [
    ("NSL-KDD", "nsl_kdd_lgb_det.csv", "nsl_kdd_matrix_summary.csv"),
    ("CSE-CIC-IDS2018", "cse_cic_ids2018_lgb_det.csv", "cse_cic_ids2018_matrix_summary.csv"),
    ("UNSW-NB15", "unsw_nb15_lgb_det.csv", "unsw_nb15_matrix_summary.csv"),
]
RED, RED_DARK = "#c0392b", "#7b0f1a"
GREEN, GREEN_DARK = "#2e7d32", "#0b4d16"


def load(lgb_file, matrix_file):
    lgb = pd.read_csv(os.path.join(RES, lgb_file))
    m = pd.read_csv(os.path.join(RES, matrix_file))
    xgb = m[m["architecture"] == "xgboost"]
    for name, df in (("LightGBM", lgb), ("XGBoost", xgb)):
        if df["seed"].nunique() != 40 or len(df) != 40:
            sys.exit(f"{lgb_file}/{matrix_file}: {name} has {len(df)} rows, expected 40 seeds")
    return lgb["aggregate_accuracy"].to_numpy(), xgb["aggregate_accuracy"].to_numpy()


def main():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.2, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.95))
    rng = np.random.default_rng(20261016)
    for ax, (title, lgb_file, matrix_file) in zip(axes, PANELS):
        lgb, xgb = load(lgb_file, matrix_file)
        for pos, vals, c, cd in ((0, lgb, RED, RED_DARK), (1, xgb, GREEN, GREEN_DARK)):
            jitter = rng.uniform(-0.09, 0.09, size=len(vals))
            ax.scatter(pos + jitter, vals, s=7, color=c, alpha=0.75, linewidths=0)
            ax.hlines(vals.mean(), pos - 0.2, pos + 0.2, color=cd, linewidth=1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["LightGBM", "XGBoost"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)
        print(f"{title:16s} LightGBM {lgb.min():.4f}-{lgb.max():.4f} mean {lgb.mean():.4f} | "
              f"XGBoost {xgb.min():.4f}-{xgb.max():.4f} mean {xgb.mean():.4f}")
    axes[0].set_ylabel("Aggregate accuracy (40 seeds)")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    pdf = os.path.join(HERE, "fig2_raw_seeds.pdf")
    fig.savefig(pdf)
    fig.savefig(pdf.replace(".pdf", ".png"), dpi=200)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
