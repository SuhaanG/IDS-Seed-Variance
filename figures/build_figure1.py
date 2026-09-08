"""
Figure 1: aggregate-accuracy signal-to-noise ratio by architecture, one panel
per dataset (NSL-KDD, CSE-CIC-IDS2018, UNSW-NB15).

STYLE
-----
Matches the submitted figure: three panels, log y-axis, dashed reference line
at SNR = 1, one colour per architecture, two-line tick labels. The submitted
version labelled bars with SNR = 0 as "N/A", but the caption defined N/A as a
degenerate case (both variances zero), and no aggregate-accuracy cell in
Tables 6-8 is degenerate; every such bar is an ordinary SNR of exactly 0.00
that cannot be drawn on a log axis. This version therefore labels them "0",
the same convention as Figure 3 and the same meaning as 0.000000 (0.00) in
the tables.

PROVENANCE OF THE PLOTTED VALUES
--------------------------------
Aggregate-accuracy SNR entries of Tables 6, 7 and 8 of the manuscript, from
the original 720-run matrix and the bootstrap decomposition of Section II-F.
Transcribed so the figure matches the tables exactly.

Logistic regression: NSL-KDD corrected to 0.00 (clean 40-seed re-run, all
seeds identical; the submitted 5.67 was an artifact of two solver
configurations mixed in one results file, see Section II-I). UNSW-NB15 was
already 0.00 in the submitted table. CSE-CIC-IDS2018 corrected to 0.00 from its
clean 40-seed re-run (all seeds identical at 0.9570989275; the submitted
16.42 was the same artifact).

Usage:
    python figures/build_figure1.py
Writes figures/fig1_aggregate_snr.pdf and a .png preview.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ARCHS = ["DNN", "Random\nForest", "XGBoost", "LightGBM", "Logistic\nRegression", "Shallow\nMLP"]
COLORS = ["#1f5fa8", "#74add1", "#2e7d32", "#b2182b", "#e8735a", "#7b2c8f"]

# Aggregate-accuracy SNR, Tables 6-8, order as ARCHS.
SNR = {
    "NSL-KDD":         [9.30,  2.89, 0.20, 1323.53,  0.00, 7.63],
    "CSE-CIC-IDS2018": [0.00,  0.00, 0.00, 1747.10,  0.00, 0.00],
    "UNSW-NB15":       [29.53, 0.00, 0.00,  378.69,  0.00, 17.45],
}

Y_FLOOR = 0.05
ZERO_STUB = Y_FLOOR * 1.12


def main():
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 6.5,
        "xtick.labelsize": 4.9, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.7), sharey=True)
    x = np.arange(len(ARCHS))
    for ax, (ds, vals) in zip(axes, SNR.items()):
        heights = [v if v > 0 else ZERO_STUB for v in vals]
        ax.bar(x, heights, 0.66, color=COLORS, edgecolor="black", linewidth=0.5)
        for xi, v in zip(x, vals):
            if v == 0:
                ax.text(xi, ZERO_STUB * 1.25, "0", ha="center", va="bottom",
                        fontsize=5.5, color="0.45")
        ax.set_yscale("log")
        ax.set_ylim(Y_FLOOR, 6000)
        ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_title(ds, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ARCHS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=-1)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Signal-to-noise ratio\n(genuine / sampling-noise variance, log scale)")
    fig.tight_layout(pad=0.5, w_pad=1.0)

    pdf = os.path.join(HERE, "fig1_aggregate_snr.pdf")
    png = os.path.join(HERE, "fig1_aggregate_snr.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print("wrote", pdf)
    print("wrote", png)


if __name__ == "__main__":
    main()
