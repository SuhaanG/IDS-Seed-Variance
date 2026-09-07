"""
Figure 3: per-category signal-to-noise ratio on NSL-KDD across the convexity
spectrum (logistic regression -> shallow MLP -> DNN).

PROVENANCE OF THE PLOTTED VALUES
--------------------------------
The DNN and shallow-MLP values are the signal-to-noise ratios printed in
Table 1 of the manuscript (tab:nsl_kdd_results), which come from the original
240-run NSL-KDD matrix and the bootstrap decomposition of Section II-F with
5,000 iterations. They are transcribed here rather than recomputed so that the
figure matches the table exactly.

Logistic regression is 0.00 in every category. The clean 40-seed re-run
(rerun_logistic_regression.py; results/nsl_kdd_matrix_summary.csv) returns
identical aggregate accuracy and identical per-category recall for all 40
seeds, so the observed between-seed variance is exactly zero, the genuine
variance estimate max(0, 0 - noise) is exactly zero, and the ratio is 0.00.
The values 7.49 / 4.77 / 14.46 that an earlier draft reported for this column
were an artifact of two solver configurations being mixed in one results file
(see Section II-I of the manuscript) and are not used.

The asterisk marking is computed, not hard-coded: a category is starred when
SNR increases strictly from logistic regression to the shallow MLP to the DNN.

Design choices (Reviewer 5, figure legibility): grayscale-safe fills with a
hatch on the lightest series, and the numeric value printed above every bar,
so the figure reads correctly in black-and-white print and no bar height has
to be estimated by eye. Zero-valued bars are labeled "0".

Usage:
    python figures/build_figure3.py
Writes figures/fig3_convexity_spectrum.pdf (for the manuscript) and a .png
preview.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Table 1 of the manuscript, signal-to-noise ratio column entries, NSL-KDD.
# Order of the inner tuple: (logistic regression, shallow MLP, DNN).
SNR = {
    "Normal": (0.00, 33.51, 70.09),
    "DoS":    (0.00,  4.24, 16.79),
    "Probe":  (0.00, 18.92, 12.17),
    "R2L":    (0.00,  5.83,  5.15),
    "U2R":    (0.00,  0.20,  1.68),
}

SERIES = [
    ("Logistic regression (convex)",      "#f2f2f2", "////"),
    ("Shallow MLP (intermediate)",        "#9a9a9a", None),
    ("DNN (strongly non-convex)",         "#2b2b2b", None),
]


def monotonic_increasing(triple):
    lr, mlp, dnn = triple
    return lr < mlp < dnn


def main():
    cats = list(SNR)
    x = np.arange(len(cats))
    width = 0.26
    starred = [monotonic_increasing(SNR[c]) for c in cats]
    tick_labels = [c + ("*" if s else "") for c, s in zip(cats, starred)]

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 6.5, "legend.fontsize": 6, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ymax = max(v for t in SNR.values() for v in t)
    for i, (label, color, hatch) in enumerate(SERIES):
        vals = [SNR[c][i] for c in cats]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label,
                      color=color, edgecolor="black", linewidth=0.6,
                      hatch=hatch)
        for b, v in zip(bars, vals):
            txt = "0" if v == 0 else f"{v:.2f}"
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.012, txt,
                    ha="center", va="bottom", fontsize=5.2, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Signal-to-noise ratio")
    ax.set_ylim(0, ymax * 1.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=1.6)
    ax.text(0.99, 0.62, "* monotonic increase across the spectrum",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
            style="italic", color="0.25")
    fig.tight_layout(pad=0.4)

    pdf = os.path.join(HERE, "fig3_convexity_spectrum.pdf")
    png = os.path.join(HERE, "fig3_convexity_spectrum.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    print("starred categories:", [c for c, s in zip(cats, starred) if s])
    print("wrote", pdf)
    print("wrote", png)


if __name__ == "__main__":
    main()
