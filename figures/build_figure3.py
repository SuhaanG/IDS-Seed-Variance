"""
Figure 3: per-category signal-to-noise ratio on NSL-KDD across the convexity
spectrum (logistic regression -> shallow MLP -> DNN).

STYLE
-----
Matches the submitted figure and the rest of the paper's SNR figures: log
y-axis, dashed reference line at SNR = 1, blue / purple / red series with the
legend above the axes, lowercase category labels. Zero-valued bars cannot be
drawn on a log axis, so they are shown as a sliver at the axis floor with a
grey "0" above them, as in the submitted version. The only addition is the
numeric value printed above each non-zero bar, requested by Reviewer 5.

PROVENANCE OF THE PLOTTED VALUES (round-3 revision)
----------------------------------------------------
Read from results/nsl_kdd_corrected_decomposition_lgbdet.csv: the September
2026 NSL-KDD matrix (run set B), decomposed with the paper's estimator at
5,000 bootstrap iterations, the same values as Table 6. Logistic regression is
0 in every category (all 40 seeds identical). Labels use three significant
figures (Reviewer 1, round 2, item 9).

The asterisk marking is computed, not hard-coded: a category is starred when
SNR increases strictly from logistic regression to the shallow MLP to the DNN.

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

import pandas as pd
_DEC = pd.read_csv(os.path.join(HERE, "..", "results", "nsl_kdd_corrected_decomposition_lgbdet.csv"))


def _snr(arch, cell):
    r = _DEC[(_DEC["architecture"] == arch) & (_DEC["cell"] == cell)]
    if len(r) != 1:
        raise SystemExit(f"missing {arch}/{cell}")
    return float(r["snr"].iloc[0])


# Inner tuple order: (logistic regression, shallow MLP, DNN).
SNR = {c: tuple(_snr(a, c) for a in ("logistic_regression", "shallow_mlp", "dnn"))
       for c in ("normal", "dos", "probe", "r2l", "u2r")}


def sig3(v):
    """Three significant figures, no scientific notation."""
    return f"{float(f'{v:.3g}'):g}"


SERIES = [("LogReg", "#1f77b4"), ("Shallow MLP", "#7b2c8f"), ("DNN", "#c0272d")]

Y_FLOOR = 0.05          # bottom of the log axis
ZERO_STUB = Y_FLOOR * 1.12   # sliver height used to make a zero bar visible


def monotonic_increasing(triple):
    lr, mlp, dnn = triple
    return lr < mlp < dnn


def main():
    cats = list(SNR)
    x = np.arange(len(cats))
    width = 0.27
    starred = [monotonic_increasing(SNR[c]) for c in cats]
    tick_labels = [c + ("*" if s else "") for c, s in zip(cats, starred)]

    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.6, 3.4))

    for i, (label, color) in enumerate(SERIES):
        vals = [SNR[c][i] for c in cats]
        heights = [v if v > 0 else ZERO_STUB for v in vals]
        xs = x + (i - 1) * width
        ax.bar(xs, heights, width, label=label, color=color,
               edgecolor="black", linewidth=0.6)
        for xi, v in zip(xs, vals):
            if v == 0:
                ax.text(xi, ZERO_STUB * 1.25, "0", ha="center", va="bottom",
                        fontsize=6, color="0.45")
            else:
                ax.text(xi, v * 1.12, sig3(v), ha="center", va="bottom",
                        fontsize=5, rotation=90, color="0.15")

    ax.set_yscale("log")
    ax.set_ylim(Y_FLOOR, 250)
    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Signal-to-noise ratio\n(log scale)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, handlelength=1.4, columnspacing=1.2)
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
