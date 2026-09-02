"""
Exports the Levene / Benjamini-Hochberg results that the submitted paper
described in Methods (Section II-F) but never tabulated in Results
(Review_Report item 6, Reviewer 5). Pure re-analysis of the matrix
summaries; produces one CSV with every row and a LaTeX supplementary table.

Usage:
    python export_levene_bh_tables.py                 # all datasets in config.DATASETS
    python export_levene_bh_tables.py --datasets nsl_kdd,unsw_nb15
"""

import os
import argparse

import numpy as np
import pandas as pd

import config
from stats_analysis import load_results, levene_with_correction
from revision_common import df_to_markdown


def fmt_p(p):
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--architectures", default=",".join(config.ARCHITECTURES))
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]

    rows = []
    for dataset in datasets:
        path = os.path.join(config.RESULTS_DIR, f"{dataset}_matrix_summary.csv")
        if not os.path.exists(path):
            print(f"  {dataset}: {path} not found, skipping.")
            continue
        summary = load_results(path)
        categories = config.DATASETS[dataset]["categories"]
        for arch in architectures:
            sub = summary[(summary["dataset"] == dataset) & (summary["architecture"] == arch)]
            if sub["seed"].nunique() < 3:
                continue
            agg_std = float(sub["aggregate_accuracy"].std(ddof=1))
            for r in levene_with_correction(summary, dataset, arch, categories):
                rows.append({
                    "dataset": dataset, "architecture": arch, "category": r["category"],
                    "recall_std": float(sub[f"{r['category']}_recall"].std(ddof=1)),
                    "aggregate_accuracy_std": agg_std,
                    "levene_stat": r["levene_stat"], "p_raw": r["p_raw"],
                    "p_bh": r["p_corrected_fdr_bh"], "significant_bh": r["significant_after_correction"],
                    "std_ratio_effect_size": r["std_ratio_effect_size"],
                })

    if not rows:
        raise SystemExit("No results found. Run this on the machine that holds results/*_matrix_summary.csv.")
    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.RESULTS_DIR, "levene_bh_all.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(config.RESULTS_DIR, "levene_bh_table.tex")
    with open(tex_path, "w") as f:
        for dataset, g in df.groupby("dataset", sort=False):
            f.write(f"% ---- {dataset} ----\n")
            f.write("\\begin{tabular}{llrrrrc}\n\\hline\n")
            f.write("Architecture & Category & Levene $W$ & $p$ (raw) & $p$ (BH) & Effect size (std ratio) & Sig.\\\\\n\\hline\n")
            for _, r in g.iterrows():
                es = "$\\infty$" if not np.isfinite(r["std_ratio_effect_size"]) else f"{r['std_ratio_effect_size']:.2f}"
                f.write(f"{r['architecture'].replace('_', ' ')} & {r['category'].replace('_', ' ')} & "
                        f"{r['levene_stat']:.2f} & {fmt_p(r['p_raw'])} & {fmt_p(r['p_bh'])} & {es} & "
                        f"{'*' if r['significant_bh'] else ''}\\\\\n")
            f.write("\\hline\n\\end{tabular}\n\n")

    print(df_to_markdown(df[["dataset", "architecture", "category", "levene_stat", "p_raw", "p_bh",
                             "significant_bh", "std_ratio_effect_size"]]))
    counts = df.groupby(["dataset", "architecture"])["significant_bh"].sum().reset_index()
    print("\nSignificant-after-BH count per (dataset, architecture):")
    print(df_to_markdown(counts))
    print(f"\nSaved: {csv_path}\nSaved: {tex_path}")


if __name__ == "__main__":
    main()
