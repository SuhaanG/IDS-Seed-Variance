"""
Overlap-stratified seed-variance analysis for the IEEE Access revision.

Addresses Review_Report item 3 (memorization confound), Reviewer 1 (#5
"separately evaluate duplicated and non-duplicated test instances") and
Reviewer 5. This is a PURE RE-ANALYSIS of the per-instance predictions the
original matrix already wrote (results/<dataset>_matrix_per_instance.csv);
no model is retrained.

For every architecture it splits the official test set into
  overlap:      test rows whose raw feature vector is identical to at least
                one training row (same hash as data._audit_train_test_overlap,
                so the counts match the 664 / 4,786 / 8,541 in the paper),
  non_overlap:  everything else,
and runs the paper's bootstrap decomposition (aggregate accuracy and every
category) on each subset separately. If instability were an artifact of
memorized duplicates, the non-overlap subset would show it weakened or
absent; if the two subsets agree, overlap is not driving the result.

Usage (on the machine that has the data and the per-instance CSVs):
    python analyze_overlap_split.py --dataset nsl_kdd
    python analyze_overlap_split.py --dataset unsw_nb15 --architectures lightgbm,xgboost
"""

import os
import argparse

import numpy as np
import pandas as pd

import config
from revision_common import (
    DEFAULT_REVISION_BOOTSTRAP, dataset_categories, compute_overlap_mask, decompose_cell,
    describe_accuracies, df_to_markdown,
)

PAPER_OVERLAP_COUNTS = {"nsl_kdd": 664, "cse_cic_ids2018": 4786, "unsw_nb15": 8541}


def load_correct_matrices(per_instance_csv, architectures, n_test, chunksize=2_000_000):
    """Returns {architecture: (seeds, bool matrix [n_seeds, n_test])}."""
    store = {a: {} for a in architectures}
    usecols = ["architecture", "seed", "instance_id", "correct"]
    for chunk in pd.read_csv(per_instance_csv, usecols=usecols, chunksize=chunksize,
                             dtype={"architecture": "category", "seed": np.int32,
                                    "instance_id": np.int32, "correct": np.int8}):
        chunk = chunk[chunk["architecture"].isin(architectures)]
        for (arch, seed), g in chunk.groupby(["architecture", "seed"], observed=True):
            vec = store[arch].setdefault(int(seed), np.zeros(n_test, dtype=bool))
            vec[g["instance_id"].values] = g["correct"].values.astype(bool)
    out = {}
    for arch, by_seed in store.items():
        if not by_seed:
            continue
        seeds = np.array(sorted(by_seed))
        out[arch] = (seeds, np.vstack([by_seed[s] for s in seeds]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nsl_kdd")
    ap.add_argument("--per-instance-csv", default=None)
    ap.add_argument("--architectures", default=",".join(config.ARCHITECTURES))
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_REVISION_BOOTSTRAP)
    args = ap.parse_args()

    dataset = args.dataset
    per_instance_csv = args.per_instance_csv or os.path.join(
        config.RESULTS_DIR, f"{dataset}_matrix_per_instance.csv")
    if not os.path.exists(per_instance_csv):
        raise SystemExit(f"Per-instance CSV not found: {per_instance_csv}. This analysis needs the "
                         f"original matrix outputs (they live on the machine that ran run_matrix.py).")
    architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]
    categories = dataset_categories(dataset)

    mask, y_test = compute_overlap_mask(dataset)
    n_test = len(y_test)
    n_overlap = int(mask.sum())
    print(f"[overlap] {dataset}: {n_overlap} of {n_test} test rows ({n_overlap / n_test * 100:.2f}%) "
          f"are feature-identical to a training row.")
    expected = PAPER_OVERLAP_COUNTS.get(dataset)
    if expected is not None and expected != n_overlap:
        print(f"  WARNING: paper reported {expected}; got {n_overlap}. Check that the data files match "
              f"the ones used for the submitted runs before trusting this analysis.")

    print("  per-category composition of the two subsets:")
    comp = pd.DataFrame({
        "category": categories,
        "overlap_n": [int(((y_test == c) & mask).sum()) for c in categories],
        "non_overlap_n": [int(((y_test == c) & ~mask).sum()) for c in categories],
    })
    print(df_to_markdown(comp))

    matrices = load_correct_matrices(per_instance_csv, architectures, n_test)
    rows = []
    for arch in architectures:
        if arch not in matrices:
            print(f"  {arch}: no per-instance rows found, skipping.")
            continue
        seeds, correct = matrices[arch]
        for subset_name, sel in (("overlap", mask), ("non_overlap", ~mask), ("all", np.ones(n_test, dtype=bool))):
            if sel.sum() < 2:
                print(f"  {arch:20s} {subset_name:12s} has {int(sel.sum())} instance(s), skipping.")
                continue
            sub_y = y_test[sel]
            sub_correct = correct[:, sel]
            present = [c for c in categories if (sub_y == c).any()]
            row = {"architecture": arch, "subset": subset_name, "n_instances": int(sel.sum()), "n_seeds": len(seeds)}
            row.update({k: v for k, v in describe_accuracies(sub_correct.mean(axis=1)).items() if k != "n_seeds"})
            row.update(decompose_cell(seeds, sub_correct, sub_y, present, dataset, f"{arch}_{subset_name}",
                                      n_bootstrap=args.n_bootstrap, include_categories=True))
            for c in categories:
                row[f"{c}_n"] = int((sub_y == c).sum())
            rows.append(row)
            print(f"  {arch:20s} {subset_name:12s} n={row['n_instances']:6d} mean={row['mean_acc']} "
                  f"CV={row['cv_pct']}% SNR(agg)={row['agg_snr']}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(config.RESULTS_DIR, f"overlap_split_{dataset}_analysis.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== Aggregate accuracy: overlap vs non-overlap ===")
    print(df_to_markdown(df[["architecture", "subset", "n_instances", "mean_acc", "cv_pct",
                             "agg_genuine_var", "agg_snr"]]))
    print("\n=== Per-category SNR: overlap vs non-overlap ===")
    print(df_to_markdown(df[["architecture", "subset"] + [f"{c}_snr" for c in categories if f"{c}_snr" in df.columns]]))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
