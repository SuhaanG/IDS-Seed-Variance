"""
Clean re-run of the logistic regression column (revision).

WHY THIS EXISTS
---------------
The submitted results file for logistic_regression contains two constant
blocks rather than one: seeds 0-9 at one accuracy and seeds 10-39 at a
different one, with the break falling exactly on the pilot/full seed
boundary. run_matrix.py resumes by skipping (dataset, architecture, seed)
cells that already appear in the summary CSV, so the ten pilot rows were
carried forward unchanged when the run was extended to 40 seeds. Any
configuration change between those two invocations is therefore recorded
as if it were seed-driven variance. It is not.

Logistic regression as configured here is a convex problem, so at
convergence its solution does not depend on the seed at all. On NSL-KDD a
clean 40-seed re-run gives exactly ONE distinct accuracy
(0.7437455642299503) and a recall standard deviation of exactly zero in
every category. That is the correct result and it is the one the paper's
convexity argument actually predicts.

This script re-runs ONLY logistic_regression, for one dataset, into its
own output files. It never writes to {dataset}_matrix_summary.csv, so the
other five architectures are untouched.

GUARD AGAINST REPEATING THE ORIGINAL BUG
----------------------------------------
Before appending to an existing output file, the script compares a
fingerprint of the solver configuration and library versions against the
one recorded when that file was created. If anything differs it aborts
instead of resuming, because resuming would once again mix two
configurations inside a single "seed variance" column.

Usage:
    python rerun_logistic_regression.py --dataset cse_cic_ids2018
    python rerun_logistic_regression.py --dataset unsw_nb15
    python rerun_logistic_regression.py --dataset nsl_kdd --seeds 0,1,2
"""

import os
import sys
import json
import argparse
import platform

import numpy as np
import pandas as pd

import config
from data import load_and_preprocess
from train import train_one_run

ARCHITECTURE = "logistic_regression"


def solver_fingerprint():
    import sklearn
    import scipy
    return {
        "architecture": ARCHITECTURE,
        "logreg_solver": config.LOGREG_SOLVER,
        "logreg_max_iter": config.LOGREG_MAX_ITER,
        "logreg_C": config.LOGREG_C,
        "results_schema_version": config.RESULTS_SCHEMA_VERSION,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def check_or_write_fingerprint(path, current):
    """Refuse to resume across a configuration change. This is the exact
    failure the original logistic regression column suffered from."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        return
    with open(path) as f:
        recorded = json.load(f)
    diffs = {k: (recorded.get(k), current.get(k))
             for k in set(recorded) | set(current)
             if recorded.get(k) != current.get(k)}
    if diffs:
        print("\nABORTING: this output file was created under a different "
              "configuration than the one running now.", file=sys.stderr)
        for k, (was, now) in sorted(diffs.items()):
            print(f"  {k}: recorded={was!r} current={now!r}", file=sys.stderr)
        print("\nResuming here would mix two configurations inside one seed "
              "column, which is the bug this re-run exists to correct. "
              "Delete the output files and start clean instead.", file=sys.stderr)
        sys.exit(2)


def completed_seeds(path, dataset):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    sub = df[(df["dataset"] == dataset) & (df["architecture"] == ARCHITECTURE)]
    return set(int(s) for s in sub["seed"].unique())


def report(path, dataset, categories):
    df = pd.read_csv(path)
    df = df[(df["dataset"] == dataset) & (df["architecture"] == ARCHITECTURE)]
    acc = df["aggregate_accuracy"]
    print(f"\n{'='*70}\nCLEAN LOGISTIC REGRESSION: {dataset}\n{'='*70}")
    print(f"seeds completed        : {len(df)}")
    print(f"distinct accuracies    : {acc.nunique()}")
    print(f"mean aggregate accuracy: {acc.mean():.10f}")
    print(f"std  aggregate accuracy: {acc.std(ddof=1):.10f}")
    print(f"min / max              : {acc.min():.10f} / {acc.max():.10f}")
    print("\nper-category recall standard deviation:")
    for cat in categories:
        col = f"{cat}_recall"
        if col in df:
            print(f"  {cat:<15} mean={df[col].mean():.6f}  std={df[col].std(ddof=1):.6f}")
    if acc.nunique() == 1:
        print("\nAll seeds identical, as a convex objective at convergence "
              "requires. Seed-driven variance for this architecture is zero.")
    else:
        print("\nNOTE: more than one distinct accuracy. Report the spread as "
              "observed; do not assume it is zero.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    ap.add_argument("--seeds", default="full",
                    help="'full' (0-39), 'pilot' (0-9), or a comma list.")
    args = ap.parse_args()

    dataset = args.dataset
    if args.seeds == "full":
        seeds = config.FULL_SEEDS
    elif args.seeds == "pilot":
        seeds = config.PILOT_SEEDS
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    stem = os.path.join(config.RESULTS_DIR, f"{dataset}_logreg_clean")
    summary_csv, per_instance_csv = f"{stem}.csv", f"{stem}_per_instance.csv"

    check_or_write_fingerprint(f"{stem}_fingerprint.json", solver_fingerprint())

    categories = config.DATASETS[dataset]["categories"]
    print(f"Loading {dataset} ...")
    X_train, y_train, X_test, y_test, _ = load_and_preprocess(dataset)
    print(f"X_train {X_train.shape}  X_test {X_test.shape}  "
          f"{len(categories)} classes")
    print(f"solver={config.LOGREG_SOLVER} max_iter={config.LOGREG_MAX_ITER} "
          f"C={config.LOGREG_C}")

    done = completed_seeds(summary_csv, dataset)
    todo = [s for s in seeds if s not in done]
    if done:
        print(f"{len(done)} seed(s) already complete under this same "
              f"configuration, skipping: {sorted(done)}")
    if not todo:
        print("Nothing left to run.")
        report(summary_csv, dataset, categories)
        return

    print(f"Running {len(todo)} seed(s): {todo}\n")
    for n, seed in enumerate(todo, 1):
        train_one_run(architecture=ARCHITECTURE, seed=seed,
                      X_train=X_train, y_train=y_train,
                      X_test=X_test, y_test=y_test,
                      categories=categories,
                      results_csv_path=summary_csv,
                      per_instance_csv_path=per_instance_csv,
                      dataset_name=dataset)
        seen = pd.read_csv(summary_csv)["aggregate_accuracy"].nunique()
        print(f"    ({n}/{len(todo)}) distinct accuracies so far: {seen}")

    report(summary_csv, dataset, categories)
    print(f"\nSaved: {summary_csv}\n       {per_instance_csv}")


if __name__ == "__main__":
    main()
