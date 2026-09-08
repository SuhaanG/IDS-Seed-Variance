"""
Full six-architecture seed matrix on an independent second CSE-CIC-IDS2018
subsample (IEEE Access revision, Reviewer 1 item 6).

WHAT THIS ANSWERS
-----------------
Reviewer 1 asked whether the CSE-CIC-IDS2018 findings depend on the particular
200,000 / 40,000 stratified subsample that was drawn. This script repeats the
submitted study on a second draw made with a different RNG seed
(config.CIC_IDS2018_SUBSAMPLE_RNG_SEED_DRAW2) but otherwise identical cleaning,
taxonomy, target sizes and stratification. If the architecture contrast and the
per-category pattern reappear on the second draw, the result is a property of
the dataset rather than of one draw.

HOW IT RUNS
-----------
It calls the SAME train_one_run() the submitted 720 runs used, with the SAME
hyperparameters from config.py, so the second draw is analysed by exactly the
pipeline that produced Table 7. Architectures run fastest first (LightGBM,
XGBoost, random forest take seconds per seed; the two neural networks and
logistic regression take minutes), so the headline contrast is available early.

Outputs (standard names; the dataset name itself carries the draw):
    results/<dataset>_matrix_summary.csv
    results/<dataset>_matrix_per_instance.csv
    results/<dataset>_matrix_fingerprint.json
    results/<dataset>_decomposition.csv          (from --analyze)

The run is resumable per (architecture, seed) and refuses to resume if the
hyperparameters or library versions differ from those recorded when the output
files were created. That guard exists because the submitted logistic
regression column was contaminated by exactly such a resume across a
configuration change (see rerun_logistic_regression.py).

--analyze routes every cell through the unmodified
stats_analysis.true_variance_decomposition, i.e. the paper's estimator at its
configured 5,000 bootstrap iterations and RNG seed, and prints a table in the
layout of Table 7 for direct comparison.

Usage:
    python run_second_subsample.py --dataset cse_cic_ids2018_draw2 --seeds full
    python run_second_subsample.py --dataset cse_cic_ids2018_draw2 --analyze
    python run_second_subsample.py --dataset cse_cic_ids2018_draw2 --architectures lightgbm,xgboost --seeds 0,1,2
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

FAST_FIRST = ["lightgbm", "xgboost", "random_forest", "shallow_mlp", "logistic_regression", "dnn"]


# --------------------------------------------------------------------------
# configuration fingerprint (refuse to resume across a change)
# --------------------------------------------------------------------------
def config_fingerprint():
    import sklearn, scipy, xgboost, lightgbm, torch
    prefixes = ("LGB_", "XGB_", "RF_", "DNN_", "LOGREG_")
    hp = {k: getattr(config, k) for k in dir(config) if k.startswith(prefixes)}
    hp = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
          for k, v in sorted(hp.items())}
    return {
        "hyperparameters": hp,
        "results_schema_version": config.RESULTS_SCHEMA_VERSION,
        "sklearn": sklearn.__version__, "scipy": scipy.__version__,
        "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
        "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(),
    }


def check_or_write_fingerprint(path, current):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        return
    with open(path) as f:
        recorded = json.load(f)
    if recorded == current:
        return
    print("\nABORTING: output files were created under a different configuration.", file=sys.stderr)
    for k in sorted(set(recorded) | set(current)):
        if recorded.get(k) != current.get(k):
            print(f"  {k}: recorded={recorded.get(k)!r} current={current.get(k)!r}", file=sys.stderr)
    print("Resuming would mix two configurations in one results file. Delete the "
          "results/<dataset>_matrix_* files to start clean.", file=sys.stderr)
    sys.exit(2)


def completed_seeds(summary_csv, dataset, architecture):
    if not os.path.exists(summary_csv):
        return set()
    df = pd.read_csv(summary_csv)
    sub = df[(df["dataset"] == dataset) & (df["architecture"] == architecture)]
    return set(int(s) for s in sub["seed"].unique())


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def run(dataset, architectures, seeds, summary_csv, per_instance_csv):
    categories = config.DATASETS[dataset]["categories"]
    print(f"Loading {dataset} ...")
    X_train, y_train, X_test, y_test, _ = load_and_preprocess(dataset)
    print(f"X_train {X_train.shape}  X_test {X_test.shape}  {len(categories)} classes")

    for arch in architectures:
        done = completed_seeds(summary_csv, dataset, arch)
        todo = [s for s in seeds if s not in done]
        print(f"\n--- {arch}: {len(done)} done, {len(todo)} to run ---")
        for n, seed in enumerate(todo, 1):
            train_one_run(architecture=arch, seed=seed,
                          X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test,
                          categories=categories,
                          results_csv_path=summary_csv,
                          per_instance_csv_path=per_instance_csv,
                          dataset_name=dataset)
            if n % 10 == 0 or n == len(todo):
                acc = pd.read_csv(summary_csv)
                acc = acc[acc["architecture"] == arch]["aggregate_accuracy"]
                spread = acc.std(ddof=1) if len(acc) > 1 else 0.0
                print(f"    [{arch}] {len(acc)} seeds so far: mean={acc.mean():.4f} "
                      f"std={spread:.5f} min={acc.min():.4f} max={acc.max():.4f}")
    print("\nRun complete. Now: --analyze")


# --------------------------------------------------------------------------
# analyze (the paper's estimator, unmodified)
# --------------------------------------------------------------------------
def analyze(dataset, summary_csv, per_instance_csv):
    from stats_analysis import load_results, load_per_instance, true_variance_decomposition
    from revision_common import df_to_markdown

    categories = config.DATASETS[dataset]["categories"]
    summary_df = load_results(summary_csv)
    per_instance_df = load_per_instance(per_instance_csv)
    archs = [a for a in FAST_FIRST if a in set(summary_df["architecture"])]
    print(f"Architectures present: {archs}")
    print(f"Bootstrap iterations: {config.BOOTSTRAP_ITERATIONS}  RNG seed: {config.BOOTSTRAP_RNG_SEED}")

    rows = []
    for arch in archs:
        sub = summary_df[(summary_df["dataset"] == dataset) & (summary_df["architecture"] == arch)]
        # Pre-filter to this architecture: the estimator filters by dataset,
        # architecture and seed again internally and builds its own RNG from
        # config.BOOTSTRAP_RNG_SEED per call, so this changes speed, not results.
        pi_arch = per_instance_df[(per_instance_df["dataset"] == dataset)
                                  & (per_instance_df["architecture"] == arch)]
        for cell in [None] + list(categories):
            d = true_variance_decomposition(summary_df, pi_arch, cell, dataset, arch)
            label = "aggregate_accuracy" if cell is None else cell
            vals = sub["aggregate_accuracy"] if cell is None else sub[f"{cell}_recall"]
            degenerate = bool(d["is_degenerate_zero_variance"])
            genuine = float(d["genuine_between_seed_variance"])
            snr = None if degenerate else float(d["signal_to_noise_ratio"])
            mean = float(vals.mean())
            rows.append({
                "dataset": dataset, "architecture": arch, "cell": label,
                "n_seeds": int(len(sub)),
                "mean": mean, "std": float(vals.std(ddof=1)),
                "cv_pct": float(vals.std(ddof=1) / mean * 100) if mean > 0 else np.nan,
                "observed_var": float(d["observed_variance_of_point_estimates"]),
                "sampling_noise_var": float(d["mean_within_seed_sampling_variance"]),
                "genuine_var": genuine, "snr": snr, "degenerate": degenerate,
                "table_entry": "Degen." if degenerate else f"{genuine:.6f} ({snr:.2f})",
            })
            print(f"  {arch:20s} {label:20s} {rows[-1]['table_entry']}")

    df = pd.DataFrame(rows)
    out = os.path.join(config.RESULTS_DIR, f"{dataset}_decomposition.csv")
    df.to_csv(out, index=False)

    order = ["aggregate_accuracy"] + list(categories)
    pivot = (df.pivot(index="cell", columns="architecture", values="table_entry")
               .reindex(order)[archs])
    print(f"\n=== {dataset}: genuine between-seed variance (SNR), layout of Table 7 ===")
    print(df_to_markdown(pivot.reset_index()))
    agg = df[df["cell"] == "aggregate_accuracy"][["architecture", "mean", "cv_pct", "snr"]]
    print("\n=== aggregate accuracy: mean, CV %, SNR ===")
    print(df_to_markdown(agg))
    print(f"\nSaved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cse_cic_ids2018_draw2", choices=list(config.DATASETS))
    ap.add_argument("--architectures", default=",".join(FAST_FIRST))
    ap.add_argument("--seeds", default="full", help="'full' (0-39), 'pilot' (0-9), or a comma list.")
    ap.add_argument("--analyze", action="store_true", help="Run the decomposition on completed runs.")
    args = ap.parse_args()

    dataset = args.dataset
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    stem = os.path.join(config.RESULTS_DIR, f"{dataset}_matrix")
    summary_csv, per_instance_csv = f"{stem}_summary.csv", f"{stem}_per_instance.csv"

    if args.analyze:
        analyze(dataset, summary_csv, per_instance_csv)
        return

    archs = [a.strip() for a in args.architectures.split(",")]
    unknown = [a for a in archs if a not in config.ARCHITECTURES]
    if unknown:
        sys.exit(f"Unknown architecture(s): {unknown}. Known: {config.ARCHITECTURES}")
    seeds = (config.FULL_SEEDS if args.seeds == "full" else
             config.PILOT_SEEDS if args.seeds == "pilot" else
             [int(s) for s in args.seeds.split(",")])

    check_or_write_fingerprint(f"{stem}_fingerprint.json", config_fingerprint())
    run(dataset, archs, seeds, summary_csv, per_instance_csv)


if __name__ == "__main__":
    main()
