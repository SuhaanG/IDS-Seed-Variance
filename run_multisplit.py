"""
Repeated train/test partition check for the IEEE Access revision.

Addresses Review_Report item 1 (MAJOR: "single fixed train/test partition
undercuts the central claim"), Reviewer 4 ("explain fixed split vs repeated
split") and Reviewer 5.

The submitted study measured seed variance on ONE fixed partition per
dataset. This script re-partitions the pooled (official train + official
test) data into K additional stratified splits with the same test fraction
as the official split, and re-runs the two architectures that carry the
paper's headline claims (LightGBM unstable, XGBoost stable) across N seeds
on each partition. Partition 0 is the official split, so the official
result is reproduced inside the same run for an apples-to-apples baseline.

Two quantities come out:
  1. within-partition seed variance (does LightGBM's instability, and
     XGBoost's stability, persist on every partition?), reported with the
     paper's own bootstrap decomposition;
  2. between-partition variance of the per-partition means (how big is
     partition variance relative to seed variance?).

Caveat that must go in the paper: pooling and re-splitting NSL-KDD removes
the deliberate train/test distribution shift of its official partition
(novel attack types in the test set), so absolute accuracies on
re-partitions 1..K are higher than on partition 0. The comparison of
interest is seed-instability WITHIN each partition, not accuracy ACROSS
partitions. Preprocessing (one-hot, scaling) is fit on each partition's
training rows only.

Usage:
    python run_multisplit.py --dataset nsl_kdd                      # 5 extra partitions, 20 seeds, lightgbm+xgboost
    python run_multisplit.py --dataset nsl_kdd --n-seeds 40
    python run_multisplit.py --dataset nsl_kdd --analyze
    python run_multisplit.py --dataset synthetic --n-partitions 2 --n-seeds 4   # smoke test only
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import config
from model import build_model
from revision_common import (
    REVISION_SCHEMA_VERSION, DEFAULT_REVISION_BOOTSTRAP, SYNTHETIC_DATASET_NAME,
    dataset_categories, load_pooled_raw, preprocess_partition, labels_to_idx, category_maps,
    reseed_everything, summary_row, append_csv_row, completed_keys, correct_dir,
    save_correct_vector, load_correct_matrix, save_y_test, load_y_test, decompose_cell,
    describe_accuracies, write_json, df_to_markdown,
)

MULTISPLIT_RNG_SEED = 880_000   # deliberately unrelated to model seeds and to the CIC subsample seed
DEFAULT_ARCHITECTURES = ["lightgbm", "xgboost"]


def _paths(dataset):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return (
        os.path.join(config.RESULTS_DIR, f"multisplit_{dataset}_summary.csv"),
        os.path.join(config.RESULTS_DIR, f"multisplit_{dataset}_partitions.json"),
        os.path.join(config.RESULTS_DIR, f"multisplit_{dataset}_analysis.csv"),
        os.path.join(config.RESULTS_DIR, f"multisplit_{dataset}_crosspartition.csv"),
    )


def make_partitions(y, n_official_train, n_partitions):
    """Returns list of (partition_id, split_seed, train_idx, test_idx)."""
    n_total = len(y)
    official = (0, None, np.arange(n_official_train), np.arange(n_official_train, n_total))
    parts = [official]
    test_fraction = (n_total - n_official_train) / n_total
    sss = StratifiedShuffleSplit(n_splits=n_partitions, test_size=test_fraction,
                                 random_state=MULTISPLIT_RNG_SEED)
    for k, (tr, te) in enumerate(sss.split(np.zeros(n_total), y), start=1):
        parts.append((k, MULTISPLIT_RNG_SEED + k, np.sort(tr), np.sort(te)))
    return parts


def run(dataset, architectures, n_partitions, seeds):
    summary_csv, partitions_json, _, _ = _paths(dataset)
    categories = dataset_categories(dataset)
    _, idx_to_cat = category_maps(categories)

    feats, y, n_official_train = load_pooled_raw(dataset)
    parts = make_partitions(y, n_official_train, n_partitions)
    print(f"[multisplit] dataset={dataset} pooled rows={len(y)} official train={n_official_train} "
          f"partitions={len(parts)} (0=official) archs={architectures} seeds={len(seeds)}")

    meta = {}
    done = completed_keys(summary_csv, ["partition_id", "architecture", "seed"])

    for pid, split_seed, tr_idx, te_idx in parts:
        y_train, y_test = y[tr_idx], y[te_idx]
        meta[pid] = {
            "split_seed": split_seed, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "test_counts": {c: int((y_test == c).sum()) for c in categories},
            "train_counts": {c: int((y_train == c).sum()) for c in categories},
        }
        write_json(partitions_json, meta)

        need = [(a, s) for a in architectures for s in seeds if (pid, a, s) not in done]
        if not need:
            print(f"  partition {pid}: all cells done, skipping.")
            continue

        X_train, X_test = preprocess_partition(feats, tr_idx, te_idx)
        y_train_idx = labels_to_idx(y_train, categories)
        y_test_idx = labels_to_idx(y_test, categories)
        print(f"  partition {pid} (split_seed={split_seed}): train={X_train.shape} test={X_test.shape}")

        for arch in architectures:
            cdir = correct_dir("multisplit", dataset, f"p{pid}_{arch}")
            save_y_test(cdir, y_test)
            for seed in seeds:
                if (pid, arch, seed) in done:
                    continue
                reseed_everything(seed)
                model = build_model(arch, input_dim=X_train.shape[1], num_classes=len(categories))
                import time
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    model.fit(X_train, y_train_idx, seed)
                    preds_idx = np.asarray(model.predict(X_test)).astype(int)
                train_time = time.time() - t0
                y_pred_labels = np.array([idx_to_cat[i] for i in preds_idx])
                row, acc = summary_row(
                    {"schema_version": REVISION_SCHEMA_VERSION, "dataset": dataset,
                     "partition_id": pid, "split_seed": split_seed if split_seed is not None else -1,
                     "architecture": arch, "seed": seed},
                    categories, y_test, y_pred_labels, preds_idx, y_test_idx, train_time)
                append_csv_row(summary_csv, row)
                save_correct_vector(cdir, seed, preds_idx == y_test_idx)
                print(f"    [{dataset} | p{pid} | {arch} | seed {seed}] {train_time:.1f}s acc={acc:.4f}")

    print(f"\n[multisplit] done. Summary: {summary_csv}\nRun with --analyze to build the tables.")


def analyze(dataset, architectures, n_bootstrap):
    _, partitions_json, analysis_csv, cross_csv = _paths(dataset)
    categories = dataset_categories(dataset)
    base = os.path.join(config.RESULTS_DIR, "multisplit_correct", dataset)
    if not os.path.isdir(base):
        raise SystemExit(f"No multisplit outputs under {base}. Run the experiment first.")

    rows = []
    for cell in sorted(os.listdir(base)):
        if not cell.startswith("p") or "_" not in cell:
            continue
        pid_str, arch = cell.split("_", 1)
        if arch not in architectures:
            continue
        cdir = os.path.join(base, cell)
        y_test = load_y_test(cdir)
        seeds, correct = load_correct_matrix(cdir, len(y_test))
        if len(seeds) < 2:
            continue
        row = {"partition_id": int(pid_str[1:]), "architecture": arch}
        row.update(describe_accuracies(correct.mean(axis=1)))
        row.update(decompose_cell(seeds, correct, y_test, categories, dataset, cell,
                                  n_bootstrap=n_bootstrap, include_categories=True))
        rows.append(row)
        print(f"  {cell}: n={row['n_seeds']} mean={row['mean_acc']} CV={row['cv_pct']}% "
              f"range={row['range_pp']}pp SNR(agg)={row['agg_snr']}")

    df = pd.DataFrame(rows).sort_values(["architecture", "partition_id"])
    df.to_csv(analysis_csv, index=False)

    cross_rows = []
    for arch, g in df.groupby("architecture"):
        cross_rows.append({
            "architecture": arch,
            "n_partitions": int(len(g)),
            "mean_within_partition_seed_std": round(float(g["std_acc"].mean()), 5),
            "std_of_partition_means": round(float(g["mean_acc"].std(ddof=1)), 5) if len(g) > 1 else float("nan"),
            "min_partition_cv_pct": float(g["cv_pct"].min()),
            "max_partition_cv_pct": float(g["cv_pct"].max()),
            "min_partition_agg_snr": _min_numeric(g["agg_snr"]),
            "max_partition_agg_snr": _max_numeric(g["agg_snr"]),
            "official_partition_cv_pct": float(g.loc[g["partition_id"] == 0, "cv_pct"].iloc[0]) if (g["partition_id"] == 0).any() else float("nan"),
        })
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(cross_csv, index=False)

    print("\n=== Per-partition seed instability (aggregate accuracy) ===")
    print(df_to_markdown(df[["architecture", "partition_id", "n_seeds", "mean_acc", "std_acc", "cv_pct",
                             "range_pp", "agg_noise_var", "agg_genuine_var", "agg_snr"]]))
    print("\n=== Cross-partition summary ===")
    print(df_to_markdown(cross))
    print(f"\nSaved: {analysis_csv}\nSaved: {cross_csv}")
    return df, cross


def _min_numeric(series):
    vals = [float(v) for v in series if isinstance(v, (int, float)) and np.isfinite(v)]
    return min(vals) if vals else float("nan")


def _max_numeric(series):
    vals = [float(v) for v in series if isinstance(v, (int, float)) and np.isfinite(v)]
    return max(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nsl_kdd",
                    help=f"One of {list(config.DATASETS)} or '{SYNTHETIC_DATASET_NAME}' (smoke test only).")
    ap.add_argument("--architectures", default=",".join(DEFAULT_ARCHITECTURES),
                    help="Comma list from model.ARCHITECTURE_REGISTRY. Default lightgbm,xgboost.")
    ap.add_argument("--n-partitions", type=int, default=5, help="Extra stratified re-partitions (plus the official one).")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_REVISION_BOOTSTRAP)
    args = ap.parse_args()

    archs = [a.strip() for a in args.architectures.split(",") if a.strip()]
    if args.analyze:
        analyze(args.dataset, archs, args.n_bootstrap)
        return
    run(args.dataset, archs, args.n_partitions, list(config.FULL_SEEDS[:args.n_seeds]))


if __name__ == "__main__":
    main()
