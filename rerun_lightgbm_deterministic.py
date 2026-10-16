"""
Deterministic 40-seed LightGBM re-run with per-seed fit diagnostics (revision,
round 2: Reviewer 1 items 2, 3 and 9).

WHY THIS EXISTS
---------------
The submitted LightGBM column was trained with n_jobs=-1 and without
LightGBM's reproducibility switch. The repeated-run diagnostic
(check_lightgbm_reproducibility.py, Table 4) showed that under those settings
the SAME seed gives a different model on every fit (6 distinct prediction
vectors from 6 fits of seed 0 on NSL-KDD, a 39 pp accuracy range). Part of
the submitted "seed variance" is therefore run-to-run nondeterminism that the
seed does not control.

  item 2: rerun LightGBM with deterministic=True and every other setting
          unchanged, so the reported spread is attributable to the seed.
  item 3: show whether the very low-accuracy seeds (0.17 on the second
          CSE-CIC-IDS2018 draw, ~0.35 to 0.44 on NSL-KDD) are failed fits.
          Every seed records its training multi-logloss at rounds 1, 10, 50,
          100 and 200, its training accuracy, the largest absolute leaf
          value, and how many classes it predicts on the test set.
  item 9: rebuild the decomposition table with the paper's estimator at
          5,000 bootstrap iterations and SNR to three significant figures.

WHAT IS CHANGED, AND NOTHING ELSE
---------------------------------
The estimator gets exactly the paper's kwargs (revision_common.
lightgbm_paper_params, asserted against model.py) plus
deterministic=True and force_row_wise=True, the pair LightGBM documents
for reproducible results. Training goes through the paper's own
train.train_one_run, so reseeding, metrics and CSV formats are identical to
the submitted matrix. Nothing in config.py / model.py / train.py is edited:
the deterministic class is registered into model.ARCHITECTURE_REGISTRY at
import time under the name "lightgbm_deterministic".

Determinism under LightGBM's flag holds for a FIXED thread count. The
thread count (OMP_NUM_THREADS) is part of the configuration fingerprint and
must be reported with the results.

SAFETY
------
  * Determinism gate: before any seed is run, seed 0 is fitted twice and the
    script aborts unless the two predicted probability matrices are
    bit-identical.
  * Fingerprint guard (same as rerun_logistic_regression.py): resuming into
    output files created under a different configuration, library version
    or thread count aborts instead of mixing two configurations in one
    seed column.
  * Commit marker: a seed counts as done only once its diagnostics row is
    written (the last of the three writes). A seed interrupted between
    writes is removed from the summary and per-instance files and rerun.

OUTPUTS (results/)
------------------
  <ds>_lgb_det.csv                 summary rows, train.py format
  <ds>_lgb_det_per_instance.csv    per-instance predictions, train.py format
  <ds>_lgb_det_diagnostics.csv     per-seed fit diagnostics (item 3)
  <ds>_lgb_det_fingerprint.json    configuration + versions + threads
  <ds>_lgb_det_gate.json           determinism gate result
  <ds>_corrected_decomposition_lgbdet.csv   (from --analyze, item 9)

Usage:
    python rerun_lightgbm_deterministic.py --dataset nsl_kdd
    python rerun_lightgbm_deterministic.py --dataset nsl_kdd --analyze
    python rerun_lightgbm_deterministic.py --dataset synthetic --seeds 0,1,2   # smoke test
"""

import os
import sys
import json
import time
import hashlib
import argparse
import platform

import numpy as np
import pandas as pd

import config
import model as model_module
from train import train_one_run
from revision_common import (
    SYNTHETIC_DATASET_NAME, lightgbm_paper_params, dataset_categories,
    load_official_split, labels_to_idx, frames_from_correct_matrix,
    reseed_everything, df_to_markdown,
)
from stats_analysis import true_variance_decomposition

ARCHITECTURE = "lightgbm_deterministic"
DETERMINISM_KWARGS = {"deterministic": True, "force_row_wise": True}
LOSS_CHECKPOINTS = [1, 10, 50, 100, 200]
SUBMITTED_LGB_LABEL = "lightgbm_submitted_nondeterministic"
ARCH_ORDER = ["dnn", "shallow_mlp", "logistic_regression", "random_forest",
              "xgboost", SUBMITTED_LGB_LABEL, ARCHITECTURE]


# ---------------------------------------------------------------------------
# Model: the paper's LightGBM plus the reproducibility switch
# ---------------------------------------------------------------------------

def deterministic_params(num_classes, seed):
    params = lightgbm_paper_params(num_classes, seed)
    params.update(DETERMINISM_KWARGS)
    return params


def _multiclass_logloss(proba, y_idx):
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def _max_abs_leaf(booster, num_classes):
    """(max |leaf| over all trees, max |leaf| excluding round 1). Round 1's
    trees carry the boost_from_average initial score, so their leaves are
    large and nearly seed-independent; the second number is the one that
    shows whether later rounds blew up."""
    overall, after_r1 = 0.0, 0.0
    for i, tree in enumerate(booster.dump_model()["tree_info"]):
        stack, best = [tree["tree_structure"]], 0.0
        while stack:
            node = stack.pop()
            if "leaf_value" in node:
                best = max(best, abs(float(node["leaf_value"])))
            else:
                stack.append(node["left_child"])
                stack.append(node["right_child"])
        overall = max(overall, best)
        if i >= num_classes:
            after_r1 = max(after_r1, best)
    return overall, after_r1


class LightGBMDeterministicModel:
    """Same interface as model.LightGBMModel. Diagnostics of the most recent
    fit are kept on the class so the runner can read them after
    train_one_run returns (train_one_run does not return the model)."""

    name = ARCHITECTURE
    last_diagnostics = None

    def __init__(self, input_dim=None, num_classes=None):
        self.model = None
        self.num_classes = num_classes
        self.epochs_run = None

    def fit(self, X_train, y_train_idx, seed):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(**deterministic_params(self.num_classes, seed))
        self.model.fit(X_train, y_train_idx)
        LightGBMDeterministicModel.last_diagnostics = self._diagnostics(X_train, y_train_idx)
        return self

    def predict(self, X_test):
        preds = np.asarray(self.model.predict(X_test)).astype(int)
        d = LightGBMDeterministicModel.last_diagnostics
        if d is not None:
            counts = np.bincount(preds, minlength=self.num_classes)
            d["test_n_predicted_classes"] = int((counts > 0).sum())
            d["test_largest_predicted_class_share"] = float(counts.max() / len(preds))
        return preds

    def _diagnostics(self, X_train, y_train_idx):
        booster = self.model.booster_
        n_iter = booster.current_iteration()
        d = {"n_trees_per_class": n_iter}
        losses = {}
        for k in LOSS_CHECKPOINTS:
            if k <= n_iter:
                proba = booster.predict(X_train, num_iteration=k)
                losses[k] = _multiclass_logloss(np.asarray(proba), y_train_idx)
                d[f"train_logloss_r{k}"] = losses[k]
            else:
                d[f"train_logloss_r{k}"] = np.nan
        final = np.asarray(booster.predict(X_train))
        d["train_accuracy"] = float(np.mean(final.argmax(axis=1) == y_train_idx))
        d["train_logloss_final"] = _multiclass_logloss(final, y_train_idx)
        best_k = min(losses, key=losses.get)
        d["best_logloss_checkpoint"] = best_k
        d["logloss_rose_after_best"] = bool(d["train_logloss_final"] > losses[best_k] + 1e-9)
        d["max_abs_leaf_value"], d["max_abs_leaf_after_r1"] = _max_abs_leaf(booster, self.num_classes)
        return d


model_module.ARCHITECTURE_REGISTRY[ARCHITECTURE] = LightGBMDeterministicModel


# ---------------------------------------------------------------------------
# Fingerprint guard
# ---------------------------------------------------------------------------

def run_fingerprint(dataset, num_classes):
    import lightgbm, sklearn, scipy
    params = deterministic_params(num_classes, seed=0)
    params.pop("random_state")
    return {
        "architecture": ARCHITECTURE,
        "dataset": dataset,
        "lightgbm_params": json.loads(json.dumps(params, default=str)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "results_schema_version": config.RESULTS_SCHEMA_VERSION,
        "lightgbm": lightgbm.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
    }


def check_or_write_fingerprint(path, current):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        return
    with open(path) as f:
        recorded = json.load(f)
    diffs = {k: (recorded.get(k), current.get(k))
             for k in set(recorded) | set(current) if recorded.get(k) != current.get(k)}
    if diffs:
        print("\nABORTING: these output files were created under a different "
              "configuration than the one running now.", file=sys.stderr)
        for k, (was, now) in sorted(diffs.items()):
            print(f"  {k}: recorded={was!r} current={now!r}", file=sys.stderr)
        print("\nResuming would mix two configurations inside one seed column. "
              "Restore the recorded setting (for threads: export "
              "OMP_NUM_THREADS=<recorded>) or delete the output files and start clean.",
              file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Determinism gate
# ---------------------------------------------------------------------------

def determinism_gate(X_train, y_train_idx, X_test, num_classes, gate_path):
    import lightgbm as lgb
    probas, hashes, times = [], [], []
    for rep in range(2):
        reseed_everything(0)
        est = lgb.LGBMClassifier(**deterministic_params(num_classes, 0))
        t0 = time.time()
        est.fit(X_train, y_train_idx)
        times.append(round(time.time() - t0, 2))
        p = np.asarray(est.predict_proba(X_test))
        probas.append(p)
        hashes.append(hashlib.sha1(np.ascontiguousarray(p).tobytes()).hexdigest()[:16])
    identical = bool(np.array_equal(probas[0], probas[1]))
    result = {"seed": 0, "fits": 2, "proba_hashes": hashes, "bit_identical": identical,
              "fit_seconds": times, "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    with open(gate_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[gate] seed 0 fitted twice: hashes {hashes[0]} / {hashes[1]} "
          f"-> {'bit-identical' if identical else 'DIFFERENT'} ({times[0]}s, {times[1]}s)")
    if not identical:
        print("\nABORTING: deterministic=True did not give bit-identical predictions for a "
              "repeated seed on this machine. Do not run the 40 seeds; the column would "
              "still contain run-to-run noise. Check that OMP_NUM_THREADS is fixed and "
              "that no other process changes the thread pool.", file=sys.stderr)
        sys.exit(3)


# ---------------------------------------------------------------------------
# Resume bookkeeping (diagnostics row is the commit marker)
# ---------------------------------------------------------------------------

def _seeds_in(path, dataset):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path, usecols=["dataset", "seed"])
    return set(int(s) for s in df.loc[df["dataset"] == dataset, "seed"].unique())


def _drop_seeds(path, seeds, chunksize=1_000_000):
    if not os.path.exists(path) or not seeds:
        return
    tmp = path + ".tmp"
    first = True
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, keep_default_na=False):
        chunk = chunk[~chunk["seed"].astype(int).isin(seeds)]
        chunk.to_csv(tmp, mode="w" if first else "a", header=first, index=False)
        first = False
    os.replace(tmp, path)


def reconcile(paths, dataset):
    committed = _seeds_in(paths["diagnostics"], dataset)
    orphans = (_seeds_in(paths["summary"], dataset) | _seeds_in(paths["per_instance"], dataset)) - committed
    if orphans:
        print(f"Seed(s) {sorted(orphans)} were interrupted before their diagnostics row was "
              f"written; removing their partial rows and rerunning them.")
        _drop_seeds(paths["summary"], orphans)
        _drop_seeds(paths["per_instance"], orphans)
    return committed


def output_paths(dataset):
    stem = os.path.join(config.RESULTS_DIR, f"{dataset}_lgb_det")
    return {"summary": f"{stem}.csv", "per_instance": f"{stem}_per_instance.csv",
            "diagnostics": f"{stem}_diagnostics.csv", "fingerprint": f"{stem}_fingerprint.json",
            "gate": f"{stem}_gate.json"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(dataset, seeds):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    paths = output_paths(dataset)
    categories = dataset_categories(dataset)
    num_classes = len(categories)

    check_or_write_fingerprint(paths["fingerprint"], run_fingerprint(dataset, num_classes))

    print(f"Loading {dataset} ...")
    X_train, y_train, X_test, y_test, _ = load_official_split(dataset)
    y_train_idx = labels_to_idx(y_train, categories)
    print(f"X_train {X_train.shape}  X_test {X_test.shape}  {num_classes} classes  "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print("LightGBM kwargs:", {k: v for k, v in deterministic_params(num_classes, 0).items()
                               if k != "random_state"})

    done = reconcile(paths, dataset)
    todo = [s for s in seeds if s not in done]
    if done:
        print(f"{len(done)} seed(s) already complete under this configuration: {sorted(done)}")
    if not todo:
        print("Nothing left to run.")
        report(dataset)
        return

    determinism_gate(X_train, y_train_idx, X_test, num_classes, paths["gate"])

    print(f"Running {len(todo)} seed(s): {todo}\n")
    for n, seed in enumerate(todo, 1):
        LightGBMDeterministicModel.last_diagnostics = None
        acc, _ = train_one_run(architecture=ARCHITECTURE, seed=seed,
                               X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                               categories=categories,
                               results_csv_path=paths["summary"],
                               per_instance_csv_path=paths["per_instance"],
                               dataset_name=dataset)
        diag = LightGBMDeterministicModel.last_diagnostics
        if diag is None:
            raise RuntimeError(f"seed {seed}: no diagnostics captured; stopping.")
        row = {"dataset": dataset, "architecture": ARCHITECTURE, "seed": seed,
               "test_accuracy": acc, **diag}
        pd.DataFrame([row]).to_csv(paths["diagnostics"], mode="a",
                                   header=not os.path.exists(paths["diagnostics"]), index=False)
        print(f"    ({n}/{len(todo)}) train_acc={diag['train_accuracy']:.4f} "
              f"logloss r1={diag['train_logloss_r1']:.4f} r200={diag['train_logloss_r200']:.4f} "
              f"max|leaf| after r1={diag['max_abs_leaf_after_r1']:.3g} "
              f"test classes predicted={diag['test_n_predicted_classes']}")

    report(dataset)
    print(f"\nSaved: {paths['summary']}\n       {paths['per_instance']}\n       {paths['diagnostics']}")


def report(dataset):
    paths = output_paths(dataset)
    if not os.path.exists(paths["diagnostics"]):
        return
    d = pd.read_csv(paths["diagnostics"]).sort_values("test_accuracy")
    acc = d["test_accuracy"]
    print(f"\n{'='*78}\nDETERMINISTIC LIGHTGBM: {dataset}\n{'='*78}")
    print(f"seeds: {len(d)}   distinct accuracies: {acc.nunique()}   mean {acc.mean():.4f}   "
          f"std {acc.std(ddof=1):.4f}   min {acc.min():.4f}   max {acc.max():.4f}")
    cols = ["seed", "test_accuracy", "train_accuracy"] + \
           [f"train_logloss_r{k}" for k in LOSS_CHECKPOINTS] + \
           ["logloss_rose_after_best", "max_abs_leaf_after_r1", "test_n_predicted_classes",
            "test_largest_predicted_class_share"]
    shown = d if len(d) <= 11 else pd.concat([d.head(8), d.tail(3)])
    print("\nSeeds by test accuracy" + ("" if len(d) <= 11 else " (worst 8, best 3)") + ":")
    print(df_to_markdown(shown[cols], floatfmt=4))
    rho = d[["test_accuracy", "train_accuracy", "train_logloss_r200", "max_abs_leaf_after_r1"]] \
        .corr(method="spearman")["test_accuracy"]
    print("\nSpearman correlation with test accuracy: "
          + ", ".join(f"{k}={v:.2f}" for k, v in rho.items() if k != "test_accuracy"))
    print("Reading (item 3): a failed fit shows a training log-loss that rises or stays high, "
          "low training accuracy, or exploding leaf values. Low test accuracy WITH a normal "
          "training curve and high training accuracy is a fit that converged to a solution "
          "that generalizes poorly to the test distribution, not a failed fit.")


# ---------------------------------------------------------------------------
# Analyze: corrected decomposition table (item 9)
# ---------------------------------------------------------------------------

def load_per_instance_matrices(path, architectures, chunksize=2_000_000):
    """{arch: (seeds, bool matrix [n_seeds, n_test])} and y_test (str array)."""
    store, labels = {}, {}
    usecols = ["architecture", "seed", "instance_id", "true_category", "correct"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                             dtype={"architecture": str, "seed": np.int32, "instance_id": np.int64,
                                    "true_category": str, "correct": np.int8}):
        chunk = chunk[chunk["architecture"].isin(architectures)]
        if chunk.empty:
            continue
        lab = chunk.drop_duplicates("instance_id")
        labels.update(zip(lab["instance_id"].values, lab["true_category"].values))
        for (arch, seed), g in chunk.groupby(["architecture", "seed"]):
            store.setdefault(arch, {}).setdefault(int(seed), []).append(
                (g["instance_id"].values, g["correct"].values.astype(bool)))
    if not labels:
        return {}, None
    n_test = max(labels) + 1
    if len(labels) != n_test:
        raise ValueError(f"{path}: instance ids are not contiguous 0..{n_test - 1}")
    y_test = np.array([labels[i] for i in range(n_test)])
    out = {}
    for arch, by_seed in store.items():
        seeds = np.array(sorted(by_seed))
        mat = np.zeros((len(seeds), n_test), dtype=bool)
        for i, s in enumerate(seeds):
            filled = 0
            for ids, corr in by_seed[s]:
                mat[i, ids] = corr
                filled += len(ids)
            if filled != n_test:
                raise ValueError(f"{path}: {arch} seed {s} has {filled} rows, expected {n_test}")
        out[arch] = (seeds, mat)
    return out, y_test


def sig3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    if x == float("inf"):
        return "inf"
    return f"{float(x):.3g}"


def decompose_all(arch_label, seeds, correct, y_test, categories, dataset, n_bootstrap):
    summary_df, per_instance_df = frames_from_correct_matrix(seeds, correct, y_test, categories,
                                                             dataset, arch_label)
    rows = []
    for cat in [None] + list(categories):
        if cat is not None and not (y_test == cat).any():
            continue
        d = true_variance_decomposition(summary_df, per_instance_df, cat, dataset, arch_label,
                                        n_bootstrap=n_bootstrap)
        metric = summary_df["aggregate_accuracy" if cat is None else f"{cat}_recall"]
        mean, std = float(metric.mean()), float(metric.std(ddof=1))
        snr = d["signal_to_noise_ratio"]
        snr_txt = "degenerate" if d["is_degenerate_zero_variance"] else sig3(snr)
        rows.append({
            "dataset": dataset, "architecture": arch_label, "cell": d["category"],
            "n_seeds": d["n_seeds"], "mean": mean, "std": std,
            "cv_pct": std / mean * 100 if mean > 0 else np.nan,
            "observed_var": d["observed_variance_of_point_estimates"],
            "sampling_noise_var": d["mean_within_seed_sampling_variance"],
            "genuine_var": d["genuine_between_seed_variance"],
            "snr": snr, "snr_3sf": snr_txt,
            "degenerate": d["is_degenerate_zero_variance"],
            "table_entry": f"{sig3(d['genuine_between_seed_variance'])} ({snr_txt})",
        })
    return rows


def analyze(dataset, n_bootstrap, matrix_csv=None):
    categories = dataset_categories(dataset)
    paths = output_paths(dataset)
    matrix_csv = matrix_csv or os.path.join(config.RESULTS_DIR, f"{dataset}_matrix_per_instance.csv")
    logreg_clean = os.path.join(config.RESULTS_DIR, f"{dataset}_logreg_clean_per_instance.csv")
    sources = []

    print(f"[analyze] {dataset}, {n_bootstrap} bootstrap iterations (paper: {config.BOOTSTRAP_ITERATIONS})")
    y_ref = None
    cells = {}

    if os.path.exists(matrix_csv):
        others = ["dnn", "shallow_mlp", "random_forest", "xgboost", "lightgbm", "logistic_regression"]
        mats, y_ref = load_per_instance_matrices(matrix_csv, others)
        for a, v in mats.items():
            label = SUBMITTED_LGB_LABEL if a == "lightgbm" else a
            cells[label] = v
            sources.append((label, matrix_csv))
    else:
        print(f"  matrix per-instance file not found ({matrix_csv}); table will contain the "
              f"deterministic LightGBM column only.")

    if os.path.exists(logreg_clean):
        mats, y_lr = load_per_instance_matrices(logreg_clean, ["logistic_regression"])
        if "logistic_regression" in mats:
            if y_ref is not None and not np.array_equal(y_ref, y_lr):
                raise ValueError("clean logistic regression file has a different test set")
            y_ref = y_lr if y_ref is None else y_ref
            cells["logistic_regression"] = mats["logistic_regression"]
            sources = [s for s in sources if s[0] != "logistic_regression"]
            sources.append(("logistic_regression", logreg_clean))

    if not os.path.exists(paths["per_instance"]):
        raise SystemExit(f"{paths['per_instance']} not found; run the 40 seeds first.")
    mats, y_det = load_per_instance_matrices(paths["per_instance"], [ARCHITECTURE])
    if y_ref is not None and not np.array_equal(y_ref, y_det):
        raise ValueError("deterministic LightGBM file has a different test set than the matrix")
    y_ref = y_det
    cells[ARCHITECTURE] = mats[ARCHITECTURE]
    sources.append((ARCHITECTURE, paths["per_instance"]))

    for label, src in sources:
        print(f"  {label:38s} <- {os.path.basename(src)}  ({len(cells[label][0])} seeds)")

    order = [a for a in ARCH_ORDER if a in cells]

    rows = []
    for label in order:
        seeds, correct = cells[label]
        t0 = time.time()
        rows.extend(decompose_all(label, seeds, correct, y_ref, categories, dataset, n_bootstrap))
        print(f"  decomposed {label} in {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    out = os.path.join(config.RESULTS_DIR, f"{dataset}_corrected_decomposition_lgbdet.csv")
    df.to_csv(out, index=False)

    wide = df.pivot(index="architecture", columns="cell", values="table_entry")
    wide = wide.reindex([a for a in order if a in wide.index])
    wide = wide[["aggregate_accuracy"] + [c for c in categories if c in wide.columns]]
    print("\n=== Genuine between-seed variance (SNR, 3 s.f.) ===")
    print(df_to_markdown(wide.reset_index()))
    agg = df[df["cell"] == "aggregate_accuracy"][["architecture", "n_seeds", "mean", "std", "cv_pct", "snr_3sf"]]
    print("\n=== Aggregate accuracy ===")
    print(df_to_markdown(agg, floatfmt=4))
    print(f"\nSaved: {out}")
    report(dataset)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=list(config.DATASETS) + [SYNTHETIC_DATASET_NAME])
    ap.add_argument("--seeds", default="full", help="'full' (0-39), 'pilot' (0-9), or a comma list.")
    ap.add_argument("--analyze", action="store_true",
                    help="Build the corrected decomposition table instead of training.")
    ap.add_argument("--n-bootstrap", type=int, default=config.BOOTSTRAP_ITERATIONS)
    ap.add_argument("--matrix-per-instance-csv", default=None,
                    help="Override the per-instance file used for the other architectures.")
    args = ap.parse_args()

    if args.analyze:
        analyze(args.dataset, args.n_bootstrap, args.matrix_per_instance_csv)
        return
    if args.seeds == "full":
        seeds = config.FULL_SEEDS
    elif args.seeds == "pilot":
        seeds = config.PILOT_SEEDS
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    run(args.dataset, seeds)


if __name__ == "__main__":
    main()
