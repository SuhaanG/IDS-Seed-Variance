"""
LightGBM / XGBoost ablation for the IEEE Access revision.

Addresses: Reviewer 1 (#2 strengthen LightGBM analysis, #3 justify
hyperparameters), Reviewer 2 (#1 is the instability an artifact of
subsampling settings?), Reviewer 4 (LightGBM-specific hyperparameters),
Reviewer 5 / Review_Report item 2 (mechanism untested).

Every configuration starts from the EXACT submitted settings of its
library (mirrored from model.py through revision_common) and changes one
factor at a time, so any change in seed-instability is attributable to
that factor. The two libraries are also pushed toward each other:
XGBoost is run with leaf-wise growth (grow_policy="lossguide") and with
LightGBM's regularization defaults, which tests the paper's stated
leaf-wise-growth hypothesis from the opposite direction.

IMPORTANT DISCLOSURE THE ABLATION IS DESIGNED TO CONFIRM: in the
submitted LightGBM configuration, subsample=0.8 has NO effect, because
LightGBM only applies row subsampling when bagging_freq (sklearn name
subsample_freq) is > 0, and it was left at its default of 0. Only
colsample_bytree=0.8 (feature_fraction) was active, which is why the
same-seed-vs-different-seed determinism test still passed. The
"lgb_rowsub_only_freq0" configuration makes this explicit: it keeps
subsample=0.8 / bagging_freq=0 but sets colsample_bytree=1.0; if the seed
then becomes inert (identical predictions for every seed), row
subsampling was inactive in the paper's runs.

Usage:
    python run_ablation.py --dataset nsl_kdd --configs core      # fast subset
    python run_ablation.py --dataset nsl_kdd --configs all       # everything
    python run_ablation.py --dataset nsl_kdd --analyze           # tables
    python run_ablation.py --dataset synthetic --configs core --n-seeds 5   # smoke test only
    python run_ablation.py --list

Idempotent: already-completed (config, seed) cells are skipped.
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd

import config
from revision_common import (
    REVISION_SCHEMA_VERSION, DEFAULT_REVISION_BOOTSTRAP, SYNTHETIC_DATASET_NAME,
    build_estimator, fit_predict_estimator, dataset_categories, load_official_split,
    labels_to_idx, category_maps, summary_row, append_csv_row, completed_keys,
    correct_dir, save_correct_vector, load_correct_matrix, save_y_test, load_y_test,
    decompose_cell, describe_accuracies, write_json, df_to_markdown,
)

# ---------------------------------------------------------------------------
# Ablation grid. `params` are applied ON TOP of the paper's exact settings.
# ---------------------------------------------------------------------------
ABLATIONS = {
    # ---- LightGBM: single-factor changes from the submitted configuration ----
    "lgb_baseline": dict(
        library="lightgbm", params={},
        desc="Submitted configuration, unchanged (subsample=0.8 configured but inactive: bagging_freq=0)."),
    "lgb_rowsub_only_freq0": dict(
        library="lightgbm", params=dict(colsample_bytree=1.0),
        desc="Feature subsampling off, row subsampling left as submitted (0.8 with bagging_freq=0). Seed inert here => row subsampling was inactive in the paper."),
    "lgb_no_subsampling": dict(
        library="lightgbm", params=dict(subsample=1.0, colsample_bytree=1.0),
        desc="No row or feature subsampling. Isolates whether the seed enters only via subsampling."),
    "lgb_bagging_freq1": dict(
        library="lightgbm", params=dict(subsample_freq=1),
        desc="Row subsampling actually activated (bagging_freq=1), otherwise as submitted."),
    "lgb_l2_reg1": dict(
        library="lightgbm", params=dict(reg_lambda=1.0),
        desc="L2 leaf regularization set to XGBoost's default (lambda=1); LightGBM default is 0."),
    "lgb_min_child_weight1": dict(
        library="lightgbm", params=dict(min_child_weight=1.0),
        desc="min_sum_hessian_in_leaf set to XGBoost's min_child_weight default (1.0); LightGBM default is 1e-3."),
    "lgb_min_child_samples100": dict(
        library="lightgbm", params=dict(min_child_samples=100),
        desc="Minimum samples per leaf raised from 20 to 100 (suppresses tiny leaves on rare classes)."),
    "lgb_num_leaves15": dict(
        library="lightgbm", params=dict(num_leaves=15),
        desc="Leaf budget halved (31 -> 15)."),
    "lgb_num_leaves64": dict(
        library="lightgbm", params=dict(num_leaves=64),
        desc="Leaf budget raised to 2^max_depth (31 -> 64), i.e. leaf-wise growth can fill a full depth-6 tree."),
    "lgb_lr005_400rounds": dict(
        library="lightgbm", params=dict(learning_rate=0.05, n_estimators=400),
        desc="Half the learning rate, double the rounds (same total step budget)."),
    "lgb_xgb_matched_reg": dict(
        library="lightgbm", params=dict(reg_lambda=1.0, min_child_weight=1.0, subsample_freq=1),
        desc="LightGBM with XGBoost's regularization defaults AND row subsampling active."),

    # ---- XGBoost: pushed toward LightGBM one factor at a time ----
    "xgb_baseline": dict(
        library="xgboost", params={},
        desc="Submitted configuration, unchanged (depth-wise growth, lambda=1, min_child_weight=1)."),
    "xgb_lossguide": dict(
        library="xgboost", params=dict(grow_policy="lossguide", max_leaves=31, tree_method="hist"),
        desc="Leaf-wise growth with the same 31-leaf budget as LightGBM. Direct test of the growth-policy hypothesis."),
    "xgb_lambda0": dict(
        library="xgboost", params=dict(reg_lambda=0.0),
        desc="XGBoost with LightGBM's L2 default (lambda=0), depth-wise growth kept."),
    "xgb_lossguide_lambda0": dict(
        library="xgboost", params=dict(grow_policy="lossguide", max_leaves=31, tree_method="hist", reg_lambda=0.0),
        desc="Leaf-wise growth AND lambda=0."),
    "xgb_lightgbm_like": dict(
        library="xgboost", params=dict(grow_policy="lossguide", max_leaves=31, tree_method="hist",
                                       reg_lambda=0.0, min_child_weight=1e-3),
        desc="XGBoost made as LightGBM-like as its parameters allow: leaf-wise, lambda=0, min_child_weight=1e-3."),
}

CONFIG_GROUPS = {
    "all": list(ABLATIONS.keys()),
    "lgb": [k for k in ABLATIONS if k.startswith("lgb_")],
    "xgb": [k for k in ABLATIONS if k.startswith("xgb_")],
    "core": ["lgb_baseline", "lgb_rowsub_only_freq0", "lgb_bagging_freq1", "lgb_l2_reg1",
             "xgb_baseline", "xgb_lossguide", "xgb_lightgbm_like"],
}


def _paths(dataset):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return (
        os.path.join(config.RESULTS_DIR, f"ablation_{dataset}_summary.csv"),
        os.path.join(config.RESULTS_DIR, f"ablation_{dataset}_determinism.json"),
        os.path.join(config.RESULTS_DIR, f"ablation_{dataset}_analysis.csv"),
    )


def _resolve_configs(spec):
    names = []
    for token in spec.split(","):
        token = token.strip()
        if token in CONFIG_GROUPS:
            names.extend(CONFIG_GROUPS[token])
        elif token in ABLATIONS:
            names.append(token)
        else:
            raise SystemExit(f"Unknown config '{token}'. Use --list to see options.")
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def determinism_check(cfg_name, num_classes, X_train, y_train_idx, X_test):
    """Same seed twice must match; seeds 0 and 1 should differ if the seed is live."""
    spec = ABLATIONS[cfg_name]
    p0a, _ = fit_predict_estimator(build_estimator(spec["library"], num_classes, 0, spec["params"]),
                                   X_train, y_train_idx, X_test, 0)
    p0b, _ = fit_predict_estimator(build_estimator(spec["library"], num_classes, 0, spec["params"]),
                                   X_train, y_train_idx, X_test, 0)
    p1, _ = fit_predict_estimator(build_estimator(spec["library"], num_classes, 1, spec["params"]),
                                  X_train, y_train_idx, X_test, 1)
    return {
        "same_seed_identical": bool(np.array_equal(p0a, p0b)),
        "different_seed_identical": bool(np.array_equal(p0a, p1)),
        "n_predictions_differing_seed0_vs_seed1": int(np.sum(p0a != p1)),
    }


def run(dataset, config_names, seeds, verbose_first_seed=False, skip_determinism=False):
    summary_csv, determinism_json, _ = _paths(dataset)
    categories = dataset_categories(dataset)
    num_classes = len(categories)
    X_train, y_train, X_test, y_test, _ = load_official_split(dataset)
    y_train_idx = labels_to_idx(y_train, categories)
    y_test_idx = labels_to_idx(y_test, categories)
    _, idx_to_cat = category_maps(categories)

    print(f"[ablation] dataset={dataset} X_train={X_train.shape} X_test={X_test.shape} "
          f"configs={len(config_names)} seeds={len(seeds)}")

    determinism = {}
    if os.path.exists(determinism_json):
        with open(determinism_json) as f:
            determinism = json.load(f)

    done = completed_keys(summary_csv, ["config", "seed"])

    for cfg_name in config_names:
        spec = ABLATIONS[cfg_name]
        cdir = correct_dir("ablation", dataset, cfg_name)
        save_y_test(cdir, y_test)

        if not skip_determinism and cfg_name not in determinism:
            determinism[cfg_name] = determinism_check(cfg_name, num_classes, X_train, y_train_idx, X_test)
            write_json(determinism_json, determinism)
            print(f"  [{cfg_name}] determinism: {determinism[cfg_name]}")

        remaining = [s for s in seeds if (cfg_name, s) not in done]
        if not remaining:
            print(f"  [{cfg_name}] all {len(seeds)} seeds already done, skipping.")
            continue
        print(f"  [{cfg_name}] {spec['desc']}")
        print(f"  [{cfg_name}] running {len(remaining)} seed(s)...")

        for seed in remaining:
            params = dict(spec["params"])
            if verbose_first_seed and seed == remaining[0] and spec["library"] == "lightgbm":
                params["verbosity"] = 1
            est = build_estimator(spec["library"], num_classes, seed, params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                preds_idx, train_time = fit_predict_estimator(est, X_train, y_train_idx, X_test, seed)
            y_pred_labels = np.array([idx_to_cat[i] for i in preds_idx])
            row, acc = summary_row(
                {"schema_version": REVISION_SCHEMA_VERSION, "dataset": dataset,
                 "config": cfg_name, "library": spec["library"], "seed": seed},
                categories, y_test, y_pred_labels, preds_idx, y_test_idx, train_time)
            append_csv_row(summary_csv, row)
            save_correct_vector(cdir, seed, preds_idx == y_test_idx)
            print(f"    [{dataset} | {cfg_name} | seed {seed}] {train_time:.1f}s acc={acc:.4f}")

    print(f"\n[ablation] done. Summary: {summary_csv}\nRun with --analyze to build the tables.")


def analyze(dataset, n_bootstrap):
    summary_csv, determinism_json, analysis_csv = _paths(dataset)
    categories = dataset_categories(dataset)
    determinism = {}
    if os.path.exists(determinism_json):
        with open(determinism_json) as f:
            determinism = json.load(f)

    rows = []
    base = os.path.join(config.RESULTS_DIR, "ablation_correct", dataset)
    if not os.path.isdir(base):
        raise SystemExit(f"No ablation outputs found under {base}. Run the ablation first.")
    for cfg_name in ABLATIONS:
        cdir = os.path.join(base, cfg_name)
        if not os.path.isdir(cdir) or not os.path.exists(os.path.join(cdir, "y_test.npy")):
            continue
        y_test = load_y_test(cdir)
        seeds, correct = load_correct_matrix(cdir, len(y_test))
        if len(seeds) < 2:
            continue
        acc = correct.mean(axis=1)
        row = {"config": cfg_name, "library": ABLATIONS[cfg_name]["library"]}
        row.update(describe_accuracies(acc))
        det = determinism.get(cfg_name, {})
        row["same_seed_identical"] = det.get("same_seed_identical", "n/a")
        row["seed_inert"] = det.get("different_seed_identical", "n/a")
        row.update(decompose_cell(seeds, correct, y_test, categories, dataset, cfg_name,
                                  n_bootstrap=n_bootstrap, include_categories=True))
        row["description"] = ABLATIONS[cfg_name]["desc"]
        rows.append(row)
        print(f"  analyzed {cfg_name}: n={row['n_seeds']} mean={row['mean_acc']} CV={row['cv_pct']}% "
              f"range={row['range_pp']}pp SNR(agg)={row['agg_snr']}")

    df = pd.DataFrame(rows)
    df.to_csv(analysis_csv, index=False)

    main_cols = ["config", "n_seeds", "mean_acc", "std_acc", "cv_pct", "min_acc", "max_acc",
                 "range_pp", "agg_noise_var", "agg_genuine_var", "agg_snr", "same_seed_identical", "seed_inert"]
    print("\n=== Aggregate-accuracy seed instability by configuration ===")
    print(df_to_markdown(df[main_cols]))
    snr_cols = ["config"] + [f"{c}_snr" for c in categories]
    print("\n=== Per-category SNR by configuration ===")
    print(df_to_markdown(df[snr_cols]))
    print(f"\nSaved: {analysis_csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nsl_kdd",
                    help=f"One of {list(config.DATASETS)} or '{SYNTHETIC_DATASET_NAME}' (smoke test only).")
    ap.add_argument("--configs", default="core", help="Comma list of config names and/or groups: all, lgb, xgb, core.")
    ap.add_argument("--n-seeds", type=int, default=len(config.FULL_SEEDS))
    ap.add_argument("--analyze", action="store_true", help="Build analysis tables from existing outputs.")
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_REVISION_BOOTSTRAP)
    ap.add_argument("--verbose-first-seed", action="store_true", help="Un-suppress LightGBM warnings on the first seed of each config.")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in ABLATIONS.items():
            print(f"{k:28s} [{v['library']}] {v['desc']}")
        print("\nGroups:", {k: len(v) for k, v in CONFIG_GROUPS.items()})
        return
    if args.analyze:
        analyze(args.dataset, args.n_bootstrap)
        return
    seeds = list(config.FULL_SEEDS[:args.n_seeds])
    run(args.dataset, _resolve_configs(args.configs), seeds,
        verbose_first_seed=args.verbose_first_seed, skip_determinism=args.skip_determinism)


if __name__ == "__main__":
    main()
