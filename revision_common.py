"""
Shared helpers for the IEEE Access revision experiments.

Everything in this module is ADDITIVE. Nothing that produced the submitted
results (config.py, data.py, model.py, train.py, run_matrix.py,
stats_analysis.py) is modified by the revision scripts, so the original
720-run matrix remains exactly reproducible from the committed code.

Provides:
  - exact copies of the paper's LightGBM / XGBoost constructor arguments
    (mirrored from model.py, asserted against config.py) so ablations start
    from the submitted configuration and vary one thing at a time;
  - compact on-disk storage of per-seed correctness vectors (bit-packed),
    so the paper's own bootstrap decomposition can be re-run on ablation /
    multi-split outputs without multi-hundred-MB per-instance CSVs;
  - a wrapper that feeds those vectors through the UNMODIFIED
    stats_analysis.true_variance_decomposition, so every SNR reported by
    the revision scripts is computed by the same function as the paper;
  - pooled raw loading + train-only-fit preprocessing per dataset, for the
    repeated-partition check;
  - a per-row train/test overlap mask that replicates the hashing used by
    data._audit_train_test_overlap, for the overlap-split analysis;
  - a small synthetic imbalanced multiclass dataset for smoke-testing the
    scripts on a machine that does not have the real data (never report
    numbers from it).
"""

import os
import json
import random
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import config
import data as data_module
from model import set_full_determinism
from stats_analysis import true_variance_decomposition
from train import compute_per_category_metrics

REVISION_SCHEMA_VERSION = "rev1.0"
SYNTHETIC_DATASET_NAME = "synthetic"
SYNTHETIC_CATEGORIES = ["c0", "c1", "c2", "c3", "c4"]

# Default bootstrap iterations for the revision analyses. The paper used
# config.BOOTSTRAP_ITERATIONS (5000). The revision scripts default to 1000
# because they decompose many more (config x seed) cells; pass
# --n-bootstrap 5000 to match the paper exactly. The within-seed variance
# estimate is already stable at 1000 for the test-set sizes involved.
DEFAULT_REVISION_BOOTSTRAP = 1000


# ---------------------------------------------------------------------------
# Paper-exact base constructor arguments (mirrored from model.py)
# ---------------------------------------------------------------------------

def lightgbm_paper_params(num_classes, seed):
    """Exactly the kwargs LightGBMModel.fit passes in model.py."""
    return dict(
        n_estimators=config.LGB_N_ESTIMATORS,
        max_depth=config.LGB_MAX_DEPTH,
        learning_rate=config.LGB_LEARNING_RATE,
        subsample=config.LGB_SUBSAMPLE,
        colsample_bytree=config.LGB_COLSAMPLE_BYTREE,
        objective="multiclass",
        num_class=num_classes,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def xgboost_paper_params(num_classes, seed):
    """Exactly the kwargs XGBoostModel.fit passes in model.py."""
    return dict(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        subsample=config.XGB_SUBSAMPLE,
        colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
        objective="multi:softmax",
        num_class=num_classes,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


def build_estimator(library, num_classes, seed, overrides):
    if library == "lightgbm":
        import lightgbm as lgb
        params = lightgbm_paper_params(num_classes, seed)
        params.update(overrides)
        return lgb.LGBMClassifier(**params)
    if library == "xgboost":
        import xgboost as xgb
        params = xgboost_paper_params(num_classes, seed)
        params.update(overrides)
        return xgb.XGBClassifier(**params)
    raise ValueError(f"Unknown library '{library}'")


def reseed_everything(seed):
    """Same reseeding train.train_one_run performs before every run."""
    set_full_determinism(seed)
    random.seed(seed)
    np.random.seed(seed)


def fit_predict_estimator(estimator, X_train, y_train_idx, X_test, seed):
    reseed_everything(seed)
    start = time.time()
    estimator.fit(X_train, y_train_idx)
    train_time = time.time() - start
    preds = np.asarray(estimator.predict(X_test)).astype(int)
    return preds, train_time


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

def category_maps(categories):
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    idx_to_cat = {i: c for c, i in cat_to_idx.items()}
    return cat_to_idx, idx_to_cat


def labels_to_idx(y, categories):
    cat_to_idx, _ = category_maps(categories)
    return np.array([cat_to_idx[c] for c in y])


def summary_row(prefix_fields, categories, y_test, y_pred_labels, preds_idx, y_test_idx, train_time):
    aggregate_accuracy = float(np.mean(preds_idx == y_test_idx))
    per_category = compute_per_category_metrics(y_test, y_pred_labels, categories)
    row = dict(prefix_fields)
    row["aggregate_accuracy"] = aggregate_accuracy
    row["train_time_sec"] = round(train_time, 2)
    for cat in categories:
        row[f"{cat}_recall"] = per_category[cat]["recall"]
        row[f"{cat}_precision"] = per_category[cat]["precision"]
        row[f"{cat}_support"] = per_category[cat]["support"]
    return row, aggregate_accuracy


def append_csv_row(path, row):
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def completed_keys(path, key_cols):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if not all(c in df.columns for c in key_cols):
        return set()
    return set(tuple(r) for r in df[key_cols].itertuples(index=False, name=None))


# ---------------------------------------------------------------------------
# Compact per-seed correctness storage
# ---------------------------------------------------------------------------

def correct_dir(kind, dataset, cell):
    d = os.path.join(config.RESULTS_DIR, f"{kind}_correct", dataset, cell)
    os.makedirs(d, exist_ok=True)
    return d


def save_correct_vector(directory, seed, correct_bool):
    packed = np.packbits(np.asarray(correct_bool, dtype=bool))
    np.save(os.path.join(directory, f"seed_{seed}.npy"), packed)


def load_correct_matrix(directory, n_test):
    """Returns (seeds sorted ascending, bool matrix [n_seeds, n_test])."""
    files = [f for f in os.listdir(directory) if f.startswith("seed_") and f.endswith(".npy")]
    seeds = sorted(int(f[len("seed_"):-len(".npy")]) for f in files)
    rows = []
    for s in seeds:
        packed = np.load(os.path.join(directory, f"seed_{s}.npy"))
        rows.append(np.unpackbits(packed)[:n_test].astype(bool))
    if not rows:
        return np.array([], dtype=int), np.zeros((0, n_test), dtype=bool)
    return np.array(seeds), np.vstack(rows)


def save_y_test(directory, y_test):
    np.save(os.path.join(directory, "y_test.npy"), np.asarray(y_test).astype(str))


def load_y_test(directory):
    return np.load(os.path.join(directory, "y_test.npy"), allow_pickle=False).astype(str)


# ---------------------------------------------------------------------------
# Decomposition through the paper's unmodified function
# ---------------------------------------------------------------------------

def frames_from_correct_matrix(seeds, correct_matrix, y_test, categories, dataset_tag, arch_tag):
    """
    Rebuilds the two DataFrames stats_analysis.true_variance_decomposition
    expects (summary + per-instance) from a bool correctness matrix, so the
    revision analyses use the identical decomposition code path as the paper.
    """
    y_test = np.asarray(y_test).astype(str)
    n_seeds, n_test = correct_matrix.shape
    summary_rows = []
    for i, seed in enumerate(seeds):
        row = {"dataset": dataset_tag, "architecture": arch_tag, "seed": int(seed),
               "aggregate_accuracy": float(correct_matrix[i].mean())}
        for cat in categories:
            mask = y_test == cat
            row[f"{cat}_recall"] = float(correct_matrix[i][mask].mean()) if mask.any() else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    per_instance_df = pd.DataFrame({
        "dataset": dataset_tag,
        "architecture": arch_tag,
        "seed": np.repeat(np.asarray(seeds, dtype=int), n_test),
        "true_category": np.tile(y_test, n_seeds),
        "correct": correct_matrix.reshape(-1).astype(int),
    })
    return summary_df, per_instance_df


def decompose_cell(seeds, correct_matrix, y_test, categories, dataset_tag, arch_tag,
                   n_bootstrap=DEFAULT_REVISION_BOOTSTRAP, include_categories=True):
    """
    Runs the paper's true_variance_decomposition for aggregate accuracy and
    (optionally) every category. Returns a flat dict of results.
    """
    summary_df, per_instance_df = frames_from_correct_matrix(
        seeds, correct_matrix, y_test, categories, dataset_tag, arch_tag)
    out = {}
    d = true_variance_decomposition(summary_df, per_instance_df, None, dataset_tag, arch_tag,
                                    n_bootstrap=n_bootstrap)
    out["agg_observed_var"] = d["observed_variance_of_point_estimates"]
    out["agg_noise_var"] = d["mean_within_seed_sampling_variance"]
    out["agg_genuine_var"] = d["genuine_between_seed_variance"]
    out["agg_snr"] = _snr_value(d)
    if include_categories:
        for cat in categories:
            dc = true_variance_decomposition(summary_df, per_instance_df, cat, dataset_tag, arch_tag,
                                             n_bootstrap=n_bootstrap)
            out[f"{cat}_genuine_var"] = dc["genuine_between_seed_variance"]
            out[f"{cat}_snr"] = _snr_value(dc)
    return out


def _snr_value(d):
    if d["is_degenerate_zero_variance"]:
        return "degenerate"
    snr = d["signal_to_noise_ratio"]
    return "inf" if snr == float("inf") else round(float(snr), 2)


def describe_accuracies(acc):
    acc = np.asarray(acc, dtype=float)
    mean = acc.mean()
    std = acc.std(ddof=1) if len(acc) > 1 else float("nan")
    return {
        "n_seeds": int(len(acc)),
        "mean_acc": round(float(mean), 4),
        "std_acc": round(float(std), 5),
        "cv_pct": round(float(std / mean * 100), 2) if mean > 0 else float("nan"),
        "min_acc": round(float(acc.min()), 4),
        "max_acc": round(float(acc.max()), 4),
        "range_pp": round(float((acc.max() - acc.min()) * 100), 1),
    }


# ---------------------------------------------------------------------------
# Dataset access: official split, pooled raw, overlap mask, synthetic
# ---------------------------------------------------------------------------

def make_synthetic_dataset(seed=0):
    """
    Small imbalanced 5-class problem for SMOKE TESTS only. Deliberately
    noisy so that seed effects are visible. Never report numbers from it.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=8000, n_features=30, n_informative=12, n_redundant=6,
        n_classes=5, n_clusters_per_class=2,
        weights=[0.55, 0.28, 0.10, 0.05, 0.02], flip_y=0.03, class_sep=0.7,
        random_state=seed,
    )
    labels = np.array(SYNTHETIC_CATEGORIES)[y]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(sss.split(X, labels))
    X_tr, y_tr, X_te, y_te = X[tr], labels[tr], X[te], labels[te]
    # Plant ~10% exact train->test duplicates so the overlap-split analysis
    # has something to split on when smoke-tested (real datasets have
    # 3-12% natural overlap; continuous synthetic features have none).
    dup_rng = np.random.default_rng(seed + 1)
    n_dup = len(X_te) // 10
    src = dup_rng.choice(len(X_tr), size=n_dup, replace=False)
    dst = dup_rng.choice(len(X_te), size=n_dup, replace=False)
    X_te = X_te.copy(); y_te = y_te.copy()
    X_te[dst] = X_tr[src]
    y_te[dst] = y_tr[src]
    return X_tr, y_tr, X_te, y_te, [f"f{i}" for i in range(X.shape[1])]


def dataset_categories(dataset):
    if dataset == SYNTHETIC_DATASET_NAME:
        return list(SYNTHETIC_CATEGORIES)
    return list(config.DATASETS[dataset]["categories"])


def load_official_split(dataset):
    if dataset == SYNTHETIC_DATASET_NAME:
        return make_synthetic_dataset()
    return data_module.load_and_preprocess(dataset)


def load_pooled_raw(dataset):
    """
    Returns (features_df, y, n_official_train). Rows 0..n_official_train-1
    are the official training partition, the rest the official test
    partition, so the official split can be reconstructed as partition 0.
    Features are RAW (pre-encoding, pre-scaling); preprocessing is fit per
    partition on that partition's training rows only.
    """
    if dataset == SYNTHETIC_DATASET_NAME:
        X_tr, y_tr, X_te, y_te, names = make_synthetic_dataset()
        feats = pd.DataFrame(np.vstack([X_tr, X_te]), columns=names)
        return feats, np.concatenate([y_tr, y_te]), len(X_tr)

    ds = config.DATASETS[dataset]
    if dataset == "nsl_kdd":
        train_df = data_module._load_nsl_kdd_raw(ds["train_path"], ds["expected_train_rows"])
        test_df = data_module._load_nsl_kdd_raw(ds["test_path"], ds["expected_test_rows"])
        for df in (train_df, test_df):
            df.drop(columns=["difficulty"], inplace=True)
            df["category"] = df["label"].apply(
                lambda l: data_module._map_attack_category(l, ds["attack_map"]))
        y = np.concatenate([train_df["category"].values, test_df["category"].values])
        feats = pd.concat([train_df, test_df], ignore_index=True).drop(columns=["label", "category"])
        return feats, y, len(train_df)

    if dataset == "cse_cic_ids2018":
        train_df = data_module._load_cic_ids2018_prepared(ds["train_path"], ds["expected_train_rows"])
        test_df = data_module._load_cic_ids2018_prepared(ds["test_path"], ds["expected_test_rows"])
        test_df = test_df[train_df.columns]
        y = np.concatenate([train_df["category"].values, test_df["category"].values])
        feats = pd.concat([train_df, test_df], ignore_index=True).drop(columns=["category"])
        return feats, y, len(train_df)

    if dataset == "unsw_nb15":
        train_df = data_module._load_unsw_nb15_raw(ds["train_path"], ds["expected_train_rows"])
        test_df = data_module._load_unsw_nb15_raw(ds["test_path"], ds["expected_test_rows"])

        def find_col(df, name):
            m = [c for c in df.columns if c.strip().lower() == name]
            return m[0] if m else None

        frames, ys = [], []
        for df in (train_df, test_df):
            cat_col = find_col(df, "attack_cat")
            ys.append(df[cat_col].astype(str).str.strip().str.lower().values)
            drop = [c for c in [find_col(df, "id"), cat_col, find_col(df, "label")] if c]
            frames.append(df.drop(columns=drop))
        frames[1] = frames[1][frames[0].columns]
        feats = pd.concat(frames, ignore_index=True)
        return feats, np.concatenate(ys), len(train_df)

    raise ValueError(f"No pooled loader for dataset '{dataset}'")


def preprocess_partition(features_df, train_idx, test_idx):
    """
    Train-only-fit one-hot encoding of non-numeric columns and
    standardization of numeric columns, mirroring the discipline of the
    original loaders. (Only tree models are evaluated in the multi-split
    check, for which scaling is immaterial, but the discipline is kept.)
    """
    categorical_cols = [c for c in features_df.columns
                        if not pd.api.types.is_numeric_dtype(features_df[c])]
    numeric_cols = [c for c in features_df.columns if c not in categorical_cols]

    train_f = features_df.iloc[train_idx]
    test_f = features_df.iloc[test_idx]

    if categorical_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(train_f[categorical_cols])
        tr_cat = enc.transform(train_f[categorical_cols])
        te_cat = enc.transform(test_f[categorical_cols])
    else:
        tr_cat = np.empty((len(train_f), 0))
        te_cat = np.empty((len(test_f), 0))

    scaler = StandardScaler()
    tr_num = scaler.fit_transform(train_f[numeric_cols].values.astype(np.float64))
    te_num = scaler.transform(test_f[numeric_cols].values.astype(np.float64))

    X_train = np.concatenate([tr_num, tr_cat], axis=1)
    X_test = np.concatenate([te_num, te_cat], axis=1)
    if not (np.all(np.isfinite(X_train)) and np.all(np.isfinite(X_test))):
        raise ValueError("Non-finite values after partition preprocessing; stopping.")
    return X_train, X_test


def compute_overlap_mask(dataset):
    """
    Boolean mask over the OFFICIAL test rows: True where the raw feature
    row is identical to at least one training row. Uses the same
    pandas hash as data._audit_train_test_overlap, so the count matches the
    figures reported in the paper (664 / 4,786 / 8,541).
    """
    feats, y, n_train = load_pooled_raw(dataset)
    train_feats = feats.iloc[:n_train]
    test_feats = feats.iloc[n_train:]
    train_hashes = pd.util.hash_pandas_object(train_feats, index=False)
    test_hashes = pd.util.hash_pandas_object(test_feats, index=False)
    mask = test_hashes.isin(set(train_hashes)).values
    return mask, y[n_train:]


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def df_to_markdown(df, floatfmt=6):
    """Dependency-free Markdown table (avoids needing tabulate)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.{floatfmt}g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
