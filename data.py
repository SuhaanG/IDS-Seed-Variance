"""
Data loading and preprocessing, generalized across the dataset registry
in config.py. NSL-KDD, CSE-CIC-IDS2018, and UNSW-NB15 are all implemented.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import config

# ---------------------------------------------------------------------------
# Shared validation helpers (dataset-agnostic)
# ---------------------------------------------------------------------------

def _validate_row_count(df, expected, path, tolerance=0):
    actual = len(df)
    if expected is not None and abs(actual - expected) > tolerance:
        raise ValueError(
            f"Row count mismatch for {path}: expected {expected}, got "
            f"{actual}. This means the file does not match what was "
            f"expected. Do not proceed, results from a mismatched file "
            f"are not trustworthy."
        )


def _check_category_completeness(y_train, y_test, expected_categories, split_name_pair):
    train_name, test_name = split_name_pair
    train_present = set(np.unique(y_train))
    test_present = set(np.unique(y_test))
    expected = set(expected_categories)

    missing_train = expected - train_present
    missing_test = expected - test_present

    if missing_train:
        raise ValueError(
            f"Expected categories {missing_train} do not appear at all in "
            f"{train_name}. This will silently break per-category metrics "
            f"for those classes (zero support), stop and investigate "
            f"before proceeding."
        )
    if missing_test:
        raise ValueError(
            f"Expected categories {missing_test} do not appear at all in "
            f"{test_name}. Per-category recall for these classes cannot "
            f"be computed at all if this is not fixed."
        )


def _check_no_nan_or_inf(X, array_name):
    if not np.all(np.isfinite(X)):
        n_bad = int(np.sum(~np.isfinite(X)))
        raise ValueError(
            f"{array_name} contains {n_bad} NaN or Inf values after "
            f"preprocessing. Do not proceed with corrupted features."
        )


def _audit_train_test_overlap(train_features_df, test_features_df):
    """
    Reports how many exact-duplicate feature rows exist between train and
    test, an honest, citable number for the Methods/Limitations section
    rather than an unverified assumption.
    """
    train_hashes = pd.util.hash_pandas_object(train_features_df, index=False)
    test_hashes = pd.util.hash_pandas_object(test_features_df, index=False)
    overlap_count = int(test_hashes.isin(set(train_hashes)).sum())
    overlap_fraction = overlap_count / len(test_features_df) if len(test_features_df) > 0 else 0.0
    print(
        f"Train/test overlap audit: {overlap_count} of {len(test_features_df)} "
        f"test rows ({overlap_fraction*100:.2f}%) have feature values "
        f"identical to some training row (hash-based check, not a "
        f"guarantee against hash collisions, but a strong signal)."
    )
    return overlap_count, overlap_fraction


def audit_dataset(y_train, y_test, categories):
    print("=== Train category distribution ===")
    for cat in categories:
        count = int((y_train == cat).sum())
        print(f"  {cat}: {count}")

    print("=== Test category distribution ===")
    for cat in categories:
        count = int((y_test == cat).sum())
        print(f"  {cat}: {count}")
        if count < config.SMALL_CLASS_SUPPORT_THRESHOLD:
            print(
                f"    WARNING: '{cat}' has only {count} test instances "
                f"(below the {config.SMALL_CLASS_SUPPORT_THRESHOLD}-instance "
                f"threshold). Use Clopper-Pearson exact intervals for this "
                f"class's recall, not naive bootstrap percentile intervals."
            )


# ---------------------------------------------------------------------------
# NSL-KDD loader
# ---------------------------------------------------------------------------

NSL_KDD_COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
NSL_KDD_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def _load_nsl_kdd_raw(path, expected_rows):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the official NSL-KDD file "
            f"and place it there before running anything."
        )
    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMN_NAMES)
    _validate_row_count(df, expected_rows, path)
    return df


def _map_attack_category(label, attack_map):
    label = label.strip().lower()
    if label not in attack_map:
        raise ValueError(
            f"Unrecognized label '{label}' not in the attack map for this "
            f"dataset. Do not silently drop this row, either the file is "
            f"non-standard or the map needs an added entry."
        )
    return attack_map[label]


def _load_and_preprocess_nsl_kdd(ds_config):
    train_df = _load_nsl_kdd_raw(ds_config["train_path"], ds_config["expected_train_rows"])
    test_df = _load_nsl_kdd_raw(ds_config["test_path"], ds_config["expected_test_rows"])

    attack_map = ds_config["attack_map"]
    expected_categories = ds_config["categories"]

    train_df = train_df.drop(columns=["difficulty"])
    test_df = test_df.drop(columns=["difficulty"])

    train_df["category"] = train_df["label"].apply(lambda l: _map_attack_category(l, attack_map))
    test_df["category"] = test_df["label"].apply(lambda l: _map_attack_category(l, attack_map))

    y_train = train_df["category"].values
    y_test = test_df["category"].values

    _check_category_completeness(
        y_train, y_test, expected_categories,
        (ds_config["train_path"], ds_config["test_path"]),
    )

    train_features = train_df.drop(columns=["label", "category"])
    test_features = test_df.drop(columns=["label", "category"])

    _audit_train_test_overlap(train_features, test_features)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_features[NSL_KDD_CATEGORICAL_COLS])

    train_cat_encoded = encoder.transform(train_features[NSL_KDD_CATEGORICAL_COLS])
    test_cat_encoded = encoder.transform(test_features[NSL_KDD_CATEGORICAL_COLS])
    cat_feature_names = list(encoder.get_feature_names_out(NSL_KDD_CATEGORICAL_COLS))

    numeric_cols = [c for c in train_features.columns if c not in NSL_KDD_CATEGORICAL_COLS]
    train_numeric = train_features[numeric_cols].values.astype(np.float64)
    test_numeric = test_features[numeric_cols].values.astype(np.float64)

    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_numeric)
    test_numeric_scaled = scaler.transform(test_numeric)

    X_train = np.concatenate([train_numeric_scaled, train_cat_encoded], axis=1)
    X_test = np.concatenate([test_numeric_scaled, test_cat_encoded], axis=1)

    _check_no_nan_or_inf(X_train, "X_train")
    _check_no_nan_or_inf(X_test, "X_test")

    feature_names = numeric_cols + cat_feature_names

    return X_train, y_train, X_test, y_test, feature_names


# ---------------------------------------------------------------------------
# CSE-CIC-IDS2018 loader
# ---------------------------------------------------------------------------

def _load_cic_ids2018_prepared(path, expected_rows):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run prepare_cicids2018.py first to "
            f"generate this file from the raw downloaded data."
        )
    df = pd.read_csv(path)
    _validate_row_count(df, expected_rows, path)
    if "category" not in df.columns:
        raise ValueError(
            f"{path} has no 'category' column. This file should have been "
            f"produced by prepare_cicids2018.py, which adds this column, "
            f"if it's missing, this isn't the correct prepared file."
        )
    return df


def _load_and_preprocess_cic_ids2018(ds_config):
    train_df = _load_cic_ids2018_prepared(ds_config["train_path"], ds_config["expected_train_rows"])
    test_df = _load_cic_ids2018_prepared(ds_config["test_path"], ds_config["expected_test_rows"])

    expected_categories = ds_config["categories"]

    y_train = train_df["category"].values
    y_test = test_df["category"].values

    _check_category_completeness(
        y_train, y_test, expected_categories,
        (ds_config["train_path"], ds_config["test_path"]),
    )

    train_features = train_df.drop(columns=["category"])
    test_features = test_df.drop(columns=["category"])

    if not set(train_features.columns) == set(test_features.columns):
        raise ValueError(
            f"Train and test feature columns don't match. Train has "
            f"{set(train_features.columns) - set(test_features.columns)} "
            f"extra, test has "
            f"{set(test_features.columns) - set(train_features.columns)} "
            f"extra. This means prepare_cicids2018.py produced "
            f"inconsistent files, do not proceed."
        )
    test_features = test_features[train_features.columns]

    _audit_train_test_overlap(train_features, test_features)

    non_numeric_cols = [
        c for c in train_features.columns
        if not pd.api.types.is_numeric_dtype(train_features[c])
    ]
    if non_numeric_cols:
        raise ValueError(
            f"Non-numeric feature columns found: {non_numeric_cols}. "
            f"prepare_cicids2018.py should have coerced all features to "
            f"numeric already, this means either that script has a gap "
            f"or the wrong file is being read."
        )

    train_numeric = train_features.values.astype(np.float64)
    test_numeric = test_features.values.astype(np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_numeric)
    X_test = scaler.transform(test_numeric)

    _check_no_nan_or_inf(X_train, "X_train")
    _check_no_nan_or_inf(X_test, "X_test")

    feature_names = list(train_features.columns)

    return X_train, y_train, X_test, y_test, feature_names


# ---------------------------------------------------------------------------
# UNSW-NB15 loader
# ---------------------------------------------------------------------------

def _load_unsw_nb15_raw(path, expected_rows):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the official pre-partitioned "
            f"UNSW-NB15 train/test CSVs and place them there before running "
            f"anything."
        )
    df = pd.read_csv(path)
    _validate_row_count(df, expected_rows, path)
    return df


def _load_and_preprocess_unsw_nb15(ds_config):
    train_df = _load_unsw_nb15_raw(ds_config["train_path"], ds_config["expected_train_rows"])
    test_df = _load_unsw_nb15_raw(ds_config["test_path"], ds_config["expected_test_rows"])

    expected_categories = ds_config["categories"]

    def _find_column(df, target_name):
        matches = [c for c in df.columns if c.strip().lower() == target_name]
        return matches[0] if matches else None

    for df, df_name in [(train_df, "train"), (test_df, "test")]:
        cat_col = _find_column(df, "attack_cat")
        if cat_col is None:
            raise ValueError(
                f"No 'attack_cat' column found in the UNSW-NB15 {df_name} "
                f"file. Columns present: {list(df.columns)}. This dataset's "
                f"category column may be named differently than expected, "
                f"check the actual downloaded file's header before "
                f"proceeding."
            )

    train_cat_col = _find_column(train_df, "attack_cat")
    test_cat_col = _find_column(test_df, "attack_cat")

    train_df["category"] = train_df[train_cat_col].astype(str).str.strip().str.lower()
    test_df["category"] = test_df[test_cat_col].astype(str).str.strip().str.lower()

    y_train = train_df["category"].values
    y_test = test_df["category"].values

    _check_category_completeness(
        y_train, y_test, expected_categories,
        (ds_config["train_path"], ds_config["test_path"]),
    )

    id_col_train = _find_column(train_df, "id")
    label_col_train = _find_column(train_df, "label")
    id_col_test = _find_column(test_df, "id")
    label_col_test = _find_column(test_df, "label")

    drop_cols_train = [c for c in [id_col_train, train_cat_col, label_col_train, "category"] if c]
    drop_cols_test = [c for c in [id_col_test, test_cat_col, label_col_test, "category"] if c]

    train_features = train_df.drop(columns=drop_cols_train)
    test_features = test_df.drop(columns=drop_cols_test)

    if not set(train_features.columns) == set(test_features.columns):
        raise ValueError(
            f"Train and test feature columns don't match after dropping "
            f"id/attack_cat/label. Train has "
            f"{set(train_features.columns) - set(test_features.columns)} "
            f"extra, test has "
            f"{set(test_features.columns) - set(train_features.columns)} "
            f"extra. Do not proceed with mismatched columns."
        )
    test_features = test_features[train_features.columns]

    _audit_train_test_overlap(train_features, test_features)

    categorical_cols = [
        c for c in train_features.columns
        if not pd.api.types.is_numeric_dtype(train_features[c])
    ]
    numeric_cols = [c for c in train_features.columns if c not in categorical_cols]

    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(train_features[categorical_cols])
        train_cat_encoded = encoder.transform(train_features[categorical_cols])
        test_cat_encoded = encoder.transform(test_features[categorical_cols])
        cat_feature_names = list(encoder.get_feature_names_out(categorical_cols))
    else:
        train_cat_encoded = np.empty((len(train_features), 0))
        test_cat_encoded = np.empty((len(test_features), 0))
        cat_feature_names = []

    train_numeric = train_features[numeric_cols].values.astype(np.float64)
    test_numeric = test_features[numeric_cols].values.astype(np.float64)

    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_numeric)
    test_numeric_scaled = scaler.transform(test_numeric)

    X_train = np.concatenate([train_numeric_scaled, train_cat_encoded], axis=1)
    X_test = np.concatenate([test_numeric_scaled, test_cat_encoded], axis=1)

    _check_no_nan_or_inf(X_train, "X_train")
    _check_no_nan_or_inf(X_test, "X_test")

    feature_names = numeric_cols + cat_feature_names

    return X_train, y_train, X_test, y_test, feature_names


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_LOADERS = {
    "nsl_kdd": _load_and_preprocess_nsl_kdd,
    "cse_cic_ids2018": _load_and_preprocess_cic_ids2018,
    "cse_cic_ids2018_draw2": _load_and_preprocess_cic_ids2018,  # same loader, second draw
    "unsw_nb15": _load_and_preprocess_unsw_nb15,
}


def load_and_preprocess(dataset_name="nsl_kdd"):
    """
    Loads and preprocesses the named dataset from the registry.
    Fitting (scaler, and encoder where applicable) is done on TRAIN ONLY
    and applied to test, avoiding leakage.

    Returns:
        X_train, y_train_category, X_test, y_test_category, feature_names
    """
    if dataset_name not in config.DATASETS:
        raise ValueError(
            f"'{dataset_name}' is not registered in config.DATASETS. "
            f"Available: {list(config.DATASETS.keys())}"
        )
    if dataset_name not in _LOADERS:
        raise NotImplementedError(
            f"'{dataset_name}' is registered but has no loader implemented "
            f"in data.py. Add one following the existing patterns."
        )

    ds_config = config.DATASETS[dataset_name]
    loader = _LOADERS[dataset_name]
    return loader(ds_config)


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "nsl_kdd"
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess(dataset_name)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    audit_dataset(y_train, y_test, config.DATASETS[dataset_name]["categories"])