"""
One-time preparation script for CSE-CIC-IDS2018.

Unlike NSL-KDD, this dataset does not ship as a single clean file with an
official train/test split, it's distributed as multiple daily CSVs, with
well-documented real-world messiness: inconsistent column name spacing
(e.g. " Label" with a leading space), Infinity/NaN values in Flow Bytes/s
and Flow Packets/s from division-by-zero during feature extraction, AND
(confirmed via this project's actual downloaded data) inconsistent
columns across daily files, one file has 4 extra columns (Src Port,
Flow ID, Src IP, Dst IP) that no other file has. This script handles all
of that and writes two clean, flat output files that data.py's loader
can then read simply.

DELIBERATE METHODOLOGICAL CHOICE, stated explicitly for the paper's
Methods section: no SMOTE or other class-balancing is applied here,
unlike the ACAFS paper's preprocessing. This study measures seed-driven
instability under NATURAL class imbalance, consistent with how NSL-KDD
is handled elsewhere in this codebase.

Usage:
    python prepare_cicids2018.py
        (first draw, exactly as used for the submitted manuscript)
    python prepare_cicids2018.py --rng-seed 888888 --output-suffix _draw2
        (independent second draw for the revision, Reviewer 1 item 6; same
         cleaning, taxonomy, target sizes and stratification, different seed)

The defaults of --rng-seed and --output-suffix equal the values that were
previously hard-coded, so running with no arguments still reproduces the
submitted files byte-for-byte.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

import config

RAW_DATA_DIR = os.path.join(config.DATA_DIR, "cicids2018_raw")
OUTPUT_TRAIN_PATH = config.DATASETS["cse_cic_ids2018"]["train_path"]
OUTPUT_TEST_PATH = config.DATASETS["cse_cic_ids2018"]["test_path"]

COLUMNS_TO_DROP_EXACT = {"timestamp", "flow id", "flow_id"}
COLUMNS_TO_DROP_SUBSTRING = ["ip"]


def _clean_column_names(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def _identify_columns_to_drop(columns):
    to_drop = []
    for col in columns:
        col_lower = col.strip().lower()
        if col_lower in COLUMNS_TO_DROP_EXACT:
            to_drop.append(col)
        elif any(sub in col_lower for sub in COLUMNS_TO_DROP_SUBSTRING) and col_lower != "label":
            to_drop.append(col)
    return to_drop


def load_all_raw_files():
    csv_paths = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DATA_DIR}. Download the dataset "
            f"there first (see the aws s3 cp command)."
        )

    print(f"Found {len(csv_paths)} raw file(s):")
    for p in csv_paths:
        print(f"  {p}")

    frames = []
    for path in csv_paths:
        print(f"Loading {path}...")
        df = pd.read_csv(path, low_memory=False)
        df = _clean_column_names(df)
        frames.append(df)

    # CRITICAL FIX, found via real data: CSE-CIC-IDS2018's daily files do
    # not all have the same columns. One file (confirmed: the DDoS-LOIC-
    # HTTP day) has 4 extra columns (Src Port, Flow ID, Src IP, Dst IP)
    # that no other file has. Naively concatenating would silently fill
    # those columns with NaN for every row from the other 9 files, and
    # since dropna() later removes any row with ANY NaN feature, this
    # wiped out 100% of every category except the two present in that
    # one file. Restricting to the column INTERSECTION across all files
    # before concatenating prevents this, rather than special-casing the
    # one column discovered so far, this also protects against any other
    # undiscovered inconsistency between files.
    common_columns = set(frames[0].columns)
    for df in frames[1:]:
        common_columns &= set(df.columns)

    all_columns = set()
    for df in frames:
        all_columns |= set(df.columns)
    dropped_for_inconsistency = all_columns - common_columns
    if dropped_for_inconsistency:
        print(
            f"WARNING: {len(dropped_for_inconsistency)} column(s) are not "
            f"present in all {len(frames)} files and are being dropped "
            f"entirely to avoid silent NaN-fill corruption during "
            f"concatenation: {dropped_for_inconsistency}. This is a "
            f"documented, disclosed data-quality issue with this "
            f"dataset's daily files, worth stating explicitly in the "
            f"paper's limitations section."
        )

    ordered_common_columns = [c for c in frames[0].columns if c in common_columns]
    frames = [df[ordered_common_columns] for df in frames]

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined raw shape: {combined.shape}")
    return combined


def check_label_mapping(df, attack_map):
    if "Label" not in df.columns:
        raise ValueError(
            f"No 'Label' column found after cleaning. Columns present: "
            f"{list(df.columns)}."
        )

    raw_labels = df["Label"].astype(str).str.strip().str.lower().unique()
    unmapped = [l for l in raw_labels if l not in attack_map]

    if unmapped:
        raise ValueError(
            f"Found {len(unmapped)} label value(s) in the data with no "
            f"entry in config.CIC_IDS2018_ATTACK_MAP: {unmapped}\n"
            f"Add these to the map in config.py before proceeding."
        )

    print(f"All {len(raw_labels)} unique raw label values map cleanly "
          f"to the 7-category taxonomy.")


def clean_and_map(df, attack_map):
    columns_to_drop = _identify_columns_to_drop(df.columns)
    if columns_to_drop:
        print(f"Dropping identifier/leakage-risk columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    if "Label" not in df.columns:
        raise ValueError(
            f"No 'Label' column found after cleaning. Columns present: "
            f"{list(df.columns)}."
        )

    label_str = df["Label"].astype(str).str.strip()
    embedded_header_mask = label_str.str.lower() == "label"
    n_embedded_headers = int(embedded_header_mask.sum())
    if n_embedded_headers > 0:
        print(
            f"Found and removed {n_embedded_headers} embedded duplicate "
            f"header row(s) (literal 'Label' value appearing as data), "
            f"a documented quirk of this dataset's file concatenation."
        )
        df = df[~embedded_header_mask].copy()

    check_label_mapping(df, attack_map)

    df["category"] = df["Label"].astype(str).str.strip().str.lower().map(attack_map)
    df = df.drop(columns=["Label"])

    feature_cols = [c for c in df.columns if c != "category"]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    n_after = len(df)
    n_dropped = n_before - n_after
    print(
        f"Dropped {n_dropped} of {n_before} rows ({n_dropped/n_before*100:.2f}%) "
        f"due to NaN/Inf values in feature columns."
    )

    return df


def stratified_split_and_subsample(df, categories, target_train_rows,
                                    target_test_rows, rng_seed):
    rng = np.random.default_rng(rng_seed)

    train_frames = []
    test_frames = []

    total_rows = len(df)
    for cat in categories:
        cat_df = df[df["category"] == cat]
        n_cat = len(cat_df)
        if n_cat == 0:
            print(f"WARNING: category '{cat}' has zero rows in the raw "
                  f"data. Check whether the expected attack file was "
                  f"actually downloaded.")
            continue

        cat_proportion = n_cat / total_rows
        cat_target_train = max(1, int(round(target_train_rows * cat_proportion)))
        cat_target_test = max(1, int(round(target_test_rows * cat_proportion)))

        shuffled_idx = rng.permutation(n_cat)
        cat_df_shuffled = cat_df.iloc[shuffled_idx]

        n_take = min(n_cat, cat_target_train + cat_target_test)
        if n_take < cat_target_train + cat_target_test:
            print(
                f"WARNING: category '{cat}' has only {n_cat} rows available, "
                f"fewer than the {cat_target_train + cat_target_test} "
                f"requested (train+test). Using all {n_cat} available."
            )
            split_point = int(n_cat * (cat_target_train / (cat_target_train + cat_target_test)))
        else:
            split_point = cat_target_train

        selected = cat_df_shuffled.iloc[:n_take]
        train_frames.append(selected.iloc[:split_point])
        test_frames.append(selected.iloc[split_point:])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(drop=True)
    test_df = test_df.iloc[rng.permutation(len(test_df))].reset_index(drop=True)

    return train_df, test_df


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rng-seed", type=int, default=config.CIC_IDS2018_SUBSAMPLE_RNG_SEED,
                    help="Seed for the stratified subsample draw. Default is the first "
                         "draw's seed; use config.CIC_IDS2018_SUBSAMPLE_RNG_SEED_DRAW2 "
                         "for the independent second draw.")
    ap.add_argument("--output-suffix", default="",
                    help="Inserted before .csv in both output file names, e.g. _draw2. "
                         "Default writes the standard file names.")
    return ap.parse_args()


def _report_overlap_with_first_draw(train_df, test_df):
    """How many rows of this draw also appear in the first draw (exact match on
    every column). Two independent draws from the same pool are expected to
    share a small fraction; this makes that fraction a reported number."""
    if not (os.path.exists(OUTPUT_TRAIN_PATH) and os.path.exists(OUTPUT_TEST_PATH)):
        print("\n(first-draw files not present here; overlap with the first draw not computed)")
        return

    def row_hashes(df):
        return set(pd.util.hash_pandas_object(df.astype(str), index=False).values)

    print("\nOverlap with the first draw (exact row match on all columns):")
    for name, new_df, first_path in [("train", train_df, OUTPUT_TRAIN_PATH),
                                     ("test", test_df, OUTPUT_TEST_PATH)]:
        first_df = pd.read_csv(first_path, low_memory=False)
        if set(new_df.columns) <= set(first_df.columns):
            first_df = first_df[list(new_df.columns)]
        shared = len(row_hashes(new_df) & row_hashes(first_df))
        print(f"  {name}: {shared} of {len(new_df)} rows "
              f"({100.0 * shared / len(new_df):.2f}%) also appear in the first draw")


def main():
    args = _parse_args()
    out_train = OUTPUT_TRAIN_PATH.replace(".csv", f"{args.output_suffix}.csv")
    out_test = OUTPUT_TEST_PATH.replace(".csv", f"{args.output_suffix}.csv")
    print(f"Subsample RNG seed: {args.rng_seed} "
          f"(first draw used {config.CIC_IDS2018_SUBSAMPLE_RNG_SEED})")
    print(f"Outputs: {out_train}\n         {out_test}")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    combined = load_all_raw_files()
    cleaned = clean_and_map(combined, config.CIC_IDS2018_ATTACK_MAP)

    print("\nCategory distribution in cleaned data:")
    print(cleaned["category"].value_counts())

    train_df, test_df = stratified_split_and_subsample(
        cleaned,
        config.CIC_IDS2018_CATEGORIES,
        config.CIC_IDS2018_TARGET_TRAIN_ROWS,
        config.CIC_IDS2018_TARGET_TEST_ROWS,
        args.rng_seed,
    )

    print(f"\nFinal train shape: {train_df.shape}")
    print("Train category distribution:")
    print(train_df["category"].value_counts())

    print(f"\nFinal test shape: {test_df.shape}")
    print("Test category distribution:")
    print(test_df["category"].value_counts())

    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)

    print(f"\nWrote {out_train}")
    print(f"Wrote {out_test}")
    if args.output_suffix:
        _report_overlap_with_first_draw(train_df, test_df)
    print(
        "\nIMPORTANT: update config.py's 'expected_train_rows' and "
        "'expected_test_rows' for cse_cic_ids2018 with the exact numbers "
        "printed above."
    )


if __name__ == "__main__":
    main()