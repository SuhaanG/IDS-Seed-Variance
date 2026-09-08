"""
Configuration for the IDS seed-variance study.

PRE-REGISTRATION NOTE:
These parameters are locked before running any experiment stage and
should not change after seeing results. If a parameter genuinely must
change after the pilot, document why and when in a dated commit message,
undocumented post-hoc changes undermine the pre-registration argument in
the paper's Methods section.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
ENV_LOG_DIR = os.path.join(PROJECT_ROOT, "environment_logs")

# ---------------------------------------------------------------------------
# Results schema version
# ---------------------------------------------------------------------------
RESULTS_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Seed configuration
# ---------------------------------------------------------------------------
PILOT_SEEDS = list(range(10))
FULL_SEEDS = list(range(40))

BOOTSTRAP_RNG_SEED = 999_999

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
NSL_KDD_ATTACK_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "worm": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "httptunnel": "u2r", "ps": "u2r",
    "sqlattack": "u2r", "xterm": "u2r",
}

NSL_KDD_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]

CIC_IDS2018_ATTACK_MAP = {
    "benign": "normal",
    "dos attacks-goldeneye": "dos",
    "dos attacks-slowloris": "dos",
    "dos attacks-slowhttptest": "dos",
    "dos attacks-hulk": "dos",
    "ddos attacks-loic-http": "ddos",
    "ddos attack-loic-udp": "ddos",
    "ddos attack-hoic": "ddos",
    "brute force -web": "web_attacks",
    "brute force -xss": "web_attacks",
    "sql injection": "web_attacks",
    "infilteration": "infiltration",
    "bot": "botnet",
    "ftp-bruteforce": "bruteforce",
    "ssh-bruteforce": "bruteforce",
}

CIC_IDS2018_CATEGORIES = [
    "normal", "bruteforce", "dos", "web_attacks", "infiltration", "botnet", "ddos",
]

CIC_IDS2018_TARGET_TRAIN_ROWS = 200_000
CIC_IDS2018_TARGET_TEST_ROWS = 40_000
CIC_IDS2018_SUBSAMPLE_RNG_SEED = 777_777
# Independent second draw for the IEEE Access revision (Reviewer 1, item 6).
# Same pool, cleaning, taxonomy, target sizes and stratification; only the
# subsample RNG seed differs.
CIC_IDS2018_SUBSAMPLE_RNG_SEED_DRAW2 = 888_888

UNSW_NB15_CATEGORIES = [
    "normal", "generic", "exploits", "fuzzers", "dos", "reconnaissance",
    "analysis", "backdoor", "shellcode", "worms",
]

DATASETS = {
    "nsl_kdd": {
        "train_path": os.path.join(DATA_DIR, "KDDTrain+.txt"),
        "test_path": os.path.join(DATA_DIR, "KDDTest+.txt"),
        "attack_map": NSL_KDD_ATTACK_MAP,
        "categories": NSL_KDD_CATEGORIES,
        "expected_train_rows": 125973,
        "expected_test_rows": 22544,
    },
    "cse_cic_ids2018": {
        "train_path": os.path.join(DATA_DIR, "CSECICIDS2018_train.csv"),
        "test_path": os.path.join(DATA_DIR, "CSECICIDS2018_test.csv"),
        "attack_map": CIC_IDS2018_ATTACK_MAP,
        "categories": CIC_IDS2018_CATEGORIES,
        "expected_train_rows": 200000,
        "expected_test_rows": 39999,
    },
    # Independent second draw of the CSE-CIC-IDS2018 subsample (revision, R1-6).
    # Produced by: python prepare_cicids2018.py --rng-seed 888888 --output-suffix _draw2
    # Expected row counts equal the first draw because the per-category targets
    # are computed from the same cleaned pool; the loader hard-fails otherwise.
    # SHA-256 of the files produced on 2026-09-07 (verify uploads against these):
    #   train 786fec38f097e34c9444a4cd69b7059022de752b179b7ab8dbfe15da33fde2c7
    #   test  1ade7cc19612d1c9af6f05188f4c42ff2b601e72e5b573ce8ca28d460ef5133d
    # Exact-row overlap with the first draw: train 10,637/200,000 (5.32%),
    # test 1,061/39,999 (2.65%), counting duplicate flows present in the pool.
    "cse_cic_ids2018_draw2": {
        "train_path": os.path.join(DATA_DIR, "CSECICIDS2018_train_draw2.csv"),
        "test_path": os.path.join(DATA_DIR, "CSECICIDS2018_test_draw2.csv"),
        "attack_map": CIC_IDS2018_ATTACK_MAP,
        "categories": CIC_IDS2018_CATEGORIES,
        "expected_train_rows": 200000,
        "expected_test_rows": 39999,
    },
    "unsw_nb15": {
        "train_path": os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"),
        "test_path": os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"),
        "attack_map": None,
        "categories": UNSW_NB15_CATEGORIES,
        "expected_train_rows": 175341,
        "expected_test_rows": 82332,
    },
}

ATTACK_CATEGORIES = NSL_KDD_CATEGORIES

# ---------------------------------------------------------------------------
# Architecture hyperparameters
# ---------------------------------------------------------------------------
ARCHITECTURES = [
    "dnn", "random_forest", "xgboost",
    "lightgbm", "logistic_regression", "shallow_mlp",
]

LGB_N_ESTIMATORS = 200
LGB_MAX_DEPTH = 6
LGB_LEARNING_RATE = 0.1
LGB_SUBSAMPLE = 0.8
LGB_COLSAMPLE_BYTREE = 0.8

LOGREG_MAX_ITER = 1000
LOGREG_C = 1.0
LOGREG_SOLVER = "saga"

SHALLOW_MLP_HIDDEN_SIZE = 64
SHALLOW_MLP_DROPOUT = 0.2
SHALLOW_MLP_LEARNING_RATE = 1e-3
SHALLOW_MLP_BATCH_SIZE = 256
SHALLOW_MLP_MAX_EPOCHS = 50
SHALLOW_MLP_EARLY_STOP_PATIENCE = 5
SHALLOW_MLP_VALIDATION_FRACTION = 0.15

DNN_HIDDEN_SIZES = [128, 64, 32]
DNN_DROPOUT = 0.2
DNN_LEARNING_RATE = 1e-3
DNN_BATCH_SIZE = 256
DNN_MAX_EPOCHS = 50
DNN_EARLY_STOP_PATIENCE = 5
DNN_VALIDATION_FRACTION = 0.15

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None

XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# ---------------------------------------------------------------------------
# Statistical analysis parameters
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERATIONS = 5000
CONFIDENCE_LEVEL = 0.95
SEED_SUBSET_CHECKPOINTS = [3, 5, 10, 20, 30, 40]
MULTIPLE_COMPARISON_METHOD = "fdr_bh"
SMALL_CLASS_SUPPORT_THRESHOLD = 500

RANDOM_STATE_FOR_SPLIT = 42