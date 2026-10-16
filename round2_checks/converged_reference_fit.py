"""Reference for Table (LightGBM training diagnostics): training accuracy of a converged LightGBM fit
(lambda=1, otherwise the paper's settings, deterministic mode) against the default, seed 0, per dataset."""
import warnings, sys, os; warnings.filterwarnings('ignore'); sys.path.insert(0, os.getcwd())
import numpy as np, lightgbm as lgb, config
from data import load_and_preprocess
for ds in sys.argv[1:]:
    Xtr, ytr, Xte, yte, _ = load_and_preprocess(ds)
    cats = config.DATASETS[ds]['categories']; idx = {c: i for i, c in enumerate(cats)}
    y = np.array([idx[c] for c in ytr]); yt = np.array([idx[c] for c in yte])
    for label, extra in (('default', {}), ('lambda=1', {'reg_lambda': 1.0})):
        m = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective='multiclass', num_class=len(cats), random_state=0, n_jobs=-1, verbosity=-1, deterministic=True, force_row_wise=True, **extra).fit(Xtr, y)
        raw = m.predict(Xtr, raw_score=True)
        print(f'{ds:16s} seed 0 {label:9s} train acc {np.mean(m.predict(Xtr) == y):.4f}  test acc {np.mean(m.predict(Xte) == yt):.4f}  max|raw| {np.abs(raw).max():.3g}', flush=True)
