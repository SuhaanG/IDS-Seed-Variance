import warnings; warnings.filterwarnings('ignore')
import numpy as np, lightgbm as lgb, config
from data import load_and_preprocess
Xtr, ytr, Xte, yte, _ = load_and_preprocess('nsl_kdd')
cats = config.DATASETS['nsl_kdd']['categories']; idx = {c: i for i, c in enumerate(cats)}
y = np.array([idx[c] for c in ytr])
def fit(seed, **extra):
    m = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective='multiclass', num_class=5, random_state=seed, n_jobs=-1, verbosity=-1, deterministic=True, force_row_wise=True, **extra)
    return m.fit(Xtr, y)
for label, seed, extra in [('seed22 default', 22, {}), ('seed2 default', 2, {}), ('seed22 lambda=1', 22, {'reg_lambda': 1.0})]:
    m = fit(seed, **extra)
    for k in (10, 50, 100, 200):
        raw = m.predict(Xtr, raw_score=True, num_iteration=k)
        fin = np.isfinite(raw)
        p = m.predict_proba(Xtr, num_iteration=k)
        print(f'{label:16s} round {k:3d}: nonfinite raw scores {int((~fin).sum()):7d}  max|raw| {np.abs(raw[fin]).max():10.3g}  nan probs {int(np.isnan(p).sum()):7d}  train acc {np.mean(p.argmax(1)==y):.4f}')
