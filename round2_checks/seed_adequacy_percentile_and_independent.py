import numpy as np, pandas as pd
from scipy import stats
src = {'lightgbm_deterministic': pd.read_csv('results/nsl_kdd_lgb_det.csv')['aggregate_accuracy'].values}
m = pd.read_csv('results/nsl_kdd_matrix_summary.csv')
for a in ['lightgbm', 'dnn', 'shallow_mlp', 'random_forest', 'xgboost']:
    src[a + ('_nondeterministic' if a == 'lightgbm' else '')] = m[m.architecture == a].sort_values('seed')['aggregate_accuracy'].values
rng = np.random.default_rng(0)
print('arch | n | t-interval excludes 40-seed mean | t-interval excludes mean of the OTHER seeds | percentile-bootstrap interval excludes 40-seed mean | percentile excludes other-seeds mean')
for name, acc in src.items():
    assert len(acc) == 40, (name, len(acc))
    full = acc.mean()
    for n in (3, 5, 10, 20):
        e = np.zeros(4); R = 1000
        for _ in range(R):
            idx = rng.choice(40, n, replace=False); s = acc[idx]; rest = np.delete(acc, idx).mean()
            h = stats.t.ppf(0.975, n - 1) * s.std(ddof=1) / np.sqrt(n); lo, hi = s.mean() - h, s.mean() + h
            bm = rng.choice(s, (2000, n), replace=True).mean(1); plo, phi = np.percentile(bm, [2.5, 97.5])
            e += [not lo <= full <= hi, not lo <= rest <= hi, not plo <= full <= phi, not plo <= rest <= phi]
        print(name, '|', n, '|', ' | '.join(f'{100*v/R:.1f}%' for v in e))
