import numpy as np, pandas as pd, config
from stats_analysis import levene_with_correction
ds = 'nsl_kdd'; cats = config.DATASETS[ds]['categories']
m = pd.read_csv('results/nsl_kdd_matrix_summary.csv'); d = pd.read_csv('results/nsl_kdd_lgb_det.csv')
s = pd.concat([m[m.architecture != 'lightgbm'], d[m.columns.intersection(d.columns)]], ignore_index=True)
dec = pd.read_csv('results/nsl_kdd_corrected_decomposition_lgbdet.csv').set_index(['architecture', 'cell'])
rows = []
for a in ['dnn', 'random_forest', 'xgboost', 'lightgbm_deterministic', 'logistic_regression', 'shallow_mlp']:
    for r in levene_with_correction(s, ds, a, cats):
        c = r['category']; ga = dec.loc[(a, 'aggregate_accuracy'), 'genuine_var']; gc = dec.loc[(a, c), 'genuine_var']
        snr = dec.loc[(a, c), 'snr']
        rows.append(dict(arch=a, cat=c, raw_sd_ratio=round(r.get('std_ratio_effect_size', np.nan), 2), p_bh=r['p_corrected_fdr_bh'], levene_sig=bool(r['significant_after_correction']),
                         genuine_sd_ratio=(round(float(np.sqrt(gc / ga)), 2) if ga > 0 else ('inf' if gc > 0 else 'both 0')), cat_snr=snr, cat_snr_gt1=bool(snr > 1)))
t = pd.DataFrame(rows); t.to_csv('results/levene_vs_decomposition_nsl_kdd_round2.csv', index=False)
print(t.to_string(index=False))
print('Levene significant:', int(t.levene_sig.sum()), 'of', len(t), '| significant AND category SNR>1:', int((t.levene_sig & t.cat_snr_gt1).sum()), '| significant but SNR<=1:', int((t.levene_sig & ~t.cat_snr_gt1).sum()))
