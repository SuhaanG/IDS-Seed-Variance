"""How far apart are the 40 clean logistic regression seeds? Counts differing test predictions
between every pair of seeds and writes a small JSON summary next to the results."""
import json, sys, itertools, numpy as np, pandas as pd
ds = sys.argv[1]
pi = pd.read_csv(f'results/{ds}_logreg_clean_per_instance.csv', usecols=['seed', 'instance_id', 'pred_category'])
piv = pi.pivot(index='instance_id', columns='seed', values='pred_category')
seeds = list(piv.columns)
d = [int((piv[a] != piv[b]).sum()) for a, b in itertools.combinations(seeds, 2)]
out = {'dataset': ds, 'n_seeds': len(seeds), 'n_test': int(len(piv)), 'max_differing_predictions_any_two_seeds': max(d),
       'pairs_identical': int(sum(x == 0 for x in d)), 'pairs_total': len(d), 'instances_ever_differing': int((piv.nunique(axis=1) > 1).sum())}
json.dump(out, open(f'results/{ds}_logreg_clean_seed_agreement.json', 'w'), indent=1); print(out)
