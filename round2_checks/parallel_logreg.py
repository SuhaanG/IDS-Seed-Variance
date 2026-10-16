"""Run rerun_logistic_regression.py for disjoint seed subsets in parallel, each worker writing to its
own folder, then merge into results/<ds>_logreg_clean*.csv after checking that every worker recorded
the same configuration fingerprint as the main file. Used for UNSW-NB15 (about 11 minutes per seed).
  python round2_checks/parallel_logreg.py worker <k> <dataset> <comma-separated seeds>
  python round2_checks/parallel_logreg.py merge <dataset>
"""
import os, sys, glob, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, os.getcwd())
mode = sys.argv[1]
if mode == 'worker':
    k, ds, seeds = sys.argv[2], sys.argv[3], sys.argv[4]
    import config
    config.RESULTS_DIR = os.path.join('results', f'lr_parallel_{ds}', f'w{k}')
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    sys.argv = ['rerun_logistic_regression.py', '--dataset', ds, '--seeds', seeds]
    import rerun_logistic_regression as r
    r.main()
elif mode == 'merge':
    ds = sys.argv[2]
    import pandas as pd
    stem = f'results/{ds}_logreg_clean'
    fp = json.load(open(stem + '_fingerprint.json'))
    S, P = [pd.read_csv(stem + '.csv')], [pd.read_csv(stem + '_per_instance.csv')]
    for wd in sorted(glob.glob(f'results/lr_parallel_{ds}/w*')):
        wf = json.load(open(os.path.join(wd, f'{ds}_logreg_clean_fingerprint.json')))
        if wf != fp: sys.exit(f'fingerprint mismatch in {wd}: {wf} vs {fp}')
        S.append(pd.read_csv(os.path.join(wd, f'{ds}_logreg_clean.csv')))
        P.append(pd.read_csv(os.path.join(wd, f'{ds}_logreg_clean_per_instance.csv')))
    S, P = pd.concat(S, ignore_index=True), pd.concat(P, ignore_index=True)
    if not S.seed.is_unique or sorted(S.seed) != list(range(40)): sys.exit(f'seeds wrong: {sorted(S.seed)}')
    c = P.seed.value_counts()
    if set(c.index) != set(range(40)) or c.nunique() != 1: sys.exit(f'per-instance counts wrong: {c.to_dict()}')
    S.sort_values('seed').to_csv(stem + '.csv', index=False)
    P.sort_values(['seed', 'instance_id']).to_csv(stem + '_per_instance.csv', index=False)
    a = S.aggregate_accuracy
    print(f'merged 40 seeds; distinct accuracies {a.nunique()}; mean {a.mean():.10f}; std {a.std(ddof=1):.3e}')
