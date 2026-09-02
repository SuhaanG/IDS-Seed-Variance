"""
Seed-count adequacy with RANDOM seed subsets (not sequential prefixes).

Addresses Reviewer 2 (#3 "evaluating seed counts using a fixed sequential
sequence introduces potential ordering bias; repeat with averaged random
subsets"), Reviewer 1 (#4 "propose an adaptive seed-count criterion based
on confidence-interval convergence") and Reviewer 1's practical-implications
point. Pure re-analysis of results/<dataset>_matrix_summary.csv.

For every (architecture, metric) and every checkpoint n it draws R random
subsets of n seeds (without replacement) from the full set and reports:
  mean_ci_half_width    average 95% t-interval half-width across draws
  sd_subset_means       spread of the n-seed point estimate itself
  p05/p95_subset_mean   5th/95th percentile of the n-seed point estimate
  frac_ci_excludes_full the fraction of n-seed studies whose CI does NOT
                        contain the full-N mean ("confidently wrong" rate,
                        the failure mode the paper reports for logistic
                        regression, now measured without ordering effects)
  prefix_ci_half_width  the sequential-prefix value the paper used, for
                        direct comparison
plus an adaptive stopping rule: the smallest checkpoint at which
mean_ci_half_width falls below a tolerance (default 1.0 and 0.5
percentage points). That gives the "seeds required for +/- tau" statement
reviewers asked for instead of a blanket "40 seeds".

Usage:
    python analyze_seed_adequacy_random_subsets.py --dataset nsl_kdd
    python analyze_seed_adequacy_random_subsets.py --dataset nsl_kdd --metrics aggregate_accuracy,u2r_recall
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import config
from revision_common import df_to_markdown

RANDOM_SUBSET_RNG_SEED = 424_242


def ci_half_width(values):
    n = len(values)
    if n < 2:
        return float("nan")
    return float(stats.sem(values) * stats.t.ppf((1 + config.CONFIDENCE_LEVEL) / 2, df=n - 1))


def analyze_metric(values, checkpoints, n_draws, rng):
    values = np.asarray(values, dtype=float)
    N = len(values)
    full_mean = values.mean()
    rows = []
    for n in checkpoints:
        if n > N:
            continue
        if n == N:
            means = np.array([full_mean])
            hws = np.array([ci_half_width(values)])
        else:
            means = np.empty(n_draws)
            hws = np.empty(n_draws)
            for r in range(n_draws):
                idx = rng.choice(N, size=n, replace=False)
                sub = values[idx]
                means[r] = sub.mean()
                hws[r] = ci_half_width(sub)
        excludes = np.abs(means - full_mean) > hws
        rows.append({
            "n_seeds": n,
            "mean_ci_half_width": float(np.nanmean(hws)),
            "median_ci_half_width": float(np.nanmedian(hws)),
            "sd_subset_means": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "p05_subset_mean": float(np.percentile(means, 5)),
            "p95_subset_mean": float(np.percentile(means, 95)),
            "frac_ci_excludes_full_mean": float(excludes.mean()),
            "prefix_ci_half_width": ci_half_width(values[:n]),
            "prefix_mean": float(values[:n].mean()),
            "full_mean": float(full_mean),
        })
    return pd.DataFrame(rows)


def seeds_required(df_metric, tolerance):
    ok = df_metric[df_metric["mean_ci_half_width"] <= tolerance]
    return int(ok["n_seeds"].iloc[0]) if len(ok) else f">{int(df_metric['n_seeds'].max())}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nsl_kdd")
    ap.add_argument("--summary-csv", default=None)
    ap.add_argument("--architectures", default=",".join(config.ARCHITECTURES))
    ap.add_argument("--metrics", default=None,
                    help="Comma list of summary columns. Default: aggregate_accuracy plus every <category>_recall.")
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--checkpoints", default=",".join(str(c) for c in config.SEED_SUBSET_CHECKPOINTS))
    ap.add_argument("--tolerances", default="0.01,0.005", help="CI half-width tolerances (absolute, e.g. 0.01 = 1 percentage point).")
    args = ap.parse_args()

    dataset = args.dataset
    summary_csv = args.summary_csv or os.path.join(config.RESULTS_DIR, f"{dataset}_matrix_summary.csv")
    if not os.path.exists(summary_csv):
        raise SystemExit(f"Summary CSV not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    df = df[df["dataset"] == dataset] if "dataset" in df.columns else df

    categories = list(config.DATASETS[dataset]["categories"]) if dataset in config.DATASETS else []
    metrics = ([m.strip() for m in args.metrics.split(",")] if args.metrics
               else ["aggregate_accuracy"] + [f"{c}_recall" for c in categories])
    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    tolerances = [float(t) for t in args.tolerances.split(",")]
    architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]
    rng = np.random.default_rng(RANDOM_SUBSET_RNG_SEED)

    all_rows, rule_rows = [], []
    for arch in architectures:
        sub = df[df["architecture"] == arch].sort_values("seed")
        if len(sub) < 3:
            print(f"  {arch}: fewer than 3 seeds, skipping.")
            continue
        for metric in metrics:
            if metric not in sub.columns:
                continue
            values = sub[metric].dropna().values
            if len(values) < 3:
                continue
            res = analyze_metric(values, checkpoints, args.n_draws, rng)
            res.insert(0, "metric", metric)
            res.insert(0, "architecture", arch)
            all_rows.append(res)
            rule = {"architecture": arch, "metric": metric, "full_mean": float(values.mean()),
                    "n_seeds_available": int(len(values))}
            for tol in tolerances:
                rule[f"seeds_for_hw_le_{tol}"] = seeds_required(res, tol)
            small = res[res["n_seeds"] == 5]
            if len(small):
                rule["frac_confidently_wrong_at_5_seeds"] = float(small["frac_ci_excludes_full_mean"].iloc[0])
            rule_rows.append(rule)

    out = pd.concat(all_rows, ignore_index=True)
    rules = pd.DataFrame(rule_rows)
    out_csv = os.path.join(config.RESULTS_DIR, f"seed_adequacy_random_{dataset}.csv")
    rule_csv = os.path.join(config.RESULTS_DIR, f"seed_adequacy_rule_{dataset}.csv")
    out.to_csv(out_csv, index=False)
    rules.to_csv(rule_csv, index=False)

    print(f"\n=== Random-subset seed adequacy, aggregate accuracy, {dataset} ===")
    agg = out[out["metric"] == "aggregate_accuracy"]
    print(df_to_markdown(agg[["architecture", "n_seeds", "mean_ci_half_width", "prefix_ci_half_width",
                              "sd_subset_means", "frac_ci_excludes_full_mean"]]))
    print(f"\n=== Adaptive stopping rule: seeds required for CI half-width <= tolerance ===")
    print(df_to_markdown(rules))

    # Compact LaTeX for the paper (aggregate accuracy only)
    tex_path = os.path.join(config.RESULTS_DIR, f"seed_adequacy_rule_{dataset}.tex")
    agg_rules = rules[rules["metric"] == "aggregate_accuracy"]
    tol_cols = [c for c in agg_rules.columns if c.startswith("seeds_for_hw_le_")]
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(tol_cols) + 1) + "}\n\\hline\n")
        f.write("Architecture & " + " & ".join(
            f"$n$ for $\\pm${float(c.split('_')[-1]) * 100:g}\\,pp" for c in tol_cols)
            + " & Confidently-wrong rate at 5 seeds \\\\\n\\hline\n")
        for _, r in agg_rules.iterrows():
            cw = r.get("frac_confidently_wrong_at_5_seeds", float("nan"))
            f.write(f"{r['architecture'].replace('_', ' ')} & "
                    + " & ".join(str(r[c]) for c in tol_cols)
                    + f" & {cw * 100:.1f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"\nSaved: {out_csv}\nSaved: {rule_csv}\nSaved: {tex_path}")


if __name__ == "__main__":
    main()
