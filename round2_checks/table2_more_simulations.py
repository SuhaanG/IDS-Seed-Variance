import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
config.RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "table2_200sims")
os.makedirs(config.RESULTS_DIR, exist_ok=True)
import validate_bootstrap_decomposition as v
t = time.time(); v.validate_small_variance_recovery(n_simulations=200, n_bootstrap=300); print("minutes", (time.time()-t)/60)
