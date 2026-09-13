import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Import shared functions from k_sweep
from analysis.k_sweep import load_prices, slice_window, build_top_k_graph, paper_style

K = 5
WINDOW = "max"
N_ITERATIONS = 500
SEED = 42

def compute_observed_modularity(G):
    partition = community_louvain.best_partition(G, random_state=SEED)
    mod = community_louvain.modularity(partition, G)
    return mod, partition

def run_null_model(G, n_iter=500):
    null_mods = []
    
    # Pre-calculate edges
    n_edges = G.number_of_edges()
    n_swaps = 5 * n_edges
    max_tries = 15 * n_edges
    
    for i in range(n_iter):
        G_null = G.copy()
        try:
            nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=max_tries, seed=SEED + i)
        except nx.NetworkXError:
            # If it fails to reach exact n_swaps, we can tolerate whatever swaps it managed
            pass
        
        partition = community_louvain.best_partition(G_null, random_state=SEED + i)
        mod = community_louvain.modularity(partition, G_null)
        null_mods.append(mod)
        
        if (i + 1) % 50 == 0:
            print(f"  Completed {i+1}/{n_iter} randomizations...")
            
    return null_mods

def main():
    print(f"Loading data for window='{WINDOW}', k={K}...")
    prices = load_prices()
    win_prices = slice_window(prices, WINDOW)
    
    log_returns = np.log(win_prices / win_prices.shift(1))
    
    # Filter valid stocks
    valid_counts = log_returns.notna().sum()
    min_valid = int(len(log_returns) * 0.9)
    valid_tickers = valid_counts[valid_counts >= min_valid].index
    log_returns = log_returns[valid_tickers].dropna()
    
    corr_matrix = log_returns.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
    
    print("Building observed top-k graph...")
    G_obs = build_top_k_graph(corr_matrix, K)
    
    print("Computing observed modularity...")
    q_obs, partition_obs = compute_observed_modularity(G_obs)
    print(f"  Observed Q = {q_obs:.4f}")
    
    print(f"Running null model ({N_ITERATIONS} iterations)...")
    null_mods = run_null_model(G_obs, N_ITERATIONS)
    
    null_mean = np.mean(null_mods)
    null_std = np.std(null_mods)
    z_score = (q_obs - null_mean) / null_std
    p_val = sum(1 for q in null_mods if q >= q_obs) / N_ITERATIONS
    
    print(f"\nResults:")
    print(f"  Observed Q: {q_obs:.4f}")
    print(f"  Null Mean Q: {null_mean:.4f}")
    print(f"  Null Std  Q: {null_std:.4f}")
    print(f"  Z-score    : {z_score:.2f}")
    print(f"  p-value    : {p_val:.4f}")
    
    # Save results
    res_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)
    pd.DataFrame({"null_modularity": null_mods}).to_csv(os.path.join(res_dir, "null_model_dist.csv"), index=False)
    
    with open(os.path.join(res_dir, "null_model_metrics.txt"), "w") as f:
        f.write(f"Observed Q: {q_obs:.4f}\n")
        f.write(f"Null Mean Q: {null_mean:.4f}\n")
        f.write(f"Null Std Q: {null_std:.4f}\n")
        f.write(f"Z-score: {z_score:.4f}\n")
        f.write(f"p-value: {p_val:.4f}\n")
    
    # Plotting
    print("\nPlotting distribution...")
    fig_dir = os.path.join(ROOT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    with paper_style():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(null_mods, bins=30, color="#56B4E9", edgecolor="black", alpha=0.7, label="Null Model (Randomized Edges)")
        ax.axvline(q_obs, color="#D55E00", linestyle="--", linewidth=2, label=f"Observed Graph (Q={q_obs:.3f})")
        
        # Add text box with stats
        p_val_str = f"< {1/N_ITERATIONS:.3f}" if p_val == 0 else f"= {p_val:.3f}"
        textstr = f"Z-score = {z_score:.2f}\np-value {p_val_str}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel("Modularity (Q)")
        ax.set_ylabel("Frequency")
        ax.set_title("Modularity Significance Test (k=5, max window)")
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "null_model_modularity.png"))
        plt.close(fig)
        
    print(f"Saved artifacts to {fig_dir} and {res_dir}.")
    print("Done.")

if __name__ == "__main__":
    main()
