import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from analysis.k_sweep import load_prices, build_top_k_graph, paper_style

K = 5
SEED = 42

def get_rolling_windows(prices, window_size_days=252, step_size_days=63):
    windows = []
    n_days = len(prices)
    for start in range(0, n_days - window_size_days + 1, step_size_days):
        end = start + window_size_days
        win = prices.iloc[start:end]
        windows.append({
            'start_date': win.index[0],
            'end_date': win.index[-1],
            'prices': win
        })
    return windows

def main():
    prices = load_prices()
    windows = get_rolling_windows(prices)
    
    results = []
    partitions = {}
    
    for i, w in enumerate(windows):
        win_prices = w['prices']
        log_returns = np.log(win_prices / win_prices.shift(1))
        
        valid_counts = log_returns.notna().sum()
        min_valid = int(len(log_returns) * 0.9)
        valid_tickers = valid_counts[valid_counts >= min_valid].index
        log_returns = log_returns[valid_tickers].dropna()
        
        corr_matrix = log_returns.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
        if len(corr_matrix) <= K:
            continue
        
        G = build_top_k_graph(corr_matrix, K)
        
        if G.number_of_nodes() == 0:
            continue
            
        partition = community_louvain.best_partition(G, random_state=SEED)
        partitions[i] = partition
        
        if i > 0 and (i-1) in partitions:
            prev_partition = partitions[i-1]
            common_nodes = list(set(partition.keys()) & set(prev_partition.keys()))
            
            if common_nodes:
                l1 = [partition[n] for n in common_nodes]
                l2 = [prev_partition[n] for n in common_nodes]
                ari = adjusted_rand_score(l1, l2)
                nmi = normalized_mutual_info_score(l1, l2)
            else:
                ari, nmi = 0.0, 0.0
                
            results.append({
                'window_end': w['end_date'],
                'ari': ari,
                'nmi': nmi,
                'common_nodes': len(common_nodes)
            })
            print(f"Window {w['start_date'].date()} to {w['end_date'].date()}: ARI = {ari:.4f}")

    df_res = pd.DataFrame(results)
    
    res_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)
    df_res.to_csv(os.path.join(res_dir, "stability_metrics.csv"), index=False)
    
    fig_dir = os.path.join(ROOT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    with paper_style():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_res['window_end'], df_res['ari'], marker='o', label='Adjusted Rand Index (ARI)')
        ax.plot(df_res['window_end'], df_res['nmi'], marker='s', label='Normalized Mutual Info (NMI)')
        
        ax.set_xlabel("Window End Date")
        ax.set_ylabel("Stability Score")
        ax.set_title("Community Stability over Time (Rolling 1-Year Windows)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "community_stability.png"))
        plt.close(fig)

if __name__ == "__main__":
    main()
