import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import zscore

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from analysis.k_sweep import load_prices, slice_window, build_top_k_graph
from data.sector_map import get_sector

def main():
    prices = load_prices()
    win_prices = slice_window(prices, "max")
    log_returns = np.log(win_prices / win_prices.shift(1))
    
    valid_counts = log_returns.notna().sum()
    min_valid = int(len(log_returns) * 0.9)
    valid_tickers = valid_counts[valid_counts >= min_valid].index
    log_returns = log_returns[valid_tickers].dropna()
    
    corr_matrix = log_returns.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
    
    results = []
    for k in [3, 5, 10]:
        G = build_top_k_graph(corr_matrix, k)
        bw = nx.betweenness_centrality(G, weight='weight')
        
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors: continue
            
            my_sector = get_sector(node)
            out_of_sector_count = sum(1 for n in neighbors if get_sector(n) != my_sector and get_sector(n) != "Unknown")
            total_known = sum(1 for n in neighbors if get_sector(n) != "Unknown")
            
            out_frac = out_of_sector_count / total_known if total_known > 0 else 0
            
            results.append({
                'k': k,
                'ticker': node,
                'sector': my_sector,
                'out_of_sector_fraction': out_frac,
                'betweenness': bw[node]
            })
            
    df = pd.DataFrame(results)
    df5 = df[df['k'] == 5].copy()
    
    bw_90th = df5['betweenness'].quantile(0.9)
    df5['is_anomaly'] = (df5['out_of_sector_fraction'] > 0.5) & (df5['betweenness'] >= bw_90th)
    
    anomalies = df5[df5['is_anomaly']].sort_values('betweenness', ascending=False)
    
    res_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)
    df5.to_csv(os.path.join(res_dir, "anomaly_scores.csv"), index=False)
    
    with open(os.path.join(res_dir, "anomalies_summary.txt"), "w") as f:
        f.write(f"Formal Anomaly Definition (k=5):\n")
        f.write(f"- Out-of-sector neighbor fraction > 0.5\n")
        f.write(f"- Betweenness centrality >= 90th percentile ({bw_90th:.4f})\n\n")
        f.write(f"Detected Anomalies:\n")
        f.write(anomalies[['ticker', 'sector', 'out_of_sector_fraction', 'betweenness']].to_string(index=False))
        
    print("Anomaly definition complete.")

if __name__ == "__main__":
    main()
