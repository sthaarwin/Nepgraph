import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from scipy.stats import pearsonr, spearmanr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from analysis.k_sweep import load_prices, slice_window, build_top_k_graph, paper_style

K = 5
WINDOW = "max"
SEED = 42

def main():
    prices = load_prices()
    win_prices = slice_window(prices, WINDOW)
    
    log_returns = np.log(win_prices / win_prices.shift(1))
    
    valid_counts = log_returns.notna().sum()
    min_valid = int(len(log_returns) * 0.9)
    valid_tickers = valid_counts[valid_counts >= min_valid].index
    log_returns = log_returns[valid_tickers].dropna()
    
    corr_matrix = log_returns.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
    
    G = build_top_k_graph(corr_matrix, K)
    partition = community_louvain.best_partition(G, random_state=SEED)
    
    print("Computing centralities...")
    # The graph built by build_top_k_graph uses absolute correlation as weight
    # Eigenvector centrality needs to be careful with weights
    ev_cent = nx.eigenvector_centrality(G, max_iter=2000, weight='weight')
    bw_cent = nx.betweenness_centrality(G, weight='weight')
    
    df = pd.DataFrame({
        'eigenvector': pd.Series(ev_cent),
        'betweenness': pd.Series(bw_cent),
        'community': pd.Series(partition)
    })
    
    pearson_corr, p_p = pearsonr(df['eigenvector'], df['betweenness'])
    spearman_corr, p_s = spearmanr(df['eigenvector'], df['betweenness'])
    
    print(f"Pearson r: {pearson_corr:.4f} (p={p_p:.4e})")
    print(f"Spearman r: {spearman_corr:.4f} (p={p_s:.4e})")
    
    ev_med = df['eigenvector'].median()
    bw_med = df['betweenness'].median()
    
    def assign_quadrant(row):
        if row['eigenvector'] >= ev_med and row['betweenness'] >= bw_med:
            return "True Hubs"
        elif row['eigenvector'] >= ev_med and row['betweenness'] < bw_med:
            return "Local Kings"
        elif row['eigenvector'] < ev_med and row['betweenness'] >= bw_med:
            return "Bridge Stocks"
        else:
            return "Peripheral"
            
    df['quadrant'] = df.apply(assign_quadrant, axis=1)
    
    res_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)
    df.to_csv(os.path.join(res_dir, "centrality_quadrants.csv"))
    
    bridges = df[df['quadrant'] == 'Bridge Stocks'].sort_values('betweenness', ascending=False)
    print("\nTop 5 Bridge Stocks:")
    print(bridges.head(5))
    
    fig_dir = os.path.join(ROOT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    with paper_style():
        fig, ax = plt.subplots(figsize=(9, 7))
        
        scatter = ax.scatter(df['eigenvector'], df['betweenness'], 
                             c=df['community'], cmap='tab20', alpha=0.7, s=40, edgecolors='none')
        
        ax.axvline(ev_med, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(bw_med, color='gray', linestyle='--', alpha=0.5)
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.95, 'True Hubs', transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        ax.text(0.95, 0.05, 'Local Kings', transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        ax.text(0.05, 0.95, 'Bridge Stocks', transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left', bbox=props)
        ax.text(0.05, 0.05, 'Peripheral', transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='left', bbox=props)
        
        for idx, row in bridges.head(3).iterrows():
            ax.annotate(idx, (row['eigenvector'], row['betweenness']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
                        
        hubs = df[df['quadrant'] == 'True Hubs'].sort_values('betweenness', ascending=False).head(3)
        for idx, row in hubs.iterrows():
            ax.annotate(idx, (row['eigenvector'], row['betweenness']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
                        
        ax.set_xlabel("Eigenvector Centrality")
        ax.set_ylabel("Betweenness Centrality")
        ax.set_title(f"Centrality Quadrants (Spearman $\\rho$ = {spearman_corr:.2f})")
        
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "eigenvector_vs_betweenness.png"))
        plt.close(fig)

if __name__ == "__main__":
    main()
