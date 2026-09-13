import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from analysis.k_sweep import load_prices, build_top_k_graph, paper_style
from data.sector_map import get_sector

K = 5
TC = 0.005 # 50 bps

def get_portfolio_returns(weights, next_prices):
    if next_prices.empty or len(next_prices) < 2: return 0
    start_p = next_prices.iloc[0]
    end_p = next_prices.iloc[-1]
    
    valid = (start_p > 0) & (end_p > 0)
    start_p, end_p = start_p[valid], end_p[valid]
    
    weights = weights.reindex(start_p.index).fillna(0)
    if weights.sum() == 0: return 0
    weights = weights / weights.sum()
    
    return np.sum(weights * (end_p / start_p - 1))

def turnover(w_old, w_new):
    if w_old is None: return 1.0
    all_assets = set(w_old.index).union(set(w_new.index))
    w_old = w_old.reindex(list(all_assets)).fillna(0)
    w_new = w_new.reindex(list(all_assets)).fillna(0)
    return np.abs(w_new - w_old).sum() / 2.0

def main():
    prices = load_prices()
    window, step = 252, 63
    dates, ret_comm, ret_sect, ret_ew = [], [], [], []
    w_comm_old, w_sect_old, w_ew_old = None, None, None
    
    for start in range(0, len(prices) - window - step, step):
        train_end = start + window
        test_end = train_end + step
        train_prices = prices.iloc[start:train_end]
        test_prices = prices.iloc[train_end:test_end]
        
        log_returns = np.log(train_prices / train_prices.shift(1))
        valid_tickers = log_returns.columns[log_returns.notna().sum() >= int(window * 0.9)]
        log_returns = log_returns[valid_tickers].dropna()
        
        if len(log_returns.columns) < 10: continue
            
        corr_matrix = log_returns.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
        if len(corr_matrix) <= K: continue
            
        G = build_top_k_graph(corr_matrix, K)
        partition = community_louvain.best_partition(G, random_state=42)
        try:
            ev = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            ev = {n: 1.0 for n in G.nodes()}
        
        # Community Port
        comm_best = {}
        for node, comm_id in partition.items():
            if comm_id not in comm_best or ev[node] > ev[comm_best[comm_id]]:
                comm_best[comm_id] = node
        w_comm = pd.Series(0.0, index=corr_matrix.columns)
        for node in comm_best.values(): w_comm[node] = 1.0
        if w_comm.sum() > 0: w_comm = w_comm / w_comm.sum()
            
        # Sector Port
        sect_best = {}
        for node in G.nodes():
            s = get_sector(node)
            if s not in sect_best or ev[node] > ev[sect_best[s]]:
                sect_best[s] = node
        w_sect = pd.Series(0.0, index=corr_matrix.columns)
        for node in sect_best.values(): w_sect[node] = 1.0
        if w_sect.sum() > 0: w_sect = w_sect / w_sect.sum()
            
        # EW Port
        w_ew = pd.Series(1.0, index=corr_matrix.columns)
        w_ew = w_ew / w_ew.sum()
        
        # Returns & Costs
        r_comm = get_portfolio_returns(w_comm, test_prices) - turnover(w_comm_old, w_comm) * TC
        r_sect = get_portfolio_returns(w_sect, test_prices) - turnover(w_sect_old, w_sect) * TC
        r_ew = get_portfolio_returns(w_ew, test_prices) - turnover(w_ew_old, w_ew) * TC
        
        w_comm_old, w_sect_old, w_ew_old = w_comm, w_sect, w_ew
        
        dates.append(test_prices.index[-1])
        ret_comm.append(r_comm)
        ret_sect.append(r_sect)
        ret_ew.append(r_ew)

    df_ret = pd.DataFrame({'date': dates, 'Community': ret_comm, 'Sector': ret_sect, 'EqualWeight': ret_ew}).set_index('date')
    
    def calc_metrics(series):
        ann_ret = (1 + series).prod() ** (4 / len(series)) - 1
        vol = series.std() * np.sqrt(4)
        sharpe = ann_ret / vol if vol > 0 else 0
        max_dd = ((1 + series).cumprod() / (1 + series).cumprod().cummax() - 1).min()
        return pd.Series([ann_ret, vol, sharpe, max_dd], index=['Ann Return', 'Volatility', 'Sharpe', 'Max DD'])

    metrics = df_ret.apply(calc_metrics)
    
    res_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(res_dir, exist_ok=True)
    metrics.to_csv(os.path.join(res_dir, "backtest_metrics.csv"))
    
    fig_dir = os.path.join(ROOT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    with paper_style():
        fig, ax = plt.subplots(figsize=(10, 6))
        (1 + df_ret['Community']).cumprod().plot(ax=ax, label='Community-Based')
        (1 + df_ret['Sector']).cumprod().plot(ax=ax, label='Sector-Based')
        (1 + df_ret['EqualWeight']).cumprod().plot(ax=ax, label='Equal Weight', color='gray', linestyle='--')
        ax.set_title("Portfolio Backtest: Cumulative Returns (Net of 50 bps Cost)")
        ax.set_ylabel("Wealth Index")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "backtest_equity_curve.png"))
        plt.close(fig)
        
    print("Backtest complete. Metrics:")
    print(metrics.round(4))

if __name__ == "__main__":
    main()
