"""
Phase 1: Parameter sweep to empirically justify the top-k sparsification threshold.

For each (k, time_window) combination, builds a top-k correlation graph,
runs Louvain community detection (10 runs, seed=42+offset), and records
structural metrics. Produces CSV results and publication-quality figures.

Usage:
    python analysis/k_sweep.py
"""

import os
import sys
import random
from contextlib import contextmanager
from itertools import product

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from cycler import cycler
import community as community_louvain

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
# Publication-quality figure style (Wong 2011 colorblind-safe palette)
# ---------------------------------------------------------------------------
@contextmanager
def paper_style():
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "grid.color": "#cccccc",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "lines.markeredgewidth": 0.8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.loc": "best",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.prop_cycle": cycler("color", [
            "#0072B2", "#E69F00", "#009E73", "#CC79A7",
            "#56B4E9", "#D55E00", "#F0E442", "#999999",
        ]),
    }
    with plt.rc_context(rc):
        yield


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_prices():
    csv_path = os.path.join(ROOT_DIR, "data", "nepse_prices.csv")
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def slice_window(prices, window):
    if window == "max":
        return prices.copy()
    days = {"1yr": 252, "3yr": 756, "5yr": 1260}[window]
    return prices.iloc[-days:].copy()


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_top_k_graph(corr_matrix, k):
    """For each node, keep edges to top-k most correlated neighbours (absolute).
    Undirected union: if A picks B, the edge exists."""
    tickers = corr_matrix.columns.tolist()
    n = len(tickers)
    edge_set = set()
    corr_vals = corr_matrix.values

    for i in range(n):
        abs_row = np.abs(corr_vals[i])
        abs_row[i] = -1  # exclude self
        top_k_idx = np.argpartition(abs_row, -k)[-k:]
        for j in top_k_idx:
            edge = (min(i, j), max(i, j))
            edge_set.add(edge)

    G = nx.Graph()
    G.add_nodes_from(tickers)
    for i, j in edge_set:
        G.add_edge(tickers[i], tickers[j], weight=abs(corr_vals[i, j]))
    return G


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SWEEP_K = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
WINDOWS = ["1yr", "3yr", "5yr", "max"]
N_RUNS = 10
BASE_SEED = 42


def run_sweep(prices):
    results = []

    for window in WINDOWS:
        win_prices = slice_window(prices, window)
        log_returns = np.log(win_prices / win_prices.shift(1))

        # Drop stocks with insufficient data (zero prices → -inf → NaN)
        valid_counts = log_returns.notna().sum()
        min_valid = int(len(log_returns) * 0.9)
        valid_tickers = valid_counts[valid_counts >= min_valid].index
        log_returns = log_returns[valid_tickers].dropna()

        corr_matrix = log_returns.corr()

        # Drop any remaining NaN columns from correlation
        corr_matrix = corr_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

        for k in SWEEP_K:
            modularities = []
            for run in range(N_RUNS):
                seed = BASE_SEED + run
                G = build_top_k_graph(corr_matrix, k)
                if G.number_of_edges() == 0:
                    modularities.append(0.0)
                    continue
                partition = community_louvain.best_partition(
                    G, random_state=seed, resolution=1.0
                )
                mod = community_louvain.modularity(partition, G)
                modularities.append(mod)

            # Use the partition from the last run for structural metrics
            G = build_top_k_graph(corr_matrix, k)
            partition = community_louvain.best_partition(
                G, random_state=BASE_SEED, resolution=1.0
            )

            # Community sizes
            comm_sizes = {}
            for node, cid in partition.items():
                comm_sizes.setdefault(cid, []).append(node)
            sizes = sorted([len(v) for v in comm_sizes.values()], reverse=True)
            n_comm = len(sizes)
            n_nodes = G.number_of_nodes()

            # Connected components
            n_components = nx.number_connected_components(G)

            results.append({
                "k": k,
                "window": window,
                "modularity_mean": np.mean(modularities),
                "modularity_std": np.std(modularities),
                "n_communities": n_comm,
                "largest_community_pct": (sizes[0] / n_nodes * 100) if sizes else 0,
                "community_size_min": min(sizes) if sizes else 0,
                "community_size_mean": np.mean(sizes) if sizes else 0,
                "community_size_median": np.median(sizes) if sizes else 0,
                "community_size_max": max(sizes) if sizes else 0,
                "n_components": n_components,
                "n_edges": G.number_of_edges(),
                "avg_degree": sum(dict(G.degree()).values()) / n_nodes if n_nodes else 0,
                "density": nx.density(G),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Optimal k selection (pre-registered criteria)
# ---------------------------------------------------------------------------
def find_optimal_k(df, window):
    sub = df[df["window"] == window].copy()

    # Compute threshold only from connected, structurally valid graphs
    valid_mask = (sub["n_components"] == 1) & (sub["n_communities"] >= 3)
    valid_sub = sub[valid_mask]
    if len(valid_sub) == 0:
        # Fallback: just pick highest modularity
        best = sub.loc[sub["modularity_mean"].idxmax()]
        return int(best["k"]), best["modularity_mean"], {"fallback": True}

    max_q = valid_sub["modularity_mean"].max()
    q_threshold = max_q * 0.99  # within 1% of max (among valid graphs)

    for _, row in valid_sub.iterrows():
        k = int(row["k"])
        q = row["modularity_mean"]
        largest = row["largest_community_pct"]
        n_comm = int(row["n_communities"])

        passes_q = q >= q_threshold
        passes_size = largest < 60
        passes_comm = 3 <= n_comm <= 20

        if passes_q and passes_size and passes_comm:
            return k, q, {
                "q": passes_q,
                "size": passes_size,
                "components": True,  # guaranteed by valid_sub filter
                "community_count": passes_comm,
            }

    # Fallback: argmax modularity among connected graphs
    best = valid_sub.loc[valid_sub["modularity_mean"].idxmax()]
    return int(best["k"]), best["modularity_mean"], {"fallback": True}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIG_DIR = os.path.join(ROOT_DIR, "figures")
RES_DIR = os.path.join(ROOT_DIR, "results")


def plot_modularity(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for window in WINDOWS:
        sub = df[df["window"] == window]
        ax1.errorbar(
            sub["k"], sub["modularity_mean"], yerr=sub["modularity_std"],
            marker="o", label=window, capsize=3
        )
        ax2.plot(sub["k"], sub["n_communities"], marker="s", label=window)

    ax1.set_xlabel("k (neighbors per node)")
    ax1.set_ylabel("Modularity Q")
    ax1.set_title("A. Modularity vs. k")
    ax1.legend(title="Window")

    ax2.set_xlabel("k (neighbors per node)")
    ax2.set_ylabel("Number of communities")
    ax2.set_title("B. Community count vs. k")
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.legend(title="Window")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "k_sweep_modularity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_community_dist(df, prices):
    selected_k = [3, 5, 10, 15]
    window = "max"

    win_prices = slice_window(prices, window)
    log_returns = np.log(win_prices / win_prices.shift(1))
    valid_counts = log_returns.notna().sum()
    min_valid = int(len(log_returns) * 0.9)
    valid_tickers = valid_counts[valid_counts >= min_valid].index
    log_returns = log_returns[valid_tickers].dropna()
    corr_matrix = log_returns.corr()
    corr_matrix = corr_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

    all_sizes = {}
    max_rank = 0
    for k in selected_k:
        G = build_top_k_graph(corr_matrix, k)
        partition = community_louvain.best_partition(G, random_state=BASE_SEED)
        comm_sizes = {}
        for node, cid in partition.items():
            comm_sizes.setdefault(cid, []).append(node)
        sizes = sorted([len(v) for v in comm_sizes.values()], reverse=True)
        all_sizes[k] = sizes
        max_rank = max(max_rank, len(sizes))

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.2
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

    for idx, k in enumerate(selected_k):
        sizes = all_sizes[k]
        x = np.arange(len(sizes))
        ax.bar(
            x + idx * bar_width, sizes, bar_width,
            label=f"k={k}", color=colors[idx], alpha=0.85
        )

    ax.set_xlabel("Community rank (sorted by size)")
    ax.set_ylabel("Number of stocks")
    ax.set_title("Community size distribution (max window)")
    ax.set_xticks(np.arange(max_rank) + bar_width * 1.5)
    ax.set_xticklabels(np.arange(1, max_rank + 1))
    ax.legend(title="k")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "k_sweep_community_dist.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_connectivity(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for window in WINDOWS:
        sub = df[df["window"] == window]
        ax.plot(sub["k"], sub["n_components"], marker="D", label=window)

    ax.set_xlabel("k (neighbors per node)")
    ax.set_ylabel("Connected components")
    ax.set_title("Graph connectivity vs. k")
    ax.set_yticks(range(0, int(df["n_components"].max()) + 2))
    ax.legend(title="Window")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "k_sweep_connectivity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    print("Loading price data...")
    prices = load_prices()
    print(f"  {prices.shape[1]} stocks, {prices.shape[0]} trading days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    print("\nRunning k-sweep (11 k x 4 windows x 10 Louvain runs = 440 iterations)...")
    df = run_sweep(prices)

    csv_path = os.path.join(RES_DIR, "k_sweep_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    print("\nOptimal k per window:")
    summary_lines = []
    for window in WINDOWS:
        k_opt, q_opt, criteria = find_optimal_k(df, window)
        status = "PASS" if not any(v == False for v in criteria.values() if isinstance(v, bool)) else "PARTIAL"
        detail = ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in criteria.items() if isinstance(v, bool))
        line = f"  {window:>4s}: k={k_opt}  Q={q_opt:.4f}  [{detail}]"
        print(line)
        summary_lines.append(line)

    summary_path = os.path.join(RES_DIR, "k_sweep_summary.md")
    with open(summary_path, "w") as f:
        f.write("# k-Sweep Results Summary\n\n")
        f.write("## Optimal k per Window\n\n")
        f.write("| Window | Optimal k | Modularity Q | Criteria |\n")
        f.write("|--------|-----------|-------------|----------|\n")
        for window in WINDOWS:
            k_opt, q_opt, criteria = find_optimal_k(df, window)
            detail = ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in criteria.items() if isinstance(v, bool))
            f.write(f"| {window} | {k_opt} | {q_opt:.4f} | {detail} |\n")
        f.write("\n## Selection Criteria\n\n")
        f.write("1. Modularity Q within 1% of maximum (plateau detection)\n")
        f.write("2. Largest community < 60% of nodes\n")
        f.write("3. Single connected component\n")
        f.write("4. 3-20 communities\n")
        f.write("5. Tiebreak: smaller k (parsimony)\n")
    print(f"Saved {summary_path}")

    print("\nGenerating figures...")
    with paper_style():
        plot_modularity(df)
        plot_community_dist(df, prices)
        plot_connectivity(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
