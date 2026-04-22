import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr
import community as community_louvain


class CorrelationNetwork:
    def __init__(self, log_returns=None, price_data=None):
        self.log_returns = log_returns
        self.price_data = price_data
        self.corr_matrix = None
        self.distance_matrix = None
        self.G = None
        self.mst = None
        self.communities = None

    def calculate_log_returns(self, price_data=None):
        if price_data is not None:
            self.price_data = price_data
        
        if self.price_data is None:
            raise ValueError("No price data available")
        
        self.log_returns = np.log(self.price_data / self.price_data.shift(1))
        self.log_returns = self.log_returns.dropna()
        return self.log_returns

    def get_correlation_matrix(self, method='pearson'):
        if self.log_returns is None:
            raise ValueError("Log returns not calculated. Call calculate_log_returns first.")
        
        if method == 'pearson':
            self.corr_matrix = self.log_returns.corr()
        elif method == 'spearman':
            corr, _ = spearmanr(self.log_returns, axis=0)
            self.corr_matrix = pd.DataFrame(corr, index=self.log_returns.columns, columns=self.log_returns.columns)
        
        return self.corr_matrix

    def get_distance_matrix(self):
        if self.corr_matrix is None:
            self.get_correlation_matrix()
        
        dist = np.sqrt(2 * (1 - self.corr_matrix.values))
        np.fill_diagonal(dist, 0)
        
        self.distance_matrix = pd.DataFrame(
            dist,
            index=self.corr_matrix.index,
            columns=self.corr_matrix.columns
        )
        
        return self.distance_matrix

    def build_correlation_graph(self, threshold=None):
        if self.corr_matrix is None:
            self.get_correlation_matrix()
        
        self.G = nx.Graph()
        tickers = self.corr_matrix.columns
        self.G.add_nodes_from(tickers)
        
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i < j:
                    corr = self.corr_matrix.loc[ticker1, ticker2]
                    if threshold is None or corr > threshold:
                        self.G.add_edge(ticker1, ticker2, weight=corr)
        
        return self.G

    def build_mst(self):
        if self.distance_matrix is None:
            self.get_distance_matrix()
        
        self.mst = nx.from_pandas_adjacency(self.distance_matrix)
        mst = nx.minimum_spanning_tree(self.mst)
        
        self.G = mst
        return mst

    def get_louvain_communities(self):
        if self.G is None:
            self.build_mst()
        
        partition = community_louvain.best_partition(self.G)
        self.communities = partition
        
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        return communities
    
    def get_community_list(self):
        if self.communities is None:
            self.get_louvain_communities()
        
        communities_list = []
        for comm_id in set(self.communities.values()):
            community = [node for node, comm in self.communities.items() if comm == comm_id]
            communities_list.append(community)
        
        return communities_list

    def get_modularity(self):
        if self.communities is None:
            self.get_louvain_communities()
        
        if self.G is None:
            self.build_mst()
        
        mod = community_louvain.modularity(self.communities, self.G)
        
        return mod

    def get_centrality(self):
        if self.G is None:
            self.build_mst()
        
        return {
            'degree': nx.degree_centrality(self.G),
            'betweenness': nx.betweenness_centrality(self.G),
            'closeness': nx.closeness_centrality(self.G),
            'eigenvector': nx.eigenvector_centrality(self.G, max_iter=1000)
        }

    def summary(self):
        return {
            'n_nodes': self.G.number_of_nodes() if self.G else 0,
            'n_edges': self.G.number_of_edges() if self.G else 0,
            'communities': self.communities,
            'modularity': self.get_modularity() if self.communities else None,
            'centrality': self.get_centrality() if self.G else None
        }


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.data_manager import DataManager
    
    dm = DataManager(data_path='data/nepse_prices.csv')
    prices = dm.get_data()
    
    cn = CorrelationNetwork()
    cn.calculate_log_returns(prices)
    
    print("=== Correlation Matrix ===")
    print(cn.get_correlation_matrix().round(3))
    
    print("\n=== Distance Matrix ===")
    print(cn.get_distance_matrix().round(3))
    
    print("\n=== MST ===")
    cn.build_mst()
    print(f"Nodes: {cn.G.number_of_nodes()}, Edges: {cn.G.number_of_edges()}")
    print("Edges:", list(cn.G.edges(data=True)))
    
    print("\n=== Communities ===")
    communities = cn.get_louvain_communities()
    for comm_id, members in communities.items():
        print(f"  Community {comm_id}: {members}")
    
    print("\n=== Modularity ===")
    print(round(cn.get_modularity(), 4))
    
    print("\n=== Centrality ===")
    centrality = cn.get_centrality()
    for metric, values in centrality.items():
        print(f"  {metric}:")
        for node, value in values.items():
            print(f"    {node}: {round(value, 4)}")