import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import tempfile
import os

from graph.network import CorrelationNetwork
from data.data_manager import DataManager


st.set_page_config(page_title="NEPGRAPH", page_icon="📈", layout="wide")


@st.cache_data
def load_data():
    dm = DataManager(data_path='data/nepse_prices.csv')
    return dm.get_data()


@st.cache_data
def build_network(prices):
    cn = CorrelationNetwork()
    cn.calculate_log_returns(prices)
    cn.get_correlation_matrix()
    cn.get_distance_matrix()
    cn.build_mst()
    cn.get_louvain_communities()
    return cn


def main():
    st.title("📈 NEPGRAPH: Hidden Market Communities")
    st.markdown("**Detecting organic market communities via Graph Topology**")
    
    prices = load_data()
    cn = build_network(prices)
    
    tab1, tab2, tab3 = st.tabs(["Network Map", "Market Insights", "Portfolio Checker"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Interactive Network Map")
            
            G = cn.G
            
            net = Network(height="500px", bgcolor="#222222", font_color="white", notebook=True)
            net.from_nx(G)
            
            colors = {
                0: "#FF6B6B",
                1: "#4ECDC4", 
                2: "#45B7D1",
                3: "#96CEB4",
                4: "#FFEAA7",
                5: "#DDA0DD",
            }
            
            for node in net.nodes:
                comm_id = cn.communities.get(node['id'], 0)
                node['color'] = colors.get(comm_id, '#FFFFFF')
                node['size'] = 20 + cn.get_centrality()['eigenvector'].get(node['id'], 0) * 50
                eigen = cn.get_centrality()['eigenvector'].get(node['id'], 0)
                betw = cn.get_centrality()['betweenness'].get(node['id'], 0)
                node['title'] = f"{node['id']}\nCommunity: {comm_id}\nEigenvector: {eigen:.4f}\nBetweenness: {betw:.4f}"
            
            for edge in net.edges:
                edge['width'] = 2
                weight = edge.get('weight', 0)
                edge['title'] = f"{weight:.4f}"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r') as f:
                    html_content = f.read()
                os.unlink(tmp.name)
            
            st.components.v1.html(html_content, height=520)
        
        with col2:
            st.subheader("Legend")
            
            communities = cn.get_louvain_communities()
            for comm_id, members in communities.items():
                color = colors.get(comm_id, '#FFF')
                st.markdown(f"**Community {comm_id}**")
                st.markdown(f"{color} ● {', '.join(members)}")
            
            st.markdown("---")
            st.markdown("**Node Size**: Eigenvector centrality")
            st.markdown("**Edges**: Distance in MST")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Communities")
            
            for comm_id, members in sorted(cn.get_louvain_communities().items()):
                with st.expander(f"Community {comm_id}"):
                    for member in members:
                        eigen = cn.get_centrality()['eigenvector'].get(member, 0)
                        betw = cn.get_centrality()['betweenness'].get(member, 0)
                        st.write(f"**{member}**")
                        st.write(f"  - Eigenvector: {eigen:.4f}")
                        st.write(f"  - Betweenness: {betw:.4f}")
        
        with col2:
            st.subheader("Network Metrics")
            
            st.metric("Modularity Score", f"{cn.get_modularity():.4f}")
            st.metric("Nodes (Stocks)", cn.G.number_of_nodes())
            st.metric("Edges (Connections)", cn.G.number_of_edges())
            
            st.markdown("**Top Hub Stocks (by Eigenvector)**")
            centrality = cn.get_centrality()['eigenvector']
            sorted_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            for stock, score in sorted_hubs:
                st.write(f"  {stock}: {score:.4f}")
    
    with tab3:
        st.subheader("Portfolio Diversification Checker")
        
        available = list(prices.columns)
        
        portfolio = st.multiselect(
            "Select stocks in your portfolio:",
            available,
            default=['ADBL', 'AHPC']
        )
        
        if len(portfolio) > 1:
            portfolio_stocks = [p for p in portfolio if p in available]
            
            if len(portfolio_stocks) > 1:
                portfolio_communities = {}
                for stock in portfolio_stocks:
                    comm = cn.communities.get(stock)
                    if comm not in portfolio_communities:
                        portfolio_communities[comm] = []
                    portfolio_communities[comm].append(stock)
                
                st.write("**Your Portfolio Distribution:**")
                for comm, members in portfolio_communities.items():
                    st.write(f"Community {comm}: {', '.join(members)}")
                
                unique_comms = len(portfolio_communities)
                diversity_score = unique_comms / len(portfolio_stocks)
                
                st.metric(
                    "Diversification Score",
                    f"{diversity_score:.2f}",
                    delta="Good" if diversity_score > 0.5 else "Concentrated",
                    delta_color="normal" if diversity_score > 0.5 else "inverse"
                )
                
                if diversity_score < 0.4:
                    st.warning("Consider diversifying across communities to reduce risk.")
                else:
                    st.success("Good diversification across communities!")
            else:
                st.info("Select at least 2 stocks to analyze diversification.")
        else:
            st.info("Select stocks to check diversification.")


if __name__ == '__main__':
    main()