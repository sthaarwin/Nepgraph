# NepGraph

**Community Detection in NEPSE Investor Networks**

NepGraph identifies organic "hidden communities" in the Nepal Stock Exchange (NEPSE) based on price movement correlation, revealing market structures invisible through traditional sector labels.

![NepGraph Dashboard](https://via.placeholder.com/800x400?text=NepGraph+Dashboard)

## Features

- **Interactive Network Map** - Draggable, zoomable MST visualization showing stock relationships
- **Community Detection** - Louvain algorithm identifies hidden market clusters
- **Anomaly Detection** - Find stocks trading outside their official NEPSE sector
- **Portfolio Analysis** - Diversification scoring and correlation analysis
- **Backtesting** - Historical portfolio performance simulation
- **Export** - Download correlation matrices and hub scores as CSV

## Installation

```bash
# Clone the repository
git clone https://github.com/sthaarwin/Nepgraph.git
cd Nepgraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

Open http://localhost:8501 in your browser.

## Usage

### Network Tab
- Explore the Minimum Spanning Tree visualization
- Node size = Eigenvector centrality (market influence)
- Edge width = Correlation strength

### Market Insights Tab
- View community breakdown with hub stocks
- Network metrics (modularity, nodes, edges)
- Technical indicators (SMA, Bollinger Bands)

### Anomaly Report Tab
- Discover "hidden links" - stocks trading with unexpected sectors
- Click each anomaly to see detailed neighbor analysis

### Portfolio Checker Tab
- Select stocks to analyze diversification
- View correlation matrix and community distribution
- Run backtests with configurable investment horizon
- Export data as CSV

## How It Works

```
Price Data → Log Returns → Correlation Matrix → Distance Matrix → MST → Louvain Communities
```

1. **Data Loading**: Fetch historical daily closing prices
2. **Signal Processing**: Compute log returns: $r_t = \ln(P_t / P_{t-1})$
3. **Correlation**: Pearson correlation between all stock pairs
4. **Sparsification**: Build Minimum Spanning Tree to remove noise
5. **Community Detection**: Apply Louvain algorithm for clustering

## Project Structure

```
Nepgraph/
├── dashboard.py          # Streamlit UI
├── graph/
│   └── network.py        # Graph engine (MST, Louvain, Centrality)
├── data/
│   ├── data_manager.py   # Data loading and processing
│   ├── sector_map.py     # NEPSE sector classifications
│   └── nepse_prices.csv  # Price data
└── requirements.txt      # Python dependencies
```

## Configuration

- **Time Lookback**: 1, 3, 5 years, or Max (sidebar)
- **Technical Indicators**: Toggle SMA and Bollinger Bands (sidebar)

## Tech Stack

- **Python 3.x** - Core language
- **Streamlit** - Web UI
- **NetworkX** - Graph algorithms
- **python-louvain** - Community detection
- **PyVis** - Interactive visualization

## Requirements

```
pandas
numpy
networkx
python-louvain
pyvis
streamlit
scipy
```

## License

MIT License

## Acknowledgments

- NEPSE for market data
- Community detection via Louvain algorithm
- Visualization via PyVis/vis.js