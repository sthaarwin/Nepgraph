import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import tempfile
import os

from graph.network import CorrelationNetwork
from data.data_manager import DataManager


st.set_page_config(
    page_title="NepGraph",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal, targeted CSS (no full-page overrides) ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Apply font globally without breaking Streamlit's layout */
[class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Tighten block padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* Stat card */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}
.stat-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: #64748b; margin-bottom: 0.25rem; }
.stat-val   { font-size: 1.9rem; font-weight: 700; color: #f1f5f9; line-height: 1.1; }
.stat-sub   { font-size: 0.72rem; color: #475569; margin-top: 0.2rem; }

/* Section header */
.sec-head {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0.4rem; margin-bottom: 0.9rem;
}

/* Community block */
.comm-block {
    border-radius: 8px; padding: 0.7rem 1rem;
    margin-bottom: 0.5rem; border: 1px solid;
}

/* Hub row */
.hub-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0.8rem; border-radius: 7px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0.3rem;
}
.hub-ticker { font-weight: 600; font-size: 0.85rem; color: #cbd5e1; }
.hub-val    { font-size: 0.8rem; font-weight: 500; color: #38bdf8; }

/* Bar */
.bar-wrap { background: rgba(255,255,255,0.05); border-radius: 3px; height: 4px; margin-top: 0.3rem; }
.bar-fill  { height: 4px; border-radius: 3px; }

/* Alert boxes */
.box-info    { background: rgba(56,189,248,0.07); border: 1px solid rgba(56,189,248,0.22); border-radius: 8px; padding: 0.75rem 1rem; color: #7dd3fc; font-size: 0.85rem; }
.box-warn    { background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.22); border-radius: 8px; padding: 0.75rem 1rem; color: #fcd34d; font-size: 0.85rem; }
.box-success { background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.22); border-radius: 8px; padding: 0.75rem 1rem; color: #6ee7b7; font-size: 0.85rem; }

/* Thin divider */
.div { height: 1px; background: rgba(255,255,255,0.07); margin: 1rem 0; }

/* Remove white border from pyvis iframe & components */
iframe { border: 0 !important; outline: 0 !important; box-shadow: none !important; }
[data-testid="stIFrame"], [data-testid="stHtml"] { border: 0 !important; outline: 0 !important; background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette for communities ───────────────────────────────────────────
COLORS = [
    "#38bdf8",  # sky
    "#f472b6",  # pink
    "#34d399",  # emerald
    "#fb923c",  # orange
    "#a78bfa",  # violet
    "#facc15",  # yellow
    "#2dd4bf",  # teal
    "#f87171",  # red
    "#818cf8",  # indigo
    "#4ade80",  # green
]

st.sidebar.markdown("## NepGraph Controls")

with st.sidebar:
    if st.button("↻ Refresh Data", use_container_width=True):
        with st.spinner("Clearing cache..."):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

st.sidebar.markdown("---")

time_regime_map = {
    "1 Year": 1,
    "3 Years": 3,
    "5 Years": 5,
    "Max": 100, # Effectively no limit
}
time_regime_key = st.sidebar.radio(
    "**Time Lookback**",
    options=time_regime_map.keys(),
    index=2, # Default to 5 years
    help="How far back to calculate correlations. Shorter periods capture recent market dynamics, longer periods capture stable, long-term relationships."
)
_LOOKBACK_YEARS = time_regime_map[time_regime_key]

st.sidebar.markdown("---")
st.sidebar.markdown("### Technical Indicators")
show_sma = st.sidebar.checkbox("Show 50/200 Day SMA", value=True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=False)

with st.sidebar.expander("💡 Tips"):
    st.markdown("""
    - **Network Tab**: Drag nodes to explore connections
    - **Insights Tab**: Click hub stocks to see details  
    - **Anomaly Tab**: Stocks trading outside their sector
    - **Portfolio Tab**: Add more stocks for better analysis
    """)


def clr(comm_id: int) -> str:
    return COLORS[comm_id % len(COLORS)]

def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ── Data / network helpers ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(lookback_years):
    try:
        dm = DataManager(data_path="data/nepse_prices.csv")
        raw = dm.get_data()
        
        if raw is None or raw.empty:
            raise ValueError("No data loaded. Please check data/nepse_prices.csv")

        raw.index = pd.to_datetime(raw.index)
        if lookback_years < 100:
            cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)
            trimmed = raw[raw.index >= cutoff]
        else:
            trimmed = raw

        trimmed = trimmed.loc[:, trimmed.isna().mean() < 0.4]
        
        if trimmed.empty:
            raise ValueError(f"No stocks with sufficient data for {lookback_years} year lookback")
        
        trimmed = trimmed.ffill().bfill().astype('float32')
        return trimmed
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False, hash_funcs={"numpy.ndarray": lambda x: id(x)}, ttl=3600)
def build_network(_prices):
    try:
        if _prices.empty:
            raise ValueError("Empty price data")
        
        prices = _prices.astype('float32')
        prices = prices.loc[:, prices.std() > 0]
        prices = prices.loc[:, prices.nunique() >= 30]

        if prices.shape[1] < 2:
            raise ValueError("Not enough valid stocks for analysis")

        cn = CorrelationNetwork()
        cn.calculate_log_returns(prices)

        if cn.log_returns is not None:
            cn.log_returns = cn.log_returns.dropna(axis=1, how='all')
            cn.log_returns = cn.log_returns.loc[:, cn.log_returns.std() > 0].astype('float32')
            cn.price_data  = prices[cn.log_returns.columns].astype('float32')

        cn.get_correlation_matrix()
        cn.get_distance_matrix()

        if cn.distance_matrix is not None:
            cn.distance_matrix = cn.distance_matrix.fillna(2.0).astype('float32')

        cn.build_mst()
        cn.get_louvain_communities()
        return cn
    except Exception as e:
        st.error(f"Network build error: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_centrality(_cn):
    """Expensive for 100+ nodes — cache it."""
    return _cn.get_centrality()


@st.cache_data(show_spinner=False)
def get_modularity(_cn_id, _cn):
    return _cn.get_modularity()


@st.cache_data(show_spinner=False)
def get_anomalies(_cn_id, _cn):
    return _cn.get_anomalies()


@st.cache_data(show_spinner=False)
def get_normalised_prices(_prices_hash, prices):
    """Pre-compute base-100 normalisation for every ticker as plain lists.
    Dict lookup is O(1); avoids copying the full DataFrame on every render."""
    first  = prices.iloc[0].replace(0, float('nan'))
    normed = (prices / first * 100).round(1)
    dates  = prices.index.astype(str).tolist()
    return {col: normed[col].tolist() for col in normed.columns}, dates


@st.cache_data(show_spinner=False)
def get_pyvis_html(_cn_id, nodes_data, edges_data, communities_map):
    """Build and cache the pyvis HTML so it isn't rebuilt on every render."""
    net = Network(height="100%", width="100%", bgcolor="#0f172a", font_color="#cbd5e1", notebook=True, cdn_resources='remote')

    for nid, size, color, border_clr, label in nodes_data:
        net.add_node(
            nid,
            size=size,
            color={"background": color, "border": border_clr,
                   "highlight": {"background": "#ffffff", "border": border_clr}},
            font={"size": 11, "color": "#cbd5e1"},
            borderWidth=1.5,
            title=label,   # plain text — no HTML tags
            label=nid,
        )

    for src, dst, width, weight in edges_data:
        net.add_edge(
            src, dst,
            width=width,
            color={"color": "rgba(148,163,184,0.15)", "highlight": "rgba(56,189,248,0.7)"},
            title=f"Dist {weight:.3f}",
        )

    # Optimized for faster load
    net.set_options("""{
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 80,
                "damping": 0.2
            },
            "stabilization": {"iterations": 80, "updateInterval": 25},
            "minVelocity": 0.8
        },
        "interaction": {"hover": true, "tooltipDelay": 100},
        "edges": {"smooth": {"type": "continuous", "roundness": 0}}
    }""")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as tmp:
        tmp_name = tmp.name
    net.save_graph(tmp_name)
    with open(tmp_name, "r") as f:
        html = f.read()
    os.unlink(tmp_name)

    # Hover-bound physics: physics is only active when mouse is over the graph. 
    # This prevents the pyvis engine from starving CPU when user is dragging line charts in other tabs.
    hover_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var ready = setInterval(function() {
            if (typeof network === 'undefined') return;
            clearInterval(ready);
            var container = document.getElementById('mynetwork');
            if (container) {
                container.addEventListener('mouseenter', function() {
                    network.setOptions({ physics: { enabled: true } });
                });
                container.addEventListener('mouseleave', function() {
                    network.setOptions({ physics: { enabled: false } });
                });
            }
        }, 200);
    });
    </script>
    """
    html = html.replace("</body>", "</body>")
    
    # Force body to have no margins and be dark, and remove the hardcoded border
    import re
    html = re.sub(r'border\s*:\s*1px\s+solid\s+lightgray\s*;?', 'border: 0 !important;', html)
    css_inject = """
    <style>
    body, html { padding: 0 !important; margin: 0 !important; border: 0 !important; outline: 0 !important; background-color: #0f172a !important; overflow: hidden !important; }
    #mynetwork { border: 0 !important; outline: 0 !important; width: 100vw !important; height: 100vh !important; }
    canvas { border: 0 !important; outline: 0 !important; }
    * { border: 0 !important; outline: 0 !important; }
    </style>
    """
    html = html.replace("</head>", css_inject + "</head>")
    return html


# ── Widgets ───────────────────────────────────────────────────────────────────
def stat_card(label, value, sub=""):
    st.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-val">{value}</div>'
        f'{"<div class=stat-sub>" + sub + "</div>" if sub else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def sec_head(text):
    st.markdown(f'<div class="sec-head">{text}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
def tab_network(cn):
    G = cn.G
    centrality  = get_centrality(cn)
    communities = cn.get_louvain_communities()

    eigen_c = centrality["eigenvector"]
    betw_c  = centrality["betweenness"]
    max_e   = max(eigen_c.values()) or 1

    # Build serialisable tuples for the cached HTML builder
    nodes_data = []
    for nid in G.nodes():
        comm_id = cn.communities.get(nid, 0)
        c       = clr(comm_id)
        bg      = hex_to_rgba(c, 0.82)
        e       = eigen_c.get(nid, 0)
        b       = betw_c.get(nid, 0)
        size    = 12 + e / max_e * 28
        # Plain-text tooltip — no HTML tags
        label   = f"{nid}  |  Comm {comm_id}\nEigen {e:.4f}  Betw {b:.4f}"
        nodes_data.append((nid, size, bg, c, label))

    edges_data = []
    for src, dst, data in G.edges(data=True):
        w     = data.get("weight", 0.5)
        width = max(0.5, 2.2 - w * 1.3)
        edges_data.append((src, dst, width, w))

    html = get_pyvis_html(id(cn), tuple(nodes_data), tuple(edges_data), None)

    col_g, col_l = st.columns([4, 1], gap="medium")

    with col_g:
        sec_head("MST Network — Minimum Spanning Tree")
        
        import base64
        html_b64 = base64.b64encode(html.encode('utf-8')).decode('utf-8')
        st.markdown(f'<iframe src="data:text/html;base64,{html_b64}" width="100%" height="600" style="border:none;border-radius:8px;"></iframe>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📐 Algorithm Pipeline")
        
        st.markdown("""
        <div style="background:linear-gradient(90deg,#38bdf820,#38bdf810);border-left:4px solid #38bdf8;border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;">
        <b style="color:#38bdf8;font-size:1rem;">1. Log Returns</b>
        <span style="color:#94a3b8;font-size:0.9rem;"> — Calculate daily returns from price data</span>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)")
        
        st.markdown("""
        <div style="background:linear-gradient(90deg,#f472b620,#f472b610);border-left:4px solid #f472b6;border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;">
        <b style="color:#f472b6;font-size:1rem;">2. Correlation</b>
        <span style="color:#94a3b8;font-size:0.9rem;"> — Compute Pearson correlation between all stock pairs</span>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\rho_{ij} = \frac{\text{Cov}(r_i,r_j)}{\sigma_i \sigma_j}")
        
        st.markdown("""
        <div style="background:linear-gradient(90deg,#34d39920,#34d39910);border-left:4px solid #34d399;border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;">
        <b style="color:#34d399;font-size:1rem;">3. Distance</b>
        <span style="color:#94a3b8;font-size:0.9rem;"> — Convert correlation to distance metric</span>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"d_{ij} = \sqrt{2(1 - \rho_{ij})}")
        
        st.markdown("""
        <div style="background:linear-gradient(90deg,#fbbf2420,#fbbf2410);border-left:4px solid #fbbf24;border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;">
        <b style="color:#fbbf24;font-size:1rem;">4. MST (Minimum Spanning Tree)</b>
        <span style="color:#94a3b8;font-size:0.9rem;"> — Kruskal's algorithm to remove noise edges</span>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"T = \text{MST}(d_{ij})")
        
        st.markdown("""
        <div style="background:linear-gradient(90deg,#a78bfa20,#a78bfa10);border-left:4px solid #a78bfa;border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;">
        <b style="color:#a78bfa;font-size:1rem;">5. Louvain Community Detection</b>
        <span style="color:#94a3b8;font-size:0.9rem;"> — Find clusters by maximizing modularity Q</span>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"Q = \frac{1}{2m}\sum_{ij}(A_{ij}-\frac{k_i k_j}{2m})\delta(c_i,c_j)")

    with col_l:
        sec_head("Communities")
        for comm_id, members in sorted(communities.items(), key=lambda x: -len(x[1])):
            c = clr(comm_id)
            bg = hex_to_rgba(c, 0.1)
            border = hex_to_rgba(c, 0.35)
            st.markdown(
                f'<div class="comm-block" style="background:{bg};border-color:{border};">'
                f'<span style="color:{c};font-weight:600;font-size:0.82rem;">● Comm {comm_id}</span>'
                f'<span style="color:#64748b;font-size:0.7rem;margin-left:0.4rem;">({len(members)})</span>'
                f'<div style="color:#94a3b8;font-size:0.72rem;margin-top:0.3rem;line-height:1.6;">'
                f'{", ".join(sorted(members))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        divider()
        st.markdown(
            '<span style="color:#475569;font-size:0.73rem;line-height:2;">'
            'Node size → Eigenvector<br>Edge width → Correlation</span>',
            unsafe_allow_html=True,
        )


def tab_insights(cn, prices):
    centrality  = get_centrality(cn)
    communities = cn.get_louvain_communities()
    eigen_all   = centrality["eigenvector"]
    betw_all    = centrality["betweenness"]

    col_c, col_m = st.columns([3, 2], gap="large")

    with col_c:
        sec_head("Community Breakdown")
        for comm_id, members in sorted(communities.items(), key=lambda x: -len(x[1])):
            c     = clr(comm_id)
            hub   = max(members, key=lambda m: eigen_all.get(m, 0))
            label = f"Community {comm_id} · {len(members)} stocks · Hub: **{hub}**"
            with st.expander(label):
                rows = sorted(members, key=lambda m: -eigen_all.get(m, 0))
                max_e = max((eigen_all.get(m, 0) for m in rows), default=1) or 1
                for m in rows:
                    e = eigen_all.get(m, 0)
                    b = betw_all.get(m, 0)
                    bar = int(e / max_e * 100)
                    st.markdown(
                        f'<div class="hub-row">'
                        f'<span class="hub-ticker"><span style="color:{c}">●</span> {m}</span>'
                        f'<span style="display:flex;gap:1rem;">'
                        f'<span style="color:#64748b;font-size:0.75rem;">eigen <span style="color:{c}">{e:.3f}</span></span>'
                        f'<span style="color:#64748b;font-size:0.75rem;">betw <span style="color:#a78bfa">{b:.3f}</span></span>'
                        f'</span></div>'
                        f'<div class="bar-wrap"><div class="bar-fill" style="width:{bar}%;background:{c};"></div></div>',
                        unsafe_allow_html=True,
                    )

    with col_m:
        sec_head("Network Metrics")
        col_a, col_b = st.columns(2)
        with col_a:
            stat_card("Modularity", f"{get_modularity(id(cn), cn):.3f}", "community separation")
            st.write("")
            stat_card("Nodes", cn.G.number_of_nodes(), "stocks in MST")
        with col_b:
            stat_card("Communities", len(communities), "Louvain clusters")
            st.write("")
            stat_card("Edges", cn.G.number_of_edges(), "MST connections")

        divider()
        sec_head("Top Hubs by Eigenvector")
        sorted_hubs = sorted(eigen_all.items(), key=lambda x: -x[1])[:8]
        max_e = sorted_hubs[0][1] if sorted_hubs else 1
        for rank, (stock, score) in enumerate(sorted_hubs, 1):
            c   = clr(cn.communities.get(stock, 0))
            bar = int(score / max_e * 100)
            st.markdown(
                f'<div class="hub-row">'
                f'<span style="color:#475569;font-size:0.73rem;width:1.2rem">{rank}</span>'
                f'<span class="hub-ticker" style="flex:1;margin-left:0.4rem;">{stock}</span>'
                f'<span class="hub-val">{score:.4f}</span>'
                f'</div>'
                f'<div class="bar-wrap"><div class="bar-fill" style="width:{bar}%;background:{c};"></div></div>',
                unsafe_allow_html=True,
            )

        divider()

        top_stock = sorted_hubs[0][0] if sorted_hubs else None
        if top_stock and top_stock in prices.columns:
            sec_head(f"{top_stock} — Close Price")
            # Resample to weekly so the interactive chart doesn't lag on pan/zoom
            chart_df = prices[[top_stock]].dropna()
            if not chart_df.empty:
                # chart_df = chart_df.resample('W').last()

                if show_sma:
                    chart_df['SMA_50'] = chart_df[top_stock].rolling(window=50).mean()
                    chart_df['SMA_200'] = chart_df[top_stock].rolling(window=200).mean()
                
                if show_bb:
                    window = 20
                    chart_df['SMA_20'] = chart_df[top_stock].rolling(window=window).mean()
                    chart_df['BB_Upper'] = chart_df['SMA_20'] + chart_df[top_stock].rolling(window=window).std() * 2
                    chart_df['BB_Lower'] = chart_df['SMA_20'] - chart_df[top_stock].rolling(window=window).std() * 2

            st.line_chart(chart_df, height=160)


def tab_anomalies(cn):
    anomalies = get_anomalies(id(cn), cn)
    
    filtered = [a for a in anomalies if a['is_anomaly']]
    unknown = [a for a in anomalies if a['official_sector'] == "Unknown"]
    
    col_h, col_a = st.columns([2, 1], gap="large")
    
    with col_h:
        sec_head("Anomaly Detection — Stocks Trading Outside Their Sector")
        st.markdown(
            '<div class="box-info" style="margin-bottom:1rem;">'
            '<b>Hidden Links:</b> These stocks are grouped by correlation with stocks from <i>different</i> official sectors. '
            'This may indicate non-obvious market relationships (e.g., Hydro stocks trading with Finance companies).'
            '</div>',
            unsafe_allow_html=True
        )
        
        if not filtered:
            st.markdown(
                '<div class="box-success">No significant anomalies detected. '
                'Most stocks are trading with their expected sector peers.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"**{len(filtered)} hidden-link anomalies detected**:")
            
            for a in filtered[:15]:
                c = clr(a['community'])
                with st.expander(f"**{a['ticker']}** — {a['official_sector']} → {a['majority_sector']} ({a['sector_match_pct']:.0f}% match)"):
                    st.markdown(f"**Community:** {a['community']} · **Connections:** {a['num_neighbors']}")
                    st.markdown(f"**Official Sector:** {a['official_sector']}")
                    st.markdown(f"**Trading With:** {a['majority_sector']}")
                    st.markdown(f"**Sector Match:** {a['sector_match_pct']:.0f}%")
                    neighbor_list = ", ".join(a['neighbors'][:12]) + ("..." if len(a['neighbors']) > 12 else "")
                    st.markdown(f"**Neighbors:** {neighbor_list}")
            
            if len(filtered) > 15:
                st.caption(f"Showing top 15 of {len(filtered)} anomalies.")
    
    with col_a:
        sec_head("Anomaly Summary")
        
        stat_card("Total Anomalies", len(filtered), "hidden links")
        st.write("")
        stat_card("Unclassified", len(unknown), "no sector data")
        
        sec_head("Anomalies by Sector")
        sector_counts = {}
        for a in filtered:
            sector = a['official_sector']
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            pct = int(count / len(filtered) * 100) if filtered else 0
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;color:#94a3b8;font-size:0.8rem;">'
                f'<span>{sector}</span><span style="color:#f472b6;">{count}</span></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct}%;background:#f472b6;"></div></div>',
                unsafe_allow_html=True
            )
        
        divider()
        sec_head("Methodology")
        st.markdown(
            '<div style="color:#64748b;font-size:0.75rem;line-height:1.6;">'
            '<b>Detection:</b> A stock is flagged if <50% of its MST neighbors share its official sector.<br><br>'
            '<b>Insight:</b> These hidden links reveal market relationships not visible in traditional sector classifications.'
            '</div>',
            unsafe_allow_html=True
        )


@st.fragment
def tab_portfolio(cn, prices):
    available = sorted(prices.columns.tolist())

    col_sel, col_res = st.columns([1, 2], gap="large")

    with col_sel:
        sec_head("Select Portfolio")
        portfolio = st.multiselect(
            "Stocks",
            available,
            default=available[:5],
            label_visibility="collapsed",
        )

    with col_res:
        if len(portfolio) == 0:
            st.markdown("""
            <div class="box-info" style="text-align:center;padding:2rem;">
            <b>📊 Select stocks to analyze</b><br>
            <span style="color:#94a3b8;">Use the selector on the left to choose stocks for portfolio analysis</span>
            </div>
            """, unsafe_allow_html=True)
            return
        
        if len(portfolio) < 2:
            st.markdown("""
            <div class="box-warn" style="text-align:center;padding:2rem;">
            <b>⚠️ Need more stocks</b><br>
            <span style="color:#94a3b8;">Select at least 2 stocks to analyze diversification and correlation</span>
            </div>
            """, unsafe_allow_html=True)
            return

        sec_head("Diversification Analysis")

        # Map each stock to community
        comm_map: dict[int, list] = {}
        for s in portfolio:
            cid = cn.communities.get(s, -1)
            comm_map.setdefault(cid, []).append(s)

        unique = len(comm_map)
        score  = unique / len(portfolio)
        pct    = int(score * 100)

        if score >= 0.6:
            box_cls, verdict, icon = "box-success", "Well diversified", "✅"
        elif score >= 0.35:
            box_cls, verdict, icon = "box-info", "Moderate diversification", "ℹ️"
        else:
            box_cls, verdict, icon = "box-warn", "Concentrated — consider spreading across communities", "⚠️"

        # Score bar
        st.markdown(
            f'<div style="margin-bottom:0.9rem;">'
            f'<div class="stat-label">Diversification Score</div>'
            f'<div class="stat-val">{pct}%</div>'
            f'<div class="bar-wrap" style="height:6px;margin-top:0.5rem;">'
            f'<div class="bar-fill" style="width:{pct}%;height:6px;background:linear-gradient(90deg,#f87171,#facc15,#34d399);"></div>'
            f'</div>'
            f'<div class="stat-sub" style="margin-top:0.3rem;">{unique} communities / {len(portfolio)} stocks</div>'
            f'</div>'
            f'<div class="{box_cls}" style="margin-bottom:1rem;">{icon} {verdict}</div>',
            unsafe_allow_html=True,
        )

        # Correlation analysis
        corr_matrix = cn.get_correlation_matrix()
        if corr_matrix is not None and all(s in corr_matrix.columns for s in portfolio):
            portfolio_corr = corr_matrix.loc[portfolio, portfolio]
            
            # Get upper triangle of the correlation matrix (excluding the diagonal)
            upper_tri = portfolio_corr.where(np.triu(np.ones(portfolio_corr.shape), k=1).astype(bool))
            
            # Calculate the average pairwise correlation
            avg_corr = upper_tri.stack().mean()

            if avg_corr > 0.6:
                st.markdown(
                    f'<div class="box-warn" style="margin-top: 1rem;">'
                    f'⚠️ <b>High Risk:</b> These stocks have an average pairwise correlation of <b>{avg_corr:.2f}</b>. '
                    f'A drop in one will likely impact the others.'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Per-community breakdown
        for cid, members in sorted(comm_map.items(), key=lambda x: -len(x[1])):
            c      = clr(cid)
            bg     = hex_to_rgba(c, 0.08)
            border = hex_to_rgba(c, 0.28)
            pct_c  = int(len(members) / len(portfolio) * 100)
            pills  = "".join(
                f'<span style="background:{hex_to_rgba(c,0.15)};border:1px solid {hex_to_rgba(c,0.4)};'
                f'color:{c};border-radius:5px;padding:0.15rem 0.55rem;font-size:0.75rem;margin:0.15rem;">{m}</span>'
                for m in members
            )
            st.markdown(
                f'<div class="comm-block" style="background:{bg};border-color:{border};margin-bottom:0.5rem;">'
                f'<span style="color:{c};font-weight:600;font-size:0.82rem;">Community {cid}</span>'
                f'<div style="margin-top:0.4rem;line-height:2;">{pills}</div>'
                f'<div class="bar-wrap" style="margin-top:0.5rem;">'
                f'<div class="bar-fill" style="width:{pct_c}%;background:{c};"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Normalised price chart — use pre-computed cache, just slice the columns
        if all(s in prices.columns for s in portfolio):
            divider()
            sec_head("Normalised Price (base = 100)")
            norm_dict, dates = get_normalised_prices(id(prices), prices)
            chart_df = pd.DataFrame(
                {s: norm_dict[s] for s in portfolio if s in norm_dict},
                index=pd.to_datetime(dates),
            )
            # Resample to weekly so the interactive chart doesn't lag on pan/zoom
            if not chart_df.empty:
                chart_df = chart_df.resample('W').last()
            st.line_chart(chart_df, height=220)


        # Backtester
        divider()
        sec_head("Portfolio Backtester")
        
        backtest_years = st.select_slider(
            "Investment Horizon (Years)",
            options=[1, 2, 3, 4, 5],
            value=3
        )
        
        initial_investment = 100_000
        
        # Filter prices for the backtest period
        cutoff_date = prices.index[-1] - pd.DateOffset(years=backtest_years)
        backtest_prices = prices[prices.index >= cutoff_date]
        
        if not backtest_prices.empty and all(s in backtest_prices.columns for s in portfolio):
            portfolio_prices = backtest_prices[portfolio]
            
            # Normalize prices to the start of the backtest period
            norm_prices = portfolio_prices / portfolio_prices.iloc[0]
            
            # Equal weight investment
            num_stocks = len(portfolio)
            investment_per_stock = initial_investment / num_stocks
            
            # Calculate the value of each stock over time
            position_values = norm_prices * investment_per_stock
            
            # Calculate total portfolio value
            portfolio_value = position_values.sum(axis=1)
            
            start_val = portfolio_value.iloc[0]
            end_val = portfolio_value.iloc[-1]
            
            st.line_chart(portfolio_value, height=200)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Initial Investment", f"Rs. {start_val:,.0f}")
            c2.metric("Final Value", f"Rs. {end_val:,.0f}")
            
            # Calculate percentage gain/loss
            pct_change = ((end_val - start_val) / start_val) * 100
            c3.metric("Total Return", f"{pct_change:.2f}%")

        divider()
        sec_head("Export Data")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            corr_matrix = cn.get_correlation_matrix()
            if corr_matrix is not None and all(s in corr_matrix.columns for s in portfolio):
                portfolio_corr = corr_matrix.loc[portfolio, portfolio]
                csv_corr = portfolio_corr.to_csv()
                st.download_button(
                    "📥 Download Correlation Matrix (CSV)",
                    data=csv_corr,
                    file_name="nepgraph_correlation_matrix.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col_exp2:
            centrality = get_centrality(cn)
            eigen_c = centrality.get("eigenvector", {})
            betw_c = centrality.get("betweenness", {})
            hub_data = []
            for s in portfolio:
                hub_data.append({
                    "Ticker": s,
                    "Community": cn.communities.get(s, -1),
                    "Eigenvector": eigen_c.get(s, 0),
                    "Betweenness": betw_c.get(s, 0),
                })
            hub_df = pd.DataFrame(hub_data)
            csv_hub = hub_df.to_csv(index=False)
            st.download_button(
                "📥 Download Hub Scores (CSV)",
                data=csv_hub,
                file_name="nepgraph_hub_scores.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import time
    start_time = time.time()

    with st.spinner("Loading market data…"):
        prices = load_data(_LOOKBACK_YEARS)
    
    if prices.empty:
        st.stop()

    with st.spinner("Building MST network…"):
        cn = build_network(prices)

    if cn is None or cn.G is None:
        st.error("Failed to build network. Try a different time range.")
        st.stop()

    communities = cn.get_louvain_communities()
    compute_time = time.time() - start_time

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 📊 NepGraph")
    st.caption("Nepal Stock Exchange · Minimum Spanning Tree · Louvain Community Detection")

    divider()

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1: stat_card("Stocks", cn.G.number_of_nodes(), "in MST")
    with c2: stat_card("Communities", len(communities), "Louvain clusters")
    with c3: 
        mod = get_modularity(id(cn), cn)
        quality = "✓ Good" if mod > 0.4 else "○ Fair"
        stat_card("Modularity", f"{mod:.3f}", quality)
    with c4: 
        d1 = pd.to_datetime(prices.index[0]).strftime("%b %Y")
        d2 = pd.to_datetime(prices.index[-1]).strftime("%b %Y")
        stat_card("Date Range", f"{d1} → {d2}")
    with c5: stat_card("Compute", f"{compute_time:.2f}s", f"{cn.G.number_of_edges()} edges")

    divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Network Map", "Market Insights", "Anomaly Report", "Portfolio Checker"])

    with tab1:
        tab_network(cn)

    with tab2:
        tab_insights(cn, prices)

    with tab3:
        tab_anomalies(cn)

    with tab4:
        tab_portfolio(cn, prices)


if __name__ == "__main__":
    main()