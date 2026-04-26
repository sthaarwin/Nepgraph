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

/* Remove white border from pyvis iframe */
iframe { border: none !important; background: transparent !important; }
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

def clr(comm_id: int) -> str:
    return COLORS[comm_id % len(COLORS)]

def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ── Data / network helpers ────────────────────────────────────────────────────
_LOOKBACK_YEARS = 5

@st.cache_data(show_spinner=False)
def load_data():
    dm = DataManager(data_path="data/nepse_prices.csv")
    raw = dm.get_data()

    # Trim to last 5 years (removes ~80 % of zero-padded rows)
    raw.index = pd.to_datetime(raw.index)
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=_LOOKBACK_YEARS)
    trimmed = raw[raw.index >= cutoff]

    # Drop columns that are >40 % missing in this window, then fill gaps
    trimmed = trimmed.loc[:, trimmed.isna().mean() < 0.4]
    trimmed = trimmed.ffill().bfill().astype('float32')
    return trimmed


@st.cache_resource(show_spinner=False)
def build_network(_prices):
    prices = _prices.astype('float64')

    # Drop constant or near-constant columns — they produce NaN correlations
    prices = prices.loc[:, prices.std() > 0]
    prices = prices.loc[:, prices.nunique() >= 30]

    cn = CorrelationNetwork()
    cn.calculate_log_returns(prices)

    # Drop any column whose full log-return series is NaN or zero-variance
    if cn.log_returns is not None:
        cn.log_returns = cn.log_returns.dropna(axis=1, how='all')
        cn.log_returns = cn.log_returns.loc[:, cn.log_returns.std() > 0]
        cn.price_data  = prices[cn.log_returns.columns]

    cn.get_correlation_matrix()
    cn.get_distance_matrix()

    # Fill any residual NaN in the distance matrix (e.g. perfect correlation = 0 dist)
    if cn.distance_matrix is not None:
        cn.distance_matrix = cn.distance_matrix.fillna(2.0)

    cn.build_mst()
    cn.get_louvain_communities()
    return cn


@st.cache_resource(show_spinner=False)
def get_centrality(_cn):
    """Expensive for 100+ nodes — cache it."""
    return _cn.get_centrality()


@st.cache_data(show_spinner=False)
def get_modularity(_cn_id, _cn):
    return _cn.get_modularity()


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
    net = Network(height="560px", bgcolor="#0f172a", font_color="#cbd5e1", notebook=True)

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

    # Fewer iterations + disable physics after stabilisation = much faster
    net.set_options("""{
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -6000,
                "centralGravity": 0.2,
                "springLength": 100,
                "damping": 0.18
            },
            "stabilization": {"iterations": 120, "updateInterval": 50},
            "minVelocity": 0.75
        },
        "interaction": {"hover": true, "tooltipDelay": 120},
        "edges": {"smooth": {"type": "continuous", "roundness": 0}}
    }""")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as tmp:
        tmp_name = tmp.name
    net.save_graph(tmp_name)
    with open(tmp_name, "r") as f:
        html = f.read()
    os.unlink(tmp_name)

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
        st.components.v1.html(html, height=580)

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
            st.line_chart(prices[[top_stock]].dropna(), height=160, use_container_width=True)


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
        if len(portfolio) < 2:
            st.markdown('<div class="box-info">Select at least 2 stocks to analyse diversification.</div>', unsafe_allow_html=True)
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
            st.line_chart(chart_df, height=220, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Loading market data…"):
        prices = load_data()

    with st.spinner("Building MST network…"):
        cn = build_network(prices)

    communities = cn.get_louvain_communities()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 📊 NepGraph")
    st.caption("Nepal Stock Exchange · Minimum Spanning Tree · Louvain Community Detection")

    divider()

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1: stat_card("Stocks", cn.G.number_of_nodes())
    with c2: stat_card("Communities", len(communities))
    with c3: stat_card("Modularity", f"{get_modularity(id(cn), cn):.3f}")
    with c4: stat_card("Date Range", f"{prices.index[0]} → {prices.index[-1]}")

    divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Network Map", "Market Insights", "Portfolio Checker"])

    with tab1:
        tab_network(cn)

    with tab2:
        tab_insights(cn, prices)

    with tab3:
        tab_portfolio(cn, prices)


if __name__ == "__main__":
    main()