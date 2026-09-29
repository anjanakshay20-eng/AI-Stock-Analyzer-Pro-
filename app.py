"""AI Stock Analyzer Pro — v5 Complete."""

from __future__ import annotations
import concurrent.futures, io, json, logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, streamlit as st, yfinance as yf

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    go = None; px = None; make_subplots = None

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

# ============ CONSTANTS ============
APP_TITLE = "AI Stock Analyzer Pro"
FUNDAMENTAL_PREFIX = "Nifty"
TECHNICAL_PREFIX = "Explore_Promising"
PRICE_PERIOD = "1y"
CACHE_TTL_SECONDS = 60 * 15
AUTO_REFRESH_MIN = 15
MAX_WORKERS = 8
DATA_DIR_NAME = "data"
INDEX_JSON_NAME = "index_constituents.json"
SNAPSHOT_DIR_NAME = "sector_snapshots"
TAB_ORDER_FILE = "tab_order.json"

RISK_FREE_RATE = 0.065
EQUITY_RISK_PREMIUM = 0.06
TERMINAL_GROWTH = 0.035
VALUATION_BAND = 0.15
BENCHMARK_TICKER = "^NSEI"
TRADING_DAYS = 252
SECTOR_COLUMN_CANDIDATES = ["Sub-Sector", "Sector", "Sector Name"]

# ⭐ CHANGE 3: Tab registry for reordering
TAB_META = {
    "screener": {"label": "🎯 Screener"},
    "dcf":      {"label": "💰 DCF"},
    "deep":     {"label": "🔬 Deep"},
    "hybrid":   {"label": "🔀 Hybrid"},  # ⭐ CHANGE 2
    "trade":    {"label": "💼 Trade"},
    "peers":    {"label": "👥 Peers"},
    "indices":  {"label": "📊 Indices"},
    "patterns": {"label": "📐 Patterns"},
    "quality":  {"label": "🔍 Quality"},
}
DEFAULT_TAB_ORDER = ["screener", "dcf", "deep", "hybrid", "trade",
                     "peers", "indices", "patterns", "quality"]

# ⭐ CHANGE 1: All chart indicators
CHART_OVERLAYS = {
    "SMA 20":  {"color": "#fbbf24", "kind": "sma", "window": 20,  "col": "20 DMA"},
    "SMA 50":  {"color": "#60a5fa", "kind": "sma", "window": 50,  "col": "50 DMA"},
    "SMA 200": {"color": "#a78bfa", "kind": "sma", "window": 200, "col": "200 DMA"},
    "EMA 20":  {"color": "#f97316", "kind": "ema", "window": 20,  "col": "EMA20"},
    "EMA 50":  {"color": "#06b6d4", "kind": "ema", "window": 50,  "col": "EMA50"},
    "EMA 200": {"color": "#ec4899", "kind": "ema", "window": 200, "col": "EMA200"},
    "BB Upper": {"color": "rgba(139,163,192,0.7)", "kind": "bb_upper", "window": 20, "col": "Upper Band"},
    "BB Middle": {"color": "rgba(139,163,192,0.45)", "kind": "bb_middle", "window": 20, "col": "Middle Band"},
    "BB Lower": {"color": "rgba(139,163,192,0.7)", "kind": "bb_lower", "window": 20, "col": "Lower Band"},
    "SuperTrend": {"color": "#10b981", "kind": "supertrend", "col": "SuperTrend_Val"},
    "VWAP": {"color": "#f59e0b", "kind": "vwap"},
    "PSAR": {"color": "#8b5cf6", "kind": "psar"},
}
CHART_SUBPLOTS = ["Volume", "RSI", "MACD", "Stochastic", "ADX", "ATR", "OBV"]

FUNDAMENTAL_THRESHOLDS = {
    "PE Ratio": (15, 25, 35, 50, True), "PB Ratio": (1.5, 3, 5, 8, True),
    "EV/EBITDA Ratio": (8, 12, 18, 25, True), "Return on Equity": (20, 15, 10, 5, False),
    "ROCE": (20, 15, 10, 5, False), "Net Profit Margin": (15, 10, 5, 2, False),
    "EBITDA Margin": (20, 15, 10, 5, False), "5Y Historical EPS Growth": (20, 15, 10, 5, False),
    "5Y Historical Revenue Growth": (15, 10, 5, 0, False),
    "5Y Historical EBITDA Growth": (15, 10, 5, 0, False), "5Y CAGR": (15, 10, 5, 0, False),
    "Debt to Equity": (0.2, 0.5, 1.0, 2.0, True), "Current Ratio": (2.0, 1.5, 1.0, 0.5, False),
    "Promoter Holding": (50, 40, 30, 20, False), "Pledged Promoter Holdings": (0, 5, 10, 25, True),
    "Dividend Yield": (3, 2, 1, 0.5, False), "Percentage Upside": (20, 10, 5, 0, False),
}
SECTOR_RELATIVE_METRICS = {
    "Return on Equity": (True, 0.18), "ROCE": (True, 0.12),
    "5Y Historical EPS Growth": (True, 0.15), "Net Profit Margin": (True, 0.10),
    "PE Ratio": (False, 0.15), "PB Ratio": (False, 0.10),
    "EV/EBITDA Ratio": (False, 0.08), "Debt to Equity": (False, 0.07),
    "Dividend Yield": (True, 0.05),
}
RED_FLAG_RULES = {
    "High Promoter Pledge": ("Pledged Promoter Holdings", "gt", 10),
    "Excessive Leverage": ("Debt to Equity", "gt", 1.5),
    "Weak Profitability": ("Return on Equity", "lt", 8),
    "Negative FCF": ("Free Cash Flow", "lt", 0),
    "Stretched Valuation": ("PE Ratio", "gt", 60),
    "Low Promoter Holding": ("Promoter Holding", "lt", 25),
    "Weak Margins": ("Net Profit Margin", "lt", 3),
    "Overbought RSI": ("RSI Exponential - 14D", "gt", 80),
    "Below 200 DMA": ("CMP", "below_dma200", 0),
    "Low Current Ratio": ("Current Ratio", "lt", 0.8),
}
PATTERN_NAMES = ["Golden Cross", "Death Cross", "52W Breakout", "52W Breakdown",
                 "Volume Spike", "Near Support", "Near Resistance", "Above All MAs"]

DEFAULT_INDEX_CONSTITUENTS = {
    "Nifty 50": ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
        "BHARTIARTL","KOTAKBANK","BAJFINANCE","LT","HCLTECH","ASIANPAINT","AXISBANK","MARUTI",
        "SUNPHARMA","TITAN","ULTRACEMCO","WIPRO","NESTLEIND","ONGC","NTPC","POWERGRID","M&M",
        "TATAMOTORS","TATASTEEL","JSWSTEEL","ADANIENT","ADANIPORTS","COALINDIA","GRASIM",
        "HINDALCO","DRREDDY","CIPLA","EICHERMOT","BRITANNIA","DIVISLAB","TECHM","INDUSINDBK",
        "BAJAJFINSV","HEROMOTOCO","SBILIFE","HDFCLIFE","APOLLOHOSP","BPCL","TATACONSUM",
        "BAJAJ-AUTO","LTIM","SHRIRAMFIN"],
    "Nifty Next 50": ["ADANIENSOL","ADANIGREEN","ADANIPOWER","AMBUJACEM","DMART","BAJAJHLDNG",
        "BANKBARODA","BERGEPAINT","BEL","BOSCHLTD","CANBK","CHOLAFIN","COLPAL","DABUR","DLF",
        "GAIL","GODREJCP","GODREJPROP","HAVELLS","HDFCAMC","HINDZINC","ICICIGI","ICICIPRULI",
        "IOC","INDHOTEL","INDIGO","JINDALSTEL","JIOFIN","LICI","LODHA","MARICO","MOTHERSON",
        "NAUKRI","PFC","PIDILITIND","PIIND","PNB","RECLTD","SIEMENS","SRF","TVSMOTOR",
        "TORNTPHARM","VBL","VEDL","ZYDUSLIFE"],
    "Bank Nifty": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","INDUSINDBK",
        "BANKBARODA","PNB","IDFCFIRSTB","FEDERALBNK","AUBANK","BANDHANBNK"],
}

# ============ TAB ORDER PERSISTENCE (CHANGE 3) ============
def _tab_order_path(p): return p / DATA_DIR_NAME / TAB_ORDER_FILE

def load_tab_order(project_dir):
    path = _tab_order_path(project_dir)
    if not path.exists(): return DEFAULT_TAB_ORDER.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        order = [k for k in data.get("order", []) if k in TAB_META]
        for k in DEFAULT_TAB_ORDER:
            if k not in order: order.append(k)
        return order
    except Exception: return DEFAULT_TAB_ORDER.copy()

def save_tab_order(project_dir, order):
    try:
        path = _tab_order_path(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"order": order}, indent=2), encoding="utf-8")
    except OSError as exc: LOGGER.warning("Tab order save failed: %s", exc)

def render_tab_order_ui(project_dir):
    if "tab_order" not in st.session_state:
        st.session_state["tab_order"] = load_tab_order(project_dir)
    order = st.session_state["tab_order"]
    with st.sidebar:
        with st.expander("📑 Reorder Tabs", expanded=False):
            st.caption("Use ⬆️/⬇️ to move tabs. Order saves automatically.")
            for i, key in enumerate(order):
                c1, c2, c3 = st.columns([6, 1, 1])
                with c1: st.markdown(f"**{i+1}.** {TAB_META[key]['label']}")
                with c2:
                    if st.button("⬆️", key=f"up_{key}", disabled=(i == 0)):
                        order[i], order[i-1] = order[i-1], order[i]
                        st.session_state["tab_order"] = order
                        save_tab_order(project_dir, order)
                        st.rerun()
                with c3:
                    if st.button("⬇️", key=f"dn_{key}", disabled=(i == len(order)-1)):
                        order[i], order[i+1] = order[i+1], order[i]
                        st.session_state["tab_order"] = order
                        save_tab_order(project_dir, order)
                        st.rerun()
            st.divider()
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state["tab_order"] = DEFAULT_TAB_ORDER.copy()
                save_tab_order(project_dir, DEFAULT_TAB_ORDER.copy())
                st.rerun()
    return order

# ============ COLUMN NORMALIZATION ============
_UNICODE_FIX = str.maketrans({"’":"'","‘":"'","“":'"',"”":'"',"–":"-","—":"-","\u00a0":" "})
def normalise_columns(df):
    return df.rename(columns={c: str(c).translate(_UNICODE_FIX).strip() for c in df.columns})
def detect_sector_column(df):
    for c in SECTOR_COLUMN_CANDIDATES:
        if c in df.columns: return c
    return None

# ============ INDEX JSON ============
def _index_json_path(p): return p / DATA_DIR_NAME / INDEX_JSON_NAME

def load_index_constituents(project_dir):
    path = _index_json_path(project_dir)
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(DEFAULT_INDEX_CONSTITUENTS, indent=2), encoding="utf-8")
        except OSError: return {k: frozenset(v) for k, v in DEFAULT_INDEX_CONSTITUENTS.items()}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        indices = {str(n): frozenset(str(t).upper().replace(".NS","").replace(".BO","").strip()
                   for t in tickers if str(t).strip()) for n, tickers in raw.items()}
        return indices or {k: frozenset(v) for k, v in DEFAULT_INDEX_CONSTITUENTS.items()}
    except Exception: return {k: frozenset(v) for k, v in DEFAULT_INDEX_CONSTITUENTS.items()}

def _normalise_ticker(t):
    return str(t).upper().replace(".NS","").replace(".BO","").strip()

# ============ HELPERS ============
def numeric_column(frame, name, default=np.nan):
    if name not in frame: return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")

def find_latest_csv(directory, prefix):
    dirs = [directory, directory / DATA_DIR_NAME]
    matches = [p for f in dirs if f.exists() for p in f.glob(f"*{prefix}*.csv")]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None

def persist_uploaded_csv(upload, prefix, directory):
    if upload is None: return None
    d = directory / DATA_DIR_NAME; d.mkdir(parents=True, exist_ok=True)
    fn = "Nifty_Fundamentals.csv" if prefix == FUNDAMENTAL_PREFIX else "Explore_Promising.csv"
    p = d / fn; p.write_bytes(upload.getvalue()); return p

def read_csv_source(upload, fallback, label):
    src = upload if upload is not None else fallback
    if src is None: raise FileNotFoundError(f"{label} CSV nahi mili.")
    try: frame = pd.read_csv(src)
    except Exception as exc: raise ValueError(f"{label} CSV read nahi hui: {exc}")
    frame = normalise_columns(frame)
    if frame.empty: raise ValueError(f"{label} CSV khaali hai.")
    if "Ticker" not in frame.columns: raise ValueError(f"{label} mein 'Ticker' nahi hai.")
    frame["Ticker"] = frame["Ticker"].astype("string").str.strip().str.upper()
    frame = frame.loc[frame["Ticker"].notna() & frame["Ticker"].ne("")].copy()
    return frame.drop_duplicates(subset="Ticker", keep="last").reset_index(drop=True)

def merge_sources(fund, tech):
    common = (set(fund.columns) & set(tech.columns)) - {"Ticker"}
    fund_clean = fund.copy()
    for col in common:
        if tech[col].isna().sum() < fund[col].isna().sum():
            fund_clean = fund_clean.drop(columns=[col])
    merged = fund_clean.merge(tech, on="Ticker", how="inner")
    if merged.empty: raise ValueError("Dono CSV mein matching Ticker nahi mile.")
    return merged

# ============ SCORING ============
def calculate_fundamental_score(frame):
    total = pd.Series(0.0, index=frame.index)
    count = pd.Series(0, index=frame.index, dtype="int64")
    for col, (best, good, fair, weak, rev) in FUNDAMENTAL_THRESHOLDS.items():
        v = numeric_column(frame, col)
        bins = ([v.le(best), v.le(good), v.le(fair), v.le(weak)] if rev
                else [v.ge(best), v.ge(good), v.ge(fair), v.ge(weak)])
        s = np.select(bins, [100, 75, 50, 25], default=0)
        valid = v.notna()
        total += pd.Series(s, index=frame.index).where(valid, 0)
        count += valid.astype("int64")
    return total.div(count.replace(0, np.nan)).fillna(50)

def calculate_sector_relative_score(frame, sector_col="Sub-Sector"):
    if sector_col not in frame.columns or frame[sector_col].isna().all():
        return pd.Series(50.0, index=frame.index)
    sectors = frame[sector_col].fillna("Unknown").astype(str)
    ws = pd.Series(0.0, index=frame.index); wt = pd.Series(0.0, index=frame.index)
    for metric, (hb, weight) in SECTOR_RELATIVE_METRICS.items():
        if metric not in frame.columns: continue
        v = pd.to_numeric(frame[metric], errors="coerce")
        pct = v.groupby(sectors).rank(pct=True, ascending=hb)
        valid = pct.notna()
        ws += (pct * 100 * weight).where(valid, 0)
        wt += pd.Series(weight, index=frame.index).where(valid, 0)
    return ws.div(wt.replace(0, np.nan)).fillna(50).clip(0, 100)

def _is_financial_sector(frame, sector_col):
    if sector_col not in frame.columns: return pd.Series(False, index=frame.index)
    s = frame[sector_col].fillna("").astype(str).str.lower()
    return (s.str.contains("bank", na=False) | s.str.contains("finance", na=False)
            | s.str.contains("insurance", na=False) | s.str.contains("nbfc", na=False))

def calculate_enhanced_2stage_dcf(frame):
    fcf = numeric_column(frame, "Free Cash Flow")
    ocf = numeric_column(frame, "Operating Cash Flow")
    capex = numeric_column(frame, "Capital Expenditure").abs()
    fcf = fcf.where(fcf.notna() & fcf.gt(0), ocf - capex)
    debt = numeric_column(frame, "Total Debt", 0).fillna(0)
    cash = numeric_column(frame, "Cash and Equivalent", 0).fillna(0)
    mcap = numeric_column(frame, "Market Cap"); close = numeric_column(frame, "Close Price")
    shares = numeric_column(frame, "Common Shares Outstanding")
    fallback = mcap.div(close.where(close.gt(0)))
    shares = shares.where(shares.gt(0), fallback).where(lambda s: s.gt(0))
    beta = numeric_column(frame, "Beta", 1.0).fillna(1.0).clip(0.5, 2.0)
    wacc = (RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM).clip(0.10, 0.14)
    g_hist = numeric_column(frame, "5Y Historical EPS Growth", 10).fillna(10).clip(5, 20) / 100
    g_near, g_far = g_hist, g_hist * 0.5
    pv_fcf = pd.Series(0.0, index=frame.index); cur = fcf.copy()
    for year in range(1, 11):
        g = g_near if year <= 5 else g_far
        cur = cur * (1 + g); pv_fcf += cur / ((1 + wacc) ** year)
    terminal_fcf = cur * (1 + TERMINAL_GROWTH)
    terminal_value = terminal_fcf / (wacc - TERMINAL_GROWTH).replace(0, np.nan)
    pv_terminal = terminal_value / ((1 + wacc) ** 10)
    eq = pv_fcf + pv_terminal - debt + cash
    intrinsic = eq / shares.where(shares.gt(0))
    sector_col = detect_sector_column(frame)
    if sector_col:
        is_fin = _is_financial_sector(frame, sector_col)
        pb = numeric_column(frame, "PB Ratio")
        bvps = close / pb.where(pb.gt(0))
        pb_med = pb.where(~is_fin).median()
        if pd.notna(pb_med) and pb_med > 0:
            intrinsic = intrinsic.where(~is_fin, bvps * pb_med)
    valid = fcf.gt(0) & shares.gt(0) & intrinsic.notna() & intrinsic.gt(0)
    intrinsic = intrinsic.where((intrinsic > close * 0.2) & (intrinsic < close * 5))
    return pd.DataFrame({"DCF Base": intrinsic.where(valid), "WACC Used": wacc.where(valid)},
                        index=frame.index)

def calculate_peer_multiples(frame, sector_col="Sub-Sector"):
    empty = pd.DataFrame({"Multiples Base": np.nan}, index=frame.index)
    if sector_col not in frame.columns or frame[sector_col].isna().all(): return empty
    sectors = frame[sector_col].fillna("Unknown").astype(str)
    close = numeric_column(frame, "Close Price"); pe = numeric_column(frame, "PE Ratio")
    pb = numeric_column(frame, "PB Ratio"); ev = numeric_column(frame, "EV/EBITDA Ratio")
    eps = close / pe.where(pe.gt(0)); bvps = close / pb.where(pb.gt(0))
    ebitda_ps = close / ev.where(ev.gt(0))
    pe_med = pe.groupby(sectors).transform("median")
    pb_med = pb.groupby(sectors).transform("median")
    ev_med = ev.groupby(sectors).transform("median")
    vals = [(ebitda_ps * ev_med, 0.5), (eps * pe_med, 0.3), (bvps * pb_med, 0.2)]
    ws = pd.Series(0.0, index=frame.index); wt = pd.Series(0.0, index=frame.index)
    for v, w in vals:
        valid = v.notna() & v.gt(0); ws += v.fillna(0) * w
        wt += pd.Series(w, index=frame.index).where(valid, 0)
    base = ws / wt.replace(0, np.nan)
    base = base.where((base > close * 0.2) & (base < close * 5))
    return pd.DataFrame({"Multiples Base": base}, index=frame.index)

def calculate_hybrid_valuation(frame):
    dcf = calculate_enhanced_2stage_dcf(frame)
    mult = calculate_peer_multiples(frame)
    dcf_base = dcf["DCF Base"]; mult_base = mult["Multiples Base"]
    blended = dcf_base * 0.6 + mult_base * 0.4
    blended = blended.fillna(dcf_base).fillna(mult_base)
    has_both = dcf_base.notna() & mult_base.notna()
    band = pd.Series(VALUATION_BAND, index=frame.index).where(has_both, 0.20)
    return pd.DataFrame({
        "Intrinsic Value (DCF)": dcf_base, "Intrinsic Value (Multiples)": mult_base,
        "Intrinsic Value (Base)": blended,
        "Lower Intrinsic Value": blended * (1 - band),
        "Upper Intrinsic Value": blended * (1 + band),
        "WACC Used": dcf["WACC Used"]}, index=frame.index)

def calculate_valuation_score(frame):
    cmp = numeric_column(frame, "CMP"); iv = numeric_column(frame, "Intrinsic Value (Base)")
    mos = (iv - cmp) / iv.where(iv.gt(0)) * 100
    score = pd.Series(np.select(
        [mos >= 40, mos >= 25, mos >= 10, mos >= -5, mos >= -15, mos >= -30],
        [100, 88, 72, 55, 38, 20], default=10), index=frame.index)
    return score.where(mos.notna(), 50)

def calculate_valuation_trigger_score(frame):
    cmp = numeric_column(frame, "CMP"); lower = numeric_column(frame, "Lower Intrinsic Value")
    base = numeric_column(frame, "Intrinsic Value (Base)")
    upper = numeric_column(frame, "Upper Intrinsic Value"); mid = (base + upper) / 2
    score = pd.Series(np.select(
        [cmp.le(lower*0.95), cmp.le(lower), cmp.le(base), cmp.le(mid), cmp.le(upper)],
        [100, 90, 70, 55, 35], default=10), index=frame.index)
    valid = cmp.notna() & lower.notna() & base.notna() & upper.notna()
    return score.where(valid, 50)

def calculate_rank_composite(frame):
    comps = []
    fs = numeric_column(frame, "Fundamental Score")
    if fs.notna().any(): comps.append((fs.where(fs > 15, fs * 10).clip(0, 100), 0.25))
    for col, w in [("Price Momentum Rank", 0.25), ("Value Momentum Rank", 0.20),
                   ("Earnings Quality Rank", 0.20), ("Price to Intrinsic Value Rank", 0.10)]:
        v = numeric_column(frame, col)
        if v.notna().any(): comps.append((v.clip(0, 100), w))
    if not comps: return pd.Series(50.0, index=frame.index)
    tw = sum(w for _, w in comps)
    return (sum(v * w for v, w in comps) / tw).fillna(50).clip(0, 100)

def calculate_analyst_consensus(frame):
    buy = numeric_column(frame, "Percentage Buy Reco's").clip(0, 100)
    sell = numeric_column(frame, "Percentage Sell Reco's").clip(0, 100)
    n = numeric_column(frame, "Total no. of analysts").fillna(0)
    base = (buy - sell * 0.5).clip(0, 100); cov = n.clip(0, 20) / 20 * 20
    return (base * 0.8 + cov).clip(0, 100).fillna(50)

def calculate_cashflow_quality(frame):
    fcf = numeric_column(frame, "Free Cash Flow"); ocf = numeric_column(frame, "Operating Cash Flow")
    capex = numeric_column(frame, "Capital Expenditure").abs()
    fin = numeric_column(frame, "Financing Cash Flow")
    g = numeric_column(frame, "5Y Hist Op. Cash Flow Growth")
    s = pd.Series(50.0, index=frame.index)
    s = s.where(fcf.isna(), s + fcf.gt(0).astype(int)*15 - 7.5)
    s = s.where(ocf.isna(), s + ocf.gt(0).astype(int)*15 - 7.5)
    r = capex / ocf.where(ocf > 0)
    s = s.where(r.isna(), s + (50 - r.clip(0, 2)*25)*0.3)
    s = s.where(fin.isna(), s + fin.lt(0).astype(int)*10 - 5)
    s = s.where(g.isna(), s + g.clip(-30, 50)*0.3)
    return s.clip(0, 100).fillna(50)

def calculate_csv_technical_score(frame):
    parts = []
    rsi = numeric_column(frame, "RSI Exponential - 14D")
    if rsi.notna().any():
        s = np.select([rsi.between(55, 70), rsi.between(45, 80), rsi.between(30, 45)],
                      [100, 70, 40], default=20)
        parts.append(pd.Series(s, index=frame.index).where(rsi.notna()))
    if "Super Trend" in frame.columns:
        st_col = frame["Super Trend"].astype(str).str.upper()
        parts.append((st_col.str.contains("BUY").astype(int)*80 + 20)
                     .where(st_col.ne("NAN") & st_col.ne("NONE")))
    k = numeric_column(frame, "Stochastic %K"); d = numeric_column(frame, "Stochastic %D")
    if k.notna().any() and d.notna().any():
        s = np.select([(k > d) & k.between(20, 80), k.gt(80), k.lt(20)], [100, 40, 60], default=50)
        parts.append(pd.Series(s, index=frame.index).where(k.notna() & d.notna()))
    wr = numeric_column(frame, "William %R")
    if wr.notna().any():
        s = np.select([wr.between(-60, -20), wr.between(-80, -60), wr.lt(-80)],
                      [100, 70, 60], default=30)
        parts.append(pd.Series(s, index=frame.index).where(wr.notna()))
    up = numeric_column(frame, "% From Upper Bollinger Band")
    lp = numeric_column(frame, "% From Lower Bollinger Band")
    if up.notna().any() or lp.notna().any():
        s = np.select([lp < 3, lp < 8, up < 5], [100, 80, 60], default=40)
        parts.append(pd.Series(s, index=frame.index).where(lp.notna() | up.notna()))
    if "MACD Line 1 - Trend Indicator" in frame.columns:
        m = frame["MACD Line 1 - Trend Indicator"].astype(str).str.upper()
        parts.append((m.str.contains("BUY").astype(int)*80 + 20).where(m.ne("NAN") & m.ne("NONE")))
    psar = numeric_column(frame, "% From Parabolic SAR")
    if psar.notna().any():
        s = np.select([psar.abs() < 2, psar.abs() < 5], [100, 70], default=40)
        parts.append(pd.Series(s, index=frame.index).where(psar.notna()))
    obv = numeric_column(frame, "1W Change in On Balance Volume")
    ad = numeric_column(frame, "1W Change in AD Line")
    if obv.notna().any() or ad.notna().any():
        s = np.select([obv.gt(0) & ad.gt(0), obv.gt(0) | ad.gt(0)], [100, 70], default=30)
        parts.append(pd.Series(s, index=frame.index).where(obv.notna() | ad.notna()))
    if not parts: return pd.Series(50.0, index=frame.index)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(50).clip(0, 100)

# ============ TECHNICAL INDICATORS ============
def _flatten_columns(data):
    if isinstance(data.columns, pd.MultiIndex):
        l0 = data.columns.get_level_values(0)
        if "Close" in l0: data = data.copy(); data.columns = l0
    return data

def _compute_indicators(data):
    data = _flatten_columns(data)
    req = {"Close", "High", "Low", "Volume"}
    if not req.issubset(data.columns): return {"Technical_Score": 50.0}
    close = pd.to_numeric(data["Close"], errors="coerce")
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce")
    data = data.assign(Close=close, High=high, Low=low, Volume=volume).dropna(subset=["Close","High","Low"])
    close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]
    if len(data) < 200 or not np.isfinite(close.iloc[-1]) or close.iloc[-1] <= 0:
        return {"Technical_Score": 50.0}
    sma = {n: close.rolling(n).mean() for n in (20, 50, 200)}
    ema = {n: close.ewm(span=n, adjust=False).mean() for n in (20, 50, 200)}
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(100)
    ml = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    ms = ml.ewm(span=9, adjust=False).mean(); mh = ml - ms
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    um, dm = high.diff(), -low.diff()
    pdm = um.where((um > dm) & (um > 0), 0.0); mdm = dm.where((dm > um) & (dm > 0), 0.0)
    pdi = 100*pdm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100*mdm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100*(pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    mid = close.rolling(20).mean(); dev = close.rolling(20).std(ddof=0) * 2
    up, lo = mid + dev, mid - dev
    bu = ((high + low) / 2 + 3 * atr).to_numpy(); bl = ((high + low) / 2 - 3 * atr).to_numpy()
    fu, fl = bu.copy(), bl.copy(); c = close.to_numpy(); d = np.ones(len(c)); an = atr.to_numpy()
    for i in range(1, len(c)):
        if np.isnan(an[i]): continue
        fu[i] = bu[i] if (np.isnan(fu[i-1]) or bu[i] < fu[i-1] or c[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = bl[i] if (np.isnan(fl[i-1]) or bl[i] > fl[i-1] or c[i-1] < fl[i-1]) else fl[i-1]
        if d[i-1] < 0 and c[i] > fu[i]: d[i] = 1
        elif d[i-1] > 0 and c[i] < fl[i]: d[i] = -1
        else: d[i] = d[i-1]
    st_val = pd.Series(fl, index=data.index).where(
        pd.Series(d, index=data.index).gt(0), pd.Series(fu, index=data.index))
    direction = pd.Series(d, index=data.index)
    prev = data.iloc[-2]; cmp = float(close.iloc[-1])
    pivot = float((prev["High"] + prev["Low"] + prev["Close"]) / 3)
    hr = float(prev["High"] - prev["Low"])
    ts = 0
    if sma[20].iloc[-1] > sma[50].iloc[-1] > sma[200].iloc[-1]: ts += 40
    elif sma[20].iloc[-1] > sma[50].iloc[-1]: ts += 25
    elif sma[20].iloc[-1] > sma[200].iloc[-1]: ts += 15
    if ema[20].iloc[-1] > ema[50].iloc[-1] > ema[200].iloc[-1]: ts += 30
    elif ema[20].iloc[-1] > ema[50].iloc[-1]: ts += 20
    elif ema[20].iloc[-1] > ema[200].iloc[-1]: ts += 10
    if direction.iloc[-1] > 0: ts += 20
    adx_v = adx.iloc[-1]
    if adx_v >= 40: ts += 10
    elif adx_v >= 30: ts += 8
    elif adx_v >= 25: ts += 6
    elif adx_v >= 20: ts += 3
    rsi_v = rsi.iloc[-1]
    ms_val = (30 if 55 <= rsi_v <= 70 else 20 if rsi_v >= 50 else 10 if rsi_v >= 40 else 0)
    if ml.iloc[-1] > ms.iloc[-1]: ms_val += 40
    if mh.iloc[-1] > mh.iloc[-2]: ms_val += 30
    av = float(volume.tail(20).mean()); cv = float(volume.iloc[-1])
    vs = 100 if cv > av * 1.5 else 70 if cv > av else 30
    atr_v = float(atr.iloc[-1]); ap = atr_v / cmp * 100 if np.isfinite(atr_v) else np.nan
    vol_s = 100 if ap <= 1.5 else 70 if ap <= 2.5 else 40 if ap <= 4 else 10
    sr_s = 100 if cmp > pivot else 40
    tech = (ts + ms_val + vs + vol_s + sr_s) / 5
    return {"CMP": cmp, "20 DMA": float(sma[20].iloc[-1]), "50 DMA": float(sma[50].iloc[-1]),
        "200 DMA": float(sma[200].iloc[-1]), "RSI": float(rsi_v),
        "EMA20": float(ema[20].iloc[-1]), "EMA50": float(ema[50].iloc[-1]),
        "EMA200": float(ema[200].iloc[-1]), "MACDSignal": float(ms.iloc[-1]),
        "Histogram": float(mh.iloc[-1]), "Upper Band": float(up.iloc[-1]),
        "Middle Band": float(mid.iloc[-1]), "Lower Band": float(lo.iloc[-1]),
        "ATR": atr_v, "DI Plus": float(pdi.iloc[-1]), "DI Minus": float(mdi.iloc[-1]),
        "ADX": float(adx_v), "SuperTrend": "BUY" if direction.iloc[-1] > 0 else "SELL",
        "SuperTrend_Val": float(st_val.iloc[-1]), "Trend Pivot": pivot,
        "R1": pivot + 0.382*hr, "R2": pivot + 0.618*hr, "R3": pivot + hr,
        "S1": pivot - 0.382*hr, "S2": pivot - 0.618*hr, "S3": pivot - hr,
        "Volume": cv, "Avg Volume": av, "52W High": float(high.max()), "52W Low": float(low.min()),
        "Technical_Score": min(tech, 100), "Trend Score": ts, "Momentum Score": ms_val,
        "Volume Score": vs, "Volatility Score": vol_s, "Support Resistance Score": sr_s}

# ============ RISK ============
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_benchmark_history():
    try:
        d = yf.download(BENCHMARK_TICKER, period=PRICE_PERIOD, progress=False,
                        auto_adjust=False, threads=False)
        if d.empty: return pd.DataFrame()
        return _flatten_columns(d).copy().dropna(subset=["Close"])
    except Exception: return pd.DataFrame()

def compute_risk_metrics(history, benchmark):
    empty = {k: np.nan for k in ["Volatility %","Sharpe","Sortino","Max Drawdown %","Beta",
        "Alpha %","Return 1M %","Return 3M %","Return 6M %","Return 1Y %","Nifty 1Y %","Relative Strength %"]}
    if history.empty or "Close" not in history.columns: return empty
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(close) < 60: return empty
    dr = close.pct_change().dropna()
    vol = float(dr.std() * np.sqrt(TRADING_DAYS) * 100)
    ex = dr - (RISK_FREE_RATE / TRADING_DAYS)
    sh = float(ex.mean() / dr.std() * np.sqrt(TRADING_DAYS)) if dr.std() > 0 else np.nan
    dn = dr[dr < 0]
    so = (float(ex.mean() / dn.std() * np.sqrt(TRADING_DAYS))
          if len(dn) > 0 and dn.std() > 0 else np.nan)
    rm = close.cummax(); dd = float(((close - rm) / rm).min() * 100)
    beta, alpha = np.nan, np.nan
    if not benchmark.empty and "Close" in benchmark.columns:
        bc = pd.to_numeric(benchmark["Close"], errors="coerce").dropna()
        al = pd.concat([close.pct_change().rename("s"), bc.pct_change().rename("b")], axis=1).dropna()
        if len(al) > 30 and al["b"].var() > 0:
            beta = float(al["s"].cov(al["b"]) / al["b"].var())
            sr = (1 + al["s"]).prod() ** (TRADING_DAYS/len(al)) - 1
            br = (1 + al["b"]).prod() ** (TRADING_DAYS/len(al)) - 1
            alpha = float((sr - RISK_FREE_RATE - beta * (br - RISK_FREE_RATE)) * 100)
    def _p(d): return float((close.iloc[-1]/close.iloc[-d-1] - 1)*100) if len(close) > d else np.nan
    n1y = np.nan
    if not benchmark.empty and "Close" in benchmark.columns:
        bc = pd.to_numeric(benchmark["Close"], errors="coerce").dropna()
        if len(bc) > TRADING_DAYS:
            n1y = float((bc.iloc[-1]/bc.iloc[-TRADING_DAYS-1] - 1)*100)
    r1y = _p(TRADING_DAYS)
    rs = (r1y - n1y) if pd.notna(r1y) and pd.notna(n1y) else np.nan
    return {"Volatility %": round(vol, 2),
        "Sharpe": round(sh, 2) if pd.notna(sh) else np.nan,
        "Sortino": round(so, 2) if pd.notna(so) else np.nan,
        "Max Drawdown %": round(dd, 2),
        "Beta": round(beta, 2) if pd.notna(beta) else np.nan,
        "Alpha %": round(alpha, 2) if pd.notna(alpha) else np.nan,
        "Return 1M %": round(_p(21), 2), "Return 3M %": round(_p(63), 2),
        "Return 6M %": round(_p(126), 2),
        "Return 1Y %": round(r1y, 2) if pd.notna(r1y) else np.nan,
        "Nifty 1Y %": round(n1y, 2) if pd.notna(n1y) else np.nan,
        "Relative Strength %": round(rs, 2) if pd.notna(rs) else np.nan}

# ============ PATTERNS ============
def detect_patterns(row):
    ps = []
    try:
        cmp = float(row.get("CMP", np.nan)); s20 = float(row.get("20 DMA", np.nan))
        s50 = float(row.get("50 DMA", np.nan)); s200 = float(row.get("200 DMA", np.nan))
        h52 = float(row.get("52W High", np.nan)); l52 = float(row.get("52W Low", np.nan))
        v = float(row.get("Volume", np.nan)); av = float(row.get("Avg Volume", np.nan))
        r1 = float(row.get("R1", np.nan)); s1 = float(row.get("S1", np.nan))
        if all(pd.notna(x) for x in [s50, s200]) and s50 > s200: ps.append("Golden Cross")
        elif all(pd.notna(x) for x in [s50, s200]) and s50 < s200: ps.append("Death Cross")
        if pd.notna(cmp) and pd.notna(h52) and cmp >= h52*0.99: ps.append("52W Breakout")
        if pd.notna(cmp) and pd.notna(l52) and cmp <= l52*1.01: ps.append("52W Breakdown")
        if pd.notna(v) and pd.notna(av) and av > 0 and v > av*2: ps.append("Volume Spike")
        if all(pd.notna(x) for x in [cmp, s1]) and s1 > 0 and abs(cmp-s1)/cmp < 0.02:
            ps.append("Near Support")
        if all(pd.notna(x) for x in [cmp, r1]) and r1 > 0 and abs(r1-cmp)/cmp < 0.02:
            ps.append("Near Resistance")
        if all(pd.notna(x) for x in [cmp, s20, s50, s200]) and cmp > s20 > s50 > s200:
            ps.append("Above All MAs")
    except Exception: pass
    return ", ".join(ps) if ps else "—"

def scan_red_flags(row):
    flags = []
    for label, (col, op, th) in RED_FLAG_RULES.items():
        if col not in row.index: continue
        v = pd.to_numeric(row[col], errors="coerce")
        if pd.isna(v): continue
        if op == "gt" and v > th: flags.append(label)
        elif op == "lt" and v < th: flags.append(label)
        elif op == "below_dma200":
            d200 = pd.to_numeric(row.get("200 DMA", np.nan), errors="coerce")
            if pd.notna(d200) and v < d200: flags.append(label)
    return flags

def build_flag_summary(frame):
    out = frame.copy()
    fl = out.apply(scan_red_flags, axis=1)
    out["Red Flag Count"] = fl.map(len)
    out["Red Flag List"] = fl.map(lambda l: ", ".join(l) if l else "✅ Clean")
    out["Patterns"] = out.apply(detect_patterns, axis=1)
    out["Quality Grade"] = np.select(
        [out["Red Flag Count"].le(0), out["Red Flag Count"].le(2), out["Red Flag Count"].le(4)],
        ["A — Clean", "B — Minor", "C — Watch"], default="D — High Risk")
    return out

# ============ TICKER ANALYSIS ============
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def analyze_ticker(ticker):
    sym = ticker if ticker.endswith((".NS", ".BO")) else f"{ticker}.NS"
    try:
        h = yf.download(sym, period=PRICE_PERIOD, progress=False,
                        auto_adjust=False, threads=False)
        if h.empty: return {"Technical_Score": 50.0, "Fetch Status": "No price history"}
        h = _flatten_columns(h).copy()
        r = _compute_indicators(h)
        if "CMP" in r:
            r.update(compute_risk_metrics(h, fetch_benchmark_history()))
            r["Fetch Status"] = "OK"
        return r
    except Exception as exc:
        return {"Technical_Score": 50.0, "Fetch Status": f"Failed: {type(exc).__name__}"}

def fetch_technicals(tickers):
    results = {}
    if not tickers: return pd.DataFrame()
    fetch_benchmark_history()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as pool:
        futures = {pool.submit(analyze_ticker, t): t for t in tickers}
        prog = st.progress(0, text="Fetching...")
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            t = futures[fut]
            try: results[t] = fut.result()
            except Exception: results[t] = {"Technical_Score": 50.0, "Fetch Status": "Error"}
            prog.progress(i / len(futures), text=f"{i}/{len(futures)}")
        prog.empty()
    return pd.DataFrame.from_dict(results, orient="index").rename_axis("Ticker").reset_index()

# ============ COMPOSITE ============
def classify_index_membership(ticker, registry):
    t = _normalise_ticker(ticker)
    ms = [n for n, m in registry.items() if t in m]
    return ms or ["Other"]

def _kelly_lite(ai, rr, cf):
    b = (ai.clip(0, 100)/100)*0.6; rb = (rr.clip(0, 5)/5)*0.25; cb = (cf.clip(0, 100)/100)*0.15
    return ((b + rb + cb)*100).clip(0, 100).round(1)

def build_scores(frame, index_registry):
    out = frame.copy()
    sector_col = detect_sector_column(out)
    out["Fundamental_Score_AI"] = calculate_fundamental_score(out)
    out["Sector_Relative_Score"] = calculate_sector_relative_score(out, sector_col or "Sector")
    out["Rank_Composite"] = calculate_rank_composite(out)
    out["Analyst_Consensus_Score"] = calculate_analyst_consensus(out)
    out["CashFlow_Quality_Score"] = calculate_cashflow_quality(out)
    out["CSV_Technical_Score"] = calculate_csv_technical_score(out)
    val = calculate_hybrid_valuation(out)
    out["Intrinsic Value (DCF)"] = val["Intrinsic Value (DCF)"]
    out["Intrinsic Value (Multiples)"] = val["Intrinsic Value (Multiples)"]
    out["Intrinsic Value (Base)"] = val["Intrinsic Value (Base)"]
    out["Lower Intrinsic Value"] = val["Lower Intrinsic Value"]
    out["Upper Intrinsic Value"] = val["Upper Intrinsic Value"]
    out["WACC Used"] = val["WACC Used"]
    out["DCF_Valuation_Score"] = calculate_valuation_score(out)
    out["Valuation_Trigger_Score"] = calculate_valuation_trigger_score(out)
    cmp_now = numeric_column(out, "CMP"); lower_now = numeric_column(out, "Lower Intrinsic Value")
    upper_now = numeric_column(out, "Upper Intrinsic Value")
    base_now = numeric_column(out, "Intrinsic Value (Base)")
    out["Entry Trigger Price"] = lower_now.round(2)
    out["Exit Trigger Price"] = upper_now.round(2)
    out["To Entry %"] = (((lower_now - cmp_now)/cmp_now.where(cmp_now.gt(0)))*100).round(2)
    out["To Exit %"] = (((upper_now - cmp_now)/cmp_now.where(cmp_now.gt(0)))*100).round(2)
    out["Valuation Action"] = np.select(
        [cmp_now.le(lower_now*0.95), cmp_now.le(lower_now), cmp_now.le(base_now),
         cmp_now.le((base_now + upper_now)/2), cmp_now.le(upper_now)],
        ["🟢🟢🟢 DEEP BUY", "🟢🟢 ENTRY TRIGGER", "🟢 ACCUMULATE",
         "⚪ FAIR / HOLD", "🟠 STRETCHED"], default="🔴 EXIT TRIGGER")
    tech = numeric_column(out, "Technical_Score", 50).fillna(50)
    sharpe_n = numeric_column(out, "Sharpe", 0).fillna(0).clip(-2, 3).add(2).mul(20).clip(0, 100)
    out["AI_Score"] = (
        tech*0.16 + out["Fundamental_Score_AI"]*0.16
        + out["Sector_Relative_Score"]*0.09 + out["Rank_Composite"]*0.12
        + out["CSV_Technical_Score"]*0.10 + out["Analyst_Consensus_Score"]*0.06
        + out["CashFlow_Quality_Score"]*0.05 + sharpe_n*0.04
        + out["DCF_Valuation_Score"]*0.10 + out["Valuation_Trigger_Score"]*0.12
    ).round().astype(int)
    out["AI Signal"] = np.select(
        [out["AI_Score"].ge(90), out["AI_Score"].ge(80), out["AI_Score"].ge(70),
         out["AI_Score"].ge(60), out["AI_Score"].ge(40)],
        ["Strong Buy", "Buy", "Accumulate", "Hold", "Reduce"], default="Sell")
    stacked = pd.concat([tech, out["Fundamental_Score_AI"], out["Sector_Relative_Score"],
                         out["Rank_Composite"], out["DCF_Valuation_Score"],
                         out["Valuation_Trigger_Score"]], axis=1)
    out["AI Confidence"] = (100 - stacked.std(axis=1).fillna(0)*1.5).clip(0, 100).round()
    out["Safety Score"] = (
        out["Fundamental_Score_AI"]*0.24 + tech*0.18
        + out["Sector_Relative_Score"]*0.09 + out["Rank_Composite"]*0.11
        + numeric_column(out, "Volatility Score", 50).fillna(50)*0.09
        + out["CashFlow_Quality_Score"]*0.06 + sharpe_n*0.04
        + out["DCF_Valuation_Score"]*0.09 + out["Valuation_Trigger_Score"]*0.10)
    out["Risk Score"] = 100 - out["Safety Score"]
    out["Risk Level"] = np.select(
        [out["Risk Score"].lt(25), out["Risk Score"].lt(50), out["Risk Score"].lt(75)],
        ["Low Risk", "Moderate Risk", "High Risk"], default="Very High Risk")
    ms = out["Ticker"].map(lambda t: classify_index_membership(t, index_registry))
    out["Index Memberships"] = ms.map(lambda l: ", ".join(l))
    out["Primary Index"] = ms.map(lambda l: l[0])
    cmp = numeric_column(out, "CMP"); stv = numeric_column(out, "SuperTrend_Val")
    atr = numeric_column(out, "ATR"); valid = stv.gt(0) & stv.lt(cmp)
    out["Entry Price"] = cmp
    out["Stop Loss"] = stv.where(valid, (cmp - atr).where(atr.gt(0), cmp*0.95))
    min_t = cmp + (cmp - out["Stop Loss"])*2
    res = pd.concat([numeric_column(out, r) for r in ("R1","R2","R3")], axis=1)
    above = res.where(res.ge(min_t, axis=0))
    out["Target Price"] = above.min(axis=1).fillna(min_t)
    out["Upside %"] = ((out["Target Price"] - cmp) / cmp.where(cmp.gt(0)) * 100).round(2)
    rps = cmp - out["Stop Loss"]
    out["Risk : Reward"] = ((out["Target Price"] - cmp) / rps.where(rps.gt(0))).round(2)
    out["Trailing Stop"] = (cmp - 3*atr).where(atr.gt(0), out["Stop Loss"])
    iv = out["Intrinsic Value (Base)"]
    out["Margin of Safety %"] = ((iv - cmp) / iv.where(iv.gt(0)) * 100).round(2)
    mos = out["Margin of Safety %"]
    out["Valuation Signal"] = np.select(
        [mos.ge(40), mos.ge(25), mos.ge(10), mos.ge(-5), mos.ge(-15), mos.ge(-30)],
        ["🟢 Deep Value", "🟢 Undervalued", "🟡 Slightly Under", "⚪ Fair Value",
         "🟠 Slightly Over", "🔴 Overvalued"], default="🔴 Highly Overvalued")
    out = build_flag_summary(out)
    out["Position Size %"] = _kelly_lite(out["AI_Score"], out["Risk : Reward"].fillna(0), out["AI Confidence"])
    out = out.sort_values("AI_Score", ascending=False, kind="stable", na_position="last").reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out

# ============ ANALYTICS ============
def analyse_index(frame, index_name, members):
    norm = frame["Ticker"].map(_normalise_ticker); sub = frame.loc[norm.isin(members)]
    if sub.empty: return {"Index": index_name, "Stocks": 0, "Buy+": 0, "Hold": 0, "Sell+": 0,
        "Avg AI Score": 0, "Avg Fundamental": 0, "Avg Technical": 0, "Avg Sector-Rel": 0,
        "Avg Upside %": 0, "Avg PE": 0}
    return {"Index": index_name, "Stocks": len(sub),
        "Buy+": int(sub["AI Signal"].isin(["Strong Buy", "Buy"]).sum()),
        "Hold": int((sub["AI Signal"] == "Hold").sum()),
        "Sell+": int(sub["AI Signal"].isin(["Sell", "Reduce"]).sum()),
        "Avg AI Score": round(sub["AI_Score"].mean(), 1),
        "Avg Fundamental": round(sub["Fundamental_Score_AI"].mean(), 1),
        "Avg Technical": round(numeric_column(sub, "Technical_Score", 50).mean(), 1),
        "Avg Sector-Rel": round(sub["Sector_Relative_Score"].mean(), 1),
        "Avg Upside %": round(sub["Upside %"].mean(), 1),
        "Avg PE": round(numeric_column(sub, "PE Ratio").mean(), 1)}

def build_index_table(frame, registry):
    return pd.DataFrame([analyse_index(frame, n, m) for n, m in registry.items()])

def analyse_sectors(frame):
    sc = detect_sector_column(frame)
    if sc is None or frame[sc].isna().all(): return pd.DataFrame()
    df = frame.copy()
    df[sc] = df[sc].fillna("Unknown").astype(str)
    df["_tech"] = numeric_column(df, "Technical_Score", 50).fillna(50)
    df["_pe"] = numeric_column(df, "PE Ratio")
    df["_buy"] = df["AI Signal"].isin(["Strong Buy", "Buy"]).astype(int)
    for c in ("AI_Score", "Fundamental_Score_AI", "Sector_Relative_Score", "Rank_Composite", "Upside %"):
        if c not in df.columns: df[c] = np.nan
    stats = (df.groupby(sc, dropna=False)
        .agg(Stocks=("Ticker", "count"),
             **{"Avg AI Score": ("AI_Score", "mean"),
                "Avg Fundamental": ("Fundamental_Score_AI", "mean"),
                "Avg Technical": ("_tech", "mean"),
                "Avg Sector-Rel": ("Sector_Relative_Score", "mean"),
                "Avg Rank Composite": ("Rank_Composite", "mean"),
                "Buy+": ("_buy", "sum"), "Avg PE": ("_pe", "mean"),
                "Avg Upside %": ("Upside %", "mean")})
        .round(1).reset_index().rename(columns={sc: "Sector"})
        .sort_values("Avg AI Score", ascending=False, kind="stable").reset_index(drop=True))
    for c in ("Sector", "Avg AI Score", "Avg Sector-Rel", "Stocks"):
        if c not in stats.columns: stats[c] = np.nan if c != "Stocks" else 0
    return stats

# ============ SNAPSHOTS ============
def _snap_dir(p): return p / DATA_DIR_NAME / SNAPSHOT_DIR_NAME

def save_sector_snapshot(project_dir, stats):
    if stats.empty: return
    try:
        d = _snap_dir(project_dir); d.mkdir(parents=True, exist_ok=True)
        stats.to_csv(d / f"sector_{datetime.now():%Y-%m-%d}.csv", index=False)
    except OSError: pass

def load_sector_snapshots(project_dir, days=60):
    d = _snap_dir(project_dir)
    if not d.exists(): return pd.DataFrame()
    cut = datetime.now() - timedelta(days=days); frames = []
    for p in sorted(d.glob("sector_*.csv")):
        try: sd = datetime.strptime(p.stem.replace("sector_", ""), "%Y-%m-%d")
        except ValueError: continue
        if sd < cut: continue
        try:
            df = pd.read_csv(p); df["Date"] = sd; frames.append(df)
        except Exception: continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def compute_sector_rotation(project_dir, current_stats):
    if current_stats.empty: return pd.DataFrame()
    req = {"Sector", "Avg AI Score", "Avg Sector-Rel"}
    missing = req - set(current_stats.columns)
    if "Sector" in missing: return pd.DataFrame()
    cur = current_stats.copy()
    for c in missing: cur[c] = np.nan
    hist = load_sector_snapshots(project_dir, 90)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    base = pd.DataFrame({"Sector": cur["Sector"].values, "Prev AI Score": np.nan, "Prev Sector-Rel": np.nan})
    if not hist.empty:
        needed = {"Sector", "Avg AI Score", "Avg Sector-Rel", "Date"}
        if needed.issubset(hist.columns):
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.normalize()
            cand = hist[hist["Date"] <= (today - timedelta(days=7))]
            if cand.empty: cand = hist[hist["Date"] < today]
            if not cand.empty:
                latest = cand["Date"].max()
                base = (cand.loc[cand["Date"] == latest, ["Sector", "Avg AI Score", "Avg Sector-Rel"]]
                    .rename(columns={"Avg AI Score": "Prev AI Score", "Avg Sector-Rel": "Prev Sector-Rel"})
                    .drop_duplicates(subset="Sector", keep="last").reset_index(drop=True))
    merged = cur.merge(base, on="Sector", how="left")
    if "Prev AI Score" not in merged.columns: merged["Prev AI Score"] = np.nan
    if "Prev Sector-Rel" not in merged.columns: merged["Prev Sector-Rel"] = np.nan
    merged["WoW AI Δ"] = (merged["Avg AI Score"] - merged["Prev AI Score"]).round(1)
    merged["WoW Sector-Rel Δ"] = (merged["Avg Sector-Rel"] - merged["Prev Sector-Rel"]).round(1)
    merged["Rotation"] = np.select(
        [merged["WoW AI Δ"].ge(3), merged["WoW AI Δ"].ge(1),
         merged["WoW AI Δ"].le(-3), merged["WoW AI Δ"].le(-1)],
        ["🚀 Leading", "📈 Improving", "🔻 Lagging", "📉 Weakening"], default="➖ Stable")
    return merged.sort_values("WoW AI Δ", ascending=False, kind="stable").reset_index(drop=True)

# ============ CHART INDICATORS (CHANGE 1) ============
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_chart_for_period(ticker, period, interval):
    sym = ticker if ticker.endswith((".NS", ".BO")) else f"{ticker}.NS"
    try:
        h = yf.download(sym, period=period, interval=interval, progress=False,
                        auto_adjust=False, threads=False)
    except Exception: return pd.DataFrame()
    if h.empty: return pd.DataFrame()
    return _flatten_columns(h).copy().dropna(subset=["Close"])

_PLOT_LAYOUT = {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "#070d18",
                "font": {"color": "#8ba3c0"}, "margin": {"t": 24, "b": 24, "l": 24, "r": 24}}

def _style_figure(fig, height=340):
    fig.update_layout(**_PLOT_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#1c2e45", zerolinecolor="#1c2e45")
    fig.update_yaxes(gridcolor="#1c2e45", zerolinecolor="#1c2e45")
    return fig

def _compute_overlay_series(h, kind, window):
    close = h["Close"]; high = h["High"]; low = h["Low"]
    if kind == "sma": return close.rolling(window).mean()
    if kind == "ema": return close.ewm(span=window, adjust=False).mean()
    if kind == "bb_upper": return close.rolling(window).mean() + 2*close.rolling(window).std(ddof=0)
    if kind == "bb_middle": return close.rolling(window).mean()
    if kind == "bb_lower": return close.rolling(window).mean() - 2*close.rolling(window).std(ddof=0)
    if kind == "supertrend":
        pc = close.shift(1)
        tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        bu = ((high + low)/2 + 3*atr).to_numpy(); bl = ((high + low)/2 - 3*atr).to_numpy()
        fu, fl = bu.copy(), bl.copy(); c = close.to_numpy(); d = np.ones(len(c)); an = atr.to_numpy()
        for i in range(1, len(c)):
            if np.isnan(an[i]): continue
            fu[i] = bu[i] if (np.isnan(fu[i-1]) or bu[i] < fu[i-1] or c[i-1] > fu[i-1]) else fu[i-1]
            fl[i] = bl[i] if (np.isnan(fl[i-1]) or bl[i] > fl[i-1] or c[i-1] < fl[i-1]) else fl[i-1]
            if d[i-1] < 0 and c[i] > fu[i]: d[i] = 1
            elif d[i-1] > 0 and c[i] < fl[i]: d[i] = -1
            else: d[i] = d[i-1]
        return pd.Series(np.where(d > 0, fl, fu), index=h.index)
    if kind == "vwap":
        tp = (high + low + close)/3
        return (tp*h["Volume"]).cumsum() / h["Volume"].cumsum()
    if kind == "psar":
        hn, ln, cn = high.to_numpy(), low.to_numpy(), close.to_numpy()
        psar = cn.copy(); bull = True; af = 0.02; ep = hn[0]; psar[0] = ln[0]
        for i in range(1, len(cn)):
            psar[i] = psar[i-1] + af*(ep - psar[i-1])
            if bull:
                if ln[i] < psar[i]:
                    bull = False; psar[i] = ep; ep = ln[i]; af = 0.02
                else:
                    if hn[i] > ep: ep = hn[i]; af = min(af+0.02, 0.2)
                    psar[i] = min(psar[i], ln[i-1], ln[i-2] if i > 1 else ln[i-1])
            else:
                if hn[i] > psar[i]:
                    bull = True; psar[i] = ep; ep = hn[i]; af = 0.02
                else:
                    if ln[i] < ep: ep = ln[i]; af = min(af+0.02, 0.2)
                    psar[i] = max(psar[i], hn[i-1], hn[i-2] if i > 1 else hn[i-1])
        return pd.Series(psar, index=h.index)
    return None

def _build_subplot_trace(h, kind):
    close = h["Close"]; high = h["High"]; low = h["Low"]; volume = h["Volume"]
    if kind == "Volume":
        colors = np.where(close >= close.shift(1), "#34d399", "#fb7185")
        return [go.Bar(x=h.index, y=volume, name="Volume", marker_color=colors, opacity=0.6)]
    if kind == "RSI":
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - 100/(1 + rs)).fillna(100)
        return [go.Scatter(x=h.index, y=rsi, name="RSI", line=dict(color="#a78bfa", width=1.5))]
    if kind == "MACD":
        ml = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        sig = ml.ewm(span=9, adjust=False).mean(); hist = ml - sig
        return [go.Bar(x=h.index, y=hist, name="Hist",
                       marker_color=np.where(hist >= 0, "#34d399", "#fb7185")),
                go.Scatter(x=h.index, y=ml, name="MACD", line=dict(color="#60a5fa", width=1.3)),
                go.Scatter(x=h.index, y=sig, name="Signal", line=dict(color="#fbbf24", width=1.3))]
    if kind == "Stochastic":
        low14 = low.rolling(14).min(); high14 = high.rolling(14).max()
        k = 100*(close - low14)/(high14 - low14).replace(0, np.nan); d = k.rolling(3).mean()
        return [go.Scatter(x=h.index, y=k, name="%K", line=dict(color="#34d399", width=1.3)),
                go.Scatter(x=h.index, y=d, name="%D", line=dict(color="#fbbf24", width=1.3))]
    if kind == "ADX":
        pc = close.shift(1)
        tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        um, dm = high.diff(), -low.diff()
        pdm = um.where((um > dm) & (um > 0), 0.0); mdm = dm.where((dm > um) & (dm > 0), 0.0)
        pdi = 100*pdm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
        mdi = 100*mdm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100*(pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        adx = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        return [go.Scatter(x=h.index, y=adx, name="ADX", line=dict(color="#f59e0b", width=1.5))]
    if kind == "ATR":
        pc = close.shift(1)
        tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        return [go.Scatter(x=h.index, y=atr, name="ATR", line=dict(color="#ec4899", width=1.5))]
    if kind == "OBV":
        sign = np.sign(close.diff().fillna(0)); obv = (sign*volume).cumsum()
        return [go.Scatter(x=h.index, y=obv, name="OBV", line=dict(color="#06b6d4", width=1.5))]
    return []

def render_price_chart(ticker, row=None):
    if go is None or make_subplots is None:
        st.info("`pip install plotly` karein."); return
    c_a, c_b = st.columns(2)
    with c_a:
        ovl_key = f"ovl_{ticker}"
        if ovl_key not in st.session_state: st.session_state[ovl_key] = ["SMA 20", "SMA 50", "SMA 200"]
        selected_ovl = st.multiselect("📊 Price Overlays",
            options=list(CHART_OVERLAYS.keys()), default=st.session_state[ovl_key],
            key=f"ms_{ovl_key}")
        st.session_state[ovl_key] = selected_ovl
    with c_b:
        sub_key = f"sub_{ticker}"
        if sub_key not in st.session_state: st.session_state[sub_key] = ["Volume", "RSI"]
        selected_sub = st.multiselect("📉 Subplots", options=CHART_SUBPLOTS,
            default=st.session_state[sub_key], key=f"ms_{sub_key}")
        st.session_state[sub_key] = selected_sub
    period_opts = [("1D","1d","5m"), ("1W","5d","30m"), ("1M","1mo","1d"),
                   ("3M","3mo","1d"), ("6M","6mo","1d"), ("1Y","1y","1d"), ("5Y","5y","1d")]
    state_key = f"chart_period_{ticker}"
    if state_key not in st.session_state: st.session_state[state_key] = "1Y"
    cols = st.columns(len(period_opts))
    for i, (label, _, _) in enumerate(period_opts):
        with cols[i]:
            active = st.session_state[state_key] == label
            if st.button(label, key=f"period_{ticker}_{label}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state[state_key] = label; st.rerun()
    selected = st.session_state[state_key]
    period, interval = next((p, i) for lbl, p, i in period_opts if lbl == selected)
    h = _fetch_chart_for_period(ticker, period, interval)
    if h.empty: st.warning(f"{ticker} chart data nahi mila."); return
    n_sub = len(selected_sub)
    if n_sub > 0:
        heights = [0.55] + [0.45/n_sub]*n_sub
        fig = make_subplots(rows=1+n_sub, cols=1, shared_xaxes=True,
                            vertical_spacing=0.025, row_heights=heights,
                            subplot_titles=[""] + selected_sub)
    else: fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=h.index, open=h["Open"], high=h["High"],
        low=h["Low"], close=h["Close"], name=ticker,
        increasing_line_color="#34d399", decreasing_line_color="#fb7185"), row=1, col=1)
    for ov_name in selected_ovl:
        meta = CHART_OVERLAYS[ov_name]
        line = _compute_overlay_series(h, meta["kind"], meta.get("window"))
        if line is None: continue
        display_val = None
        if row is not None and meta.get("col") in row.index:
            v = row.get(meta["col"])
            if pd.notna(v): display_val = float(v)
        if display_val is None and not line.dropna().empty:
            display_val = float(line.dropna().iloc[-1])
        legend_name = f"{ov_name}: ₹{display_val:,.2f}" if display_val is not None else ov_name
        if meta["kind"] == "psar":
            fig.add_trace(go.Scatter(x=h.index, y=line, mode="markers", name=legend_name,
                marker=dict(color=meta["color"], size=4)), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=h.index, y=line, mode="lines", name=legend_name,
                line=dict(color=meta["color"], width=1.6)), row=1, col=1)
    for i, sp_name in enumerate(selected_sub, start=2):
        for tr in _build_subplot_trace(h, sp_name):
            fig.add_trace(tr, row=i, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=f"{ticker} — {selected}",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, font=dict(size=10)))
    fig.update_xaxes(gridcolor="#1c2e45", zerolinecolor="#1c2e45")
    fig.update_yaxes(gridcolor="#1c2e45", zerolinecolor="#1c2e45")
    _style_figure(fig, 460 + n_sub*120)
    st.plotly_chart(fig, use_container_width=True)

# ============ UI HELPERS ============
def style_signal(v):
    c = "#34d399" if "Buy" in str(v) else "#fbbf24" if v in {"Accumulate","Hold"} else "#fb7185"
    return f"color: {c}; font-weight: 700"

def style_risk(v):
    c = "#34d399" if "Low" in str(v) else "#fbbf24" if "Moderate" in str(v) else "#fb7185"
    return f"color: {c}; font-weight: 700"

def style_valuation(v):
    s = str(v)
    if "Deep" in s or "Under" in s or "DEEP BUY" in s or "ENTRY" in s:
        return "color: #34d399; font-weight: 700"
    if "Fair" in s or "Slightly" in s or "HOLD" in s or "ACCUMULATE" in s:
        return "color: #fbbf24; font-weight: 700"
    return "color: #fb7185; font-weight: 700"

def fmt_money(v): return f"₹{v:,.2f}" if pd.notna(v) and v > 0 else "—"

def inject_mobile_css():
    st.markdown("""<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 5rem !important; max-width: 100% !important; }
    body, .main, section.main { overflow-x: hidden !important; }
    footer { display: none !important; }
    .kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 12px 0 20px 0; }
    .kpi-card { background: linear-gradient(145deg, #0f1729 0%, #0a1120 100%);
        border: 1px solid rgba(139,163,192,0.12); border-radius: 14px; padding: 14px 16px;
        position: relative; overflow: hidden; transition: all 0.2s ease; }
    .kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa); }
    .kpi-card.kpi-green::before { background: linear-gradient(90deg, #34d399, #10b981); }
    .kpi-card.kpi-red::before { background: linear-gradient(90deg, #fb7185, #ef4444); }
    .kpi-card.kpi-amber::before { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
    .kpi-card:hover { transform: translateY(-2px); border-color: rgba(96,165,250,0.4);
        box-shadow: 0 8px 24px rgba(96,165,250,0.1); }
    .kpi-card .kpi-label { color: #8ba3c0; font-size: 0.72rem; font-weight: 500;
        margin-bottom: 6px; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-card .kpi-value { color: #e5eefb; font-size: 1.75rem; font-weight: 800;
        line-height: 1; letter-spacing: -0.02em; }
    .kpi-card .kpi-delta { font-size: 0.7rem; margin-top: 4px; font-weight: 600; }
    .kpi-card .kpi-delta.up { color: #34d399; }
    .kpi-card .kpi-delta.down { color: #fb7185; }
    .kpi-card .kpi-delta.neutral { color: #fbbf24; }
    .stock-card { background: linear-gradient(145deg, #0d1424 0%, #0a0f1c 100%);
        border: 1px solid rgba(139,163,192,0.12); border-radius: 14px; padding: 14px;
        margin-bottom: 10px; position: relative; overflow: hidden; transition: all 0.2s ease; }
    .stock-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.4); }
    .stock-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; }
    .stock-card.signal-strong-buy::before { background: #10b981; }
    .stock-card.signal-buy::before { background: #34d399; }
    .stock-card.signal-accumulate::before { background: #84cc16; }
    .stock-card.signal-hold::before { background: #fbbf24; }
    .stock-card.signal-reduce::before { background: #fb923c; }
    .stock-card.signal-sell::before { background: #fb7185; }
    .stock-card .stock-header { display: flex; justify-content: space-between;
        align-items: flex-start; margin-bottom: 8px; }
    .stock-card .stock-ticker { font-weight: 800; font-size: 1rem; color: #e5eefb; }
    .stock-card .stock-name { font-size: 0.7rem; color: #8ba3c0; margin-top: 2px; opacity: 0.85; }
    .stock-card .stock-score-badge { background: rgba(96,165,250,0.15);
        border: 1px solid rgba(96,165,250,0.4); color: #60a5fa; padding: 4px 10px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 800; }
    .stock-card .stock-price-row { display: flex; align-items: baseline; gap: 8px; margin: 6px 0 8px 0; }
    .stock-card .stock-price { font-size: 1.5rem; font-weight: 800; color: #fff; }
    .stock-card .stock-change { font-size: 0.8rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; }
    .stock-card .stock-change.up { color: #34d399; background: rgba(52,211,153,0.12); }
    .stock-card .stock-change.down { color: #fb7185; background: rgba(251,113,133,0.12); }
    .stock-card .stock-metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;
        margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(139,163,192,0.08); }
    .stock-card .stock-metric-label { font-size: 0.62rem; color: #8ba3c0; opacity: 0.75;
        text-transform: uppercase; margin-bottom: 2px; }
    .stock-card .stock-metric-value { font-size: 0.82rem; font-weight: 700; color: #e5eefb; }
    .stock-card .stock-action { display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.68rem; font-weight: 700; margin-top: 8px; }
    .stock-card .stock-action.entry { background: rgba(52,211,153,0.15); color: #34d399;
        border: 1px solid rgba(52,211,153,0.3); }
    .stock-card .stock-action.exit { background: rgba(251,113,133,0.15); color: #fb7185;
        border: 1px solid rgba(251,113,133,0.3); }
    .stock-card .stock-action.hold { background: rgba(251,191,36,0.15); color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.3); }
    .stock-scroll { display: flex !important; gap: 12px !important; overflow-x: auto !important;
        overflow-y: hidden !important; padding: 4px 0 14px 0 !important;
        scroll-snap-type: x proximity !important; -webkit-overflow-scrolling: touch !important;
        scrollbar-width: thin !important;
        scrollbar-color: rgba(96,165,250,0.5) rgba(255,255,255,0.05) !important; }
    .stock-scroll::-webkit-scrollbar { height: 8px !important; display: block !important; }
    .stock-scroll::-webkit-scrollbar-track { background: rgba(255,255,255,0.04) !important; border-radius: 4px !important; }
    .stock-scroll::-webkit-scrollbar-thumb { background: rgba(96,165,250,0.5) !important; border-radius: 4px !important; }
    .stock-scroll .stock-card { min-width: 240px !important; max-width: 240px !important;
        flex: 0 0 240px !important; margin-bottom: 0 !important; }
    .section-header { display: flex; align-items: center; justify-content: space-between; margin: 16px 0 8px 0; }
    .section-header h3 { font-size: 1rem !important; font-weight: 700 !important; margin: 0 !important; color: #e5eefb; }
    .section-header .see-all { font-size: 0.75rem; color: #60a5fa; font-weight: 600; }
    @media (max-width: 640px) {
        .block-container { padding-left: 0.65rem !important; padding-right: 0.65rem !important;
            padding-top: 0.5rem !important; padding-bottom: 5rem !important; }
        h1 { font-size: 1.2rem !important; }
        .kpi-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 10px 0 16px 0; }
        .kpi-card { padding: 10px 12px; }
        .kpi-card .kpi-label { font-size: 0.6rem; }
        .kpi-card .kpi-value { font-size: 1.35rem; }
        .stock-scroll .stock-card { min-width: 220px !important; max-width: 220px !important; flex: 0 0 220px !important; }
        .stTabs [data-baseweb="tab-list"] { overflow-x: auto !important; white-space: nowrap !important; scrollbar-width: none !important; }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
        .stTabs [data-baseweb="tab"] { padding: 8px 12px !important; font-size: 0.78rem !important;
            white-space: nowrap !important; flex-shrink: 0 !important; }
        div[data-testid="stHorizontalBlock"] { display: flex !important; flex-wrap: wrap !important; gap: 6px !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 0 0 calc(50% - 3px) !important; min-width: calc(50% - 3px) !important;
            max-width: calc(50% - 3px) !important; }
    }
    </style>""", unsafe_allow_html=True)

DISPLAY_COLUMNS = ["Rank","Ticker","Name","Sector","CMP","AI_Score","AI Signal","AI Confidence",
    "Valuation Action","Valuation_Trigger_Score","Entry Trigger Price","Exit Trigger Price",
    "Margin of Safety %","Intrinsic Value (Base)","Quality Grade","Risk Level","Sharpe",
    "Target Price","Upside %","Position Size %"]

def _signal_class(s):
    s = str(s).lower().replace(" ", "-")
    return f"signal-{s}" if s in {"strong-buy","buy","accumulate","hold","reduce","sell"} else "signal-hold"

def _action_class(a):
    a = str(a)
    if "ENTRY" in a or "DEEP BUY" in a: return "entry"
    if "EXIT" in a: return "exit"
    return "hold"

def render_stock_card_html(row):
    signal_cls = _signal_class(row.get("AI Signal", "Hold"))
    action_cls = _action_class(row.get("Valuation Action", ""))
    cmp_val = row.get("CMP", 0); cmp_str = f"₹{cmp_val:,.0f}" if pd.notna(cmp_val) else "—"
    upside = row.get("Upside %", 0)
    upside_cls = "up" if pd.notna(upside) and upside > 0 else "down"
    upside_str = f"{upside:+.1f}%" if pd.notna(upside) else "—"
    ai_score = int(row.get("AI_Score", 0)) if pd.notna(row.get("AI_Score", 0)) else 0
    mos = row.get("Margin of Safety %", 0)
    mos_str = f"{mos:+.0f}%" if pd.notna(mos) else "—"
    entry_p = row.get("Entry Trigger Price", 0)
    entry_str = f"₹{entry_p:,.0f}" if pd.notna(entry_p) and entry_p > 0 else "—"
    return f"""<div class="stock-card {signal_cls}"><div class="stock-header">
    <div><div class="stock-ticker">{row.get('Ticker','?')}</div>
    <div class="stock-name">{str(row.get('Name',''))[:28]}</div></div>
    <div class="stock-score-badge">{ai_score}</div></div>
    <div class="stock-price-row"><div class="stock-price">{cmp_str}</div>
    <div class="stock-change {upside_cls}">{upside_str}</div></div>
    <div class="stock-metrics">
    <div><div class="stock-metric-label">MoS</div><div class="stock-metric-value">{mos_str}</div></div>
    <div><div class="stock-metric-label">Entry</div><div class="stock-metric-value">{entry_str}</div></div>
    <div><div class="stock-metric-label">Signal</div><div class="stock-metric-value">{row.get('AI Signal','—')}</div></div>
    </div><div class="stock-action {action_cls}">{row.get('Valuation Action','⚪ HOLD')}</div></div>"""

def render_stock_card_grid(df, max_items=20):
    if df.empty: st.info("Koi stock nahi."); return
    html = "".join(render_stock_card_html(row) for _, row in df.head(max_items).iterrows())
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">{html}</div>',
                unsafe_allow_html=True)

def render_horizontal_scroll(df, title, max_items=15):
    if df.empty: return
    html = "".join(render_stock_card_html(row) for _, row in df.head(max_items).iterrows())
    st.markdown(f'<div class="section-header"><h3>{title}</h3>'
                f'<span class="see-all">{len(df)} total →</span></div>'
                f'<div class="stock-scroll">{html}</div>', unsafe_allow_html=True)

def _render_dcf_card(row):
    signal_cls = _signal_class(row.get("AI Signal", "Hold"))
    action_cls = _action_class(row.get("Valuation Action", ""))
    def _f(v): return f"₹{v:,.0f}" if pd.notna(v) and v > 0 else "—"
    mos = row.get("Margin of Safety %", 0)
    mos_cls = "up" if pd.notna(mos) and mos > 0 else "down"
    mos_str = f"{mos:+.1f}%" if pd.notna(mos) else "—"
    ai_score = int(row.get("AI_Score", 0)) if pd.notna(row.get("AI_Score", 0)) else 0
    return f"""<div class="stock-card {signal_cls}"><div class="stock-header">
    <div><div class="stock-ticker">{row.get('Ticker','?')}</div>
    <div class="stock-name">{str(row.get('Name',''))[:30]}</div></div>
    <div class="stock-score-badge">{ai_score}</div></div>
    <div class="stock-price-row"><div class="stock-price">{_f(row.get('CMP',0))}</div>
    <div class="stock-change {mos_cls}">MoS {mos_str}</div></div>
    <div class="stock-metrics">
    <div><div class="stock-metric-label">Entry</div><div class="stock-metric-value">{_f(row.get('Entry Trigger Price',0))}</div></div>
    <div><div class="stock-metric-label">Exit</div><div class="stock-metric-value">{_f(row.get('Exit Trigger Price',0))}</div></div>
    <div><div class="stock-metric-label">Base IV</div><div class="stock-metric-value">{_f(row.get('Intrinsic Value (Base)',0))}</div></div>
    </div><div class="stock-metrics">
    <div><div class="stock-metric-label">DCF IV</div><div class="stock-metric-value">{_f(row.get('Intrinsic Value (DCF)',0))}</div></div>
    <div><div class="stock-metric-label">Mult IV</div><div class="stock-metric-value">{_f(row.get('Intrinsic Value (Multiples)',0))}</div></div>
    <div><div class="stock-metric-label">Signal</div><div class="stock-metric-value">{row.get('AI Signal','—')}</div></div>
    </div><div class="stock-action {action_cls}">{row.get('Valuation Action','⚪ HOLD')}</div></div>"""

def _render_dcf_stock_cards(df, max_items=200):
    if df.empty: st.info("Koi stock nahi."); return
    html = "".join(_render_dcf_card(row) for _, row in df.head(max_items).iterrows())
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;">{html}</div>',
                unsafe_allow_html=True)

def render_kpi_row(scored):
    inject_mobile_css()
    total = len(scored); avg = scored["AI_Score"].mean()
    sb = int((scored["AI Signal"] == "Strong Buy").sum())
    bl = int(scored["AI Signal"].isin(["Strong Buy", "Buy"]).sum())
    entry = int(scored.get("Valuation Action", pd.Series()).astype(str)
                .str.contains("DEEP BUY|ENTRY", na=False).sum())
    exit_ = int(scored.get("Valuation Action", pd.Series()).astype(str)
                .str.contains("EXIT", na=False).sum())
    st.markdown(f"""<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">Total Stocks</div>
    <div class="kpi-value">{total}</div><div class="kpi-delta neutral">Portfolio</div></div>
    <div class="kpi-card"><div class="kpi-label">Avg AI Score</div>
    <div class="kpi-value">{avg:.1f}</div><div class="kpi-delta neutral">0-100</div></div>
    <div class="kpi-card kpi-green"><div class="kpi-label">Entry Triggers</div>
    <div class="kpi-value">{entry}</div><div class="kpi-delta up">🟢 Buy</div></div>
    <div class="kpi-card kpi-red"><div class="kpi-label">Exit Triggers</div>
    <div class="kpi-value">{exit_}</div><div class="kpi-delta down">🔴 Sell</div></div>
    <div class="kpi-card kpi-amber"><div class="kpi-label">Strong Buys</div>
    <div class="kpi-value">{sb}</div><div class="kpi-delta neutral">AI ≥ 90</div></div>
    <div class="kpi-card"><div class="kpi-label">Buy + Strong</div>
    <div class="kpi-value">{bl}</div><div class="kpi-delta neutral">AI ≥ 80</div></div>
    </div>""", unsafe_allow_html=True)
    entry_df = scored[scored.get("Valuation Action", pd.Series()).astype(str)
                      .str.contains("DEEP BUY|ENTRY", na=False)].sort_values(
                      "Margin of Safety %", ascending=False)
    if not entry_df.empty: render_horizontal_scroll(entry_df, "🎯 Top Entry Opportunities", 15)
    top_ai = scored.sort_values("AI_Score", ascending=False).head(15)
    if not top_ai.empty: render_horizontal_scroll(top_ai, "⭐ Top AI-Rated Stocks", 15)

# ============ TAB RENDERERS ============
def render_screener_tab(scored, registry, project_dir):
    with st.expander("🔎 Filters", expanded=True):
        c1, c2 = st.columns(2)
        ms = c1.slider("Min AI Score", 0, 100, 0, 5)
        po = ["All"] + sorted(scored["Primary Index"].dropna().unique().tolist())
        pr = c2.selectbox("Primary Index", po)
        sg = st.multiselect("Signals", sorted(scored["AI Signal"].unique().tolist()),
                            default=sorted(scored["AI Signal"].unique().tolist()))
        act_opts = sorted(scored.get("Valuation Action", pd.Series()).dropna().unique().tolist())
        ac = st.multiselect("Valuation Action", act_opts, default=act_opts) if act_opts else []
    f = scored[scored["AI_Score"].ge(ms) & scored["AI Signal"].isin(sg)]
    if "Valuation Action" in scored.columns and ac: f = f[f["Valuation Action"].isin(ac)]
    if pr != "All": f = f[f["Primary Index"] == pr]
    f = f.copy()
    sc = detect_sector_column(f)
    if sc and "Sector" not in f.columns: f = f.rename(columns={sc: "Sector"})
    st.subheader(f"📋 Screener ({len(f)})")
    view_mode = st.radio("View", ["📇 Cards", "📊 Table"], horizontal=True,
                        key="screener_view", label_visibility="collapsed")
    if view_mode == "📇 Cards":
        if not f.empty:
            col1, col2 = st.columns(2)
            sort_by = col1.selectbox("Sort by", ["AI_Score","Margin of Safety %","Upside %","Sharpe"], key="card_sort")
            max_cards = col2.slider("Max cards", 5, 50, 20, 5, key="max_cards")
            render_stock_card_grid(f.sort_values(sort_by, ascending=False, na_position="last"), max_cards)
    else:
        cols = [c for c in DISPLAY_COLUMNS if c in f.columns]
        styled = (f[cols].style
            .map(style_signal, subset=["AI Signal"])
            .map(style_valuation, subset=["Valuation Action"])
            .map(style_risk, subset=["Risk Level"])
            .format({"CMP": "₹{:,.2f}", "AI_Score": "{:.0f}", "AI Confidence": "{:.0f}",
                "Valuation_Trigger_Score": "{:.0f}",
                "Entry Trigger Price": "₹{:,.2f}", "Exit Trigger Price": "₹{:,.2f}",
                "Margin of Safety %": "{:+.2f}%", "Intrinsic Value (Base)": "₹{:,.2f}",
                "Sharpe": "{:.2f}", "Target Price": "₹{:,.2f}",
                "Upside %": "{:+.2f}%", "Position Size %": "{:.1f}%"}, na_rep="—"))
        st.dataframe(styled, use_container_width=True, height=560, hide_index=True)
    ss = analyse_sectors(scored)
    rot = compute_sector_rotation(project_dir, ss) if not ss.empty else pd.DataFrame()
    idf = build_index_table(scored, registry)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV", data=f.to_csv(index=False).encode("utf-8"),
                       file_name=f"screener_{datetime.now():%Y%m%d}.csv", mime="text/csv",
                       use_container_width=True)
    try:
        xlsx = _build_excel(f, ss, rot, idf)
        d2.download_button("📊 Excel", data=xlsx,
                           file_name=f"screener_{datetime.now():%Y%m%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    except Exception: pass

def render_dcf_valuation_tab(scored):
    st.subheader("💰 DCF Valuation")
    if "Intrinsic Value (Base)" not in scored.columns: st.warning("No valuation data."); return
    val = scored.dropna(subset=["Intrinsic Value (Base)", "CMP"]).copy()
    if val.empty: st.info("No data."); return
    dv = int((val["Margin of Safety %"] >= 40).sum())
    uv = int(((val["Margin of Safety %"] >= 10) & (val["Margin of Safety %"] < 40)).sum())
    fv = int((val["Margin of Safety %"].abs() < 10).sum())
    ov = int((val["Margin of Safety %"] <= -10).sum())
    am = val["Margin of Safety %"].mean()
    aw = val["WACC Used"].mean() if "WACC Used" in val else np.nan
    st.markdown(f"""<div class="kpi-grid">
    <div class="kpi-card kpi-green"><div class="kpi-label">🟢 Deep Value</div><div class="kpi-value">{dv}</div></div>
    <div class="kpi-card kpi-green"><div class="kpi-label">🟢 Undervalued</div><div class="kpi-value">{uv}</div></div>
    <div class="kpi-card kpi-amber"><div class="kpi-label">⚪ Fair</div><div class="kpi-value">{fv}</div></div>
    <div class="kpi-card kpi-red"><div class="kpi-label">🔴 Overvalued</div><div class="kpi-value">{ov}</div></div>
    <div class="kpi-card"><div class="kpi-label">Avg MoS</div><div class="kpi-value">{am:+.1f}%</div></div>
    <div class="kpi-card"><div class="kpi-label">Avg WACC</div><div class="kpi-value">{f"{aw:.1%}" if pd.notna(aw) else "—"}</div></div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🎯 Trigger Alerts by Valuation Signal")
    sig_order = ["🟢 Deep Value", "🟢 Undervalued", "🟡 Slightly Under", "⚪ Fair Value",
                 "🟠 Slightly Over", "🔴 Overvalued", "🔴 Highly Overvalued"]
    existing = [s for s in sig_order if s in val["Valuation Signal"].dropna().unique()]
    if existing:
        sig_tabs = st.tabs([f"{s} ({int((val['Valuation Signal'] == s).sum())})" for s in existing])
        for tab, sig in zip(sig_tabs, existing):
            with tab:
                sub = val[val["Valuation Signal"] == sig].sort_values("Margin of Safety %", ascending=False)
                if sub.empty: st.info("No stocks.")
                else: _render_dcf_stock_cards(sub, 200)
    st.divider()
    with st.expander("🔎 Filters", expanded=False):
        mms = st.slider("Min MoS %", -100, 100, -100, 5)
        sigs = st.multiselect("Valuation Signal",
                              sorted(val["Valuation Signal"].dropna().unique().tolist()),
                              default=sorted(val["Valuation Signal"].dropna().unique().tolist()))
        sc = detect_sector_column(val)
        if sc:
            so = ["All"] + sorted(val[sc].dropna().unique().tolist())
            sf = st.selectbox("Sub-Sector", so)
        else: sf = "All"
    f = val[val["Margin of Safety %"].ge(mms) & val["Valuation Signal"].isin(sigs)]
    if sf != "All" and sc: f = f[f[sc] == sf]
    f = f.copy().sort_values("Margin of Safety %", ascending=False).reset_index(drop=True)
    if "Sector" not in f.columns:
        sc2 = detect_sector_column(f)
        if sc2: f = f.rename(columns={sc2: "Sector"})
    table_tab, chart_tab = st.tabs([f"📋 Full Table ({len(f)})", "📊 Price vs IV"])
    with table_tab:
        cols = ["Rank","Ticker","Name","Sector","CMP","Lower Intrinsic Value",
                "Intrinsic Value (Base)","Upper Intrinsic Value","Intrinsic Value (DCF)",
                "Intrinsic Value (Multiples)","WACC Used","Margin of Safety %","Valuation Signal",
                "Valuation Action","Entry Trigger Price","Exit Trigger Price","AI_Score","AI Signal"]
        cols = [c for c in cols if c in f.columns]
        st.dataframe(f[cols].style
            .map(style_valuation, subset=["Valuation Signal"])
            .map(style_valuation, subset=["Valuation Action"])
            .map(style_signal, subset=["AI Signal"])
            .format({"CMP": "₹{:,.2f}", "Margin of Safety %": "{:+.2f}%", "AI_Score": "{:.0f}"},
                    na_rep="—"),
            use_container_width=True, hide_index=True, height=560)
    with chart_tab:
        if go is not None:
            pd_ = f.dropna(subset=["CMP", "Intrinsic Value (Base)"]).copy()
            if not pd_.empty:
                pd_["Zone"] = np.where(pd_["Margin of Safety %"] >= 10, "Undervalued",
                    np.where(pd_["Margin of Safety %"] <= -10, "Overvalued", "Fair"))
                fig = px.scatter(pd_, x="CMP", y="Intrinsic Value (Base)", color="Zone",
                    size="AI_Score", hover_data=["Ticker","Name","Margin of Safety %"],
                    color_discrete_map={"Undervalued": "#34d399", "Fair": "#fbbf24",
                                        "Overvalued": "#fb7185"})
                mv = float(max(pd_["CMP"].max(), pd_["Intrinsic Value (Base)"].max()))
                fig.add_trace(go.Scatter(x=[0,mv], y=[0,mv], mode="lines", name="Fair Value",
                    line=dict(color="#8ba3c0", dash="dash", width=1.5)))
                _style_figure(fig, 500); st.plotly_chart(fig, use_container_width=True)

def render_deep_tab(scored):
    st.subheader("🔬 Deep Dive")
    if scored.empty: return
    t = st.selectbox("Ticker", scored["Ticker"].tolist())
    if not t: return
    row = scored.loc[scored["Ticker"] == t].iloc[0]
    st.markdown(render_stock_card_html(row), unsafe_allow_html=True)
    tabs = st.tabs(["📈 Chart", "💰 Valuation", "📊 Technical", "🧮 Fundamentals"])
    with tabs[0]: render_price_chart(t, row=row)
    with tabs[1]:
        vd = {k: row.get(k) for k in ["Intrinsic Value (DCF)","Intrinsic Value (Multiples)",
              "Intrinsic Value (Base)","Lower Intrinsic Value","Upper Intrinsic Value",
              "WACC Used","Margin of Safety %","Valuation Signal","Valuation Action",
              "Entry Trigger Price","Exit Trigger Price"] if k in row}
        st.json({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int,float,np.number)) else v))
                 for k, v in vd.items()})
    with tabs[2]:
        tech = {k: row.get(k) for k in ["CMP","RSI","ADX","ATR","MACDSignal","Histogram",
                "20 DMA","50 DMA","200 DMA","EMA20","EMA50","EMA200","Upper Band","Middle Band",
                "Lower Band","DI Plus","DI Minus","SuperTrend","SuperTrend_Val","Trend Pivot",
                "R1","R2","R3","S1","S2","S3","52W High","52W Low","Trend Score","Momentum Score",
                "Volume Score","Volatility Score","Support Resistance Score",
                "Technical_Score","Analyst_Consensus_Score","Fundamental_Score_AI",
                "Sector_Relative_Score","DCF_Valuation_Score","Valuation_Trigger_Score"] if k in row}
        st.json({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int,float,np.number)) else v))
                 for k, v in tech.items()})
    with tabs[3]:
        fd = {k: row.get(k) for k in ["PE Ratio","PB Ratio","EV/EBITDA Ratio","Return on Equity",
              "ROCE","Net Profit Margin","EBITDA Margin","5Y Historical EPS Growth",
              "Debt to Equity","Current Ratio","Promoter Holding","Dividend Yield","Free Cash Flow",
              "Operating Cash Flow","Total Debt","Cash and Equivalent","Market Cap"] if k in row}
        st.json({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int,float,np.number)) else v))
                 for k, v in fd.items()})

# ⭐ CHANGE 2: Hybrid tab
def render_hybrid_tab(scored, project_dir):
    st.subheader("🔀 Hybrid Analytics")
    st.caption("Risk • Red Flags • Sectors • Rotation")
    hyb_tabs = st.tabs(["⚠️ Risk", "🚩 Flags", "🏭 Sectors", "🔥 Rotation"])
    with hyb_tabs[0]:
        if "Sharpe" not in scored.columns or scored["Sharpe"].isna().all():
            st.info("Risk data missing.")
        else:
            m = st.columns(4)
            m[0].metric("Avg Sharpe", f"{scored['Sharpe'].mean():.2f}")
            m[1].metric("Avg Beta", f"{scored['Beta'].mean():.2f}")
            m[2].metric("Beating Nifty", f"{(scored['Return 1Y %'] > scored['Nifty 1Y %']).sum()}/{len(scored)}")
            m[3].metric("Avg Alpha", f"{scored['Alpha %'].mean():+.2f}%")
            cs = ["Rank","Ticker","Name","Sharpe","Sortino","Volatility %","Max Drawdown %",
                  "Beta","Alpha %","Return 1Y %","Nifty 1Y %","Relative Strength %"]
            cs = [c for c in cs if c in scored.columns]
            st.dataframe(scored.sort_values("Sharpe", ascending=False, na_position="last")
                .head(25)[cs].style.format({"Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                "Volatility %": "{:.2f}", "Max Drawdown %": "{:.2f}", "Beta": "{:.2f}",
                "Alpha %": "{:+.2f}%", "Return 1Y %": "{:+.2f}%",
                "Nifty 1Y %": "{:+.2f}%", "Relative Strength %": "{:+.2f}%"}, na_rep="—"),
                use_container_width=True, hide_index=True)
    with hyb_tabs[1]:
        if "Red Flag Count" not in scored.columns: st.info("No data.")
        else:
            fl = scored[scored["Red Flag Count"] > 0].sort_values(
                ["Red Flag Count","AI_Score"], ascending=[False, False])
            if fl.empty: st.success("✅ No red flags.")
            else: render_stock_card_grid(fl, 30)
    with hyb_tabs[2]:
        ss = analyse_sectors(scored)
        if ss.empty: st.info("No sub-sector data.")
        else:
            save_sector_snapshot(project_dir, ss)
            st.dataframe(ss.style.background_gradient(
                subset=["Avg AI Score","Avg Sector-Rel"], cmap="RdYlGn")
                .format({"Avg AI Score": "{:.1f}", "Avg Fundamental": "{:.1f}",
                         "Avg Technical": "{:.1f}", "Avg Sector-Rel": "{:.1f}",
                         "Avg Rank Composite": "{:.1f}", "Avg PE": "{:.1f}",
                         "Avg Upside %": "{:+.1f}%"}, na_rep="—"),
                use_container_width=True, hide_index=True, height=min(640, 40 + 32*len(ss)))
    with hyb_tabs[3]:
        ss = analyse_sectors(scored)
        if ss.empty: st.info("No sub-sector data.")
        else:
            save_sector_snapshot(project_dir, ss)
            rot = compute_sector_rotation(project_dir, ss)
            if rot.empty: st.info("Rotation unavailable.")
            elif not rot["Prev AI Score"].notna().any():
                st.info("📌 Pehla run — WoW next run se populate hoga.")
            else:
                dc = [c for c in ["Sector","Stocks","Avg AI Score","Prev AI Score",
                                  "WoW AI Δ","Avg Sector-Rel","WoW Sector-Rel Δ","Rotation"]
                      if c in rot.columns]
                st.dataframe(rot[dc].style.background_gradient(
                    subset=["WoW AI Δ","WoW Sector-Rel Δ"], cmap="RdYlGn")
                    .format({"Avg AI Score": "{:.1f}", "Prev AI Score": "{:.1f}",
                             "WoW AI Δ": "{:+.1f}", "Avg Sector-Rel": "{:.1f}",
                             "WoW Sector-Rel Δ": "{:+.1f}"}, na_rep="—"),
                    use_container_width=True, hide_index=True)

def render_trade_tab(scored):
    st.subheader("💼 Trade Plan")
    if "Position Size %" not in scored.columns: return
    cand = scored[scored["AI Signal"].isin(["Strong Buy","Buy","Accumulate"])].copy()
    if cand.empty: st.info("No buy-rated stocks."); return
    render_stock_card_grid(cand, 30)
    st.info(f"💡 Total: **{cand['Position Size %'].sum():.1f}%** across {len(cand)} ideas.")

def render_peer_tab(scored):
    st.markdown("### 👥 Peer Comparison")
    sc = detect_sector_column(scored)
    if sc is None or scored[sc].isna().all(): st.info("No data."); return
    ss = sorted(scored[sc].dropna().unique().tolist())
    if not ss: return
    sec = st.selectbox("Sub-Sector", ss, key="peer_sec")
    mo = ["AI_Score","Fundamental_Score_AI","Sector_Relative_Score","Technical_Score",
          "Rank_Composite","CSV_Technical_Score","Analyst_Consensus_Score",
          "CashFlow_Quality_Score","DCF_Valuation_Score","Valuation_Trigger_Score",
          "Margin of Safety %","PE Ratio","Return on Equity","Upside %","Sharpe"]
    av = [m for m in mo if m in scored.columns]
    met = st.selectbox("Metric", av, key="peer_met")
    peers = scored.loc[scored[sc] == sec].copy()
    if peers.empty: return
    v = pd.to_numeric(peers[met], errors="coerce")
    peers["_v"] = v
    peers = peers.sort_values("_v", ascending=False, na_position="last").reset_index(drop=True)
    if go is not None:
        av_v = v.mean()
        colors = ["#34d399" if x >= av_v else "#fb7185" for x in peers["_v"].fillna(0)]
        fig = go.Figure(go.Bar(x=peers["Ticker"], y=peers["_v"], marker_color=colors,
                               text=peers["_v"].round(2), textposition="outside"))
        if pd.notna(av_v):
            fig.add_hline(y=av_v, line_dash="dash", line_color="#60a5fa",
                          annotation_text=f"Avg: {av_v:.2f}")
        fig.update_layout(title=f"{met} — {sec}", xaxis_title="", yaxis_title=met)
        _style_figure(fig, 400); st.plotly_chart(fig, use_container_width=True)
    render_stock_card_grid(peers, 20)

def render_index_tab(scored, registry):
    st.subheader("📊 Index Analysis")
    idf = build_index_table(scored, registry)
    if idf.empty or idf["Stocks"].sum() == 0: st.info("No constituents."); return
    st.dataframe(idf.style.format({"Avg AI Score": "{:.1f}", "Avg Fundamental": "{:.1f}",
        "Avg Technical": "{:.1f}", "Avg Sector-Rel": "{:.1f}",
        "Avg Upside %": "{:+.1f}%", "Avg PE": "{:.1f}"}, na_rep="—"),
        use_container_width=True, hide_index=True)
    if go is None: return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idf["Index"], y=idf["Buy+"], name="Buy+", marker_color="#34d399"))
    fig.add_trace(go.Bar(x=idf["Index"], y=idf["Hold"], name="Hold", marker_color="#fbbf24"))
    fig.add_trace(go.Bar(x=idf["Index"], y=idf["Sell+"], name="Sell+", marker_color="#fb7185"))
    fig.update_layout(barmode="stack", title="Breadth")
    _style_figure(fig, 380); st.plotly_chart(fig, use_container_width=True)

def render_patterns_tab(scored):
    st.subheader("📐 Pattern Scanner")
    if "Patterns" not in scored.columns: return
    cnt = {p: scored["Patterns"].str.contains(p, regex=False).sum() for p in PATTERN_NAMES}
    pick = st.selectbox("Filter", ["All"] + [p for p, c in cnt.items() if c > 0], key="pat_f")
    f = scored if pick == "All" else scored[scored["Patterns"].str.contains(pick, regex=False)]
    render_stock_card_grid(f, 30)

def render_quality_tab(scored):
    st.subheader("🔍 Data Quality")
    tot = len(scored)
    fok = int((scored.get("Fetch Status", pd.Series()) == "OK").sum())
    sc = detect_sector_column(scored)
    val_ok = scored.get('Intrinsic Value (Base)', pd.Series()).notna().sum()
    st.markdown(f"""<div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr);">
    <div class="kpi-card"><div class="kpi-label">Total</div><div class="kpi-value">{tot}</div></div>
    <div class="kpi-card kpi-green"><div class="kpi-label">Fetch OK</div><div class="kpi-value">{fok}/{tot}</div></div>
    <div class="kpi-card kpi-amber"><div class="kpi-label">Sector Data</div><div class="kpi-value">{scored[sc].notna().sum() if sc else 0}/{tot}</div></div>
    <div class="kpi-card kpi-green"><div class="kpi-label">Valuation OK</div><div class="kpi-value">{val_ok}/{tot}</div></div>
    </div>""", unsafe_allow_html=True)

# ============ EXCEL EXPORT ============
def _build_excel(scored, ss, rot, idf):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        scored.to_excel(w, sheet_name="Screener", index=False)
        if not ss.empty: ss.to_excel(w, sheet_name="Sectors", index=False)
        if not rot.empty: rot.to_excel(w, sheet_name="Rotation", index=False)
        if not idf.empty: idf.to_excel(w, sheet_name="Indices", index=False)
    return buf.getvalue()

# ============ DASHBOARD ============
def render_dashboard(scored, registry, project_dir):
    render_kpi_row(scored)
    st.divider()
    order = st.session_state.get("tab_order", DEFAULT_TAB_ORDER)
    labels = [TAB_META[k]["label"] for k in order]
    tabs = st.tabs(labels)
    for tab, key in zip(tabs, order):
        with tab:
            try:
                if key == "screener": render_screener_tab(scored, registry, project_dir)
                elif key == "dcf": render_dcf_valuation_tab(scored)
                elif key == "deep": render_deep_tab(scored)
                elif key == "hybrid": render_hybrid_tab(scored, project_dir)
                elif key == "trade": render_trade_tab(scored)
                elif key == "peers": render_peer_tab(scored)
                elif key == "indices": render_index_tab(scored, registry)
                elif key == "patterns": render_patterns_tab(scored)
                elif key == "quality": render_quality_tab(scored)
            except Exception as exc:
                st.error(f"Error in {key}: {exc}")

# ============ SIDEBAR ============
def render_sidebar(project_dir):
    with st.sidebar:
        st.header("📁 Data Sources")
        st.caption("Upload CSVs — `./data/` me save honge.")
        fu = st.file_uploader("Nifty Fundamentals CSV", type=["csv"], key="f")
        tu = st.file_uploader("Explore Promising CSV", type=["csv"], key="t")
    return fu, tu

# ============ MAIN ============
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide",
                       initial_sidebar_state="collapsed")
    inject_mobile_css()

    # ⭐ Auto-refresh (15 min)
    if AUTOREFRESH_AVAILABLE:
        refresh_count = st_autorefresh(interval=AUTO_REFRESH_MIN * 60 * 1000,
                                       limit=None, key="auto_refresh_tick")
        if refresh_count % 4 == 0 and refresh_count > 0:
            st.cache_data.clear()
    else:
        refresh_count = 0

    st.title(f"📈 {APP_TITLE}")
    st.caption("10-dim AI Score + Hybrid DCF + Entry/Exit Triggers.")

    project_dir = Path(__file__).parent
    registry = load_index_constituents(project_dir)
    fu, tu = render_sidebar(project_dir)

    # ⭐ Tab reorder UI + auto-refresh status
    order = render_tab_order_ui(project_dir)
    with st.sidebar:
        st.divider()
        if AUTOREFRESH_AVAILABLE:
            st.caption(f"🔄 Auto-refresh: every {AUTO_REFRESH_MIN} min")
            st.caption(f"⏱️ Last tick: #{refresh_count}")
        else:
            st.caption("⚠️ Auto-refresh off")
        st.caption(f"🕐 Loaded: {datetime.now():%H:%M:%S}")

    if fu: persist_uploaded_csv(fu, FUNDAMENTAL_PREFIX, project_dir)
    if tu: persist_uploaded_csv(tu, TECHNICAL_PREFIX, project_dir)
    try:
        fund = read_csv_source(fu, find_latest_csv(project_dir, FUNDAMENTAL_PREFIX), "Fundamentals")
        _ = read_csv_source(tu, find_latest_csv(project_dir, TECHNICAL_PREFIX), "Technicals")
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc)); st.stop()
    tickers = fund["Ticker"].dropna().unique().tolist()
    st.info(f"📥 {len(tickers)} tickers ka data fetch ho raha hai...")
    tech = fetch_technicals(tickers)
    try: merged = merge_sources(fund, tech)
    except ValueError as exc: st.error(str(exc)); st.stop()
    with st.spinner("Scoring + valuation compute ho rahe hain..."):
        scored = build_scores(merged, registry)
    render_dashboard(scored, registry, project_dir)

if __name__ == "__main__":
    main()
