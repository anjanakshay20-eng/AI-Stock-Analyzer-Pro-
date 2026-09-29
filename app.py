"""AI Stock Analyzer Pro — TradingView-Style Native UI (v3).

3 targeted changes:
  1. Fixed horizontal scroll on Top Entry Opportunities
  2. DCF Trigger Alerts by Valuation Signal with full data cards
  3. Deep Dive: Technical Analysis tab + Chart time ranges (1D/1W/1M/3M/6M/1Y/5Y)
"""

from __future__ import annotations
import concurrent.futures, io, json, logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, streamlit as st, yfinance as yf

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None; px = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

# ============= CONSTANTS =============
APP_TITLE = "AI Stock Analyzer Pro"
FUNDAMENTAL_PREFIX = "Nifty"
TECHNICAL_PREFIX = "Explore_Promising"
PRICE_PERIOD = "1y"
CACHE_TTL_SECONDS = 60 * 60
MAX_WORKERS = 8
DATA_DIR_NAME = "data"
INDEX_JSON_NAME = "index_constituents.json"
SNAPSHOT_DIR_NAME = "sector_snapshots"

RISK_FREE_RATE = 0.065
EQUITY_RISK_PREMIUM = 0.06
TERMINAL_GROWTH = 0.035
VALUATION_BAND = 0.15
BENCHMARK_TICKER = "^NSEI"
TRADING_DAYS = 252

SECTOR_COLUMN_CANDIDATES = ["Sub-Sector", "Sector", "Sector Name"]

FUNDAMENTAL_THRESHOLDS = {
    "PE Ratio": (15, 25, 35, 50, True),
    "PB Ratio": (1.5, 3, 5, 8, True),
    "EV/EBITDA Ratio": (8, 12, 18, 25, True),
    "Return on Equity": (20, 15, 10, 5, False),
    "ROCE": (20, 15, 10, 5, False),
    "Net Profit Margin": (15, 10, 5, 2, False),
    "EBITDA Margin": (20, 15, 10, 5, False),
    "5Y Historical EPS Growth": (20, 15, 10, 5, False),
    "5Y Historical Revenue Growth": (15, 10, 5, 0, False),
    "5Y Historical EBITDA Growth": (15, 10, 5, 0, False),
    "5Y CAGR": (15, 10, 5, 0, False),
    "Debt to Equity": (0.2, 0.5, 1.0, 2.0, True),
    "Current Ratio": (2.0, 1.5, 1.0, 0.5, False),
    "Promoter Holding": (50, 40, 30, 20, False),
    "Pledged Promoter Holdings": (0, 5, 10, 25, True),
    "Dividend Yield": (3, 2, 1, 0.5, False),
    "Percentage Upside": (20, 10, 5, 0, False),
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
    "Nifty 50": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
        "SBIN", "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "LT", "HCLTECH", "ASIANPAINT",
        "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
        "ONGC", "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL",
        "ADANIENT", "ADANIPORTS", "COALINDIA", "GRASIM", "HINDALCO", "DRREDDY", "CIPLA",
        "EICHERMOT", "BRITANNIA", "DIVISLAB", "TECHM", "INDUSINDBK", "BAJAJFINSV",
        "HEROMOTOCO", "SBILIFE", "HDFCLIFE", "APOLLOHOSP", "BPCL", "TATACONSUM",
        "BAJAJ-AUTO", "LTIM", "SHRIRAMFIN"],
    "Nifty Next 50": ["ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "DMART",
        "BAJAJHLDNG", "BANKBARODA", "BERGEPAINT", "BEL", "BOSCHLTD", "CANBK", "CHOLAFIN",
        "COLPAL", "DABUR", "DLF", "GAIL", "GODREJCP", "GODREJPROP", "HAVELLS", "HDFCAMC",
        "HINDZINC", "ICICIGI", "ICICIPRULI", "IOC", "INDHOTEL", "INDIGO", "JINDALSTEL",
        "JIOFIN", "LICI", "LODHA", "MARICO", "MOTHERSON", "NAUKRI", "PFC", "PIDILITIND",
        "PIIND", "PNB", "RECLTD", "SIEMENS", "SRF", "TVSMOTOR", "TORNTPHARM", "VBL",
        "VEDL", "ZYDUSLIFE"],
    "Bank Nifty": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK",
        "BANKBARODA", "PNB", "IDFCFIRSTB", "FEDERALBNK", "AUBANK", "BANDHANBNK"],
}

# ============= COLUMN NORMALIZATION =============
_UNICODE_FIX = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                              "–": "-", "—": "-", "\u00a0": " "})

def normalise_columns(df):
    return df.rename(columns={c: str(c).translate(_UNICODE_FIX).strip()
                              for c in df.columns})

def detect_sector_column(df):
    for c in SECTOR_COLUMN_CANDIDATES:
        if c in df.columns: return c
    return None

# ============= INDEX JSON =============
def _index_json_path(project_dir):
    return project_dir / DATA_DIR_NAME / INDEX_JSON_NAME

def load_index_constituents(project_dir):
    path = _index_json_path(project_dir)
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(DEFAULT_INDEX_CONSTITUENTS, indent=2),
                            encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Could not seed %s: %s", path, exc)
            return {k: frozenset(v) for k, v in DEFAULT_INDEX_CONSTITUENTS.items()}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        indices = {str(n): frozenset(str(t).upper().replace(".NS", "").replace(".BO", "").strip()
                                     for t in tickers if str(t).strip())
                   for n, tickers in raw.items()}
        return indices or {k: frozenset(v) for k, v in DEFAULT_INDEX_CONSTITUENTS.items()}
    except (OSError, json.JSONDecodeError, ValueError):
        return {k: frozenset(v) for k, v in DEFAULT_INDEX_CONSTITUENTS.items()}

def _normalise_ticker(ticker):
    return str(ticker).upper().replace(".NS", "").replace(".BO", "").strip()

# ============= GENERIC HELPERS =============
def numeric_column(frame, name, default=np.nan):
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")

def find_latest_csv(directory, prefix):
    dirs = [directory, directory / DATA_DIR_NAME]
    matches = [p for f in dirs if f.exists() for p in f.glob(f"*{prefix}*.csv")]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None

def persist_uploaded_csv(upload, prefix, directory):
    if upload is None: return None
    d = directory / DATA_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    fn = "Nifty_Fundamentals.csv" if prefix == FUNDAMENTAL_PREFIX else "Explore_Promising.csv"
    p = d / fn
    p.write_bytes(upload.getvalue())
    return p

def read_csv_source(upload, fallback, label):
    src = upload if upload is not None else fallback
    if src is None:
        raise FileNotFoundError(f"{label} CSV file nahi mili. Upload karein.")
    try:
        frame = pd.read_csv(src)
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"{label} CSV read nahi hui: {exc}") from exc
    frame = normalise_columns(frame)
    if frame.empty: raise ValueError(f"{label} CSV khaali hai.")
    if "Ticker" not in frame.columns:
        raise ValueError(f"{label} CSV mein 'Ticker' column nahi hai.")
    frame["Ticker"] = frame["Ticker"].astype("string").str.strip().str.upper()
    frame = frame.loc[frame["Ticker"].notna() & frame["Ticker"].ne("")].copy()
    return frame.drop_duplicates(subset="Ticker", keep="last").reset_index(drop=True)

def merge_sources(fundamentals, technicals):
    common = (set(fundamentals.columns) & set(technicals.columns)) - {"Ticker"}
    fund_clean = fundamentals.copy()
    for col in common:
        if technicals[col].isna().sum() < fundamentals[col].isna().sum():
            fund_clean = fund_clean.drop(columns=[col])
    merged = fund_clean.merge(technicals, on="Ticker", how="inner")
    if merged.empty:
        raise ValueError("Dono CSV files mein matching Ticker nahi mile.")
    return merged

# ============= FUNDAMENTAL SCORING =============
def calculate_fundamental_score(frame):
    total = pd.Series(0.0, index=frame.index)
    count = pd.Series(0, index=frame.index, dtype="int64")
    for col, (best, good, fair, weak, reverse) in FUNDAMENTAL_THRESHOLDS.items():
        v = numeric_column(frame, col)
        bins = ([v.le(best), v.le(good), v.le(fair), v.le(weak)] if reverse
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
    ws = pd.Series(0.0, index=frame.index)
    wt = pd.Series(0.0, index=frame.index)
    for metric, (higher_better, weight) in SECTOR_RELATIVE_METRICS.items():
        if metric not in frame.columns: continue
        v = pd.to_numeric(frame[metric], errors="coerce")
        pct = v.groupby(sectors).rank(pct=True, ascending=higher_better)
        valid = pct.notna()
        ws += (pct * 100 * weight).where(valid, 0)
        wt += pd.Series(weight, index=frame.index).where(valid, 0)
    return ws.div(wt.replace(0, np.nan)).fillna(50).clip(0, 100)

# ============= HYBRID VALUATION =============
def _is_financial_sector(frame, sector_col):
    if sector_col not in frame.columns:
        return pd.Series(False, index=frame.index)
    s = frame[sector_col].fillna("").astype(str).str.lower()
    return (s.str.contains("bank", na=False) | s.str.contains("finance", na=False)
            | s.str.contains("insurance", na=False) | s.str.contains("nbfc", na=False))

def calculate_enhanced_2stage_dcf(frame):
    fcf = numeric_column(frame, "Free Cash Flow")
    ocf = numeric_column(frame, "Operating Cash Flow")
    capex = numeric_column(frame, "Capital Expenditure").abs()
    derived_fcf = ocf - capex
    fcf = fcf.where(fcf.notna() & fcf.gt(0), derived_fcf)
    debt = numeric_column(frame, "Total Debt", 0).fillna(0)
    cash = numeric_column(frame, "Cash and Equivalent", 0).fillna(0)
    mcap = numeric_column(frame, "Market Cap")
    close = numeric_column(frame, "Close Price")
    shares = numeric_column(frame, "Common Shares Outstanding")
    fallback = mcap.div(close.where(close.gt(0)))
    shares = shares.where(shares.gt(0), fallback).where(lambda s: s.gt(0))
    beta = numeric_column(frame, "Beta", 1.0).fillna(1.0).clip(0.5, 2.0)
    wacc = (RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM).clip(0.10, 0.14)
    g_hist = numeric_column(frame, "5Y Historical EPS Growth", 10).fillna(10).clip(5, 20) / 100
    g_near, g_far = g_hist, g_hist * 0.5
    pv_fcf = pd.Series(0.0, index=frame.index)
    cur = fcf.copy()
    for year in range(1, 11):
        g = g_near if year <= 5 else g_far
        cur = cur * (1 + g)
        pv_fcf += cur / ((1 + wacc) ** year)
    terminal_fcf = cur * (1 + TERMINAL_GROWTH)
    terminal_value = terminal_fcf / (wacc - TERMINAL_GROWTH).replace(0, np.nan)
    pv_terminal = terminal_value / ((1 + wacc) ** 10)
    ev = pv_fcf + pv_terminal
    eq = ev - debt + cash
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
    return pd.DataFrame({"DCF Base": intrinsic.where(valid),
                         "WACC Used": wacc.where(valid)}, index=frame.index)

def calculate_peer_multiples(frame, sector_col="Sub-Sector"):
    empty = pd.DataFrame({"Multiples Base": np.nan}, index=frame.index)
    if sector_col not in frame.columns or frame[sector_col].isna().all():
        return empty
    sectors = frame[sector_col].fillna("Unknown").astype(str)
    close = numeric_column(frame, "Close Price")
    pe = numeric_column(frame, "PE Ratio")
    pb = numeric_column(frame, "PB Ratio")
    ev = numeric_column(frame, "EV/EBITDA Ratio")
    eps = close / pe.where(pe.gt(0))
    bvps = close / pb.where(pb.gt(0))
    ebitda_ps = close / ev.where(ev.gt(0))
    pe_med = pe.groupby(sectors).transform("median")
    pb_med = pb.groupby(sectors).transform("median")
    ev_med = ev.groupby(sectors).transform("median")
    vals = [(ebitda_ps * ev_med, 0.5), (eps * pe_med, 0.3), (bvps * pb_med, 0.2)]
    ws = pd.Series(0.0, index=frame.index)
    wt = pd.Series(0.0, index=frame.index)
    for v, w in vals:
        valid = v.notna() & v.gt(0)
        ws += v.fillna(0) * w
        wt += pd.Series(w, index=frame.index).where(valid, 0)
    base = ws / wt.replace(0, np.nan)
    base = base.where((base > close * 0.2) & (base < close * 5))
    return pd.DataFrame({"Multiples Base": base}, index=frame.index)

def calculate_hybrid_valuation(frame):
    dcf = calculate_enhanced_2stage_dcf(frame)
    mult = calculate_peer_multiples(frame)
    dcf_base = dcf["DCF Base"]
    mult_base = mult["Multiples Base"]
    blended = dcf_base * 0.6 + mult_base * 0.4
    blended = blended.fillna(dcf_base).fillna(mult_base)
    has_both = dcf_base.notna() & mult_base.notna()
    band = pd.Series(VALUATION_BAND, index=frame.index).where(has_both, 0.20)
    return pd.DataFrame({
        "Intrinsic Value (DCF)": dcf_base,
        "Intrinsic Value (Multiples)": mult_base,
        "Intrinsic Value (Base)": blended,
        "Lower Intrinsic Value": blended * (1 - band),
        "Upper Intrinsic Value": blended * (1 + band),
        "WACC Used": dcf["WACC Used"],
    }, index=frame.index)

def calculate_valuation_score(frame):
    cmp = numeric_column(frame, "CMP")
    iv = numeric_column(frame, "Intrinsic Value (Base)")
    mos = (iv - cmp) / iv.where(iv.gt(0)) * 100
    score = pd.Series(np.select(
        [mos >= 40, mos >= 25, mos >= 10, mos >= -5, mos >= -15, mos >= -30],
        [100, 88, 72, 55, 38, 20], default=10), index=frame.index)
    return score.where(mos.notna(), 50)

def calculate_valuation_trigger_score(frame):
    cmp = numeric_column(frame, "CMP")
    lower = numeric_column(frame, "Lower Intrinsic Value")
    base = numeric_column(frame, "Intrinsic Value (Base)")
    upper = numeric_column(frame, "Upper Intrinsic Value")
    mid = (base + upper) / 2
    score = pd.Series(np.select(
        [cmp.le(lower * 0.95), cmp.le(lower), cmp.le(base), cmp.le(mid), cmp.le(upper)],
        [100, 90, 70, 55, 35], default=10), index=frame.index)
    valid = cmp.notna() & lower.notna() & base.notna() & upper.notna()
    return score.where(valid, 50)

# ============= OTHER SCORING =============
def calculate_rank_composite(frame):
    comps = []
    fs = numeric_column(frame, "Fundamental Score")
    if fs.notna().any():
        comps.append((fs.where(fs > 15, fs * 10).clip(0, 100), 0.25))
    for col, w in [("Price Momentum Rank", 0.25), ("Value Momentum Rank", 0.20),
                   ("Earnings Quality Rank", 0.20), ("Price to Intrinsic Value Rank", 0.10)]:
        v = numeric_column(frame, col)
        if v.notna().any():
            comps.append((v.clip(0, 100), w))
    if not comps: return pd.Series(50.0, index=frame.index)
    tw = sum(w for _, w in comps)
    return (sum(v * w for v, w in comps) / tw).fillna(50).clip(0, 100)

def calculate_analyst_consensus(frame):
    buy = numeric_column(frame, "Percentage Buy Reco's").clip(0, 100)
    sell = numeric_column(frame, "Percentage Sell Reco's").clip(0, 100)
    n = numeric_column(frame, "Total no. of analysts").fillna(0)
    base = (buy - sell * 0.5).clip(0, 100)
    cov = n.clip(0, 20) / 20 * 20
    return (base * 0.8 + cov).clip(0, 100).fillna(50)

def calculate_cashflow_quality(frame):
    fcf = numeric_column(frame, "Free Cash Flow")
    ocf = numeric_column(frame, "Operating Cash Flow")
    capex = numeric_column(frame, "Capital Expenditure").abs()
    fin = numeric_column(frame, "Financing Cash Flow")
    g = numeric_column(frame, "5Y Hist Op. Cash Flow Growth")
    s = pd.Series(50.0, index=frame.index)
    s = s.where(fcf.isna(), s + fcf.gt(0).astype(int) * 15 - 7.5)
    s = s.where(ocf.isna(), s + ocf.gt(0).astype(int) * 15 - 7.5)
    r = capex / ocf.where(ocf > 0)
    s = s.where(r.isna(), s + (50 - r.clip(0, 2) * 25) * 0.3)
    s = s.where(fin.isna(), s + fin.lt(0).astype(int) * 10 - 5)
    s = s.where(g.isna(), s + g.clip(-30, 50) * 0.3)
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
        parts.append((st_col.str.contains("BUY").astype(int) * 80 + 20)
                     .where(st_col.ne("NAN") & st_col.ne("NONE")))
    k = numeric_column(frame, "Stochastic %K"); d = numeric_column(frame, "Stochastic %D")
    if k.notna().any() and d.notna().any():
        s = np.select([(k > d) & k.between(20, 80), k.gt(80), k.lt(20)],
                      [100, 40, 60], default=50)
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
        parts.append((m.str.contains("BUY").astype(int) * 80 + 20)
                     .where(m.ne("NAN") & m.ne("NONE")))
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

# ============= TECHNICAL INDICATORS =============
def _flatten_columns(data):
    if isinstance(data.columns, pd.MultiIndex):
        l0 = data.columns.get_level_values(0)
        if "Close" in l0:
            data = data.copy(); data.columns = l0
    return data

def _compute_indicators(data):
    data = _flatten_columns(data)
    req = {"Close", "High", "Low", "Volume"}
    if not req.issubset(data.columns): return {"Technical_Score": 50.0}
    close = pd.to_numeric(data["Close"], errors="coerce")
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce")
    data = data.assign(Close=close, High=high, Low=low, Volume=volume).dropna(
        subset=["Close", "High", "Low"])
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
    ms = ml.ewm(span=9, adjust=False).mean()
    mh = ml - ms
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    um, dm = high.diff(), -low.diff()
    pdm = um.where((um > dm) & (um > 0), 0.0)
    mdm = dm.where((dm > um) & (dm > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    mid = close.rolling(20).mean()
    dev = close.rolling(20).std(ddof=0) * 2
    up, lo = mid + dev, mid - dev
    bu = ((high + low) / 2 + 3 * atr).to_numpy()
    bl = ((high + low) / 2 - 3 * atr).to_numpy()
    fu, fl = bu.copy(), bl.copy()
    c = close.to_numpy()
    d = np.ones(len(c))
    an = atr.to_numpy()
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
    prev = data.iloc[-2]
    cmp = float(close.iloc[-1])
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
    av = float(volume.tail(20).mean())
    cv = float(volume.iloc[-1])
    vs = 100 if cv > av * 1.5 else 70 if cv > av else 30
    atr_v = float(atr.iloc[-1])
    ap = atr_v / cmp * 100 if np.isfinite(atr_v) else np.nan
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
        "R1": pivot + 0.382 * hr, "R2": pivot + 0.618 * hr, "R3": pivot + hr,
        "S1": pivot - 0.382 * hr, "S2": pivot - 0.618 * hr, "S3": pivot - hr,
        "Volume": cv, "Avg Volume": av, "52W High": float(high.max()),
        "52W Low": float(low.min()), "Technical_Score": min(tech, 100),
        "Trend Score": ts, "Momentum Score": ms_val, "Volume Score": vs,
        "Volatility Score": vol_s, "Support Resistance Score": sr_s}

# ============= RISK METRICS =============
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_benchmark_history():
    try:
        d = yf.download(BENCHMARK_TICKER, period=PRICE_PERIOD, progress=False,
                        auto_adjust=False, threads=False)
        if d.empty: return pd.DataFrame()
        return _flatten_columns(d).copy().dropna(subset=["Close"])
    except Exception as exc:
        LOGGER.warning("Benchmark fetch failed: %s", exc)
        return pd.DataFrame()

def compute_risk_metrics(history, benchmark):
    empty = {k: np.nan for k in ["Volatility %", "Sharpe", "Sortino", "Max Drawdown %",
        "Beta", "Alpha %", "Return 1M %", "Return 3M %", "Return 6M %", "Return 1Y %",
        "Nifty 1Y %", "Relative Strength %"]}
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
    rm = close.cummax()
    dd = float(((close - rm) / rm).min() * 100)
    beta, alpha = np.nan, np.nan
    if not benchmark.empty and "Close" in benchmark.columns:
        bc = pd.to_numeric(benchmark["Close"], errors="coerce").dropna()
        al = pd.concat([close.pct_change().rename("s"), bc.pct_change().rename("b")],
                       axis=1).dropna()
        if len(al) > 30 and al["b"].var() > 0:
            beta = float(al["s"].cov(al["b"]) / al["b"].var())
            sr = (1 + al["s"]).prod() ** (TRADING_DAYS / len(al)) - 1
            br = (1 + al["b"]).prod() ** (TRADING_DAYS / len(al)) - 1
            alpha = float((sr - RISK_FREE_RATE - beta * (br - RISK_FREE_RATE)) * 100)
    def _p(d):
        return float((close.iloc[-1] / close.iloc[-d-1] - 1) * 100) if len(close) > d else np.nan
    n1y = np.nan
    if not benchmark.empty and "Close" in benchmark.columns:
        bc = pd.to_numeric(benchmark["Close"], errors="coerce").dropna()
        if len(bc) > TRADING_DAYS:
            n1y = float((bc.iloc[-1] / bc.iloc[-TRADING_DAYS-1] - 1) * 100)
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

# ============= PATTERN & RED FLAGS =============
def detect_patterns(row):
    ps = []
    try:
        cmp = float(row.get("CMP", np.nan))
        s20 = float(row.get("20 DMA", np.nan)); s50 = float(row.get("50 DMA", np.nan))
        s200 = float(row.get("200 DMA", np.nan))
        h52 = float(row.get("52W High", np.nan)); l52 = float(row.get("52W Low", np.nan))
        v = float(row.get("Volume", np.nan)); av = float(row.get("Avg Volume", np.nan))
        r1 = float(row.get("R1", np.nan)); s1 = float(row.get("S1", np.nan))
        if all(pd.notna(x) for x in [s50, s200]) and s50 > s200: ps.append("Golden Cross")
        elif all(pd.notna(x) for x in [s50, s200]) and s50 < s200: ps.append("Death Cross")
        if pd.notna(cmp) and pd.notna(h52) and cmp >= h52 * 0.99: ps.append("52W Breakout")
        if pd.notna(cmp) and pd.notna(l52) and cmp <= l52 * 1.01: ps.append("52W Breakdown")
        if pd.notna(v) and pd.notna(av) and av > 0 and v > av * 2: ps.append("Volume Spike")
        if all(pd.notna(x) for x in [cmp, s1]) and s1 > 0 and abs(cmp - s1)/cmp < 0.02:
            ps.append("Near Support")
        if all(pd.notna(x) for x in [cmp, r1]) and r1 > 0 and abs(r1 - cmp)/cmp < 0.02:
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

# ============= TICKER ANALYSIS =============
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
        LOGGER.warning("Failed %s: %s", sym, exc)
        return {"Technical_Score": 50.0, "Fetch Status": f"Failed: {type(exc).__name__}"}

def fetch_technicals(tickers):
    results = {}
    if not tickers: return pd.DataFrame()
    fetch_benchmark_history()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as pool:
        futures = {pool.submit(analyze_ticker, t): t for t in tickers}
        prog = st.progress(0, text="Technical + risk indicators load ho rahe hain...")
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            t = futures[fut]
            try: results[t] = fut.result()
            except Exception as exc:
                LOGGER.exception("Error %s", t)
                results[t] = {"Technical_Score": 50.0, "Fetch Status": f"Failed: {type(exc).__name__}"}
            prog.progress(i / len(futures), text=f"Analyzed: {i}/{len(futures)}")
        prog.empty()
    return pd.DataFrame.from_dict(results, orient="index").rename_axis("Ticker").reset_index()

# ============= COMPOSITE SCORING =============
def classify_index_membership(ticker, registry):
    t = _normalise_ticker(ticker)
    ms = [n for n, m in registry.items() if t in m]
    return ms or ["Other"]

def _kelly_lite(ai, rr, cf):
    b = (ai.clip(0, 100) / 100) * 0.6
    rb = (rr.clip(0, 5) / 5) * 0.25
    cb = (cf.clip(0, 100) / 100) * 0.15
    return ((b + rb + cb) * 100).clip(0, 100).round(1)

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
    cmp_now = numeric_column(out, "CMP")
    lower_now = numeric_column(out, "Lower Intrinsic Value")
    upper_now = numeric_column(out, "Upper Intrinsic Value")
    base_now = numeric_column(out, "Intrinsic Value (Base)")
    out["Entry Trigger Price"] = lower_now.round(2)
    out["Exit Trigger Price"] = upper_now.round(2)
    out["To Entry %"] = (((lower_now - cmp_now) / cmp_now.where(cmp_now.gt(0))) * 100).round(2)
    out["To Exit %"] = (((upper_now - cmp_now) / cmp_now.where(cmp_now.gt(0))) * 100).round(2)
    out["Valuation Action"] = np.select(
        [cmp_now.le(lower_now * 0.95), cmp_now.le(lower_now),
         cmp_now.le(base_now), cmp_now.le((base_now + upper_now) / 2),
         cmp_now.le(upper_now)],
        ["🟢🟢🟢 DEEP BUY", "🟢🟢 ENTRY TRIGGER", "🟢 ACCUMULATE",
         "⚪ FAIR / HOLD", "🟠 STRETCHED"],
        default="🔴 EXIT TRIGGER")
    tech = numeric_column(out, "Technical_Score", 50).fillna(50)
    sharpe_n = numeric_column(out, "Sharpe", 0).fillna(0).clip(-2, 3).add(2).mul(20).clip(0, 100)
    out["AI_Score"] = (
        tech * 0.16 + out["Fundamental_Score_AI"] * 0.16
        + out["Sector_Relative_Score"] * 0.09 + out["Rank_Composite"] * 0.12
        + out["CSV_Technical_Score"] * 0.10 + out["Analyst_Consensus_Score"] * 0.06
        + out["CashFlow_Quality_Score"] * 0.05 + sharpe_n * 0.04
        + out["DCF_Valuation_Score"] * 0.10
        + out["Valuation_Trigger_Score"] * 0.12
    ).round().astype(int)
    out["AI Signal"] = np.select(
        [out["AI_Score"].ge(90), out["AI_Score"].ge(80), out["AI_Score"].ge(70),
         out["AI_Score"].ge(60), out["AI_Score"].ge(40)],
        ["Strong Buy", "Buy", "Accumulate", "Hold", "Reduce"], default="Sell")
    stacked = pd.concat([tech, out["Fundamental_Score_AI"], out["Sector_Relative_Score"],
                         out["Rank_Composite"], out["DCF_Valuation_Score"],
                         out["Valuation_Trigger_Score"]], axis=1)
    out["AI Confidence"] = (100 - stacked.std(axis=1).fillna(0) * 1.5).clip(0, 100).round()
    out["Safety Score"] = (
        out["Fundamental_Score_AI"] * 0.24 + tech * 0.18
        + out["Sector_Relative_Score"] * 0.09 + out["Rank_Composite"] * 0.11
        + numeric_column(out, "Volatility Score", 50).fillna(50) * 0.09
        + out["CashFlow_Quality_Score"] * 0.06 + sharpe_n * 0.04
        + out["DCF_Valuation_Score"] * 0.09
        + out["Valuation_Trigger_Score"] * 0.10)
    out["Risk Score"] = 100 - out["Safety Score"]
    out["Risk Level"] = np.select(
        [out["Risk Score"].lt(25), out["Risk Score"].lt(50), out["Risk Score"].lt(75)],
        ["Low Risk", "Moderate Risk", "High Risk"], default="Very High Risk")
    ms = out["Ticker"].map(lambda t: classify_index_membership(t, index_registry))
    out["Index Memberships"] = ms.map(lambda l: ", ".join(l))
    out["Primary Index"] = ms.map(lambda l: l[0])
    cmp = numeric_column(out, "CMP")
    stv = numeric_column(out, "SuperTrend_Val")
    atr = numeric_column(out, "ATR")
    valid = stv.gt(0) & stv.lt(cmp)
    out["Entry Price"] = cmp
    out["Stop Loss"] = stv.where(valid, (cmp - atr).where(atr.gt(0), cmp * 0.95))
    min_t = cmp + (cmp - out["Stop Loss"]) * 2
    res = pd.concat([numeric_column(out, r) for r in ("R1", "R2", "R3")], axis=1)
    above = res.where(res.ge(min_t, axis=0))
    out["Target Price"] = above.min(axis=1).fillna(min_t)
    out["Upside %"] = ((out["Target Price"] - cmp) / cmp.where(cmp.gt(0)) * 100).round(2)
    rps = cmp - out["Stop Loss"]
    out["Risk : Reward"] = ((out["Target Price"] - cmp) / rps.where(rps.gt(0))).round(2)
    out["Trailing Stop"] = (cmp - 3 * atr).where(atr.gt(0), out["Stop Loss"])
    iv = out["Intrinsic Value (Base)"]
    out["Margin of Safety %"] = ((iv - cmp) / iv.where(iv.gt(0)) * 100).round(2)
    mos = out["Margin of Safety %"]
    out["Valuation Signal"] = np.select(
        [mos.ge(40), mos.ge(25), mos.ge(10), mos.ge(-5), mos.ge(-15), mos.ge(-30)],
        ["🟢 Deep Value", "🟢 Undervalued", "🟡 Slightly Under", "⚪ Fair Value",
         "🟠 Slightly Over", "🔴 Overvalued"],
        default="🔴 Highly Overvalued")
    out = build_flag_summary(out)
    out["Position Size %"] = _kelly_lite(out["AI_Score"], out["Risk : Reward"].fillna(0),
                                         out["AI Confidence"])
    out = out.sort_values("AI_Score", ascending=False, kind="stable",
                          na_position="last").reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out

# ============= ANALYTICS =============
def analyse_index(frame, index_name, members):
    norm = frame["Ticker"].map(_normalise_ticker)
    sub = frame.loc[norm.isin(members)]
    if sub.empty:
        return {"Index": index_name, "Stocks": 0, "Buy+": 0, "Hold": 0, "Sell+": 0,
                "Avg AI Score": 0, "Avg Fundamental": 0, "Avg Technical": 0,
                "Avg Sector-Rel": 0, "Avg Upside %": 0, "Avg PE": 0}
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
    for c in ("AI_Score", "Fundamental_Score_AI", "Sector_Relative_Score",
              "Rank_Composite", "Upside %"):
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
             .sort_values("Avg AI Score", ascending=False, kind="stable")
             .reset_index(drop=True))
    for c in ("Sector", "Avg AI Score", "Avg Sector-Rel", "Stocks"):
        if c not in stats.columns:
            stats[c] = np.nan if c != "Stocks" else 0
    return stats

# ============= SNAPSHOTS =============
def _snap_dir(p): return p / DATA_DIR_NAME / SNAPSHOT_DIR_NAME

def save_sector_snapshot(project_dir, stats):
    if stats.empty: return
    try:
        d = _snap_dir(project_dir); d.mkdir(parents=True, exist_ok=True)
        stats.to_csv(d / f"sector_{datetime.now():%Y-%m-%d}.csv", index=False)
    except OSError as exc: LOGGER.warning("Snapshot fail: %s", exc)

def load_sector_snapshots(project_dir, days=60):
    d = _snap_dir(project_dir)
    if not d.exists(): return pd.DataFrame()
    cut = datetime.now() - timedelta(days=days)
    frames = []
    for p in sorted(d.glob("sector_*.csv")):
        try: sd = datetime.strptime(p.stem.replace("sector_", ""), "%Y-%m-%d")
        except ValueError: continue
        if sd < cut: continue
        try:
            df = pd.read_csv(p); df["Date"] = sd; frames.append(df)
        except (OSError, pd.errors.ParserError): continue
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
    base = pd.DataFrame({"Sector": cur["Sector"].values,
                         "Prev AI Score": np.nan, "Prev Sector-Rel": np.nan})
    if not hist.empty:
        needed = {"Sector", "Avg AI Score", "Avg Sector-Rel", "Date"}
        if needed.issubset(hist.columns):
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.normalize()
            cand = hist[hist["Date"] <= (today - timedelta(days=7))]
            if cand.empty: cand = hist[hist["Date"] < today]
            if not cand.empty:
                latest = cand["Date"].max()
                base = (cand.loc[cand["Date"] == latest,
                                 ["Sector", "Avg AI Score", "Avg Sector-Rel"]]
                        .rename(columns={"Avg AI Score": "Prev AI Score",
                                         "Avg Sector-Rel": "Prev Sector-Rel"})
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

# ============= CHARTS =============
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_chart_history(ticker, period=PRICE_PERIOD):
    sym = ticker if ticker.endswith((".NS", ".BO")) else f"{ticker}.NS"
    try:
        h = yf.download(sym, period=period, progress=False, auto_adjust=False, threads=False)
    except Exception as exc:
        LOGGER.warning("Chart fail %s: %s", sym, exc); return pd.DataFrame()
    if h.empty: return pd.DataFrame()
    h = _flatten_columns(h).copy().dropna(subset=["Close"])
    for w in (20, 50, 200):
        h[f"SMA{w}"] = h["Close"].rolling(w).mean()
    return h

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_chart_for_period(ticker, period, interval):
    """Fetch OHLC for specific period+interval with SMA overlays."""
    sym = ticker if ticker.endswith((".NS", ".BO")) else f"{ticker}.NS"
    try:
        h = yf.download(sym, period=period, interval=interval,
                        progress=False, auto_adjust=False, threads=False)
    except Exception as exc:
        LOGGER.warning("Chart fail %s: %s", sym, exc)
        return pd.DataFrame()
    if h.empty:
        return pd.DataFrame()
    h = _flatten_columns(h).copy().dropna(subset=["Close"])
    for w in (20, 50, 200):
        if len(h) >= w:
            h[f"SMA{w}"] = h["Close"].rolling(w).mean()
    return h

_PLOT_LAYOUT = {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "#070d18",
                "font": {"color": "#8ba3c0"},
                "margin": {"t": 24, "b": 24, "l": 24, "r": 24}}

def _style_figure(fig, height=340):
    fig.update_layout(**_PLOT_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#1c2e45", zerolinecolor="#1c2e45")
    fig.update_yaxes(gridcolor="#1c2e45", zerolinecolor="#1c2e45")
    return fig

# ============= ⭐ CHANGE 3: CHART WITH TIME RANGES =============
def render_price_chart(ticker):
    if go is None:
        st.info("`pip install plotly` karein.")
        return

    # Time range buttons
    period_opts = [
        ("1D", "1d", "5m"), ("1W", "5d", "30m"), ("1M", "1mo", "1d"),
        ("3M", "3mo", "1d"), ("6M", "6mo", "1d"), ("1Y", "1y", "1d"),
        ("5Y", "5y", "1d"),
    ]
    state_key = f"chart_period_{ticker}"
    if state_key not in st.session_state:
        st.session_state[state_key] = "1Y"

    cols = st.columns(len(period_opts))
    for i, (label, _, _) in enumerate(period_opts):
        with cols[i]:
            active = st.session_state[state_key] == label
            if st.button(label, key=f"period_{ticker}_{label}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state[state_key] = label
                st.rerun()

    selected = st.session_state[state_key]
    period, interval = next((p, i) for lbl, p, i in period_opts if lbl == selected)

    h = _fetch_chart_for_period(ticker, period, interval)
    if h.empty:
        st.warning(f"{ticker} ka chart data nahi mila for {selected}.")
        return

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=h.index, open=h["Open"], high=h["High"],
        low=h["Low"], close=h["Close"], name=ticker,
        increasing_line_color="#34d399",
        decreasing_line_color="#fb7185"))
    for w, c in ((20, "#fbbf24"), (50, "#60a5fa"), (200, "#a78bfa")):
        col = f"SMA{w}"
        if col in h and h[col].notna().any():
            fig.add_trace(go.Scatter(x=h.index, y=h[col], mode="lines", name=col,
                                     line=dict(color=c, width=1.4)))
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=f"{ticker} — {selected} Price Action")
    _style_figure(fig, 460)
    st.plotly_chart(fig, use_container_width=True)


# ============= ⭐ CHANGE 3: TECHNICAL GAUGES =============
def _gauge_chart(value, title):
    """TradingView-style semicircle gauge."""
    try:
        value = float(value) if pd.notna(value) else 50.0
    except Exception:
        value = 50.0

    if value >= 75:
        color, label = "#10b981", "Strong Buy"
    elif value >= 60:
        color, label = "#34d399", "Buy"
    elif value >= 45:
        color, label = "#fbbf24", "Neutral"
    elif value >= 30:
        color, label = "#fb923c", "Sell"
    else:
        color, label = "#fb7185", "Strong Sell"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"color": "#e5eefb", "size": 26}},
        title={"text": f"{title}<br><span style='font-size:0.75em;color:{color};font-weight:700'>{label}</span>",
               "font": {"color": "#8ba3c0", "size": 13}},
        gauge={
            "shape": "angular",
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#8ba3c0", "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(251, 113, 133, 0.18)"},
                {"range": [30, 45], "color": "rgba(251, 146, 60, 0.18)"},
                {"range": [45, 60], "color": "rgba(251, 191, 36, 0.18)"},
                {"range": [60, 75], "color": "rgba(52, 211, 153, 0.18)"},
                {"range": [75, 100], "color": "rgba(16, 185, 129, 0.18)"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 2},
                          "thickness": 0.8, "value": value}
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#8ba3c0"},
        height=250,
        margin={"t": 55, "b": 10, "l": 20, "r": 20}
    )
    return fig


def render_technical_analysis(row):
    """Technical analysis tab — gauges + detailed metrics."""
    st.markdown("### 📊 Technical Analysis")

    def _safe(v):
        try:
            if pd.isna(v): return 50.0
            return float(v)
        except Exception:
            return 50.0

    tech_score = _safe(row.get("Technical_Score", 50))
    analyst_score = _safe(row.get("Analyst_Consensus_Score", 50))
    fund_score = _safe(row.get("Fundamental_Score_AI", 50))
    dcf_score = _safe(row.get("DCF_Valuation_Score", 50))
    sector_score = _safe(row.get("Sector_Relative_Score", 50))
    trigger_score = _safe(row.get("Valuation_Trigger_Score", 50))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_gauge_chart(tech_score, "Technical Rating"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(_gauge_chart(analyst_score, "Analyst Rating"),
                        use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(_gauge_chart(fund_score, "Fundamental Score"),
                        use_container_width=True)
    with c4:
        st.plotly_chart(_gauge_chart(dcf_score, "Valuation Score"),
                        use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(_gauge_chart(sector_score, "Sector Relative"),
                        use_container_width=True)
    with c6:
        st.plotly_chart(_gauge_chart(trigger_score, "Entry/Exit Trigger"),
                        use_container_width=True)

    # Analyst rating summary
    st.markdown("### 👥 Analyst Rating")
    ac1, ac2, ac3 = st.columns(3)
    buy_pct = row.get("Percentage Buy Reco's", 0)
    sell_pct = row.get("Percentage Sell Reco's", 0)
    n_analysts = row.get("Total no. of analysts", 0)
    ac1.metric("Buy %", f"{(buy_pct if pd.notna(buy_pct) else 0):.0f}%")
    ac2.metric("Sell %", f"{(sell_pct if pd.notna(sell_pct) else 0):.0f}%")
    ac3.metric("Analysts", f"{int(n_analysts if pd.notna(n_analysts) else 0)}")

    st.markdown("### 📈 Detailed Indicators")
    key_indicators = {
        "CMP": row.get("CMP"), "RSI (14D)": row.get("RSI"),
        "ADX": row.get("ADX"), "ATR": row.get("ATR"),
        "MACD Signal": row.get("MACDSignal"), "Histogram": row.get("Histogram"),
        "20 DMA": row.get("20 DMA"), "50 DMA": row.get("50 DMA"),
        "200 DMA": row.get("200 DMA"), "EMA20": row.get("EMA20"),
        "EMA50": row.get("EMA50"), "EMA200": row.get("EMA200"),
        "DI Plus": row.get("DI Plus"), "DI Minus": row.get("DI Minus"),
        "SuperTrend": row.get("SuperTrend"), "SuperTrend Val": row.get("SuperTrend_Val"),
        "Trend Pivot": row.get("Trend Pivot"),
        "R1": row.get("R1"), "R2": row.get("R2"), "R3": row.get("R3"),
        "S1": row.get("S1"), "S2": row.get("S2"), "S3": row.get("S3"),
        "52W High": row.get("52W High"), "52W Low": row.get("52W Low"),
        "Volume": row.get("Volume"), "Avg Volume": row.get("Avg Volume"),
        "Trend Score": row.get("Trend Score"), "Momentum Score": row.get("Momentum Score"),
        "Volume Score": row.get("Volume Score"),
        "Volatility Score": row.get("Volatility Score"),
        "Support Resistance Score": row.get("Support Resistance Score"),
    }
    display = {}
    for k, v in key_indicators.items():
        if k not in row.index: continue
        if pd.isna(v): display[k] = None
        elif isinstance(v, (int, float, np.number)): display[k] = float(v)
        else: display[k] = str(v)
    st.json(display)

    # Pattern pills
    if "Patterns" in row.index and pd.notna(row.get("Patterns")):
        st.markdown("### 📐 Detected Patterns")
        patterns = str(row.get("Patterns", "")).split(", ")
        pill_html = ""
        for p in patterns:
            if p and p != "—":
                pill_html += f'<span style="display:inline-block;padding:4px 12px;margin:4px;background:rgba(96,165,250,0.15);border:1px solid rgba(96,165,250,0.4);border-radius:12px;font-size:0.75rem;color:#60a5fa;font-weight:600;">{p}</span>'
        if pill_html:
            st.markdown(f"<div>{pill_html}</div>", unsafe_allow_html=True)

# ============= UI HELPERS =============
def style_signal(v):
    c = "#34d399" if "Buy" in str(v) else "#fbbf24" if v in {"Accumulate", "Hold"} else "#fb7185"
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

def fmt_money(v):
    return f"₹{v:,.2f}" if pd.notna(v) and v > 0 else "—"

# ============= CSS WITH SCROLL FIX (CHANGE 1) =============
def inject_mobile_css():
    """TradingView-style native mobile UI + fixed scrollbar."""
    st.markdown("""
    <style>
    html { scroll-behavior: smooth; -webkit-text-size-adjust: 100%; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: 100% !important;
    }
    body, .main, section.main { overflow-x: hidden !important; }
    footer { display: none !important; }

    /* KPI GRID */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
        margin: 12px 0 20px 0;
    }
    .kpi-card {
        background: linear-gradient(145deg, #0f1729 0%, #0a1120 100%);
        border: 1px solid rgba(139, 163, 192, 0.12);
        border-radius: 14px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
    }
    .kpi-card.kpi-green::before { background: linear-gradient(90deg, #34d399, #10b981); }
    .kpi-card.kpi-red::before { background: linear-gradient(90deg, #fb7185, #ef4444); }
    .kpi-card.kpi-amber::before { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 8px 24px rgba(96, 165, 250, 0.1);
    }
    .kpi-card .kpi-label {
        color: #8ba3c0;
        font-size: 0.72rem;
        font-weight: 500;
        margin-bottom: 6px;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-card .kpi-value {
        color: #e5eefb;
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .kpi-card .kpi-delta {
        font-size: 0.7rem;
        margin-top: 4px;
        font-weight: 600;
    }
    .kpi-card .kpi-delta.up { color: #34d399; }
    .kpi-card .kpi-delta.down { color: #fb7185; }
    .kpi-card .kpi-delta.neutral { color: #fbbf24; }

    /* STOCK CARDS */
    .stock-card {
        background: linear-gradient(145deg, #0d1424 0%, #0a0f1c 100%);
        border: 1px solid rgba(139, 163, 192, 0.12);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    .stock-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.4); }
    .stock-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
    }
    .stock-card.signal-strong-buy::before { background: #10b981; }
    .stock-card.signal-buy::before { background: #34d399; }
    .stock-card.signal-accumulate::before { background: #84cc16; }
    .stock-card.signal-hold::before { background: #fbbf24; }
    .stock-card.signal-reduce::before { background: #fb923c; }
    .stock-card.signal-sell::before { background: #fb7185; }

    .stock-card .stock-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 8px;
    }
    .stock-card .stock-ticker {
        font-weight: 800;
        font-size: 1rem;
        color: #e5eefb;
        letter-spacing: 0.02em;
    }
    .stock-card .stock-name {
        font-size: 0.7rem;
        color: #8ba3c0;
        margin-top: 2px;
        opacity: 0.85;
        line-height: 1.2;
    }
    .stock-card .stock-score-badge {
        background: rgba(96, 165, 250, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.4);
        color: #60a5fa;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
    }
    .stock-card .stock-price-row {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin: 6px 0 8px 0;
    }
    .stock-card .stock-price {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .stock-card .stock-change {
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .stock-card .stock-change.up {
        color: #34d399;
        background: rgba(52, 211, 153, 0.12);
    }
    .stock-card .stock-change.down {
        color: #fb7185;
        background: rgba(251, 113, 133, 0.12);
    }
    .stock-card .stock-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(139, 163, 192, 0.08);
    }
    .stock-card .stock-metric-label {
        font-size: 0.62rem;
        color: #8ba3c0;
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .stock-card .stock-metric-value {
        font-size: 0.82rem;
        font-weight: 700;
        color: #e5eefb;
    }
    .stock-card .stock-action {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.68rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .stock-card .stock-action.entry {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .stock-card .stock-action.exit {
        background: rgba(251, 113, 133, 0.15);
        color: #fb7185;
        border: 1px solid rgba(251, 113, 133, 0.3);
    }
    .stock-card .stock-action.hold {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* ⭐ CHANGE 1: FIXED HORIZONTAL SCROLL */
    .stock-scroll {
        display: flex !important;
        gap: 12px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        padding: 4px 0 14px 0 !important;
        scroll-snap-type: x proximity !important;
        -webkit-overflow-scrolling: touch !important;
        touch-action: pan-x !important;
        scrollbar-width: thin !important;
        scrollbar-color: rgba(96, 165, 250, 0.5) rgba(255, 255, 255, 0.05) !important;
    }
    .stock-scroll::-webkit-scrollbar {
        height: 8px !important;
        display: block !important;
    }
    .stock-scroll::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.04) !important;
        border-radius: 4px !important;
    }
    .stock-scroll::-webkit-scrollbar-thumb {
        background: rgba(96, 165, 250, 0.5) !important;
        border-radius: 4px !important;
    }
    .stock-scroll::-webkit-scrollbar-thumb:hover {
        background: rgba(96, 165, 250, 0.8) !important;
    }
    .stock-scroll .stock-card {
        min-width: 240px !important;
        max-width: 240px !important;
        flex: 0 0 240px !important;
        scroll-snap-align: start !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="stMarkdownContainer"] {
        overflow: visible !important;
    }

    /* Section header */
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 16px 0 8px 0;
    }
    .section-header h3 {
        font-size: 1rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: #e5eefb;
    }
    .section-header .see-all {
        font-size: 0.75rem;
        color: #60a5fa;
        font-weight: 600;
    }

    /* MOBILE */
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-top: 0.5rem !important;
            padding-bottom: 5rem !important;
        }
        h1 { font-size: 1.2rem !important; line-height: 1.25 !important;
             margin-bottom: 0.3rem !important; }
        h2 { font-size: 1.02rem !important; }
        h3 { font-size: 0.92rem !important; }

        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin: 10px 0 16px 0;
        }
        .kpi-card { padding: 10px 12px; border-radius: 12px; }
        .kpi-card .kpi-label { font-size: 0.6rem; margin-bottom: 4px; }
        .kpi-card .kpi-value { font-size: 1.35rem; }

        .stock-scroll .stock-card {
            min-width: 220px !important;
            max-width: 220px !important;
            flex: 0 0 220px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            white-space: nowrap !important;
            scrollbar-width: none !important;
            padding-bottom: 4px !important;
            gap: 2px !important;
            border-bottom: 1px solid rgba(139, 163, 192, 0.1) !important;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.78rem !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
            border-radius: 8px !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(96, 165, 250, 0.12) !important;
        }
        .stTabs [data-baseweb="tab"] p { font-size: 0.78rem !important; }

        div[data-testid="stDataFrame"] { overflow-x: auto !important; }
        div[data-testid="stDataFrame"] * { font-size: 0.72rem !important; }

        .stButton > button, .stDownloadButton > button {
            padding: 8px 12px !important;
            font-size: 0.78rem !important;
            min-height: 38px !important;
            width: 100% !important;
        }

        section[data-testid="stSidebar"] {
            min-width: 82% !important;
            max-width: 88% !important;
        }

        details summary { font-size: 0.82rem !important; padding: 6px 10px !important; }
        .js-plotly-plot, .plot-container { max-height: 340px !important; }
        div[data-testid="stAlert"] { font-size: 0.75rem !important; padding: 8px 10px !important; }
        div[data-baseweb="select"] { font-size: 0.78rem !important; }
        div.element-container { margin-bottom: 0.3rem !important; }
        hr { margin: 0.5rem 0 !important; }
        .stTabs { margin-top: 0.5rem !important; }

        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 6px !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 0 0 calc(50% - 3px) !important;
            min-width: calc(50% - 3px) !important;
            max-width: calc(50% - 3px) !important;
        }
    }

    @media (max-width: 380px) {
        h1 { font-size: 1.1rem !important; }
        .kpi-card .kpi-value { font-size: 1.15rem; }
        .kpi-card .kpi-label { font-size: 0.55rem; }
        .stock-scroll .stock-card {
            min-width: 190px !important;
            max-width: 190px !important;
            flex: 0 0 190px !important;
        }
    }

    @media (min-width: 641px) and (max-width: 1024px) {
        .kpi-grid { grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .kpi-card .kpi-value { font-size: 1.5rem; }
        h1 { font-size: 1.6rem !important; }
    }

    @media (min-width: 1025px) {
        .block-container { padding-left: 3rem !important; padding-right: 3rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

# ============= DISPLAY COLUMNS =============
DISPLAY_COLUMNS = [
    "Rank", "Ticker", "Name", "Sector",
    "CMP", "AI_Score", "AI Signal", "AI Confidence",
    "Valuation Action", "Valuation_Trigger_Score",
    "Entry Trigger Price", "Exit Trigger Price",
    "Margin of Safety %", "Intrinsic Value (Base)",
    "Quality Grade", "Risk Level", "Sharpe",
    "Target Price", "Upside %", "Position Size %",
]

# ============= SIGNAL HELPERS =============
def _signal_class(signal):
    s = str(signal).lower().replace(" ", "-")
    return f"signal-{s}" if s in {"strong-buy", "buy", "accumulate", "hold", "reduce", "sell"} else "signal-hold"

def _action_class(action):
    a = str(action)
    if "ENTRY" in a or "DEEP BUY" in a:
        return "entry"
    if "EXIT" in a:
        return "exit"
    return "hold"

def render_stock_card_html(row):
    signal_cls = _signal_class(row.get("AI Signal", "Hold"))
    action_cls = _action_class(row.get("Valuation Action", ""))
    cmp_val = row.get("CMP", 0)
    cmp_str = f"₹{cmp_val:,.0f}" if pd.notna(cmp_val) else "—"
    upside = row.get("Upside %", 0)
    if pd.notna(upside):
        upside_cls = "up" if upside > 0 else "down"
        upside_str = f"{upside:+.1f}%"
    else:
        upside_cls = "up"
        upside_str = "—"
    ai_score = int(row.get("AI_Score", 0)) if pd.notna(row.get("AI_Score", 0)) else 0
    mos = row.get("Margin of Safety %", 0)
    mos_str = f"{mos:+.0f}%" if pd.notna(mos) else "—"
    entry_p = row.get("Entry Trigger Price", 0)
    entry_str = f"₹{entry_p:,.0f}" if pd.notna(entry_p) and entry_p > 0 else "—"
    ticker = row.get("Ticker", "?")
    name = str(row.get("Name", ""))[:28]
    action = row.get("Valuation Action", "⚪ HOLD")
    signal = row.get("AI Signal", "—")
    return f"""
    <div class="stock-card {signal_cls}">
        <div class="stock-header">
            <div>
                <div class="stock-ticker">{ticker}</div>
                <div class="stock-name">{name}</div>
            </div>
            <div class="stock-score-badge">{ai_score}</div>
        </div>
        <div class="stock-price-row">
            <div class="stock-price">{cmp_str}</div>
            <div class="stock-change {upside_cls}">{upside_str}</div>
        </div>
        <div class="stock-metrics">
            <div>
                <div class="stock-metric-label">MoS</div>
                <div class="stock-metric-value">{mos_str}</div>
            </div>
            <div>
                <div class="stock-metric-label">Entry</div>
                <div class="stock-metric-value">{entry_str}</div>
            </div>
            <div>
                <div class="stock-metric-label">Signal</div>
                <div class="stock-metric-value">{signal}</div>
            </div>
        </div>
        <div class="stock-action {action_cls}">{action}</div>
    </div>
    """

def render_stock_card_grid(df, max_items=20):
    if df.empty:
        st.info("Koi stock nahi mila.")
        return
    subset = df.head(max_items)
    cards_html = "".join(render_stock_card_html(row) for _, row in subset.iterrows())
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">
        {cards_html}
    </div>
    """, unsafe_allow_html=True)

# ⭐ CHANGE 1: render_horizontal_scroll with fixed scroll
def render_horizontal_scroll(df, title, max_items=10):
    if df.empty:
        return
    subset = df.head(max_items)
    cards_html = "".join(render_stock_card_html(row) for _, row in subset.iterrows())
    st.markdown(f"""
    <div class="section-header">
        <h3>{title}</h3>
        <span class="see-all">{len(df)} total →</span>
    </div>
    <div class="stock-scroll">{cards_html}</div>
    """, unsafe_allow_html=True)

# ============= ⭐ CHANGE 2: DCF CARD HELPERS =============
def _render_dcf_card(row):
    """DCF-specific card with entry/exit/IV values."""
    signal_cls = _signal_class(row.get("AI Signal", "Hold"))
    action_cls = _action_class(row.get("Valuation Action", ""))

    def _fmt(v):
        return f"₹{v:,.0f}" if pd.notna(v) and v > 0 else "—"

    cmp_str = _fmt(row.get("CMP", 0))
    entry_str = _fmt(row.get("Entry Trigger Price", 0))
    exit_str = _fmt(row.get("Exit Trigger Price", 0))
    iv_base_str = _fmt(row.get("Intrinsic Value (Base)", 0))
    iv_dcf_str = _fmt(row.get("Intrinsic Value (DCF)", 0))
    iv_mult_str = _fmt(row.get("Intrinsic Value (Multiples)", 0))

    ai_score = int(row.get("AI_Score", 0)) if pd.notna(row.get("AI_Score", 0)) else 0
    mos = row.get("Margin of Safety %", 0)
    if pd.notna(mos):
        mos_cls = "up" if mos > 0 else "down"
        mos_str = f"{mos:+.1f}%"
    else:
        mos_cls = "up"
        mos_str = "—"

    ticker = row.get("Ticker", "?")
    name = str(row.get("Name", ""))[:30]
    action = row.get("Valuation Action", "⚪ HOLD")
    signal = row.get("AI Signal", "—")

    return f"""
    <div class="stock-card {signal_cls}">
        <div class="stock-header">
            <div>
                <div class="stock-ticker">{ticker}</div>
                <div class="stock-name">{name}</div>
            </div>
            <div class="stock-score-badge">{ai_score}</div>
        </div>
        <div class="stock-price-row">
            <div class="stock-price">{cmp_str}</div>
            <div class="stock-change {mos_cls}">MoS {mos_str}</div>
        </div>
        <div class="stock-metrics">
            <div>
                <div class="stock-metric-label">Entry</div>
                <div class="stock-metric-value">{entry_str}</div>
            </div>
            <div>
                <div class="stock-metric-label">Exit</div>
                <div class="stock-metric-value">{exit_str}</div>
            </div>
            <div>
                <div class="stock-metric-label">Base IV</div>
                <div class="stock-metric-value">{iv_base_str}</div>
            </div>
        </div>
        <div class="stock-metrics">
            <div>
                <div class="stock-metric-label">DCF IV</div>
                <div class="stock-metric-value">{iv_dcf_str}</div>
            </div>
            <div>
                <div class="stock-metric-label">Mult IV</div>
                <div class="stock-metric-value">{iv_mult_str}</div>
            </div>
            <div>
                <div class="stock-metric-label">Signal</div>
                <div class="stock-metric-value">{signal}</div>
            </div>
        </div>
        <div class="stock-action {action_cls}">{action}</div>
    </div>
    """


def _render_dcf_stock_cards(df, max_items=200):
    """Render DCF cards in a 2-column grid."""
    if df.empty:
        st.info("Koi stock nahi mila.")
        return
    subset = df.head(max_items)
    cards_html = "".join(_render_dcf_card(row) for _, row in subset.iterrows())
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;">
        {cards_html}
    </div>
    """, unsafe_allow_html=True)


# ============= KPI ROW =============
def render_kpi_row(scored):
    inject_mobile_css()
    total = len(scored)
    avg = scored["AI_Score"].mean()
    sb = int((scored["AI Signal"] == "Strong Buy").sum())
    bl = int(scored["AI Signal"].isin(["Strong Buy", "Buy"]).sum())
    entry = int(scored.get("Valuation Action", pd.Series()).astype(str)
                .str.contains("DEEP BUY|ENTRY", na=False).sum())
    exit_ = int(scored.get("Valuation Action", pd.Series()).astype(str)
                .str.contains("EXIT", na=False).sum())
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total Stocks</div>
            <div class="kpi-value">{total}</div>
            <div class="kpi-delta neutral">Portfolio</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg AI Score</div>
            <div class="kpi-value">{avg:.1f}</div>
            <div class="kpi-delta neutral">0-100 scale</div>
        </div>
        <div class="kpi-card kpi-green">
            <div class="kpi-label">Entry Triggers</div>
            <div class="kpi-value">{entry}</div>
            <div class="kpi-delta up">🟢 Buy zone</div>
        </div>
        <div class="kpi-card kpi-red">
            <div class="kpi-label">Exit Triggers</div>
            <div class="kpi-value">{exit_}</div>
            <div class="kpi-delta down">🔴 Sell zone</div>
        </div>
        <div class="kpi-card kpi-amber">
            <div class="kpi-label">Strong Buys</div>
            <div class="kpi-value">{sb}</div>
            <div class="kpi-delta neutral">AI ≥ 90</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Buy + Strong</div>
            <div class="kpi-value">{bl}</div>
            <div class="kpi-delta neutral">AI ≥ 80</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    entry_df = scored[scored.get("Valuation Action", pd.Series()).astype(str)
                      .str.contains("DEEP BUY|ENTRY", na=False)].sort_values(
                      "Margin of Safety %", ascending=False)
    if not entry_df.empty:
        render_horizontal_scroll(entry_df, "🎯 Top Entry Opportunities", 15)
    top_ai = scored.sort_values("AI_Score", ascending=False).head(15)
    if not top_ai.empty:
        render_horizontal_scroll(top_ai, "⭐ Top AI-Rated Stocks", 15)

# ============= SCREENER TAB =============
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
    if "Valuation Action" in scored.columns and ac:
        f = f[f["Valuation Action"].isin(ac)]
    if pr != "All": f = f[f["Primary Index"] == pr]
    f = f.copy()
    sc = detect_sector_column(f)
    if sc and "Sector" not in f.columns: f = f.rename(columns={sc: "Sector"})

    st.subheader(f"📋 Screener ({len(f)} stocks)")

    view_mode = st.radio("View", ["📇 Cards", "📊 Table"],
                        horizontal=True, key="screener_view",
                        label_visibility="collapsed")

    if view_mode == "📇 Cards":
        if f.empty:
            st.info("Koi stock filters ke according nahi mila.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by",
                    ["AI_Score", "Margin of Safety %", "Upside %", "Sharpe"],
                    key="card_sort")
            with col2:
                max_cards = st.slider("Max cards", 5, 50, 20, 5, key="max_cards")
            f_sorted = f.sort_values(sort_by, ascending=False, na_position="last")
            render_stock_card_grid(f_sorted, max_cards)
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
                       file_name=f"ai_screener_{datetime.now():%Y%m%d_%H%M}.csv",
                       mime="text/csv", use_container_width=True)
    try:
        xlsx = _build_excel(f, ss, rot, idf)
        d2.download_button("📊 Excel", data=xlsx,
                           file_name=f"ai_screener_{datetime.now():%Y%m%d_%H%M}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    except Exception as exc:
        d2.warning(f"Excel fail: {exc}")

# ============= ⭐ CHANGE 2: DCF TAB WITH SIGNAL TABS =============
def render_dcf_valuation_tab(scored):
    st.subheader("💰 DCF Valuation")
    st.caption("Hybrid IV: DCF (60%) + Peer-Multiples (40%)")
    if "Intrinsic Value (Base)" not in scored.columns:
        st.warning("Valuation data missing."); return
    val = scored.dropna(subset=["Intrinsic Value (Base)", "CMP"]).copy()
    if val.empty: st.info("Koi valuation data nahi."); return
    dv = int((val["Margin of Safety %"] >= 40).sum())
    uv = int(((val["Margin of Safety %"] >= 10) & (val["Margin of Safety %"] < 40)).sum())
    fv = int((val["Margin of Safety %"].abs() < 10).sum())
    ov = int((val["Margin of Safety %"] <= -10).sum())
    am = val["Margin of Safety %"].mean()
    aw = val["WACC Used"].mean() if "WACC Used" in val else np.nan
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card kpi-green"><div class="kpi-label">🟢 Deep Value</div>
            <div class="kpi-value">{dv}</div><div class="kpi-delta up">MoS ≥ 40%</div></div>
        <div class="kpi-card kpi-green"><div class="kpi-label">🟢 Undervalued</div>
            <div class="kpi-value">{uv}</div><div class="kpi-delta up">MoS 10-40%</div></div>
        <div class="kpi-card kpi-amber"><div class="kpi-label">⚪ Fair</div>
            <div class="kpi-value">{fv}</div><div class="kpi-delta neutral">MoS ±10%</div></div>
        <div class="kpi-card kpi-red"><div class="kpi-label">🔴 Overvalued</div>
            <div class="kpi-value">{ov}</div><div class="kpi-delta down">MoS ≤ -10%</div></div>
        <div class="kpi-card"><div class="kpi-label">Avg MoS</div>
            <div class="kpi-value">{am:+.1f}%</div><div class="kpi-delta neutral">Universe</div></div>
        <div class="kpi-card"><div class="kpi-label">Avg WACC</div>
            <div class="kpi-value">{f"{aw:.1%}" if pd.notna(aw) else "—"}</div>
            <div class="kpi-delta neutral">CAPM-lite</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎯 Trigger Alerts by Valuation Signal")

    if "Valuation Signal" in val.columns:
        sig_order = ["🟢 Deep Value", "🟢 Undervalued", "🟡 Slightly Under",
                     "⚪ Fair Value", "🟠 Slightly Over", "🔴 Overvalued",
                     "🔴 Highly Overvalued"]
        existing_sigs = [s for s in sig_order
                         if s in val["Valuation Signal"].dropna().unique()]
        if existing_sigs:
            tab_labels = [f"{s} ({int((val['Valuation Signal'] == s).sum())})"
                          for s in existing_sigs]
            sig_tabs = st.tabs(tab_labels)
            for tab, sig in zip(sig_tabs, existing_sigs):
                with tab:
                    sub = val[val["Valuation Signal"] == sig].sort_values(
                        "Margin of Safety %", ascending=False).reset_index(drop=True)
                    if sub.empty:
                        st.info("Koi stock nahi mila.")
                    else:
                        _render_dcf_stock_cards(sub, max_items=200)
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
    f = f.copy().sort_values("Margin of Safety %", ascending=False,
                             kind="stable").reset_index(drop=True)
    sc2 = detect_sector_column(f)
    if sc2 and "Sector" not in f.columns: f = f.rename(columns={sc2: "Sector"})

    table_tab, chart_tab = st.tabs([f"📋 Full Table ({len(f)})", "📊 Price vs IV"])
    with table_tab:
        cols = ["Rank", "Ticker", "Name", "Sector", "CMP", "Lower Intrinsic Value",
                "Intrinsic Value (Base)", "Upper Intrinsic Value",
                "Intrinsic Value (DCF)", "Intrinsic Value (Multiples)", "WACC Used",
                "Margin of Safety %", "Valuation Signal", "Valuation Action",
                "Entry Trigger Price", "Exit Trigger Price", "To Entry %", "To Exit %",
                "AI_Score", "AI Signal", "DCF_Valuation_Score", "Valuation_Trigger_Score"]
        cols = [c for c in cols if c in f.columns]
        st.dataframe(f[cols].style
            .map(style_valuation, subset=["Valuation Signal"])
            .map(style_valuation, subset=["Valuation Action"])
            .map(style_signal, subset=["AI Signal"])
            .format({"CMP": "₹{:,.2f}", "Lower Intrinsic Value": "₹{:,.2f}",
                     "Intrinsic Value (Base)": "₹{:,.2f}", "Upper Intrinsic Value": "₹{:,.2f}",
                     "Intrinsic Value (DCF)": "₹{:,.2f}",
                     "Intrinsic Value (Multiples)": "₹{:,.2f}",
                     "Entry Trigger Price": "₹{:,.2f}", "Exit Trigger Price": "₹{:,.2f}",
                     "WACC Used": "{:.1%}", "Margin of Safety %": "{:+.2f}%",
                     "To Entry %": "{:+.2f}%", "To Exit %": "{:+.2f}%",
                     "AI_Score": "{:.0f}", "DCF_Valuation_Score": "{:.0f}",
                     "Valuation_Trigger_Score": "{:.0f}"}, na_rep="—"),
            use_container_width=True, hide_index=True, height=560)
        st.download_button("⬇️ Download Report",
                           data=f[cols].to_csv(index=False).encode("utf-8"),
                           file_name=f"dcf_valuation_{datetime.now():%Y%m%d_%H%M}.csv",
                           mime="text/csv")
    with chart_tab:
        if go is not None:
            pd_ = f.dropna(subset=["CMP", "Intrinsic Value (Base)"]).copy()
            if not pd_.empty:
                pd_["Zone"] = np.where(pd_["Margin of Safety %"] >= 10, "Undervalued",
                    np.where(pd_["Margin of Safety %"] <= -10, "Overvalued", "Fair"))
                fig = px.scatter(pd_, x="CMP", y="Intrinsic Value (Base)",
                    color="Zone", size="AI_Score",
                    hover_data=["Ticker", "Name", "Margin of Safety %"],
                    color_discrete_map={"Undervalued": "#34d399", "Fair": "#fbbf24",
                                        "Overvalued": "#fb7185"})
                mv = float(max(pd_["CMP"].max(), pd_["Intrinsic Value (Base)"].max()))
                fig.add_trace(go.Scatter(x=[0, mv], y=[0, mv], mode="lines",
                    name="Fair Value",
                    line=dict(color="#8ba3c0", dash="dash", width=1.5)))
                fig.update_layout(title="Above dashed line = undervalued",
                    xaxis_title="Market Price (₹)", yaxis_title="Intrinsic Value (₹)")
                _style_figure(fig, 500); st.plotly_chart(fig, use_container_width=True)

# ============= OTHER TABS =============
def render_sector_tab(scored, project_dir):
    ss = analyse_sectors(scored)
    if ss.empty: st.info("Sub-Sector data nahi."); return
    save_sector_snapshot(project_dir, ss)
    st.subheader("🏭 Sub-Sector Analysis")
    st.dataframe(ss.style.background_gradient(
        subset=["Avg AI Score", "Avg Sector-Rel"], cmap="RdYlGn")
        .format({"Avg AI Score": "{:.1f}", "Avg Fundamental": "{:.1f}",
                 "Avg Technical": "{:.1f}", "Avg Sector-Rel": "{:.1f}",
                 "Avg Rank Composite": "{:.1f}", "Avg PE": "{:.1f}",
                 "Avg Upside %": "{:+.1f}%"}, na_rep="—"),
        use_container_width=True, hide_index=True, height=min(640, 40 + 32 * len(ss)))

def render_rotation_tab(scored, project_dir):
    ss = analyse_sectors(scored)
    if ss.empty: st.info("Sub-Sector data nahi."); return
    save_sector_snapshot(project_dir, ss)
    rot = compute_sector_rotation(project_dir, ss)
    st.markdown("### 🔥 Sub-Sector Rotation")
    if rot.empty: st.info("Rotation unavailable."); return
    if not rot["Prev AI Score"].notna().any():
        st.info("📌 Pehla run — WoW next run se populate hoga."); return
    dc = [c for c in ["Sector", "Stocks", "Avg AI Score", "Prev AI Score",
                      "WoW AI Δ", "Avg Sector-Rel", "WoW Sector-Rel Δ", "Rotation"]
          if c in rot.columns]
    st.dataframe(rot[dc].style.background_gradient(
        subset=["WoW AI Δ", "WoW Sector-Rel Δ"], cmap="RdYlGn")
        .format({"Avg AI Score": "{:.1f}", "Prev AI Score": "{:.1f}",
                 "WoW AI Δ": "{:+.1f}", "Avg Sector-Rel": "{:.1f}",
                 "WoW Sector-Rel Δ": "{:+.1f}"}, na_rep="—"),
        use_container_width=True, hide_index=True)

def render_peer_tab(scored):
    st.markdown("### 👥 Peer Comparison")
    sc = detect_sector_column(scored)
    if sc is None or scored[sc].isna().all(): st.info("Data nahi."); return
    ss = sorted(scored[sc].dropna().unique().tolist())
    if not ss: return
    sec = st.selectbox("Sub-Sector", ss, key="peer_sec")
    mo = ["AI_Score", "Fundamental_Score_AI", "Sector_Relative_Score", "Technical_Score",
          "Rank_Composite", "CSV_Technical_Score", "Analyst_Consensus_Score",
          "CashFlow_Quality_Score", "DCF_Valuation_Score", "Valuation_Trigger_Score",
          "Margin of Safety %", "PE Ratio", "Return on Equity", "Upside %", "Sharpe"]
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
    st.markdown("##### Peer Stocks")
    render_stock_card_grid(peers, 20)

def render_index_tab(scored, registry):
    st.subheader("📊 Index Analysis")
    idf = build_index_table(scored, registry)
    if idf.empty or idf["Stocks"].sum() == 0:
        st.info("No index constituents."); return
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

def render_risk_tab(scored):
    st.subheader("⚖️ Risk Analytics")
    if "Sharpe" not in scored.columns or scored["Sharpe"].isna().all():
        st.info("Risk data missing."); return
    cs = ["Rank", "Ticker", "Name", "Sharpe", "Sortino", "Volatility %",
          "Max Drawdown %", "Beta", "Alpha %", "Return 1Y %", "Nifty 1Y %",
          "Relative Strength %"]
    cs = [c for c in cs if c in scored.columns]
    st.dataframe(scored.sort_values("Sharpe", ascending=False, na_position="last")
        .head(25)[cs].style.format({"Sharpe": "{:.2f}", "Sortino": "{:.2f}",
        "Volatility %": "{:.2f}", "Max Drawdown %": "{:.2f}", "Beta": "{:.2f}",
        "Alpha %": "{:+.2f}%", "Return 1Y %": "{:+.2f}%", "Nifty 1Y %": "{:+.2f}%",
        "Relative Strength %": "{:+.2f}%"}, na_rep="—"),
        use_container_width=True, hide_index=True)

def render_flags_tab(scored):
    st.subheader("🚩 Red Flag Scanner")
    if "Red Flag Count" not in scored.columns: return
    fl = scored[scored["Red Flag Count"] > 0].sort_values(
        ["Red Flag Count", "AI_Score"], ascending=[False, False])
    cs = ["Rank", "Ticker", "Name", "Sector", "AI_Score", "AI Signal", "Red Flag Count",
          "Quality Grade", "Red Flag List"]
    cs = [c for c in cs if c in fl.columns]
    if fl.empty: st.success("✅ No red flags.")
    else:
        st.dataframe(fl[cs].style.map(style_signal, subset=["AI Signal"])
            .format({"AI_Score": "{:.0f}"}, na_rep="—"),
            use_container_width=True, hide_index=True, height=min(600, 40 + 35 * len(fl)))

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
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr);">
        <div class="kpi-card"><div class="kpi-label">Total</div>
            <div class="kpi-value">{tot}</div></div>
        <div class="kpi-card kpi-green"><div class="kpi-label">Fetch OK</div>
            <div class="kpi-value">{fok}/{tot}</div></div>
        <div class="kpi-card kpi-amber"><div class="kpi-label">Sector Data</div>
            <div class="kpi-value">{scored[sc].notna().sum() if sc else 0}/{tot}</div></div>
        <div class="kpi-card kpi-green"><div class="kpi-label">Valuation OK</div>
            <div class="kpi-value">{val_ok}/{tot}</div></div>
    </div>
    """, unsafe_allow_html=True)

def render_trade_tab(scored):
    st.subheader("💼 Trade Plan")
    if "Position Size %" not in scored.columns: return
    cand = scored[scored["AI Signal"].isin(["Strong Buy", "Buy", "Accumulate"])].copy()
    if cand.empty: st.info("No Buy-rated stocks."); return
    render_stock_card_grid(cand, 30)
    total = cand["Position Size %"].sum()
    st.info(f"💡 Total: **{total:.1f}%** across {len(cand)} ideas.")

# ⭐ CHANGE 3: Deep Dive with Technical tab
def render_deep_tab(scored):
    st.subheader("🔬 Deep Dive")
    if scored.empty:
        return
    t = st.selectbox("Ticker", scored["Ticker"].tolist())
    if not t:
        return
    row = scored.loc[scored["Ticker"] == t].iloc[0]

    st.markdown(render_stock_card_html(row), unsafe_allow_html=True)

    tabs = st.tabs(["📈 Chart", "💰 Valuation", "📊 Technical", "🧮 Fundamentals"])

    with tabs[0]:
        render_price_chart(t)

    with tabs[1]:
        vd = {k: row.get(k) for k in
              ["Intrinsic Value (DCF)", "Intrinsic Value (Multiples)",
               "Intrinsic Value (Base)", "Lower Intrinsic Value",
               "Upper Intrinsic Value", "WACC Used", "Margin of Safety %",
               "Valuation Signal", "DCF_Valuation_Score",
               "Valuation Action", "Entry Trigger Price", "Exit Trigger Price",
               "To Entry %", "To Exit %"] if k in row}
        st.json({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else v))
                 for k, v in vd.items()})

    with tabs[2]:
        render_technical_analysis(row)

    with tabs[3]:
        fd = {k: row.get(k) for k in
              ["PE Ratio", "PB Ratio", "EV/EBITDA Ratio", "Return on Equity", "ROCE",
               "Net Profit Margin", "EBITDA Margin", "5Y Historical EPS Growth",
               "5Y Historical Revenue Growth", "Debt to Equity", "Current Ratio",
               "Promoter Holding", "Dividend Yield", "Free Cash Flow",
               "Operating Cash Flow", "Total Debt", "Cash and Equivalent", "Market Cap"]
              if k in row}
        st.json({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.number)) else v))
                 for k, v in fd.items()})

# ============= EXCEL EXPORT =============
def _build_excel(scored, ss, rot, idf):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        scored.to_excel(w, sheet_name="Screener", index=False)
        val_cols = [c for c in ["Rank", "Ticker", "Name", "Sector", "CMP",
            "Lower Intrinsic Value", "Intrinsic Value (Base)", "Upper Intrinsic Value",
            "Intrinsic Value (DCF)", "Intrinsic Value (Multiples)", "WACC Used",
            "Margin of Safety %", "Valuation Signal", "Valuation Action",
            "Entry Trigger Price", "Exit Trigger Price", "To Entry %", "To Exit %",
            "DCF_Valuation_Score", "Valuation_Trigger_Score",
            "AI_Score", "AI Signal"] if c in scored.columns]
        scored[val_cols].to_excel(w, sheet_name="DCF Valuation", index=False)
        entry = scored[scored.get("Valuation Action", pd.Series()).astype(str)
                       .str.contains("DEEP BUY|ENTRY", na=False)]
        if not entry.empty: entry.to_excel(w, sheet_name="Entry Alerts", index=False)
        exit_ = scored[scored.get("Valuation Action", pd.Series()).astype(str)
                       .str.contains("EXIT", na=False)]
        if not exit_.empty: exit_.to_excel(w, sheet_name="Exit Alerts", index=False)
        if not ss.empty: ss.to_excel(w, sheet_name="Sectors", index=False)
        if not rot.empty: rot.to_excel(w, sheet_name="Rotation", index=False)
        if not idf.empty: idf.to_excel(w, sheet_name="Indices", index=False)
    return buf.getvalue()

# ============= DASHBOARD =============
def render_dashboard(scored, registry, project_dir):
    render_kpi_row(scored)
    st.divider()
    tabs = st.tabs(["🎯 Screener", "💰 DCF", "💼 Trade", "🏭 Sectors",
                    "🔥 Rotation", "👥 Peers", "📊 Indices", "⚖️ Risk",
                    "🚩 Flags", "📐 Patterns", "🔬 Deep", "🔍 Quality"])
    with tabs[0]: render_screener_tab(scored, registry, project_dir)
    with tabs[1]: render_dcf_valuation_tab(scored)
    with tabs[2]: render_trade_tab(scored)
    with tabs[3]: render_sector_tab(scored, project_dir)
    with tabs[4]: render_rotation_tab(scored, project_dir)
    with tabs[5]: render_peer_tab(scored)
    with tabs[6]: render_index_tab(scored, registry)
    with tabs[7]: render_risk_tab(scored)
    with tabs[8]: render_flags_tab(scored)
    with tabs[9]: render_patterns_tab(scored)
    with tabs[10]: render_deep_tab(scored)
    with tabs[11]: render_quality_tab(scored)

# ============= SIDEBAR & MAIN =============
def render_sidebar(project_dir):
    with st.sidebar:
        st.header("📁 Data Sources")
        st.caption("Upload CSVs — `./data/` me save honge.")
        fu = st.file_uploader("Nifty Fundamentals CSV", type=["csv"], key="f")
        tu = st.file_uploader("Explore Promising CSV", type=["csv"], key="t")
        st.divider()
        st.markdown("#### 🔧 Index Config")
        jp = _index_json_path(project_dir)
        st.caption(f"Edit: `{jp.relative_to(project_dir)}`")
        if jp.exists():
            with st.expander("Preview"):
                try: st.json(json.loads(jp.read_text(encoding="utf-8")))
                except Exception: st.warning("JSON read fail.")
    return fu, tu

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide",
                       initial_sidebar_state="collapsed")
    inject_mobile_css()
    st.title(f"📈 {APP_TITLE}")
    st.caption("10-dim AI Score + Hybrid DCF + Entry/Exit Triggers.")
    project_dir = Path(__file__).parent
    registry = load_index_constituents(project_dir)
    fu, tu = render_sidebar(project_dir)
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
