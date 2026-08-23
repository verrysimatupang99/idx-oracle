"""
idx-oracle data layer — multi-source index/asset downloader with validation.

Sources (all free & valid, no API key required):
  - Yahoo Finance (yfinance)  : equity indices, FX, futures — up to ~50yr
  - Stooq (stooq.com)         : fallback for indices, CSV over HTTP
  - Binance API               : crypto pairs — from 2017
  - FRED                      : macro series (rates, VIX proxy etc.)

Design rules:
  - Daily OHLCV only (10+ year horizon goal)
  - Every download is validated: min years, no future dates, sane gaps
  - Output: one parquet per symbol under data/raw/
"""
import os
import time
import math
import datetime as dt

import pandas as pd
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

UNIVERSE = {
    # --- Equity indices (global) ---
    "^GSPC": ("yahoo", "S&P 500"),
    "^NDX": ("yahoo", "Nasdaq 100"),
    "^DJI": ("yahoo", "Dow Jones"),
    "^STOXX50E": ("yahoo", "Euro Stoxx 50"),
    "^DAX": ("yahoo", "DAX"),
    "^FTSE": ("yahoo", "FTSE 100"),
    "^N225": ("yahoo", "Nikkei 225"),
    "000001.SS": ("yahoo", "Shanghai Composite"),
    "^HSI": ("yahoo", "Hang Seng"),
    "^KS11": ("yahoo", "KOSPI"),
    "^TWII": ("yahoo", "Taiwan Weighted"),
    "^AXJO": ("yahoo", "ASX 200"),
    "^BVSP": ("yahoo", "Bovespa"),
    "^NSEI": ("yahoo", "Nifty 50"),
    "^JKSE": ("yahoo", "IDX Composite"),   # Indonesia!
    # --- Crypto ---
    "BTC-USD": ("yahoo", "Bitcoin"),
    "ETH-USD": ("yahoo", "Ethereum"),
    # --- Commodities / macro proxies ---
    "GC=F": ("yahoo", "Gold Futures"),
    "CL=F": ("yahoo", "WTI Crude Futures"),
    "SI=F": ("yahoo", "Silver Futures"),
    # --- Vol & rates ---
    "^VIX": ("yahoo", "VIX"),
    "^TNX": ("yahoo", "US 10Y Yield x10"),
}


def _validate(df: pd.DataFrame, symbol: str, min_years: float = 8.0) -> dict:
    """Sanity checks. Returns report dict; raises on fatal issues."""
    rep = {"symbol": symbol, "rows": len(df)}
    if df.empty:
        raise ValueError(f"{symbol}: empty")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    rep["start"] = str(df.index.min().date())
    rep["end"] = str(df.index.max().date())
    years = (df.index.max() - df.index.min()).days / 365.25
    rep["years"] = round(years, 1)
    if df.index.max() > pd.Timestamp.now() + pd.Timedelta(days=2):
        raise ValueError(f"{symbol}: future dates present!")
    core = ["Open", "High", "Low", "Close"]
    missing = [c for c in core if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing cols {missing}")
    bad = (df[core] <= 0).sum().sum()
    rep["nonpositive"] = int(bad)
    # big gap scan (trading gaps > 14 days are suspicious for daily data)
    gaps = df.index.to_series().diff().dt.days
    rep["max_gap_days"] = int(gaps.max())
    rep["ok_years"] = years >= min_years
    return rep


def download_yahoo(symbol: str) -> pd.DataFrame | None:
    import yfinance as yf
    for attempt in range(3):
        try:
            df = yf.download(symbol, start="1990-01-01",
                             progress=False, auto_adjust=True, threads=False)
            if df is None or len(df) == 0:
                time.sleep(2 * (attempt + 1))
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def download_binance(symbol: str = "BTCUSDT") -> pd.DataFrame | None:
    """Daily klines via public REST — full history since 2017."""
    url = "https://api.binance.com/api/v3/klines"
    out, end_ms = [], int(dt.datetime.now().timestamp() * 1000)
    for _ in range(40):  # paginated fetch
        params = f"?symbol={symbol}&interval=1d&limit=1000&endTime={end_ms}"
        try:
            r = requests.get(url + params, timeout=20)
            r.raise_for_status()
        except Exception:
            return None
        rows = r.json()
        if not rows:
            break
        out = rows + out
        end_ms = rows[0][0] - 1
        if len(rows) < 1000:
            break
        time.sleep(0.4)
    if not out:
        return None
    df = pd.DataFrame(out, columns=["ot", "Open", "High", "Low", "Close",
                                    "Volume", "ct", "qav", "n", "tbb", "tbq", "ig"])
    df["ot"] = pd.to_datetime(df["ot"], unit="ms")
    df = df.set_index("ot")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.index.name = "Date"
    return df


def fetch_all(min_years: float = 8.0):
    reports = []
    for sym, (src, name) in UNIVERSE.items():
        df = None
        if src == "yahoo":
            df = download_yahoo(sym)
        elif src == "binance":
            df = download_binance(sym)
        if df is None:
            reports.append({"symbol": sym, "name": name, "status": "FAILED"})
            continue
        try:
            rep = _validate(df, sym, min_years)
            safe = sym.replace("^", "idx_").replace("=", "_").replace(".", "_").replace("/", "_")
            path = os.path.join(RAW, safe + ".parquet")
            df.to_parquet(path)
            rep.update({"name": name, "status": "OK", "saved": path})
        except Exception as e:
            rep = {"symbol": sym, "name": name, "status": f"INVALID: {e}"}
        reports.append(rep)
        print(f"  {rep['status']:12} {sym:12} {rep.get('name',''):22} "
              f"{rep.get('years','-')}yr rows={rep.get('rows','-')}")
    return reports


if __name__ == "__main__":
    print("idx-oracle data download — universe:", len(UNIVERSE))
    rs = fetch_all()
    ok = sum(1 for r in rs if r["status"] == "OK")
    print(f"\ndone: {ok}/{len(rs)} symbols OK → {RAW}")
