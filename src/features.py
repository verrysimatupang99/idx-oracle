"""
idx-oracle feature engineering — all features use ONLY past information (no lookahead).

Per-symbol daily features:
  returns:      1d, 5d, 21d log returns
  volatility:   rolling std of 1d returns (5/21/63d), Parkinson vol from HL
  momentum:     close vs SMA(50/200), RSI(14), MACD-ish spread
  regime:       vol-of-vol, drawdown from 252d high, distance from SMA200 (normalized)
  calendar:     day-of-week, month (cyclical encoding)
  volume:       z-scored volume, OBV slope proxy

Target (for direction model):
  y = sign(next-day forward return) — strictly t+1, never leaked into X.
"""
import os
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
PROC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "proc")
os.makedirs(PROC, exist_ok=True)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    r1 = np.log(c / c.shift(1))
    feats = pd.DataFrame(index=df.index)
    feats["ret_1d"] = r1
    feats["ret_5d"] = np.log(c / c.shift(5))
    feats["ret_21d"] = np.log(c / c.shift(21))

    for n in (5, 21, 63):
        feats[f"vol_{n}"] = r1.rolling(n).std() * np.sqrt(252)
    hl_ratio = (np.log(h / l)) ** 2
    feats["park_vol_21"] = np.sqrt((hl_ratio.rolling(21).mean()) / (4 * np.log(2))) * np.sqrt(252)
    feats["vol_of_vol"] = feats["vol_21"].rolling(63).std()

    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    feats["px_vs_sma50"] = c / sma50 - 1
    feats["px_vs_sma200"] = c / sma200 - 1
    feats["sma50_vs_sma200"] = sma50 / sma200 - 1
    feats["rsi_14"] = rsi(c)

    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    feats["macd_spread"] = (ema12 - ema26) / c

    roll_max = c.rolling(252).max()
    feats["drawdown_252"] = c / roll_max - 1

    vz = (v - v.rolling(63).mean()) / v.rolling(63).std()
    feats["vol_z"] = vz.replace([np.inf, -np.inf], np.nan)
    feats["obv_slope_10"] = (np.sign(r1) * v).cumsum().diff(10) / v.rolling(10).mean().clip(lower=1)

    # V4-2: signed flow-imbalance features (coarse OHLCV proxies) [arXiv 2608.07690-A]
    # All strictly backward-looking (rolling on past bars only) — no lookahead.
    up_vol = v.where(r1 > 0, 0.0).rolling(21).sum()
    dn_vol = v.where(r1 < 0, 0.0).rolling(21).sum()
    feats["flow_imb_21"] = ((up_vol - dn_vol) / (up_vol + dn_vol)).replace([np.inf, -np.inf], np.nan)
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)          # close location value ∈ [-1,1]
    feats["clv_ma_10"] = clv.rolling(10).mean()
    feats["clv_vol_weighted_21"] = (clv * v).rolling(21).mean() / v.rolling(21).mean().clip(lower=1)
    signed_vol = np.sign(r1) * v
    tot = v.rolling(63).sum().clip(lower=1)
    feats["signed_vol_ratio_63"] = signed_vol.rolling(63).sum() / tot

    idx = feats.index
    dow = idx.dayofweek.astype(float); mon = idx.month.astype(float)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7); feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feats["mon_sin"] = np.sin(2 * np.pi * mon / 12); feats["mon_cos"] = np.cos(2 * np.pi * mon / 12)

    # target: NEXT day direction (strictly forward)
    fwd = np.log(c.shift(-1) / c)
    y = (fwd > 0).astype(int)
    y[fwd.isna()] = np.nan

    feats = feats.replace([np.inf, -np.inf], np.nan)
    valid = feats.dropna(how="any").index.intersection(y.dropna().index)
    return feats.loc[valid], y.loc[valid]


def process_all() -> dict:
    out = {}
    for f in sorted(os.listdir(RAW)):
        if not f.endswith(".parquet"):
            continue
        sym = f.replace(".parquet", "")
        df = pd.read_parquet(os.path.join(RAW, f))
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        X, y = make_features(df)
        if len(X) < 1500:   # need enough history for stable training
            print(f"  skip {sym}: only {len(X)} rows after features")
            continue
        X.to_parquet(os.path.join(PROC, sym + "_X.parquet"))
        y.to_frame("y").to_parquet(os.path.join(PROC, sym + "_y.parquet"))
        out[sym] = len(X)
        print(f"  {sym:16} features={X.shape[1]:3} rows={len(X)}")
    return out


if __name__ == "__main__":
    print("feature engineering →", PROC)
    res = process_all()
    print(f"done: {len(res)} symbols processed")
