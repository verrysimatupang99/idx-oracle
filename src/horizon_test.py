"""Uji horizon: apakah memprediksi 5 hari (bukan 1 hari) memotong biaya cukup banyak?

Dasar: sweep ambang menunjukkan menaikkan ambang TIDAK menolong (transisi turun 6x,
return turun 6x juga — impas tetap 5-6 bps). Tapi autokorelasi 5-hari +0,789 menunjukkan
tren bertahan. Kalau sinyal yang sama dipakai untuk posisi 5 hari, transisi turun ~5x
TANPA menuntut keyakinan lebih tinggi.

Uji: latih ulang target = arah 5 hari ke depan (bukan 1 hari), lalu bandingkan
transisi/tahun dan titik impas biaya.
"""
import os, numpy as np, pandas as pd, lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "proc")
MIN_TRAIN, STEP, EMBARGO = 1250, 250, 5
SEEDS = (11, 23)
HORIZONS = (1, 3, 5, 10)
THRESH = 0.55


def run(df, feats, horizon):
    """Walk-forward dengan target arah `horizon` hari."""
    fwd = df["ret_1d"].shift(-1)
    for k in range(2, horizon + 1):
        fwd = fwd + df["ret_1d"].shift(-k)
    y = (fwd > 0).astype(int)
    d = df.copy()
    d["_y"] = y
    d = d.dropna(subset=feats + ["_y", "ret_1d"])
    d = d.iloc[:-horizon] if horizon > 1 else d       # buang ekor tanpa label
    if len(d) < MIN_TRAIN + 250:
        return None
    rets, poss = [], []
    for start in range(MIN_TRAIN, len(d) - EMBARGO - horizon, STEP):
        train = d.iloc[: start - EMBARGO]
        test = d.iloc[start: start + STEP]
        if len(train) < MIN_TRAIN or len(test) < 20:
            continue
        preds = np.zeros(len(test))
        for seed in SEEDS:
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                   min_child_samples=30, subsample=0.8,
                                   colsample_bytree=0.8, random_state=seed, verbose=-1)
            m.fit(train[feats], train["_y"])
            preds += m.predict_proba(test[feats])[:, 1]
        preds /= len(SEEDS)
        pos = (preds > THRESH).astype(int)
        r = test["ret_1d"].to_numpy()
        # posisi ditahan horizon hari: hasil = pos * return harian selama horizon
        for i in range(len(pos) - 1):
            rets.append(pos[i] * r[i + 1])
            poss.append(pos[i])
    if not rets:
        return None
    r = np.array(rets); p = np.array(poss)
    turns = int(np.abs(np.diff(p)).sum())
    years = len(r) / 252.0
    tpy = turns / years
    rpy = float(r.sum()) / years
    be = (rpy / tpy * 1e4 / 2) if tpy > 0 else 0.0
    return {"turns_per_year": round(tpy, 1), "ret_per_year": round(rpy, 4),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0, 3),
            "breakeven_bps_per_side": round(be, 2)}


def main():
    files = sorted(f for f in os.listdir(PROC) if f.endswith("_X.parquet"))
    out = []
    for f in files:
        sym = f.replace("_X.parquet", "")
        try:
            X = pd.read_parquet(os.path.join(PROC, f))
            y = pd.read_parquet(os.path.join(PROC, f.replace("_X", "_y")))
        except Exception:
            continue
        df = X.copy()
        df["y"] = y["y"].to_numpy()
        if "ret_1d" not in df.columns:
            continue
        feats = [c for c in df.columns if c not in ("y", "ret_1d")]
        df = df.dropna(subset=feats + ["y", "ret_1d"]).reset_index(drop=True)
        for h in HORIZONS:
            res = run(df, feats, h)
            if res:
                out.append({"symbol": sym, "horizon": h, **res})
        print(f"  {sym} selesai")
    res_df = pd.DataFrame(out)
    res_df.to_csv(os.path.join(BASE, "results", "horizon-test.csv"), index=False)
    print(f"\nsaved → results/horizon-test.csv ({len(res_df)} baris)")
    print(f"\n{'horizon':>8}{'trans/thn':>11}{'ret/thn':>10}{'sharpe':>8}{'impas bps/sisi':>16}{'layak?':>9}")
    for h in HORIZONS:
        sub = res_df[res_df["horizon"] == h]
        if sub.empty:
            continue
        be = sub["breakeven_bps_per_side"].mean()
        print(f"{h:>7}h{sub['turns_per_year'].mean():>11.0f}{sub['ret_per_year'].mean():>10.1%}"
              f"{sub['sharpe'].mean():>8.2f}{be:>16.1f}{'YA' if be >= 15 else 'tidak':>9}")


if __name__ == "__main__":
    main()
