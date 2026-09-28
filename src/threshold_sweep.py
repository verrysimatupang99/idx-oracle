"""
Uji ambang: berapa frekuensi transisi pada ambang keyakinan berbeda, dan berapa
titik impas biaya yang dihasilkan?

Pertanyaan yang dijawab: strategi idx-oracle punya impas biaya 4-13 bps/sisi pada
ambang 0.55, sementara retail IDX/ETF 15-30 bps/sisi. Kalau ambang dinaikkan,
transisi turun drastis — tapi apakah return ikut turun lebih cepat?

Cara: walk-forward yang sama dengan train_v4 (expanding window, min 5y train, refit
tahunan, embargo 5 hari, target t+1), tapi ambang di-sweep. Untuk tiap ambang dicatat:
  - transisi per tahun (penentu biaya)
  - return per tahun (kotor)
  - titik impas biaya per sisi = (return/tahun) / (transisi/tahun) / 2

Semua angka jujur: tidak ada parameter yang dipilih karena hasilnya bagus.
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "proc")
RESULTS = os.path.join(BASE, "results")

MIN_TRAIN = 1250
STEP = 250
EMBARGO = 5
SEEDS = (11, 23)
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
FEATURES_DROP = ("y", "ret_1d")


def walk_forward_threshold(df: pd.DataFrame, feats: list[str], thr: float) -> dict | None:
    """Satu lintasan walk-forward untuk satu ambang. Mengembalikan agregat."""
    n = len(df)
    if n < MIN_TRAIN + 250:
        return None
    oos_ret: list[float] = []
    oos_pos: list[int] = []
    for start in range(MIN_TRAIN, n - EMBARGO - 1, STEP):
        train = df.iloc[: start - EMBARGO]
        test = df.iloc[start: start + STEP]
        if len(train) < MIN_TRAIN or len(test) < 20:
            continue
        preds = np.zeros(len(test))
        for seed in SEEDS:
            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                random_state=seed, verbose=-1,
            )
            model.fit(train[feats], train["y"])
            preds += model.predict_proba(test[feats])[:, 1]
        preds /= len(SEEDS)
        pos = (preds > thr).astype(int)
        ret = test["ret_1d"].to_numpy()
        # posisi dibentuk di t, hasilnya t→t+1
        oos_ret.extend((pos[:-1] * ret[1:]).tolist())
        oos_pos.extend(pos[:-1].tolist())
    if not oos_ret:
        return None
    r = np.array(oos_ret)
    p = np.array(oos_pos)
    turns = int(np.abs(np.diff(p)).sum()) if len(p) > 1 else 0
    days = len(r)
    years = days / 252.0
    total = float(r.sum())
    return {
        "days": days,
        "turns": turns,
        "turns_per_year": turns / years if years > 0 else 0.0,
        "total_ret": total,
        "ret_per_year": total / years if years > 0 else 0.0,
        "coverage": float(p.mean()),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
    }


def main() -> int:
    files = sorted(f for f in os.listdir(PROC) if f.endswith("_X.parquet"))
    out_rows: list[dict] = []
    for f in files:
        sym = f.replace("_X.parquet", "")
        try:
            X = pd.read_parquet(os.path.join(PROC, f))
            y = pd.read_parquet(os.path.join(PROC, f.replace("_X", "_y")))
        except Exception as exc:
            print(f"skip {sym}: {exc}")
            continue
        df = X.copy()
        df["y"] = y["y"].to_numpy()
        if "ret_1d" not in df.columns:
            print(f"skip {sym}: tidak ada ret_1d")
            continue
        feats = [c for c in df.columns if c not in FEATURES_DROP]
        df = df.dropna(subset=feats + ["y", "ret_1d"]).reset_index(drop=True)
        for thr in THRESHOLDS:
            res = walk_forward_threshold(df, feats, thr)
            if res is None:
                continue
            be_bps = ((res["ret_per_year"] / res["turns_per_year"]) * 1e4 / 2
                      if res["turns_per_year"] > 0 else 0.0)
            out_rows.append({"symbol": sym, "threshold": thr, **res,
                             "breakeven_bps_per_side": round(be_bps, 2)})
        print(f"  {sym:14s} selesai ({len(THRESHOLDS)} ambang)")

    out = pd.DataFrame(out_rows)
    path = os.path.join(RESULTS, "threshold-sweep.csv")
    out.to_csv(path, index=False)
    print(f"\nsaved → {path}  ({len(out)} baris)")

    # Ringkas: rata-rata lintas simbol per ambang
    print(f"\n{'ambang':>7}{'trans/thn':>11}{'ret/thn':>10}{'sharpe':>8}{'impas bps/sisi':>16}{'layak retail?':>15}")
    for thr in THRESHOLDS:
        sub = out[out["threshold"] == thr]
        if sub.empty:
            continue
        tpy = sub["turns_per_year"].mean()
        rpy = sub["ret_per_year"].mean()
        sh = sub["sharpe"].mean()
        be = sub["breakeven_bps_per_side"].mean()
        verdict = "YA" if be >= 15 else ("mepet" if be >= 10 else "tidak")
        print(f"{thr:>7.2f}{tpy:>11.0f}{rpy:>10.1%}{sh:>8.2f}{be:>16.1f}{verdict:>15}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
