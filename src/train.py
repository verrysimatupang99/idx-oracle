"""
idx-oracle baseline — walk-forward LightGBM direction classifier per symbol.

Protocol (leak-free):
  - Expanding window walk-forward: train on [0, t), predict day t+1
  - Re-fit every `step` days, min train 5y, embargo 5 days between train/test
  - Report: accuracy vs 50%, precision/recall on UP class, and a simple
    long-flat PnL simulation with next-day returns (no compounding tricks)

This is a BASELINE, not a holy grail. Direction prediction on daily index data
is near-efficient; expect small edges. The point is honest measurement.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "proc")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

MIN_TRAIN = 1250   # ~5y trading days
STEP = 250         # re-fit yearly
EMBARGO = 5        # days between train end and test start


def walk_forward(sym: str) -> dict | None:
    X = pd.read_parquet(os.path.join(PROC, sym + "_X.parquet"))
    y = pd.read_parquet(os.path.join(PROC, sym + "_y.parquet"))["y"]
    df = X.join(y.rename("y")).dropna()
    if len(df) < MIN_TRAIN + STEP:
        return None
    Xd, yd = df.drop(columns=["y"]), df["y"].astype(int)

    preds = pd.Series(index=yd.index, dtype=float)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31,
                  min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                  n_estimators=300, n_jobs=4, verbosity=-1)

    t0 = MIN_TRAIN
    while t0 < len(df) - 1:
        t_end = min(t0 + STEP, len(df) - 1)
        Xtr, ytr = Xd.iloc[:t0 - EMBARGO], yd.iloc[:t0 - EMBARGO]
        Xte = Xd.iloc[t0:t_end]
        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, ytr)
        preds.iloc[t0:t_end] = model.predict_proba(Xte)[:, 1]
        t0 = t_end

    eval_df = df.iloc[MIN_TRAIN:].copy()
    eval_df["p_up"] = preds
    eval_df = eval_df.dropna(subset=["p_up"])

    up = (eval_df["p_up"] > 0.5).astype(int)
    acc = (up == eval_df["y"]).mean()

    # long-when-confident simulation: long if p_up>thr else flat
    r1 = Xd["ret_1d"].reindex(eval_df.index)  # same-day return as realized fwd proxy
    thr = 0.55
    pos = (eval_df["p_up"] > thr).astype(int)
    strat_ret = pos.shift(1).fillna(0) * r1   # position formed on t, realized t→t+1
    bh_ret = r1
    def _sharpe(x):
        x = x.dropna()
        return 0.0 if x.std() == 0 else x.mean() / x.std() * np.sqrt(252)
    res = {
        "symbol": sym,
        "test_days": len(eval_df),
        "acc": round(float(acc), 4),
        "acc_up_recall": round(float(((up == 1) & (eval_df["y"] == 1)).sum() / max((eval_df["y"] == 1).sum(), 1)), 4),
        "coverage": round(float(pos.mean()), 3),
        "strat_sharpe": round(float(_sharpe(strat_ret)), 3),
        "bh_sharpe": round(float(_sharpe(bh_ret)), 3),
        "strat_total_ret": round(float(strat_ret.sum()), 4),
        "bh_total_ret": round(float(bh_ret.sum()), 4),
    }
    return res


def main():
    rows = []
    for f in sorted(os.listdir(PROC)):
        if f.endswith("_X.parquet"):
            sym = f.replace("_X.parquet", "")
            try:
                r = walk_forward(sym)
            except Exception as e:
                r = {"symbol": sym, "error": str(e)[:60]}
            if r:
                rows.append(r)
                print(f"  {sym:16} days={r.get('test_days','-')} acc={r.get('acc','-')} "
                      f"sharpe={r.get('strat_sharpe','-')} vs bh={r.get('bh_sharpe','-')}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "baseline_lgbm.csv"), index=False)
    n = (out["strat_sharpe"] > out["bh_sharpe"]).sum() if "strat_sharpe" in out else 0
    print(f"\nbeats buy&hold on sharpe: {n}/{len(out)} symbols")
    print("saved → results/baseline_lgbm.csv")


if __name__ == "__main__":
    main()
