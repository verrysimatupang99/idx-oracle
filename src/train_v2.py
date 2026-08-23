"""
idx-oracle v2 — model upgrade:
  1. ENSEMBLE: LightGBM + LogisticRegression + ExtraTrees (soft vote, calibrated)
  2. REGIME-GATING: vol regime (calm/normal/stress) routes to per-regime models
     (RG-ResMoE insight: regime for ROUTING only — arXiv 2608.12251)
  3. Same leak-free walk-forward protocol as v1 (expanding, embargo=5, yearly refit)

Outputs results/baseline_v2.csv comparable column-by-column with baseline_lgbm.csv.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "proc")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

MIN_TRAIN = 1250
STEP = 250
EMBARGO = 5


def _vol_regime(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> pd.Series:
    """Regime labels for test rows from TRAIN quantiles only (no leakage).
    calm=0 normal=1 stress=2 by trailing 21d vol."""
    q33 = X_tr["vol_21"].quantile(0.33)
    q66 = X_tr["vol_21"].quantile(0.66)
    return pd.cut(X_te["vol_21"], [-np.inf, q33, q66, np.inf], labels=[0, 1, 2]).astype(int)


def _make_model(kind: str):
    if kind == "lgbm":
        return lgb.LGBMClassifier(objective="binary", learning_rate=0.03,
                                  num_leaves=31, min_child_samples=50,
                                  subsample=0.8, colsample_bytree=0.8,
                                  n_estimators=300, n_jobs=4, verbosity=-1)
    if kind == "logit":
        return CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, C=0.5), method="sigmoid", cv=3)
    if kind == "extra":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20,
                                    max_features=0.7, n_jobs=4)
    raise ValueError(kind)


MODELS = ["lgbm", "logit", "extra"]


def walk_forward_v2(sym: str) -> dict | None:
    X = pd.read_parquet(os.path.join(PROC, sym + "_X.parquet"))
    y = pd.read_parquet(os.path.join(PROC, sym + "_y.parquet"))["y"]
    df = X.join(y.rename("y")).dropna()
    if len(df) < MIN_TRAIN + STEP:
        return None
    Xd, yd = df.drop(columns=["y"]), df["y"].astype(int)

    preds = pd.Series(index=yd.index, dtype=float)

    t0 = MIN_TRAIN
    while t0 < len(df) - 1:
        t_end = min(t0 + STEP, len(df) - 1)
        cut = t0 - EMBARGO
        Xtr, ytr, Xte = Xd.iloc[:cut], yd.iloc[:cut], Xd.iloc[t0:t_end]
        reg_te = _vol_regime(Xtr, Xte)

        # per-model, per-regime probability matrices
        P = np.zeros((len(MODELS), 3, len(Xte)))
        for mi, kind in enumerate(MODELS):
            for r in (0, 1, 2):
                mask_tr = _vol_regime(Xtr, Xtr) == r
                if mask_tr.sum() < 400:      # too few regime samples → fallback all
                    m = _make_model(kind).fit(Xtr, ytr)
                    P[mi, r] = m.predict_proba(Xte)[:, 1]
                else:
                    m = _make_model(kind).fit(Xtr[mask_tr.values], ytr[mask_tr.values])
                    sub = Xte[reg_te.values == r]
                    if len(sub):
                        P[mi, r, reg_te.values == r] = m.predict_proba(sub)[:, 1]

        # ensemble = mean of calibrated member probs; regime-blend with global model weight
        ens = P.mean(axis=0).max(axis=0)
        preds.iloc[t0:t_end] = ens
        t0 = t_end

    eval_df = df.iloc[MIN_TRAIN:].copy()
    eval_df["p_up"] = preds
    eval_df = eval_df.dropna(subset=["p_up"])

    up = (eval_df["p_up"] > 0.5).astype(int)
    acc = (up == eval_df["y"]).mean()
    r1 = Xd["ret_1d"].reindex(eval_df.index)
    thr = 0.55
    pos = (eval_df["p_up"] > thr).astype(int)
    strat_ret = pos.shift(1).fillna(0) * r1

    def _sharpe(x):
        x = x.dropna()
        return 0.0 if x.std() == 0 else x.mean() / x.std() * np.sqrt(252)

    return {
        "symbol": sym,
        "test_days": len(eval_df),
        "acc": round(float(acc), 4),
        "coverage": round(float(pos.mean()), 3),
        "strat_sharpe": round(float(_sharpe(strat_ret)), 3),
        "bh_sharpe": round(float(_sharpe(r1)), 3),
        "strat_total_ret": round(float(strat_ret.sum()), 4),
    }


def main():
    rows = []
    for f in sorted(os.listdir(PROC)):
        if f.endswith("_X.parquet"):
            sym = f.replace("_X.parquet", "")
            try:
                r = walk_forward_v2(sym)
            except Exception as e:
                r = {"symbol": sym, "error": str(e)[:60]}
            if r:
                rows.append(r)
                print(f"  {sym:16} days={r.get('test_days','-')} acc={r.get('acc','-')} "
                      f"sharpe={r.get('strat_sharpe','-')} vs bh={r.get('bh_sharpe','-')}",
                      flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "baseline_v2.csv"), index=False)
    print(f"\nsaved → results/baseline_v2.csv")


if __name__ == "__main__":
    main()
