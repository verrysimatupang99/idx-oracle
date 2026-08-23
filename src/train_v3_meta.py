"""
idx-oracle v3 — META-SELECTION: honest per-market choice between v1 (LGBM) and
v2 (Ensemble+Regime) using ONLY past performance at each walk-forward point.

CRITICAL HONESTY RULE: we do NOT pick the winner per market by looking at the full
backtest (that would be in-sample cherry-picking). Instead, inside the same expanding
walk-forward, both models predict; a rolling tracker (last 250 out-of-sample days)
credits the better model; the tracker's current leader makes each day's call.
The meta layer only ever sees information available at time t.

Output: results/baseline_v3_meta.csv + comparison vs v1/v2/B&H.
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

MIN_TRAIN = 1250
STEP = 250
EMBARGO = 5
TRACK_WINDOW = 250  # rolling window for meta-selection


def _model(kind):
    if kind == "lgbm":
        return lgb.LGBMClassifier(objective="binary", learning_rate=0.03,
                                  num_leaves=31, min_child_samples=50,
                                  subsample=0.8, colsample_bytree=0.8,
                                  n_estimators=300, n_jobs=4, verbosity=-1)
    if kind == "logit":
        return CalibratedClassifierCV(LogisticRegression(max_iter=1000, C=0.5),
                                      method="sigmoid", cv=3)
    if kind == "extra":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20,
                                    max_features=0.7, n_jobs=4)
    raise ValueError(kind)


def _vol_regime(X_tr, X_te):
    q33 = X_tr["vol_21"].quantile(0.33)
    q66 = X_tr["vol_21"].quantile(0.66)
    return pd.cut(X_te["vol_21"], [-np.inf, q33, q66, np.inf], labels=[0, 1, 2]).astype(int)


def _predict_v2(Xtr, ytr, Xte):
    """v2 = ensemble with regime routing (same as train_v2)."""
    reg_te = _vol_regime(Xtr, Xte)
    P = np.zeros((3, 3, len(Xte)))
    for mi, kind in enumerate(["lgbm", "logit", "extra"]):
        for r in (0, 1, 2):
            mask = (_vol_regime(Xtr, Xtr) == r).values
            m = _model(kind)
            if mask.sum() < 400:
                m.fit(Xtr, ytr)
                P[mi, r] = m.predict_proba(Xte)[:, 1]
            else:
                m.fit(Xtr[mask], ytr[mask])
                sub_idx = reg_te.values == r
                if sub_idx.any():
                    P[mi, r, sub_idx] = m.predict_proba(Xte[sub_idx])[:, 1]
    return P.mean(axis=0).max(axis=0)


def walk_forward_meta(sym: str) -> dict | None:
    X = pd.read_parquet(os.path.join(PROC, sym + "_X.parquet"))
    y = pd.read_parquet(os.path.join(PROC, sym + "_y.parquet"))["y"]
    df = X.join(y.rename("y")).dropna()
    if len(df) < MIN_TRAIN + STEP:
        return None
    Xd, yd = df.drop(columns=["y"]), df["y"].astype(int)

    preds = pd.Series(index=yd.index, dtype=float)
    picks = pd.Series(index=yd.index, dtype=float)  # which model made the call

    # rolling correctness trackers (brier-based; lower = better)
    score_v1 = score_v2 = 0.5
    recent = []  # list of (err_v1, err_v2), trimmed to TRACK_WINDOW

    t0 = MIN_TRAIN
    while t0 < len(df) - 1:
        t_end = min(t0 + STEP, len(df) - 1)
        cut = t0 - EMBARGO
        Xtr, ytr, Xte = Xd.iloc[:cut], yd.iloc[:cut], Xd.iloc[t0:t_end]
        yte = yd.iloc[t0:t_end]

        p1 = _model("lgbm").fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        p2 = _predict_v2(Xtr, ytr, Xte)

        # meta choice per day from tracker state BEFORE seeing these outcomes
        use_v2 = score_v2 < score_v1
        block_pred = p2 if use_v2 else p1
        preds.iloc[t0:t_end] = block_pred
        picks.iloc[t0:t_end] = 2 if use_v2 else 1

        # update trackers with realized outcomes
        err1 = np.abs(p1 - yte.values)
        err2 = np.abs(p2 - yte.values)
        recent.extend(zip(err1, err2))
        recent = recent[-TRACK_WINDOW:]
        if recent:
            r1, r2 = zip(*recent)
            score_v1 = float(np.mean(r1))
            score_v2 = float(np.mean(r2))
        t0 = t_end

    eval_df = df.iloc[MIN_TRAIN:].copy()
    eval_df["p_up"] = preds
    eval_df["pick"] = picks
    eval_df = eval_df.dropna(subset=["p_up"])

    up = (eval_df["p_up"] > 0.5).astype(int)
    acc = (up == eval_df["y"]).mean()
    r1 = Xd["ret_1d"].reindex(eval_df.index)
    pos = (eval_df["p_up"] > 0.55).astype(int)
    strat_ret = pos.shift(1).fillna(0) * r1

    def _sharpe(x):
        x = x.dropna()
        return 0.0 if x.std() == 0 else x.mean() / x.std() * np.sqrt(252)

    v2_days = float((eval_df["pick"] == 2).mean())
    return {
        "symbol": sym,
        "test_days": len(eval_df),
        "acc": round(float(acc), 4),
        "coverage": round(float(pos.mean()), 3),
        "strat_sharpe": round(float(_sharpe(strat_ret)), 3),
        "bh_sharpe": round(float(_sharpe(r1)), 3),
        "v2_share": round(v2_days, 2),
    }


def main():
    rows = []
    for f in sorted(os.listdir(PROC)):
        if f.endswith("_X.parquet"):
            sym = f.replace("_X.parquet", "")
            try:
                r = walk_forward_meta(sym)
            except Exception as e:
                r = {"symbol": sym, "error": str(e)[:60]}
            if r:
                rows.append(r)
                print(f"  {sym:16} days={r.get('test_days','-')} acc={r.get('acc','-')} "
                      f"sharpe={r.get('strat_sharpe','-')} bh={r.get('bh_sharpe','-')} "
                      f"v2share={r.get('v2_share','-')}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "baseline_v3_meta.csv"), index=False)
    print("\nsaved → results/baseline_v3_meta.csv")


if __name__ == "__main__":
    main()
