"""
idx-oracle v4 — walk-forward LightGBM + paper-informed upgrades (Aug 2026).

Changes vs v1 (src/train.py), from 3-paper deep-dive (see ~/research/V4-ACTION-LIST.md):
  V4-1  Eval rigor: CVaR 5% reported alongside Sharpe; >=3 seeds per config,
        seed-band (mean +/- std) treated as replication units [arXiv 2608.19389-A]
  V4-2  Signed flow-imbalance features (coarse OHLCV proxies):
        rolling up/down volume ratio, CLV, signed volume imbalance [arXiv 2608.07690-A]
  V4-3  Cost-aware act-or-hold gate: carry yesterday's position unless
        expected edge > roundtrip cost estimate [arXiv 2608.19389-B]

Protocol unchanged (leak-free): expanding window, min 5y train, yearly refit,
5-day embargo, t+1 target. Honest baseline: expect small edges.
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
SEEDS = (11, 23, 47)     # V4-1: multi-seed replication
COST_BPS_ROUNDTRIP = 5   # V4-3: ~5 bps roundtrip cost estimate for liquid index ETFs


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return 0.0 if x.std() == 0 else float(x.mean() / x.std() * np.sqrt(252))


def cvar_5(x: pd.Series) -> float:
    """V4-1: mean of worst 5% daily returns (tail risk)."""
    x = x.dropna()
    if len(x) < 20:
        return np.nan
    q = x.quantile(0.05)
    return float(x[x <= q].mean())


def walk_forward(sym: str) -> dict | None:
    X = pd.read_parquet(os.path.join(PROC, sym + "_X.parquet"))
    y = pd.read_parquet(os.path.join(PROC, sym + "_y.parquet"))["y"]
    df = X.join(y.rename("y")).dropna()
    if len(df) < MIN_TRAIN + STEP:
        return None
    Xd, yd = df.drop(columns=["y"]), df["y"].astype(int)

    params = dict(objective="binary", learning_rate=0.03, num_leaves=31,
                  min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                  n_estimators=300, n_jobs=4, verbosity=-1)

    # per-seed probability predictions
    seed_preds = {}
    for seed in SEEDS:
        preds = pd.Series(index=yd.index, dtype=float)
        p = dict(params, random_state=seed)
        t0 = MIN_TRAIN
        while t0 < len(df) - 1:
            t_end = min(t0 + STEP, len(df) - 1)
            Xtr, ytr = Xd.iloc[:t0 - EMBARGO], yd.iloc[:t0 - EMBARGO]
            Xte = Xd.iloc[t0:t_end]
            model = lgb.LGBMClassifier(**p)
            model.fit(Xtr, ytr)
            preds.iloc[t0:t_end] = model.predict_proba(Xte)[:, 1]
            t0 = t_end
        seed_preds[seed] = preds

    eval_df = df.iloc[MIN_TRAIN:].copy()
    pmat = pd.DataFrame({s: pr.reindex(eval_df.index) for s, pr in seed_preds.items()})
    eval_df["p_up"] = pmat.mean(axis=1)          # seed-mean probability
    eval_df["p_std"] = pmat.std(axis=1)          # V4-1: seed dispersion
    eval_df = eval_df.dropna(subset=["p_up"])

    up = (eval_df["p_up"] > 0.5).astype(int)
    acc = (up == eval_df["y"]).mean()

    r1 = Xd["ret_1d"].reindex(eval_df.index)

    # --- strategy A: v1-style static threshold 0.55 (for apples-to-apples)
    pos_static = (eval_df["p_up"] > 0.55).astype(int)

    # --- strategy B: V4-3 cost-aware act-or-hold gate
    # expected edge proxy: |p_up - 0.5| * 2 * typical_daily_move, in bps
    typical_move_bps = float(r1.abs().rolling(63).mean().iloc[-1]) * 1e4 or 50.0
    edge_bps = (eval_df["p_up"] - 0.5).abs() * 2 * typical_move_bps
    want_long = eval_df["p_up"] > 0.55
    act = edge_bps.shift(1).fillna(0) > COST_BPS_ROUNDTRIP   # decide on t info
    prev_pos = pd.Series(0, index=eval_df.index)
    positions = []
    cur = 0
    for i in range(len(eval_df)):
        if bool(act.iloc[i]):
            cur = int(bool(want_long.iloc[i]))
        positions.append(cur)
        prev_pos.iloc[i] = cur
    pos_gated = prev_pos

    def simulate(pos: pd.Series) -> dict:
        strat_ret = pos.shift(1).fillna(0) * r1
        turns = int((pos.diff().abs() > 0).sum())
        return {
            "sharpe": round(sharpe(strat_ret), 3),
            "cvar5": round(cvar_5(strat_ret), 5),
            "total_ret": round(float(strat_ret.sum()), 4),
            "coverage": round(float(pos.mean()), 3),
            "turns": turns,
        }

    static_sim = simulate(pos_static)
    gated_sim = simulate(pos_gated)
    bh_ret = r1

    res = {
        "symbol": sym,
        "test_days": len(eval_df),
        "acc": round(float(acc), 4),
        "seed_std_mean": round(float(eval_df["p_std"].mean()), 4),
        # static threshold (v1-comparable)
        **{f"static_{k}": v for k, v in static_sim.items()},
        # cost-aware gate (new)
        **{f"gated_{k}": v for k, v in gated_sim.items()},
        "bh_sharpe": round(sharpe(bh_ret), 3),
        "bh_cvar5": round(cvar_5(bh_ret), 5),
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
                if "acc" in r:
                    print(f"  {sym:16} acc={r['acc']:.3f} "
                          f"static_sh={r['static_sharpe']:+.2f} gated_sh={r['gated_sharpe']:+.2f} "
                          f"(turns {r['static_turns']}->{r['gated_turns']}) vs bh={r['bh_sharpe']:+.2f}")
                else:
                    print(f"  {sym}: ERROR {r.get('error')}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "baseline_v4.csv"), index=False)
    ns = (out["static_sharpe"] > out["bh_sharpe"]).sum()
    ng = (out["gated_sharpe"] > out["bh_sharpe"]).sum()
    print(f"\nbeats buy&hold on sharpe: static {ns}/{len(out)} | gated {ng}/{len(out)}")
    print(f"avg CVaR5: static {out['static_cvar5'].mean():.5f} | gated {out['gated_cvar5'].mean():.5f} | bh {out['bh_cvar5'].mean():.5f}")
    print("saved → results/baseline_v4.csv")


if __name__ == "__main__":
    main()
