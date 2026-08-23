# idx-oracle 🔮

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-informational)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-yellow)](https://lightgbm.readthedocs.io/)
[![Walk-Forward](https://img.shields.io/badge/eval-walk--forward%20%2B%20embargo-success)]()
[![Assets](https://img.shields.io/badge/markets-21%20global-ff69b4)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)]()


Open-source market direction research toolkit — **multi-asset, 10-35+ years of data, honest walk-forward evaluation.**

> "Prediction is very difficult, especially about the future." — build the measurement first, the oracle second.

## What is this?

`idx-oracle` downloads daily history for **21 global markets** (equity indices, crypto, commodities, vol), engineers leak-free features, and trains **walk-forward LightGBM direction classifiers** with honest out-of-sample evaluation.

Built to be:
- **Open** — MIT license, free data sources only (Yahoo Finance, Binance public API)
- **Honest** — expanding-window walk-forward, 5-day embargo, no lookahead anywhere
- **Multi-market** — 15 equity indices (incl. IDX Composite 🇮🇩), BTC/ETH, gold/oil/silver, VIX
- **Reproducible** — one command each step, deterministic splits

## Data Coverage (verified 2026-08-22)

| Asset class | Symbols | History |
|---|---|---|
| Equity indices | 15 (S&P 500, NDX, DJI, FTSE, Nikkei, HSI, KOSPI, **IDX**, ...) | up to 36.6 yr |
| Crypto | BTC, ETH | 11.9 / 8.8 yr |
| Commodities | Gold, WTI, Silver futures | 26 yr |
| Vol/Rates | VIX, US10Y | 36.6 yr |

## Quickstart

```bash
uv venv .venv
uv pip install --python .venv/bin/python pandas numpy scikit-learn lightgbm yfinance requests pyarrow

# 1. download + validate (21/22 OK on first run)
python src/data.py

# 2. leak-free features (20 per symbol)
python src/features.py

# 3. walk-forward baseline (LightGBM, yearly refit, 5d embargo)
python src/train.py
```

## Methodology (the honest part)

- **Features** (20/symbol): multi-horizon log returns, realized + Parkinson vol, vol-of-vol,
  SMA ratios, RSI, MACD spread, 252d drawdown, volume z-score, OBV slope, cyclical calendar
- **Target**: next-day direction `sign(ret_{t+1})` — never in X
- **Protocol**: expanding window, min 5y train, refit every 250d, **5-day embargo**,
  metrics only on out-of-sample days
- **Baseline bar**: must beat buy-and-hold Sharpe, not just 50% accuracy

## Results (out-of-sample, verified 2026-08-23)

| Version | Approach | Beats B&H | Avg Sharpe |
|---|---|---|---|
| v1 | LightGBM single (`baseline_lgbm.csv`) | **13/19** | **0.437** |
| v2 | Ensemble+Regime routing (`baseline_v2.csv`) | 9/19 | 0.360 |
| v3 | Meta-selection, rolling tracker (`baseline_v3_meta.csv`) | 11/19 | 0.410 |
| v4 | Multi-seed + flow-imbalance feats + cost-aware gate (`baseline_v4.csv`) | **14/19** (static) / 11/19 (gated) | 0.433 static / 0.383 gated |

v4 notes (paper-informed, see `docs/paper-digests` upstream):
- 3-seed ensembling: seed dispersion is small (avg p_std ~0.03) — results are stable across seeds
- CVaR 5% now reported: strategies cut tail risk vs B&H (-0.029 vs -0.039 avg daily)
- Cost-aware act-or-hold gate cuts turnover ~15% and improves CVaR on **19/19 symbols**,
  but trades some mean Sharpe — the classic risk/return trade-off, now measurable

Highlights:
- **IDX Composite is the most predictable market in the set** — meta sharpe **1.06**
  vs B&H 0.51 over ~17 years of out-of-sample days; holds across ALL model families.
- Meta-selection wins where it should: IDX (1.06, best overall), Shanghai (0.64),
  Nifty (0.53) — it adapts per market instead of betting on one architecture.
- Honest lesson: a simple single LightGBM is hard to beat across the board; complex
  ensembles only pay off under regime shifts.
- Direction prediction on liquid daily indices is *near-efficient* — expect small
  edges and treat any big backtest number with suspicion.

## Roadmap

- [~] RL agent (PPO) on top of features — next up
- [ ] Open-weight LLM feature: news/macro reasoning via local models
- [x] Regime-gated MoE routing (RG-ResMoE method) — v2
- [ ] Calibration-per-time-to-expiry study (Kalshi method, arXiv 2607.14430)
- [ ] FlowLOB synthetic LOB augmentation

## License

MIT — use freely, cite kindly, trade carefully.
