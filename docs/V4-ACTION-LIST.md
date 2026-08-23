# V4 ACTION LIST — Synthesized from 3 Paper Deep-Dives
**Generated:** 2026-08-23 | **Sources:** arXiv 2608.19389 (CLP-RL), 2608.13096 (FlowLOB), 2608.07690 (Order Imbalance Skew/Width)
**Full digests:** `~/research/paper-digests/` (3 files, ~26KB total)

## Konteks Honest
Ketiga paper BUKAN sumber sinyal langsung (tidak ada price-prediction alpha yang bisa dicopy).
Nilainya: metodologi evaluasi, struktur keputusan cost-aware, dan fitur mikrostruktur murah.
Baseline realistis tetap: akurasi arah 50–53%; edge kecil = normal, angka besar = curiga.

---

## RANKED TOP-5 ACTIONS untuk v4

### 1. CVaR 5% + multi-seed eval di harness [ADOPT-NOW | S]
**Sumber:** 19389-A. Tambahkan CVaR 5% di samping Sharpe pada laporan OOS; latih ≥3 seed per
config, laporkan band antar-seed sebagai unit replikasi.
**Kenapa #1:** mengubah cara kita MENILAI semua eksperimen berikutnya — termasuk menangkap
perbaikan tail (mean −4 / CVaR +27) yang Sharpe-only miss. Murah, zero-risk.
**File target:** `src/train.py` + `src/train_v3_meta.py`.

### 2. Signed flow-imbalance features (coarse) [ADOPT-NOW | S]
**Sumber:** 07690-A. Rolling up/down volume ratio, CLV (close-location-value), signed
volume imbalance — horizon harian dari OHLCV yang sudah ada.
**Kenapa:** orthogonal terhadap momentum features yang sudah ada; auto-dievaluasi harness;
kalau null, buang tanpa drama. JANGAN tick-grade OFI (butuh data L2 yang tidak kita punya).

### 3. Cost-aware act-or-hold gate [ADOPT-NOW | S–M]
**Sumber:** 19389-B. Ganti threshold statis `p_up > 0.55` dengan: carry posisi kemarin
kecuali `expected_edge > roundtrip_cost_estimate`.
**Catatan:** ini juga template untuk Polymarket engine (inaction region = abstain yang
learned, bukan hard-coded). Perlu estimasi cost per trade dulu (spread + slippage historis).

### 4. Pooling rule: continuous signed features, bukan model per-regime [ADOPT-NOW | S design]
**Sumber:** 07690-D. Standing rule feature engineering: kalau mau split regime, pakai
fitur kontinu signed — jangan refactor jadi N model per-regime.
**Kenapa:** mencegah refactor mahal yang foreseeable; konsisten dengan pelajaran v2→v3
(regime routing via MoE kalah vs meta-selection sederhana).

### 5. Drift monitoring W1/KS antar-refit [TEST-LATER | S]
**Sumber:** 13096-C. Monitor marginal drift feature/target/prediction antar window tahunan.
Jalan kalau #1 sudah masuk (infra reporting sama). Bukan urgent tapi murah setelahnya.

---

## BACKLOG (TEST-LATER)
- Counterfactual feature-sanity check (clamp feature ke tail 5%, cek conditional outcome rate) [13096-B] — paling transferable, tapi butuh harness tambahan
- Leave-one-symbol-out generalization pass [13096-D] — kalau universe mau diperluas
- Imbalance-conditioned effective carry dalam gate #3 [07690-B/C]
- Per-decision edge ledger: pisahkan flow-reading vs inventory management [07690-E]

## DO-NOT-IMPORT (jangan buang waktu)
- ❌ LP fee harvesting sbg alpha (19389: referensinya sendiri bilang LP UniswapV3 rug rata-rata)
- ❌ Synthetic LOB augmentation untuk training (bias simulator bocor ke label t+1) [13096]
- ❌ Flow matching / sampler tech (irrelevan untuk GBT) [13096]
- ❌ Market-making leg lengkap / win-curve estimation (no data quotes/fills) [07690]
- ❌ Asumsi GBM stasioner environment 19389

## URUTAN EKSEKUSI SARAN
Minggu depan (post-cron riset Senin): #1 → #2 (paralel, dua-duanya S, satu PR masing-masing)
→ jalankan ulang walk-forward penuh → baru #3 kalau #1/#2 stabil.
Verdict gate Polymarket (n≥10) jalan sendiri — jangan campur siklusnya.
