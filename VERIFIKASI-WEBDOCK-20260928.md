# idx-oracle di Webdock: verifikasi reproducible + temuan biaya

**Tanggal:** 28 Sep 2026 · **Sumber:** Tencent `~/projects/idx-oracle` → `/home/admin/idx-oracle`

## Kenapa dipindah

Perintah user: *"pakai tapi pindahkan ke Webdock"*. Tencent tetap read-only; sekarang
salinan kerja ada di Webdock dan bisa dijalankan/diubah tanpa menyentuh Tencent.

## Reproducible — dijalankan dari nol di Webdock

| Langkah | Hasil |
|---|---|
| `git clone` | 3 commit, repo publik GitHub `verrysimatupang99/idx-oracle` |
| `pip install` | pandas 3.0.6, numpy 2.5.3, sklearn 1.9.1, lightgbm 4.7.0, pyarrow 25.0.1 |
| `src/data.py` | **21/22 simbol OK** — IDX Composite 36,5 thn (8.878 baris), S&P 36,7 thn |
| `src/features.py` | 18 simbol diproses, 24 fitur/simbol |
| `src/train.py` | walk-forward LightGBM, **11/18 simbol mengalahkan buy & hold** pada Sharpe |

Hasil tersimpan di `results/baseline_lgbm.csv`. Klaim Tencent **terverifikasi**: IDX
Composite sharpe 1,034 vs B&H 0,585.

## Temuan yang membatasi: edge tidak bertahan setelah biaya

Strategi yang diuji adalah *long-when-confident* (`p_up > 0.55`, selain itu flat).
Coverage ~40% artinya posisi berubah ~120 kali per tahun. Dengan biaya retail
IDX/ETF **0,20%/sisi** (konservatif):

| Simbol | Coverage | Sharpe | Return/thn | Transisi/thn | Biaya/thn | Bersih |
|---|---|---|---|---|---|---|
| GC_F | 0,39 | 1,13 | +12,5% | 120 | 24,0% | **−11,5%** |
| idx_JKSE | 0,48 | 1,03 | +13,5% | 126 | 25,2% | **−11,6%** |
| 000001_SS | 0,41 | 0,68 | +10,6% | 121 | 24,3% | **−13,7%** |
| idx_NDX | 0,44 | 0,64 | +12,2% | 124 | 24,9% | **−12,6%** |

**Semua simbol teratas menjadi negatif setelah biaya.** Ini pola yang sama dengan dua
sistem lain di mesin ini: GMGN (friksi 6% vs TP +100%), live-trading crypto
(taker −0,033R), Polymarket (BE 77% vs WR 79% tapi margin tipis).

### Yang belum diuji dan bisa mengubah kesimpulan

1. **Ambang lebih tinggi.** `p_up > 0.55` menghasilkan ~120 transisi/tahun. Ambang 0,65
   atau 0,70 akan memotong transisi drastis — belum diukur apakah return ikut turun
   lebih cepat dari biayanya. Ini uji paling murah dan paling menentukan.
2. **Biaya nyata IDX.** 0,20%/sisi adalah asumsi; broker retail Indonesia berkisar
   0,15%–0,30%. Titik impas ada di ~0,10%/sisi untuk JKSE.
3. **Horizon lebih panjang.** Model memprediksi arah 1 hari. Horizon 5–10 hari akan
   memotong transisi ~5× dengan mengorbankan akurasi — trade-off belum diukur.

## Cara menjalankan

```bash
cd /home/admin/idx-oracle
.venv/bin/python src/data.py        # unduh 21 simbol (cache di data/raw/)
.venv/bin/python src/features.py    # 24 fitur, leak-free
.venv/bin/python src/train.py       # walk-forward, tulis results/baseline_lgbm.csv
.venv/bin/python src/train_v4.py    # varian v4 (multi-seed, CVaR5, cost-aware gate)
```

`src/train_v4.py` sudah punya *cost-aware gate* — belum dijalankan di Webdock;
itu kandidat pertama untuk uji ambang biaya di atas.
