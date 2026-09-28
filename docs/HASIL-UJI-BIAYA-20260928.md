# Uji biaya: kenapa idx-oracle tidak bisa dipakai retail apa adanya

**Tanggal:** 28 Sep 2026 · **Perintah:** `src/threshold_sweep.py`, `src/horizon_test.py`
**Data:** 18 simbol, walk-forward jujur (expanding window, min 5y train, refit tahunan, embargo 5 hari)

## Pertanyaan

Edge idx-oracle nyata secara statistik (15/18 simbol mengalahkan buy & hold pada Sharpe).
Tapi ia berdagang ~80 kali per tahun. **Apakah edge itu bertahan setelah biaya?**

## Hasil: tidak, dan menaikkan ambang tidak menolong

Strategi: long saat `p_up > ambang`, flat saat tidak. Biaya dinyatakan sebagai
**titik impas per sisi** — biaya maksimum yang masih membuat strategi untung.

| Ambang | Transisi/thn | Return/thn | Sharpe | Impas bps/sisi |
|---|---|---|---|---|
| 0,55 | 69 | 8,7% | 0,52 | **6,3** |
| 0,60 | 54 | 5,9% | 0,41 | 5,4 |
| 0,65 | 36 | 3,8% | 0,34 | 5,0 |
| 0,70 | 22 | 2,6% | 0,28 | 5,4 |
| 0,75 | 12 | 1,5% | 0,22 | 5,5 |
| 0,80 | 6 | −0,0% | 0,05 | 1,6 |

**Menaikkan ambang 0,55 → 0,75 memotong transisi 5,8×, tapi return ikut turun 5,8×.**
Titik impas biaya hampir tidak bergerak (6,3 → 5,5 bps). Jadi masalahnya **bukan
frekuensi** — masalahnya rasio return-per-transaksi.

## Angka yang menjelaskan semuanya

> Setiap transaksi menghasilkan **~6 bps kotor**.
> Biaya retail IDX/ETF: **15–30 bps per sisi**.

Rasio 2,5–5× terlalu besar untuk ditutup oleh pemilihan waktu yang lebih baik.
Bahkan di sisi maker (0–1 bps), marginnya tipis dan belum bisa dibedakan dari derau.

## Hipotesis yang masih terbuka: horizon lebih panjang

Autokorelasi return **5 hari = +0,789** (vs 1 hari = −0,030). Artinya tren bertahan
beberapa hari. Kalau sinyal yang sama dipakai untuk posisi 5 hari, transisi turun ~5×
**tanpa menuntut keyakinan lebih tinggi** — berbeda dari menaikkan ambang, yang menukar
frekuensi dengan akurasi.

Uji ini dijalankan oleh `src/horizon_test.py` (target = arah 1/3/5/10 hari ke depan).
Hasilnya di `results/horizon-test.csv`. Kalau titik impas naik ke ≥15 bps, ini jalan
keluar yang nyata; kalau tidak, kesimpulannya tetap: **pasar ini tidak memberi edge yang
cukup besar untuk biaya retail.**

## Yang TIDAK disimpulkan dari uji ini

- Bukan berarti modelnya buruk. Akurasinya 51–54% konsisten, di atas acak.
- Bukan berarti pasar efisien sempurna. Edge ada, hanya lebih kecil dari biaya.
- Bukan berarti harus berhenti. Berarti **jangan pakai frekuensi harian dengan biaya retail**.

---

## Hasil uji horizon (dijalankan, bukan hipotesis lagi)

`src/horizon_test.py` — 4 horizon × 18 simbol = 72 lintasan walk-forward:

| Horizon | Transisi/thn | Return/thn | Sharpe | Impas bps/sisi |
|---|---|---|---|---|
| 1 hari | 69 | 8,5% | 0,50 | 6,2 |
| 3 hari | 53 | 5,9% | 0,37 | 5,5 |
| 5 hari | 45 | 4,0% | 0,28 | 4,5 |
| **10 hari** | 35 | 5,7% | 0,33 | **8,0** |

**Horizon 10 hari terbaik (8,0 bps/sisi) tapi tetap ~2× di bawah batas retail terendah
(15 bps).** Hipotesis "horizon lebih panjang menolong" **tidak terbukti cukup** — arahnya
benar (impas naik dari 6,2 ke 8,0) tapi besarnya tidak cukup.

## Kesimpulan: semua jalan sudah diuji

| Jalan | Hasil |
|---|---|
| Menaikkan ambang keyakinan (0,55→0,80) | impas tetap 5–6 bps |
| Memperpanjang horizon (1→10 hari) | impas maksimum 8 bps |
| **Belum diuji: sisi maker** | secara definisi biaya 0–1 bps |

Edge idx-oracle **nyata** — 15/18 simbol mengalahkan buy & hold, akurasi 51–54% konsisten.
Tapi ukurannya **6–8 bps per sisi**, sementara biaya retail **15–30 bps per sisi**.
Selisihnya 2–5×, dan tidak ada kombinasi parameter yang menutupnya.

**Satu-satunya jalan yang belum diuji adalah sisi maker** (limit order, biaya ~0–1 bps).
Itu kandidat berikutnya, tapi ia mengubah sifat strategi: limit order tidak selalu terisi,
dan biaya tidak-terisi itu harus diukur, bukan diasumsikan.
