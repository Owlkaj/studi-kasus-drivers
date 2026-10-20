"""
Multi-Sensor Satellite AI
Klasifikasi Penyebab Kerusakan Vegetasi Pesisir dan Hutan Penyangga Bengkulu
Menggunakan Fusi Data Radar (Sentinel-1) dan Optik (Sentinel-2)

Tugas kuliah: "Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI"
Referensi teori: materi kuliah praktisi Dr. Bart Slagter (Wageningen University & Research).

PENTING: seluruh citra pada skrip ini adalah DATA SINTETIS (simulasi) untuk keperluan akademik.
Angka akurasi menggambarkan perilaku pada data simulasi, bukan performa pada citra satelit nyata.

Alur program
  Tahap 1  Generator patch sintetis (800 sampel, 8x8 piksel, 6 band, 4 kelas imbalanced)
  Tahap 2  Simulasi tutupan awan tropis pada 30% sampel (hanya Band 1-4 optik yang rusak)
  Tahap 3  Eksperimen: Skenario 1 (Optik saja) vs Skenario 2 (Fusi Optik + Radar)
  Tahap 4  Evaluasi: Overall Accuracy vs Macro F1-Score + Classification Report per kelas
  Tahap 5  Visualisasi: hasil_analisis_satelit.png

Cara menjalankan:  python multisensor_driver_ai.py
"""

from __future__ import annotations

import argparse
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")  # backend tanpa layar agar bisa dijalankan di server / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# KONFIGURASI
# =============================================================================
SEED = 42
N_SAMPEL = 800
UKURAN_PATCH = 8          # 8 x 8 piksel (Sentinel 10 m -> sekitar 80 m x 80 m di lapangan)
FRAKSI_AWAN = 0.30
BAND = ["Red", "Green", "Blue", "NIR", "VV", "VH"]   # indeks 0-3 optik (S2), 4-5 radar (S1)

KELAS = [
    "Perkebunan_Skala_Kecil",
    "Abrasi_dan_Banjir_Pesisir",
    "Jalan_dan_Penebangan_Liar",
    "Pertambangan_Terbuka",
]
PROPORSI = [0.45, 0.25, 0.18, 0.12]       # dataset imbalanced
JENIS_AWAN = {0: "Cerah", 1: "Awan tebal", 2: "Kabut tipis", 3: "Bayangan awan"}

OUTPUT_GAMBAR = "hasil_analisis_satelit.png"


# =============================================================================
# TAHAP 1 - GENERATOR PATCH SINTETIS MULTI-SENSOR
# Optik dalam reflektansi (0-1); radar dalam desibel (dB).
# =============================================================================
def _blur(a: np.ndarray, iters: int = 1) -> np.ndarray:
    """Pelembutan 3x3 sederhana (rata-rata tetangga) untuk membuat medan acak yang halus."""
    h, w = a.shape
    for _ in range(iters):
        p = np.pad(a, 1, mode="edge")
        a = sum(p[i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9.0
    return a


def _medan_halus(rng: np.random.Generator, iters: int = 2) -> np.ndarray:
    """Medan acak halus ternormalisasi (rata-rata 0, simpangan baku 1)."""
    f = _blur(rng.normal(size=(UKURAN_PATCH, UKURAN_PATCH)), iters)
    return (f - f.mean()) / (f.std() + 1e-9)


def _grid_koordinat() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:UKURAN_PATCH, 0:UKURAN_PATCH].astype(float)
    return yy, xx


def _mask_grid(rng: np.random.Generator) -> np.ndarray:
    """Pola kotak teratur (jalur antar petak kebun, periode 4 piksel); sebagian hanya satu arah."""
    yy, xx = _grid_koordinat()
    r0, c0 = rng.integers(0, 4, size=2)
    baris = ((yy.astype(int) + r0) % 4) == 3
    kolom = ((xx.astype(int) + c0) % 4) == 3
    r = rng.random()
    if r < 0.80:
        return baris | kolom
    return baris if r < 0.90 else kolom


def _mask_garis(rng: np.random.Generator, jumlah: int = 1) -> np.ndarray:
    """Pola garis memanjang (jalan angkut / jalur penebangan) dengan orientasi acak."""
    yy, xx = _grid_koordinat()
    mask = np.zeros_like(yy, dtype=bool)
    c = (UKURAN_PATCH - 1) / 2
    for _ in range(jumlah):
        theta = rng.uniform(0, np.pi)
        cx, cy = c + rng.uniform(-2.5, 2.5), c + rng.uniform(-2.5, 2.5)
        jarak = np.abs((xx - cx) * np.sin(theta) - (yy - cy) * np.cos(theta))
        mask |= jarak <= rng.uniform(0.6, 1.1)
    return mask


def _mask_air(rng: np.random.Generator, fraksi: float) -> np.ndarray:
    """Pembagian patch darat-air oleh garis pantai/genangan; `fraksi` = porsi piksel air."""
    yy, xx = _grid_koordinat()
    theta = rng.uniform(0, 2 * np.pi)
    proyeksi = xx * np.cos(theta) + yy * np.sin(theta) + rng.normal(0, 0.5, size=xx.shape)
    ambang = np.quantile(proyeksi, 1 - fraksi)
    return proyeksi >= ambang


def _isi(mask: np.ndarray, fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Komposisi per band: nilai latar (bg) dan objek (fg); fg/bg berukuran (n_band,)."""
    return np.where(mask[None, :, :], fg[:, None, None], bg[:, None, None])


def _kontras(fg: np.ndarray, bg: np.ndarray, c: float) -> np.ndarray:
    """Kurangi kontras objek terhadap latar (c=1 jelas, c<1 samar): sumber ambiguitas antar kelas."""
    return bg + c * (fg - bg)


def _patch_perkebunan(rng) -> tuple[np.ndarray, np.ndarray]:
    """NIR sedang, pola kotak teratur (jalur antar petak)."""
    muda = rng.random() < 0.3   # kebun muda: tajuk jarang, tanah lebih terlihat
    kanopi = np.array([rng.normal(0.075, 0.015) * (1.5 if muda else 1.0), 0.11, 0.05,
                       rng.normal(0.38, 0.07) * (0.85 if muda else 1.0)])
    jalur = np.array([0.15, 0.16, 0.10, 0.24])
    rad_kanopi = np.array([rng.normal(-10.0, 1.2), rng.normal(-16.0, 1.2)])
    rad_jalur = np.array([-12.5, -19.0])
    c = rng.uniform(0.5, 1.0)
    m = _mask_grid(rng)
    opt = _isi(m, _kontras(jalur, kanopi, c), kanopi)
    rad = _isi(m, _kontras(rad_jalur, rad_kanopi, c), rad_kanopi)
    return opt, rad


def _patch_abrasi(rng) -> tuple[np.ndarray, np.ndarray]:
    """Genangan air: NIR sangat rendah dan hamburan balik radar VV/VH sangat rendah."""
    air = _mask_air(rng, rng.uniform(0.35, 1.0))
    opt_air = np.array([rng.uniform(0.05, 0.14), rng.uniform(0.07, 0.11),
                        rng.uniform(0.06, 0.10), rng.uniform(0.02, 0.08)])
    if rng.random() < 0.5:   # sisa darat berupa pasir pantai
        opt_darat, rad_darat = np.array([0.30, 0.28, 0.24, 0.22]), np.array([-15.0, -24.0])
    else:                    # sisa darat berupa vegetasi rusak
        opt_darat, rad_darat = np.array([0.08, 0.11, 0.06, 0.33]), np.array([-9.0, -15.0])
    opt = _isi(air, opt_air, opt_darat)
    rad = _isi(air, np.array([rng.normal(-21.0, 1.8), rng.normal(-28.0, 1.8)]), rad_darat)
    return opt, rad


def _patch_jalan(rng) -> tuple[np.ndarray, np.ndarray]:
    """Latar vegetasi dengan jalur tanah terbuka yang sempit dan memanjang."""
    latar = np.array([0.06, 0.10, 0.05, rng.normal(0.43, 0.06)])
    garis = np.array([0.22, 0.20, 0.16, 0.24])
    rad_latar = np.array([rng.normal(-9.0, 1.0), rng.normal(-15.0, 1.0)])
    c = rng.uniform(0.45, 1.0)
    m = _mask_garis(rng, jumlah=2 if rng.random() < 0.2 else 1)
    opt = _isi(m, _kontras(garis, latar, c), latar)
    rad = _isi(m, _kontras(np.array([-15.0, -21.5]), rad_latar, c), rad_latar)
    return opt, rad


def _patch_tambang(rng) -> tuple[np.ndarray, np.ndarray]:
    """Tanah terbuka: Red/Blue tinggi, NIR sangat rendah; teras tambang; kadang kolam atau revegetasi."""
    tanah = np.array([rng.normal(0.27, 0.05), 0.25, rng.normal(0.20, 0.04), rng.normal(0.11, 0.035)])
    rad_tanah = np.array([rng.normal(-11.0, 1.4), rng.normal(-19.0, 1.4)])
    w = rng.uniform(0.0, 0.45) if rng.random() < 0.4 else 0.0   # revegetasi sebagian
    tanah = (1 - w) * tanah + w * np.array([0.07, 0.11, 0.05, 0.40])
    rad_tanah = (1 - w) * rad_tanah + w * np.array([-9.5, -15.5])
    teras = _medan_halus(rng, 2)
    opt = np.repeat(tanah[:, None, None], UKURAN_PATCH, axis=1).repeat(UKURAN_PATCH, axis=2)
    opt = opt * (1 + 0.15 * teras[None])
    rad = np.repeat(rad_tanah[:, None, None], UKURAN_PATCH, axis=1).repeat(UKURAN_PATCH, axis=2) + 1.5 * teras[None]
    if rng.random() < 0.25:  # kolam bekas galian
        kolam = _medan_halus(rng, 2) > np.quantile(_medan_halus(rng, 2), rng.uniform(0.7, 0.9))
        kolam = _blur(kolam.astype(float), 1) > 0.5
        opt = np.where(kolam[None], np.array([0.06, 0.08, 0.08, 0.03])[:, None, None], opt)
        rad = np.where(kolam[None], np.array([-20.0, -27.0])[:, None, None], rad)
    return opt, rad


_GENERATOR = [_patch_perkebunan, _patch_abrasi, _patch_jalan, _patch_tambang]


def generate_patches(n: int = N_SAMPEL, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Tahap 1. Mengembalikan X (n, 6, 8, 8) dan y (n,).
    Urutan channel: Red, Green, Blue, NIR (Sentinel-2) lalu VV, VH (Sentinel-1).
    """
    rng = np.random.default_rng(seed)
    jumlah = np.round(np.array(PROPORSI) * n).astype(int)
    jumlah[0] += n - jumlah.sum()   # pastikan total tepat n
    X = np.zeros((n, 6, UKURAN_PATCH, UKURAN_PATCH), dtype=np.float32)
    y = np.zeros(n, dtype=int)

    i = 0
    for k, (gen, jml) in enumerate(zip(_GENERATOR, jumlah)):
        for _ in range(jml):
            opt, rad = gen(rng)
            # Optik: gain pencahayaan/fenologi per patch + derau multiplikatif per piksel
            gain = rng.lognormal(0, 0.18) * rng.lognormal(0, 0.06, size=(4, 1, 1))
            opt = opt * gain * rng.lognormal(0, 0.14, size=opt.shape)
            # Radar: offset per patch (kelembapan/sudut datang) + speckle per piksel (dB)
            rad = rad + rng.normal(0, 1.6) + rng.normal(0, [[[2.4]], [[2.6]]], size=rad.shape)
            X[i, :4] = np.clip(opt, 0.005, 1.0)
            X[i, 4:] = rad
            y[i] = k
            i += 1

    acak = rng.permutation(n)
    return X[acak], y[acak]


# =============================================================================
# TAHAP 2 - SIMULASI TUTUPAN AWAN TROPIS
# Hanya Band 1-4 (optik) yang diganggu; Band 5-6 (radar) tidak disentuh.
# =============================================================================
def simulasi_awan(X: np.ndarray, fraksi: float = FRAKSI_AWAN, seed: int = SEED):
    """
    Mengembalikan (X_berawan, jenis_awan). jenis_awan: 0 cerah, 1 awan tebal,
    2 kabut tipis, 3 bayangan awan. Radar dijamin tetap identik dengan aslinya.
    """
    rng = np.random.default_rng(seed + 1000)
    n = len(X)
    Xc = X.copy()
    jenis = np.zeros(n, dtype=int)
    terdampak = rng.choice(n, size=int(round(fraksi * n)), replace=False)

    for i in terdampak:
        tipe = rng.choice([1, 2, 3], p=[0.55, 0.25, 0.20])
        jenis[i] = tipe
        opt = Xc[i, :4].astype(float)
        medan = _medan_halus(rng, 2)
        if tipe == 3:   # bayangan awan: gelap, bukan putih
            faktor = np.clip(rng.uniform(0.25, 0.5) + 0.05 * medan, 0.15, 0.7)
            opt = opt * faktor[None]
        else:
            alpha = rng.uniform(0.85, 1.0) if tipe == 1 else rng.uniform(0.35, 0.70)
            alpha = np.clip(alpha + 0.04 * medan, 0, 1)
            dasar = rng.uniform(0.50, 0.75)
            reflek_awan = np.array([dasar, dasar * rng.uniform(0.95, 1.05),
                                    dasar * rng.uniform(0.95, 1.05), dasar * rng.uniform(0.8, 1.0)])
            reflek_awan = reflek_awan[:, None, None] * (1 + 0.04 * medan[None])
            opt = (1 - alpha)[None] * opt + alpha[None] * reflek_awan
        Xc[i, :4] = np.clip(opt, 0.005, 1.0)

    assert np.array_equal(Xc[:, 4:], X[:, 4:]), "Radar tidak boleh berubah oleh awan"
    return Xc, jenis


# =============================================================================
# EKSTRAKSI FITUR (termasuk fitur spasial kontekstual)
# =============================================================================
def _autokorelasi(img: np.ndarray, lag: int) -> float:
    x = img - img.mean()
    v = (x * x).mean() + 1e-12
    return 0.5 * ((x * np.roll(x, lag, axis=0)).mean() + (x * np.roll(x, lag, axis=1)).mean()) / v


def fitur_spasial(img: np.ndarray) -> list[float]:
    """
    Fitur spasial kontekstual dari satu band:
      - energi gradien   : seberapa banyak tepi/tekstur
      - koherensi arah   : dominasi satu orientasi tepi (tinggi untuk garis/jalan, rendah untuk kotak)
      - periodisitas     : autokorelasi lag-4 dikurangi lag-2 (tinggi untuk pola kotak periode 4 piksel)
    """
    gy, gx = np.gradient(img)
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    koherensi = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-12)
    periodisitas = _autokorelasi(img, 4) - _autokorelasi(img, 2)
    return [float(np.hypot(gx, gy).mean()), float(koherensi), float(periodisitas)]


def _fitur_optik(p: np.ndarray) -> dict:
    r, g, b, n = p[0], p[1], p[2], p[3]
    ndvi = (n - r) / (n + r + 1e-6)
    ndwi = (g - n) / (g + n + 1e-6)
    f = {}
    for nama, band in zip(["Red", "Green", "Blue", "NIR"], [r, g, b, n]):
        f[f"{nama}_mean"], f[f"{nama}_std"] = band.mean(), band.std()
    f["NDVI_mean"], f["NDVI_std"], f["NDWI_mean"] = ndvi.mean(), ndvi.std(), ndwi.mean()
    for nama, band in (("NIR", n), ("Red", r)):
        f[f"{nama}_grad"], f[f"{nama}_koheren"], f[f"{nama}_periodik"] = fitur_spasial(band)
    return f


def _fitur_radar(p: np.ndarray) -> dict:
    vv, vh = p[4], p[5]
    f = {"VV_mean": vv.mean(), "VV_std": vv.std(), "VH_mean": vh.mean(), "VH_std": vh.std(),
         "VV_minus_VH": (vv - vh).mean(), "VV_frac_rendah": (vv < -18).mean()}
    for nama, band in (("VV", vv), ("VH", vh)):
        f[f"{nama}_grad"], f[f"{nama}_koheren"], f[f"{nama}_periodik"] = fitur_spasial(band)
    return f


def ekstrak_fitur(X: np.ndarray, pakai_radar: bool) -> pd.DataFrame:
    """Skenario 1 (pakai_radar=False): 17 fitur dari 4 band optik. Skenario 2: + 12 fitur radar."""
    baris = []
    for p in X:
        f = _fitur_optik(p)
        if pakai_radar:
            f.update(_fitur_radar(p))
        baris.append(f)
    return pd.DataFrame(baris)


# =============================================================================
# TAHAP 3 & 4 - EKSPERIMEN DAN EVALUASI
# =============================================================================
def buat_model(jenis: str, seed: int):
    if jenis == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500,
                                           early_stopping=False, random_state=seed))
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def jalankan_skenario(F: pd.DataFrame, y, idx_tr, idx_te, jenis: str, seed: int) -> dict:
    model = buat_model(jenis, seed).fit(F.iloc[idx_tr], y[idx_tr])
    pred = model.predict(F.iloc[idx_te])
    yt = y[idx_te]
    return dict(
        y_pred=pred, y_true=yt,
        accuracy=accuracy_score(yt, pred),
        macro_f1=f1_score(yt, pred, average="macro", zero_division=0),
        f1_kelas=f1_score(yt, pred, average=None, labels=range(len(KELAS)), zero_division=0),
        cm=confusion_matrix(yt, pred, labels=range(len(KELAS))),
    )


def eksperimen(seed: int, jenis: str, n: int = N_SAMPEL, fraksi_awan: float = FRAKSI_AWAN):
    """Satu putaran penuh: generate -> awan -> split 80/20 -> 3 model. Dipakai juga untuk pengulangan."""
    X, y = generate_patches(n, seed)
    X_awan, jenis_awan = simulasi_awan(X, fraksi_awan, seed)
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)

    F_optik_cerah = ekstrak_fitur(X, pakai_radar=False)        # referensi: langit cerah
    F_optik_awan = ekstrak_fitur(X_awan, pakai_radar=False)    # Skenario 1
    F_fusi_awan = ekstrak_fitur(X_awan, pakai_radar=True)      # Skenario 2

    hasil = {
        "Referensi: Optik (tanpa awan)": jalankan_skenario(F_optik_cerah, y, idx_tr, idx_te, jenis, seed),
        "Skenario 1: Optik Saja (berawan)": jalankan_skenario(F_optik_awan, y, idx_tr, idx_te, jenis, seed),
        "Skenario 2: Fusi Optik+Radar (berawan)": jalankan_skenario(F_fusi_awan, y, idx_tr, idx_te, jenis, seed),
    }
    return dict(X=X, X_awan=X_awan, y=y, jenis_awan=jenis_awan, idx_tr=idx_tr, idx_te=idx_te,
                hasil=hasil, n_fitur=(F_optik_awan.shape[1], F_fusi_awan.shape[1]))


def nama_pendek(k: str, lebar: int = 14) -> str:
    return "\n".join(textwrap.wrap(k.replace("_", " "), lebar))


# =============================================================================
# TAHAP 5 - VISUALISASI
# =============================================================================
def _rgb(patch: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.stack([patch[0], patch[1], patch[2]], axis=-1) / 0.8, 0, 1) ** 0.55
    return rgb


def buat_gambar(R: dict, path: str) -> None:
    sns.set_theme(style="white", font_scale=0.95)
    X, X_awan, y, jenis = R["X"], R["X_awan"], R["y"], R["jenis_awan"]
    hasil = R["hasil"]
    nama_ref, nama_s1, nama_s2 = list(hasil.keys())

    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(2, 5, top=0.88, bottom=0.57, hspace=0.35, wspace=0.28)      # contoh patch
    gs_bawah = fig.add_gridspec(1, 3, top=0.46, bottom=0.07, wspace=0.40)              # metrik + confusion matrix
    fig.suptitle("Fusi Multi-Sensor Sentinel-1 (Radar) + Sentinel-2 (Optik): Penyebab Kerusakan Vegetasi di Bengkulu\n"
                 "(data sintetis untuk keperluan akademik)", fontsize=15, fontweight="bold", y=0.98)

    # (1) Contoh patch optik vs radar: 4 kelas cerah + 1 contoh berawan
    contoh = []
    for k in range(len(KELAS)):
        contoh.append((np.where((y == k) & (jenis == 0))[0][0], KELAS[k], False))
    idx_awan = np.where((y == 0) & (jenis == 1))[0]
    contoh.append((idx_awan[0] if len(idx_awan) else np.where(jenis == 1)[0][0], KELAS[0], True))

    for col, (i, k, berawan) in enumerate(contoh):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(_rgb(X_awan[i]), interpolation="nearest")
        ax.set_title(nama_pendek(k) + ("\n[tertutup awan tebal]" if berawan else ""), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Optik S2\n(RGB)", fontsize=11, fontweight="bold")
        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(X_awan[i, 4], cmap="gray", vmin=-25, vmax=-3, interpolation="nearest")
        ax2.set_xticks([]); ax2.set_yticks([])
        if col == 0:
            ax2.set_ylabel("Radar S1\n(VV, dB)", fontsize=11, fontweight="bold")
        if berawan:
            ax2.set_xlabel("radar tetap menembus awan", fontsize=9, color="#C0392B")

    # (2) Bar chart Accuracy vs Macro F1
    ax = fig.add_subplot(gs_bawah[0, 0])
    metrik = ["Overall Accuracy", "Macro F1-Score"]
    skenario = [(nama_ref, "#B8C4D6", "Referensi: Optik tanpa awan"),
                (nama_s1, "#E67E22", "Skenario 1: Optik saja (berawan)"),
                (nama_s2, "#1F77B4", "Skenario 2: Fusi Optik+Radar (berawan)")]
    lebar = 0.26
    for j, (kunci, warna, label) in enumerate(skenario):
        nilai = [hasil[kunci]["accuracy"], hasil[kunci]["macro_f1"]]
        pos = np.arange(2) + (j - 1) * lebar
        batang = ax.bar(pos, nilai, lebar, color=warna, label=label, edgecolor="white")
        for b, v in zip(batang, nilai):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(metrik, fontsize=11)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Skor")
    ax.set_title("Perbandingan Accuracy vs Macro F1-Score", fontweight="bold")
    ax.legend(loc="lower center", fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.32), ncol=1)
    sns.despine(ax=ax)

    # (3) Confusion matrix Skenario 1 dan Skenario 2
    label_kelas = [nama_pendek(k, 12) for k in KELAS]
    for col, (kunci, judul, cmap) in enumerate([(nama_s1, "Confusion Matrix\nSkenario 1: Optik Saja", "Oranges"),
                                                 (nama_s2, "Confusion Matrix\nSkenario 2: Fusi Multi-Sensor", "Blues")]):
        ax = fig.add_subplot(gs_bawah[0, 1 + col])
        cm = hasil[kunci]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax, linewidths=0.5, linecolor="white",
                    xticklabels=label_kelas, yticklabels=label_kelas, annot_kws={"fontsize": 12})
        ax.set_title(judul, fontweight="bold")
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Kelas sebenarnya" if col == 0 else "")
        ax.tick_params(axis="x", labelsize=8, rotation=0); ax.tick_params(axis="y", labelsize=8, rotation=0)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# PELAPORAN KONSOL
# =============================================================================
def garis(judul: str) -> None:
    print("\n" + "=" * 78 + f"\n{judul}\n" + "=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fusi Sentinel-1 + Sentinel-2 untuk atribusi penyebab kerusakan vegetasi.")
    ap.add_argument("--seed", type=int, default=SEED, help="seed acak (default 42)")
    ap.add_argument("--model", choices=["rf", "mlp"], default="rf", help="rf = RandomForest, mlp = Neural Network")
    ap.add_argument("--ulangi", type=int, default=5, help="jumlah pengulangan uji kestabilan (0 = lewati)")
    ap.add_argument("--output", default=OUTPUT_GAMBAR, help="nama file gambar keluaran")
    args = ap.parse_args()
    nama_model = "RandomForestClassifier" if args.model == "rf" else "MLPClassifier (Neural Network)"

    R = eksperimen(args.seed, args.model)
    y, jenis, idx_te = R["y"], R["jenis_awan"], R["idx_te"]
    hasil = R["hasil"]
    nama_ref, nama_s1, nama_s2 = list(hasil.keys())

    # ---------------------------------------------------------------- Tahap 1 & 2
    garis("TAHAP 1 - Dataset Sintetis Multi-Sensor")
    print(f"Bentuk data      : {R['X'].shape}  (sampel, band, tinggi, lebar)")
    print(f"Urutan band      : {BAND}  -> Band 1-4 = Sentinel-2 (optik), Band 5-6 = Sentinel-1 (radar)")
    dist = pd.Series(y).map(dict(enumerate(KELAS))).value_counts().reindex(KELAS)
    print("\nDistribusi kelas (imbalanced):")
    for k, v in dist.items():
        print(f"  {k:<28s}{v:>5d}  ({v / len(y) * 100:4.1f}%)")

    garis("TAHAP 2 - Simulasi Tutupan Awan Tropis")
    n_awan = int((jenis > 0).sum())
    print(f"Sampel terdampak awan : {n_awan} dari {len(y)} ({n_awan / len(y) * 100:.0f}%)")
    for kode in (1, 2, 3):
        print(f"  - {JENIS_AWAN[kode]:<14s}: {int((jenis == kode).sum())} sampel")
    print("Verifikasi: Band radar (5-6) identik sebelum/sesudah awan -> "
          f"{np.array_equal(R['X'][:, 4:], R['X_awan'][:, 4:])}")
    print(f"Split: {len(R['idx_tr'])} latih / {len(idx_te)} uji (80/20, stratified). "
          f"Sampel uji berawan: {int((jenis[idx_te] > 0).sum())}")
    print(f"Jumlah fitur: Skenario 1 = {R['n_fitur'][0]} (optik), Skenario 2 = {R['n_fitur'][1]} (optik + radar)")

    # ---------------------------------------------------------------- Tahap 3 & 4
    garis(f"TAHAP 3 & 4 - Perbandingan Model ({nama_model}) dan Evaluasi Metrik")
    tabel = pd.DataFrame({
        "Overall Accuracy": {k: v["accuracy"] for k, v in hasil.items()},
        "Macro F1-Score": {k: v["macro_f1"] for k, v in hasil.items()},
        "F1 Pertambangan": {k: v["f1_kelas"][3] for k, v in hasil.items()},
    })
    tabel["Selisih (Acc - MacroF1)"] = tabel["Overall Accuracy"] - tabel["Macro F1-Score"]
    print(tabel.round(3).to_string())

    s1, s2, ref = hasil[nama_s1], hasil[nama_s2], hasil[nama_ref]
    print(f"\nDampak awan pada optik : Accuracy {ref['accuracy']:.3f} -> {s1['accuracy']:.3f} "
          f"({(s1['accuracy'] - ref['accuracy']) * 100:+.1f} poin), "
          f"Macro F1 {ref['macro_f1']:.3f} -> {s1['macro_f1']:.3f} ({(s1['macro_f1'] - ref['macro_f1']) * 100:+.1f} poin)")
    print(f"Manfaat fusi radar     : Accuracy {s1['accuracy']:.3f} -> {s2['accuracy']:.3f} "
          f"({(s2['accuracy'] - s1['accuracy']) * 100:+.1f} poin), "
          f"Macro F1 {s1['macro_f1']:.3f} -> {s2['macro_f1']:.3f} ({(s2['macro_f1'] - s1['macro_f1']) * 100:+.1f} poin)")

    # Akurasi terpisah untuk sampel uji cerah vs berawan
    berawan_te = jenis[idx_te] > 0
    print("\nAkurasi pada sampel uji (dipisah berdasarkan kondisi atmosfer):")
    for nama, h in ((nama_s1, s1), (nama_s2, s2)):
        a_cerah = accuracy_score(h["y_true"][~berawan_te], h["y_pred"][~berawan_te])
        a_awan = accuracy_score(h["y_true"][berawan_te], h["y_pred"][berawan_te])
        print(f"  {nama:<42s} cerah: {a_cerah:.3f} | berawan: {a_awan:.3f}")

    # Penjelasan Accuracy vs Macro F1 dengan ilustrasi pembanding "selalu tebak kelas mayoritas"
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(R["idx_tr"]), 1)), y[R["idx_tr"]])
    yd = dummy.predict(np.zeros((len(idx_te), 1)))
    acc_d, f1_d = accuracy_score(y[idx_te], yd), f1_score(y[idx_te], yd, average="macro", zero_division=0)
    garis("MENGAPA MACRO F1-SCORE LEBIH JUJUR UNTUK KELAS MINORITAS?")
    print(textwrap.fill(
        "Overall Accuracy = proporsi prediksi benar dari SELURUH sampel, sehingga didominasi kelas besar "
        "(Perkebunan 45%). Macro F1-Score = rata-rata F1 tiap kelas dengan bobot SAMA, sehingga kelas kecil "
        "seperti Pertambangan_Terbuka (12%) memiliki suara yang setara dengan kelas besar.", 78))
    print(f"\nIlustrasi: model 'selalu menebak {KELAS[0]}' memperoleh Accuracy {acc_d:.3f} "
          f"tetapi Macro F1 hanya {f1_d:.3f}.")
    print(textwrap.fill(
        "Untuk atribusi penyebab deforestasi, kelas minoritas (tambang, jalan/penebangan liar) justru sering "
        "paling penting bagi penegakan hukum. Akurasi tinggi yang menutupi kegagalan pada kelas tersebut "
        "menyesatkan; selisih Accuracy - Macro F1 yang besar adalah tanda peringatan.", 78))

    for nama, h in ((nama_s1, s1), (nama_s2, s2)):
        garis(f"CLASSIFICATION REPORT - {nama}")
        print(classification_report(h["y_true"], h["y_pred"], target_names=KELAS, digits=3, zero_division=0))

    # ---------------------------------------------------------------- Uji kestabilan
    if args.ulangi > 0:
        garis(f"UJI KESTABILAN ({args.ulangi} pengulangan: dataset, awan, dan split berbeda)")
        rekap = {k: {"acc": [], "f1": [], "f1_tambang": []} for k in hasil}
        for r in range(args.ulangi):
            hr = eksperimen(args.seed + r, args.model)["hasil"] if r else hasil
            for k, v in hr.items():
                rekap[k]["acc"].append(v["accuracy"])
                rekap[k]["f1"].append(v["macro_f1"])
                rekap[k]["f1_tambang"].append(v["f1_kelas"][3])
        baris = {k: {"Overall Accuracy": f"{np.mean(v['acc']):.3f} ± {np.std(v['acc']):.3f}",
                     "Macro F1-Score": f"{np.mean(v['f1']):.3f} ± {np.std(v['f1']):.3f}",
                     "F1 Pertambangan": f"{np.mean(v['f1_tambang']):.3f} ± {np.std(v['f1_tambang']):.3f}"}
                 for k, v in rekap.items()}
        print(pd.DataFrame(baris).T.to_string())
        print("(rata-rata ± simpangan baku; set uji hanya 160 sampel dan Pertambangan hanya sekitar 19 sampel uji, "
              "sehingga satu kali split cukup berfluktuasi)")

    # ---------------------------------------------------------------- Visualisasi
    garis("TAHAP 5 - Visualisasi")
    buat_gambar(R, args.output)
    print(f"Gambar tersimpan: {args.output}")
    print("\nCatatan: seluruh hasil berasal dari data sintetis; jangan dibaca sebagai performa pada citra nyata.")


if __name__ == "__main__":
    main()
