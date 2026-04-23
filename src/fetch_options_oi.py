import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  BLACK-SCHOLES GAMMA
# ──────────────────────────────────────────────
def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Hitung gamma satu kontrak options pakai rumus Black-Scholes.

    S     = spot price underlying
    K     = strike price
    T     = time to expiry dalam satuan tahun
    r     = risk-free rate (desimal, misal 0.05)
    sigma = implied volatility (desimal, misal 0.20)

    Return: gamma (float), atau 0.0 kalau input invalid
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
#  FETCH & CLEAN OPTIONS DATA
# ──────────────────────────────────────────────
def fetch_and_clean_options(
    ticker_symbol: str = "SPY",
    days_back: int = 0,
    days_forward: int = 45,
) -> pd.DataFrame | None:
    """
    Ambil option chain SPY dari yfinance, filter expiration,
    bersihkan data, tambah kolom IV dan waktu ke expiry.

    Return: DataFrame bersih, atau None kalau gagal.
    """
    print(f"\n[FETCH] Mengambil data options {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    all_expirations = ticker.options

    if not all_expirations:
        print("  Tidak ada data options tersedia.")
        return None

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days_back)
    end_date   = today + datetime.timedelta(days=days_forward)

    filtered_exps = [
        e for e in all_expirations
        if start_date <= datetime.datetime.strptime(e, "%Y-%m-%d").date() <= end_date
    ]

    print(f"  Total expiration tersedia : {len(all_expirations)}")
    print(f"  Expiration dalam rentang  : {len(filtered_exps)}")

    records = []
    for exp_str in filtered_exps:
        try:
            chain = ticker.option_chain(exp_str)
            exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
            T = max((exp_date - today).days / 365.0, 1 / 365)  # min 1 hari

            for df_side, tipe in [(chain.calls, "Call"), (chain.puts, "Put")]:
                df_side = df_side.copy()
                df_side["tipe"]               = tipe
                df_side["tanggal_kedaluwarsa"] = exp_str
                df_side["T"]                  = T
                records.append(df_side)
        except Exception as e:
            print(f"  [SKIP] {exp_str}: {e}")

    if not records:
        return None

    raw = pd.concat(records, ignore_index=True)

    # ── Kolom yang dibutuhkan ──────────────────
    needed = ["strike", "openInterest", "impliedVolatility",
              "tipe", "tanggal_kedaluwarsa", "T"]
    raw = raw[[c for c in needed if c in raw.columns]].copy()

    # ── Cleaning ──────────────────────────────
    raw = raw.dropna(subset=["strike", "openInterest", "impliedVolatility"])
    raw = raw[raw["openInterest"] > 0]
    raw = raw[raw["impliedVolatility"] > 0]
    raw = raw[raw["impliedVolatility"] < 5.0]   # buang IV outlier ekstrem (>500%)
    raw["strike"] = raw["strike"].astype(float)

    print(f"  Baris bersih              : {len(raw)}")
    return raw.reset_index(drop=True)


# ──────────────────────────────────────────────
#  HITUNG GEX
# ──────────────────────────────────────────────
def compute_gex(df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    """
    Hitung GEX per baris, lalu agregasi per strike.

    Formula GEX per kontrak:
        GEX_call = +Gamma × OI × 100 × spot²  (dealer long gamma)
        GEX_put  = -Gamma × OI × 100 × spot²  (dealer short gamma)

    Satuan: dollar gamma (sensitivity terhadap 1% move underlying)
    """
    print("\n[GEX] Menghitung Black-Scholes Gamma...")

    gammas = []
    for _, row in df.iterrows():
        g = bs_gamma(
            S=spot,
            K=row["strike"],
            T=row["T"],
            r=r,
            sigma=row["impliedVolatility"],
        )
        gammas.append(g)

    df = df.copy()
    df["gamma"] = gammas

    # GEX per kontrak (dalam dollar gamma)
    # Kalikan spot² supaya satuan = $ sensitivity per 1% move
    multiplier = 100 * (spot ** 2) * 0.01  # 0.01 = 1%

    df["gex"] = np.where(
        df["tipe"] == "Call",
        +df["gamma"] * df["openInterest"] * multiplier,
        -df["gamma"] * df["openInterest"] * multiplier,
    )

    # Agregasi per strike
    gex_by_strike = (
        df.groupby("strike", as_index=False)["gex"]
        .sum()
        .rename(columns={"gex": "net_gex"})
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # Tandai zona flip
    gex_by_strike["zona"] = np.where(
        gex_by_strike["net_gex"] >= 0, "Positive GEX", "Negative GEX"
    )

    print(f"  Jumlah strike unik: {len(gex_by_strike)}")
    print(f"  GEX total (net)   : ${gex_by_strike['net_gex'].sum():,.0f}")

    # Temukan Gamma Flip Level
    sorted_df = gex_by_strike.sort_values("strike")
    flip_candidates = sorted_df[
        sorted_df["net_gex"].diff().apply(lambda x: x != 0) &
        (sorted_df["net_gex"].shift(1) * sorted_df["net_gex"] < 0)
    ]
    if not flip_candidates.empty:
        flip_level = flip_candidates.iloc[0]["strike"]
        print(f"  Gamma Flip Level  : ~${flip_level:.2f}")
    else:
        flip_level = None
        print("  Gamma Flip Level  : tidak ditemukan dalam range ini")

    return gex_by_strike, flip_level


# ──────────────────────────────────────────────
#  VISUALISASI
# ──────────────────────────────────────────────
def plot_gex(gex_df: pd.DataFrame, spot: float, flip_level: float | None,
             ticker_symbol: str = "SPY") -> None:
    """
    Bar chart horizontal GEX per strike.
    Warna hijau = Positive GEX (dealer long gamma / resistance)
    Warna merah = Negative GEX (dealer short gamma / accelerator)
    """
    # Filter: tampilkan strike dalam ±10% dari spot supaya chart readable
    lower = spot * 0.90
    upper = spot * 1.10
    plot_df = gex_df[(gex_df["strike"] >= lower) & (gex_df["strike"] <= upper)].copy()

    if plot_df.empty:
        print("[PLOT] Tidak ada data dalam range ±10% spot.")
        return

    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in plot_df["net_gex"]]

    fig, ax = plt.subplots(figsize=(14, max(8, len(plot_df) * 0.28)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars = ax.barh(
        plot_df["strike"].astype(str),
        plot_df["net_gex"] / 1e6,   # satuan juta
        color=colors,
        edgecolor="none",
        height=0.75,
    )

    # ── Garis spot price ──
    # Cari posisi index dari strike terdekat ke spot
    strikes_arr = plot_df["strike"].values
    closest_idx = np.argmin(np.abs(strikes_arr - spot))
    ax.axhline(y=closest_idx, color="#FFD700", linewidth=1.5,
               linestyle="--", alpha=0.8, label=f"Spot: ${spot:.2f}")

    # ── Garis gamma flip ──
    if flip_level is not None:
        flip_mask = plot_df["strike"] == flip_level
        if flip_mask.any():
            flip_idx = plot_df[flip_mask].index[0] - plot_df.index[0]
            ax.axhline(y=flip_idx, color="#FF9800", linewidth=1.5,
                       linestyle=":", alpha=0.9, label=f"Gamma Flip: ${flip_level:.2f}")

    # ── Zero line ──
    ax.axvline(x=0, color="#555", linewidth=1.0)

    # ── Styling ──
    ax.set_xlabel("Net GEX (Juta USD)", color="#aaaaaa", fontsize=11)
    ax.set_ylabel("Strike Price", color="#aaaaaa", fontsize=11)
    ax.set_title(
        f"{ticker_symbol} — Gamma Exposure (GEX) per Strike\n"
        f"Spot: ${spot:.2f}  |  Range: ±10%  |  {datetime.date.today()}",
        color="white", fontsize=13, fontweight="bold", pad=15
    )

    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))

    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)

    # ── Annotasi nilai bar terbesar ──
    top_n = plot_df["net_gex"].abs().nlargest(5).index
    for idx in top_n:
        row = plot_df.loc[idx]
        bar_pos = row["net_gex"] / 1e6
        ax.text(
            bar_pos + (0.02 if bar_pos >= 0 else -0.02),
            str(row["strike"]),
            f"${bar_pos:.1f}M",
            va="center",
            ha="left" if bar_pos >= 0 else "right",
            color="white",
            fontsize=7.5,
            alpha=0.85,
        )

    plt.tight_layout()
    chart_path = "SPY_GEX_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[CHART] Disimpan ke: {chart_path}")
    plt.show()


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    TICKER = "SPY"
    RISK_FREE_RATE = 0.05   # 5% — bisa diganti sesuai Fed Funds Rate terkini

    # ── 1. Ambil spot price ──
    print(f"[INIT] Mengambil spot price {TICKER}...")
    spot_data = yf.Ticker(TICKER).history(period="1d")
    if spot_data.empty:
        raise RuntimeError("Gagal mengambil spot price SPY.")
    SPOT = float(spot_data["Close"].iloc[-1])
    print(f"       Spot price: ${SPOT:.2f}")

    # ── 2. Fetch & clean options ──
    df_clean = fetch_and_clean_options(
        ticker_symbol=TICKER,
        days_back=0,
        days_forward=45,
    )
    if df_clean is None:
        raise RuntimeError("Tidak ada data options yang valid.")

    # ── 3. Simpan data bersih ke CSV ──
    clean_csv = "SPY_options_clean.csv"
    df_clean.to_csv(clean_csv, index=False)
    print(f"\n[CSV] Data bersih disimpan ke: {clean_csv}")

    # ── 4. Hitung GEX ──
    df_gex, gamma_flip = compute_gex(df_clean, spot=SPOT, r=RISK_FREE_RATE)

    # ── 5. Simpan GEX ke CSV ──
    gex_csv = "SPY_GEX_by_strike.csv"
    df_gex.to_csv(gex_csv, index=False)
    print(f"[CSV] Data GEX disimpan ke: {gex_csv}")

    # ── 6. Tampilkan summary ──
    print("\n" + "="*55)
    print(f"  SUMMARY GEX {TICKER}")
    print("="*55)
    print(f"  Spot Price        : ${SPOT:.2f}")
    print(f"  Gamma Flip Level  : {'${:.2f}'.format(gamma_flip) if gamma_flip else 'N/A'}")
    print(f"  Total Net GEX     : ${df_gex['net_gex'].sum():,.0f}")
    print(f"  Strike terbesar   : ${df_gex.loc[df_gex['net_gex'].abs().idxmax(), 'strike']:.2f}")
    print("="*55)

    print("\nTop 10 Strike by |GEX|:")
    top10 = df_gex.nlargest(10, "net_gex")[["strike", "net_gex", "zona"]]
    top10["net_gex"] = top10["net_gex"].apply(lambda x: f"${x:,.0f}")
    print(top10.to_string(index=False))

    # ── 7. Visualisasi ──
    plot_gex(df_gex, spot=SPOT, flip_level=gamma_flip, ticker_symbol=TICKER)