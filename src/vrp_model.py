"""
VRP (Volatility Risk Premium) Model
=====================================
Pipeline:
  Collect Data → Hitung RV → Extract IV → Hitung VRP → Signal

Data source  : Yahoo Finance (yfinance)
Update mode  : Batch (non-realtime, jeda ~2 detik antar update)
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "ticker"         : "SPY",   # S&P 500 (ganti ke saham lain kalau mau)
    "risk_free_rate" : 0.00375,       # ~Fed Funds Rate saat ini
    "trading_days"   : 252,
    "min_candles"    : 30,         # threshold blending
    "vrp_roll_days"  : 60,         # rolling window normalisasi VRP
    "har_history"    : 22,         # minimal hari untuk HAR-RV
    "update_interval": 2,          # jeda antar "update" (detik)
    "n_updates"      : 5,          # jumlah simulasi update loop
}


# ─────────────────────────────────────────────
# MODULE 1: DATA COLLECTION
# Strategi: coba Yahoo Finance dulu.
# Kalau gagal (network / domain block), fallback ke
# synthetic data yang realistis agar pipeline tetap jalan.
# ─────────────────────────────────────────────

def _generate_synthetic_daily(ticker: str, n: int = 130) -> pd.DataFrame:
    """
    Generate synthetic OHLC harian yang realistis.
    Pakai Geometric Brownian Motion dengan vol ~18% annualized.
    """
    np.random.seed(42)
    spot0  = 5200.0  # harga awal (proxy S&P 500)
    mu     = 0.08 / 252
    sigma  = 0.18 / np.sqrt(252)

    dates  = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    closes = [spot0]
    for _ in range(n - 1):
        closes.append(closes[-1] * np.exp(np.random.normal(mu, sigma)))

    opens  = [closes[0]] + [closes[i] * np.exp(np.random.normal(0, sigma * 0.3))
                             for i in range(n - 1)]
    highs  = [max(o, c) * (1 + abs(np.random.normal(0, sigma * 0.5)))
               for o, c in zip(opens, closes)]
    lows   = [min(o, c) * (1 - abs(np.random.normal(0, sigma * 0.5)))
               for o, c in zip(opens, closes)]

    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes},
                      index=dates)
    return df


def _generate_synthetic_intraday(spot: float, n: int = 26) -> pd.DataFrame:
    """
    Generate synthetic OHLC intraday 15-menit (≈ 1 sesi trading = 26 candle).
    """
    np.random.seed(int(time.time()) % 1000)
    sigma_15m = 0.18 / np.sqrt(252 * 26)

    now    = pd.Timestamp.now().normalize() + pd.Timedelta(hours=9, minutes=30)
    times  = [now + pd.Timedelta(minutes=15 * i) for i in range(n)]
    closes = [spot]
    for _ in range(n - 1):
        closes.append(closes[-1] * np.exp(np.random.normal(0, sigma_15m)))

    opens  = [closes[0]] + closes[:-1]
    highs  = [max(o, c) * (1 + abs(np.random.normal(0, sigma_15m * 0.4)))
               for o, c in zip(opens, closes)]
    lows   = [min(o, c) * (1 - abs(np.random.normal(0, sigma_15m * 0.4)))
               for o, c in zip(opens, closes)]

    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes},
                        index=times)


def _generate_synthetic_vix(n: int = 130, base_vix: float = 0.19) -> pd.Series:
    """VIX synthetic: mean-reverting around base_vix (Ornstein-Uhlenbeck)."""
    np.random.seed(7)
    dates  = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    vix    = [base_vix]
    kappa, theta, sv = 0.1, base_vix, 0.02
    for _ in range(n - 1):
        dv = kappa * (theta - vix[-1]) + sv * np.random.normal()
        vix.append(max(0.05, vix[-1] + dv))
    return pd.Series(vix, index=dates)


def fetch_daily_ohlc(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Ambil OHLC harian. Fallback ke synthetic kalau Yahoo gagal."""
    print(f"  [DATA] Fetching daily OHLC for {ticker} ...")
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        df.dropna(inplace=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if df.empty:
            raise ValueError("Empty dataframe")
        print(f"  [DATA] Got {len(df)} daily candles (Yahoo Finance).")
        return df
    except Exception as e:
        print(f"  [WARN] Yahoo Finance gagal ({e}). Pakai synthetic data.")
        df = _generate_synthetic_daily(ticker)
        print(f"  [DATA] Got {len(df)} daily candles (synthetic).")
        return df


def fetch_intraday_ohlc(ticker: str, period: str = "5d", interval: str = "15m",
                         spot: float = 5200.0) -> pd.DataFrame:
    """Ambil OHLC intraday. Fallback ke synthetic kalau Yahoo gagal."""
    print(f"  [DATA] Fetching intraday {interval} OHLC for {ticker} ...")
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        df.dropna(inplace=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if not df.empty:
            last_date = df.index[-1].date()
            df = df[df.index.date == last_date]
        if df.empty:
            raise ValueError("Empty dataframe")
        print(f"  [DATA] Got {len(df)} intraday candles (Yahoo Finance).")
        return df
    except Exception as e:
        print(f"  [WARN] Yahoo Finance gagal ({e}). Pakai synthetic data.")
        df = _generate_synthetic_intraday(spot)
        print(f"  [DATA] Got {len(df)} intraday candles (synthetic, 15m).")
        return df


def get_vix_as_iv(period: str = "6mo") -> pd.Series:
    """Ambil VIX sebagai IV proxy. Fallback ke synthetic kalau Yahoo gagal."""
    print("  [DATA] Fetching VIX as IV proxy ...")
    try:
        vix = yf.download("^VIX", period=period, interval="1d",
                           progress=False, auto_adjust=True)
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
        iv_series = vix["Close"] / 100.0
        if iv_series.empty:
            raise ValueError("Empty series")
        print(f"  [DATA] VIX latest: {iv_series.iloc[-1]:.2%} (Yahoo Finance).")
        return iv_series
    except Exception as e:
        print(f"  [WARN] Yahoo Finance gagal ({e}). Pakai synthetic VIX.")
        iv_series = _generate_synthetic_vix()
        print(f"  [DATA] VIX (synthetic) latest: {iv_series.iloc[-1]:.2%}")
        return iv_series


# ─────────────────────────────────────────────
# MODULE 2: RV ESTIMATORS
# ─────────────────────────────────────────────

def rv_garman_klass(df: pd.DataFrame, trading_days: int = 252) -> float:
    """
    Garman-Klass estimator.
    Formula: sqrt(252 * mean(0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2))
    Efisiensi 7.4x lebih tinggi dari close-to-close.
    """
    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values

    hl = 0.5 * np.log(highs / lows) ** 2
    co = (2 * np.log(2) - 1) * np.log(closes / opens) ** 2
    gk_variance = np.mean(hl - co)
    return np.sqrt(trading_days * gk_variance)


def rv_parkinson(df: pd.DataFrame, trading_days: int = 252) -> float:
    """
    Parkinson estimator.
    Pakai kalau tidak ada open price.
    4x lebih efisien dari close-to-close.
    """
    highs = df["High"].values
    lows  = df["Low"].values
    pk_sum = np.sum(np.log(highs / lows) ** 2)
    return np.sqrt(trading_days * pk_sum / (4 * len(highs) * np.log(2)))


def rv_close_to_close(df: pd.DataFrame, trading_days: int = 252) -> float:
    """Close-to-close standard RV, sebagai fallback."""
    returns = np.diff(np.log(df["Close"].values))
    return np.std(returns, ddof=1) * np.sqrt(trading_days)


def rv_blended(rv_intraday: float, hv20: float, n_candles: int, min_candles: int = 30) -> float:
    """
    Blend RV intraday dengan HV20 untuk early session.
    w naik linear dari 0 → 1 seiring candle bertambah.
    """
    w = min(n_candles / min_candles, 1.0)
    return w * rv_intraday + (1 - w) * hv20


def hv20_from_daily(df: pd.DataFrame, trading_days: int = 252) -> float:
    """
    Hitung HV20 dari 20 hari terakhir daily close.
    Anchor baseline untuk early session.
    """
    closes  = df["Close"].values[-21:]  # 20 return = 21 harga
    returns = np.diff(np.log(closes))
    return np.std(returns, ddof=1) * np.sqrt(trading_days)


# ─────────────────────────────────────────────
# MODULE 3: HAR-RV MODEL
# ─────────────────────────────────────────────

def build_rv_history(df_daily: pd.DataFrame, trading_days: int = 252) -> np.ndarray:
    """
    Bangun time series RV harian dari daily OHLC.
    Pakai rolling Garman-Klass per hari.
    """
    rv_list = []
    closes = df_daily["Close"].values

    # RV harian dari close-to-close (per hari, bukan rolling)
    returns = np.diff(np.log(closes))
    # Annualized, per hari (1 return per hari = 1 observasi)
    for r in returns:
        rv_list.append(abs(r) * np.sqrt(trading_days))

    return np.array(rv_list)


def har_rv_predict(rv_history: np.ndarray) -> float:
    """
    HAR-RV (Corsi 2009):
    RV_t = α + β_d*RV_{t-1} + β_w*RV̄_{t-5} + β_m*RV̄_{t-22} + ε

    Koefisien empiris dari literatur.
    """
    if len(rv_history) < 22:
        return float(rv_history[-1])

    rv_d = rv_history[-1]
    rv_w = np.mean(rv_history[-5:])
    rv_m = np.mean(rv_history[-22:])

    alpha  = 0.0001
    beta_d = 0.35
    beta_w = 0.25
    beta_m = 0.30

    return alpha + beta_d * rv_d + beta_w * rv_w + beta_m * rv_m


# ─────────────────────────────────────────────
# MODULE 4: IV EXTRACTION (BSM + Newton-Raphson)
# ─────────────────────────────────────────────

def bsm_price(S, K, T, r, sigma, option_type="call") -> float:
    """Black-Scholes-Merton pricing formula."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_vega(S, K, T, r, sigma) -> float:
    """Vega untuk Newton-Raphson step."""
    if T <= 0 or sigma <= 0:
        return 1e-10
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol_newton(market_price, S, K, T, r, option_type="call",
                       max_iter=100, tol=1e-6) -> float:
    """
    Newton-Raphson IV solver.
    σ_{n+1} = σ_n - (BSM(σ_n) - market_price) / Vega(σ_n)
    Konvergen 3-5 iterasi.
    """
    # Initial guess
    sigma = max(0.01, min((market_price / S) * np.sqrt(2 * np.pi / max(T, 1e-6)), 5.0))

    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type)
        vega  = bsm_vega(S, K, T, r, sigma)
        diff  = price - market_price

        if abs(diff) < tol:
            break
        if abs(vega) < 1e-10:
            break

        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 5.0))

    return sigma


def simulate_atm_options(S: float, iv_market: float, T: float, r: float):
    """
    Karena Yahoo Finance free tidak sediakan options chain real-time,
    kita simulate harga ATM call/put menggunakan VIX sebagai IV,
    lalu kita "recover" IV dari harga itu.

    Ini memvalidasi bahwa Newton-Raphson pipeline berjalan benar.
    Di production: ganti dengan options chain nyata dari broker API.
    """
    K = round(S)  # ATM strike = spot
    call_price = bsm_price(S, K, T, r, iv_market, "call")
    put_price  = bsm_price(S, K, T, r, iv_market, "put")
    return call_price, put_price, K


def iv_from_straddle(call_price, put_price, S, K, T, r) -> float:
    """Extract IV dari ATM straddle (rata-rata call + put IV)."""
    iv_call = implied_vol_newton(call_price, S, K, T, r, "call")
    iv_put  = implied_vol_newton(put_price,  S, K, T, r, "put")
    return (iv_call + iv_put) / 2.0


# ─────────────────────────────────────────────
# MODULE 5: VRP CALCULATOR + SIGNAL
# ─────────────────────────────────────────────

def calculate_vrp(iv: float, rv: float, vrp_history: list):
    """
    VRP_raw = IV - RV
    VRP_zscore = (VRP_raw - μ) / σ  (rolling 60 hari)
    """
    vrp_raw = iv - rv

    if len(vrp_history) >= 10:
        mu    = np.mean(vrp_history)
        sigma = np.std(vrp_history) + 1e-10
        vrp_z = (vrp_raw - mu) / sigma
    else:
        vrp_z = 0.0

    return vrp_raw, vrp_z


def vrp_signal(vrp_z: float) -> tuple:
    """Return (label, description) berdasarkan z-score VRP."""
    if vrp_z > 1.5:
        return ("STRONG SHORT VOL", "IV sangat mahal relatif ke RV → jual volatilitas")
    elif vrp_z > 0.5:
        return ("MILD SHORT VOL",   "IV sedikit elevated → potensi short vol")
    elif vrp_z < -1.5:
        return ("STRONG LONG VOL",  "IV sangat murah relatif ke RV → beli volatilitas")
    elif vrp_z < -0.5:
        return ("MILD LONG VOL",    "IV sedikit murah → potensi long vol")
    else:
        return ("NEUTRAL",          "IV fairly priced, tidak ada sinyal kuat")


SIGNAL_EMOJI = {
    "STRONG SHORT VOL" : "🔴",
    "MILD SHORT VOL"   : "🟠",
    "STRONG LONG VOL"  : "🟢",
    "MILD LONG VOL"    : "🟡",
    "NEUTRAL"          : "⚪",
}


# ─────────────────────────────────────────────
# MODULE 6: DASHBOARD PRINTER
# ─────────────────────────────────────────────

def print_dashboard(result: dict, update_num: int):
    """Print formatted dashboard ke terminal."""
    sig_label, sig_desc = result["signal"]
    emoji = SIGNAL_EMOJI.get(sig_label, "⚪")

    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"║       VRP DASHBOARD  —  Update #{update_num:<3}              ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Ticker        : {result['ticker']:<10}                    ║")
    print(f"║  Spot Price    : {result['spot']:>10.2f}                    ║")
    print(f"║  Timestamp     : {result['timestamp']:<32}║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  IV  (VIX)     : {result['iv']:>8.2%}                      ║")
    print(f"║  RV  (GK blend): {result['rv']:>8.2%}                      ║")
    print(f"║  RV  (HAR fcst): {result['rv_har']:>8.2%}                      ║")
    print(f"║  HV20 baseline : {result['hv20']:>8.2%}                      ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  VRP raw       : {result['vrp_raw']:>+8.2%}                      ║")
    print(f"║  VRP z-score   : {result['vrp_z']:>+8.2f}                      ║")
    print(f"║  VRP vs HAR    : {result['vrp_vs_har']:>+8.2%}                      ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  SIGNAL  :  {emoji} {sig_label:<38}║")
    print(f"║  DESC    :  {sig_desc:<45}║")
    print("╚══════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────
# MODULE 7: MAIN LOOP
# ─────────────────────────────────────────────

def run_vrp_model():
    cfg = CONFIG

    print("\n" + "═" * 52)
    print("   VRP MODEL — Starting up...")
    print("═" * 52)

    # ── STEP 1: Ambil semua data sekali di awal ──────────
    print("\n[STEP 1] Collecting data from Yahoo Finance...")
    df_daily    = fetch_daily_ohlc(cfg["ticker"], period="6mo")
    spot_init   = float(df_daily["Close"].iloc[-1]) if not df_daily.empty else 5200.0
    df_intraday = fetch_intraday_ohlc(cfg["ticker"], period="5d", interval="15m", spot=spot_init)
    iv_series   = get_vix_as_iv(period="6mo")

    if df_daily.empty or iv_series.empty:
        print("[ERROR] Data kosong, cek koneksi atau ticker.")
        return

    # ── STEP 2: Precompute baseline data ─────────────────
    print("\n[STEP 2] Precomputing baselines...")

    # HV20 dari daily
    hv20 = hv20_from_daily(df_daily, cfg["trading_days"])

    # RV history (60 hari) untuk normalisasi VRP
    rv_history_all = build_rv_history(df_daily, cfg["trading_days"])
    rv_history_60  = rv_history_all[-cfg["vrp_roll_days"]:]
    rv_history_22  = rv_history_all[-cfg["har_history"]:]

    # IV history dari VIX (align ke daily df)
    iv_history = iv_series.reindex(df_daily.index, method="ffill").dropna()

    # Build VRP history (untuk rolling normalisasi)
    vrp_history = []
    min_len = min(len(rv_history_60), len(iv_history))
    for i in range(min_len):
        vrp_history.append(float(iv_history.iloc[-(min_len - i)]) - rv_history_60[-(min_len - i)])

    print(f"  HV20            : {hv20:.2%}")
    print(f"  RV history pts  : {len(rv_history_60)}")
    print(f"  VRP history pts : {len(vrp_history)}")

    # ── STEP 3: Simulasi update loop (non-realtime) ───────
    print(f"\n[STEP 3] Starting update loop ({cfg['n_updates']} updates, "
          f"jeda {cfg['update_interval']}s)...\n")

    ticker_obj = yf.Ticker(cfg["ticker"])

    for update_num in range(1, cfg["n_updates"] + 1):

        # Ambil harga spot terkini
        spot_info = ticker_obj.fast_info
        try:
            spot = float(spot_info["last_price"])
        except Exception:
            spot = float(df_daily["Close"].iloc[-1])

        # IV terkini dari VIX
        iv_current = float(iv_series.iloc[-1])

        # RV intraday (Garman-Klass dari candle hari ini)
        if len(df_intraday) >= 2:
            rv_intraday_raw = rv_garman_klass(df_intraday, cfg["trading_days"])
            n_candles       = len(df_intraday)
        else:
            # Fallback: pakai HV20 kalau intraday tidak tersedia
            rv_intraday_raw = hv20
            n_candles       = 0

        rv_blended_val = rv_blended(rv_intraday_raw, hv20, n_candles, cfg["min_candles"])

        # HAR-RV forecast
        rv_har = har_rv_predict(rv_history_22)

        # Simulate ATM options (validasi Newton-Raphson pipeline)
        T = 30 / 365  # 30 hari ke expiry (proxy)
        r = cfg["risk_free_rate"]
        call_p, put_p, strike = simulate_atm_options(spot, iv_current, T, r)
        iv_extracted = iv_from_straddle(call_p, put_p, spot, strike, T, r)

        # Hitung VRP
        vrp_raw, vrp_z = calculate_vrp(iv_extracted, rv_blended_val, vrp_history)
        vrp_vs_har     = iv_extracted - rv_har
        signal         = vrp_signal(vrp_z)

        # Update rolling VRP history
        vrp_history.append(vrp_raw)
        if len(vrp_history) > cfg["vrp_roll_days"]:
            vrp_history.pop(0)

        result = {
            "ticker"     : cfg["ticker"],
            "spot"       : spot,
            "timestamp"  : pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iv"         : iv_extracted,
            "rv"         : rv_blended_val,
            "rv_har"     : rv_har,
            "hv20"       : hv20,
            "vrp_raw"    : vrp_raw,
            "vrp_z"      : vrp_z,
            "vrp_vs_har" : vrp_vs_har,
            "signal"     : signal,
        }

        print_dashboard(result, update_num)

        if update_num < cfg["n_updates"]:
            print(f"\n  ⏱  Jeda {cfg['update_interval']} detik sebelum update berikutnya...\n")
            time.sleep(cfg["update_interval"])

    print("\n" + "═" * 52)
    print("   VRP MODEL — Done.")
    print("═" * 52 + "\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_vrp_model()
