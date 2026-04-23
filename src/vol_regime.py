import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import json
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
TICKER         = "^SPX"      # ganti ke QQQ, GLD, AAPL, dll
IV_TICKER      = "^VIX"     # proxy IV index — ganti ke ^VXN untuk QQQ
LOOKBACK_DAYS  = 252        # seberapa jauh historical data diambil
RV_WINDOWS     = [10, 21, 63]
OUTPUT_HTML    = "vol_regime_SPX_dashboard.html"

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility pakai rolling std of log returns."""
    return returns.rolling(window).std() * np.sqrt(252) * 100  # dalam persen


def iv_percentile(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """IV Percentile (IVP) — posisi IV sekarang vs N hari terakhir."""
    return iv_series.rolling(window).apply(
        lambda x: round(pd.Series(x).rank(pct=True).iloc[-1] * 100, 1),
        raw=False
    )


def iv_rank(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """IV Rank (IVR) — (current - min) / (max - min) × 100."""
    def _ivr(x):
        mn, mx = x.min(), x.max()
        if mx == mn:
            return 50.0
        return round((x.iloc[-1] - mn) / (mx - mn) * 100, 1)
    return iv_series.rolling(window).apply(_ivr, raw=False)


def classify_regime(iv_val: float, rv_21: float, ivp: float) -> dict:
    """
    Klasifikasi regime berdasarkan:
    - IV level absolut
    - IV vs RV spread (vol premium)
    - IV Percentile

    Returns dict: regime, color, description, strategies
    """
    vp = iv_val - rv_21  # vol premium (positive = IV > RV = expensive options)

    # ── Regime classification ──────────────────────────────────────
    if ivp >= 70 and vp > 3:
        regime = "HIGH VOL / FEAR"
        color  = "#ff4060"
        desc   = "IV jauh di atas RV dan berada di persentil tinggi. Dealer pricing fear premium besar. Opsi mahal. Market cenderung mean-revert atau sudah dalam trending move besar."
        strats = [
            ("Premium Selling", "Sell iron condor / strangle di luar 1σ. IV crush post-event akan benefit short vega positions.", "#ff4060"),
            ("Defined Risk Spreads", "Kalau mau directional, pakai vertical spread — bayar lebih sedikit daripada beli naked option.", "#f4a261"),
            ("Avoid Buying Options", "Naked long call/put sangat mahal. Kecuali lu punya strong directional conviction + catalyst spesifik.", "#64748b"),
            ("Gamma Scalping", "Kalau punya edge intraday, long gamma bisa profitable karena realized moves besar. Tapi butuh aktif.", "#7b8cde"),
        ]
    elif ivp >= 50 and vp > 0:
        regime = "ELEVATED VOL / CAUTION"
        color  = "#f4a261"
        desc   = "IV di atas rata-rata dan masih premium terhadap RV. Pasar dalam mode waspada. GEX cenderung negatif — dealer short gamma, moves bisa impulsif."
        strats = [
            ("Neutral Spreads", "Iron condor / butterfly dengan adjustment kalau breakout. Width lebih lebar dari biasa.", "#f4a261"),
            ("Reduce Size", "Volatilitas yang elevated = P&L swings lebih besar. Kurangi size 20-30% dari normal.", "#64748b"),
            ("Watch Gamma Flip", "Perhatikan apakah spot di atas atau bawah gamma flip. Di bawah flip = regime accelerator aktif.", "#7b8cde"),
            ("Calendar Spreads", "Long back-month / short front-month jika term structure dalam contango.", "#00c896"),
        ]
    elif ivp <= 30 and vp < -2:
        regime = "LOW VOL / COMPLACENCY"
        color  = "#00c896"
        desc   = "IV rendah secara historis dan discount terhadap RV. Opsi murah. Dealer long gamma — mereka jual saat naik, beli saat turun = pasar teredam dan trending."
        strats = [
            ("Buy Options / Long Gamma", "Opsi murah secara historis. Long straddle / strangle sebelum catalyst bisa sangat profitable.", "#00c896"),
            ("Trend Following", "Low vol = dealer pinning aktif. Gunakan momentum — buy breakout dari GEX levels.", "#00c896"),
            ("Avoid Premium Selling", "Menjual opsi murah = risk/reward jelek. Theta kecil, tapi tail risk tetap ada.", "#ff4060"),
            ("Vanna Trade Setup", "Ekspektasi vol expansion kedepannya = Vanna akan jadi tailwind kalau IV naik. Long vega.", "#7b8cde"),
        ]
    elif ivp <= 50 and abs(vp) <= 3:
        regime = "NORMAL VOL / BALANCED"
        color  = "#7b8cde"
        desc   = "IV mendekati nilai wajar relatif terhadap RV dan berada di persentil tengah. Kondisi paling 'normal' — tidak ada edge spesifik dari vol mispricing."
        strats = [
            ("Standard Approach", "Tidak ada vol edge spesifik. Fokus ke price action, GEX levels, dan order flow.", "#7b8cde"),
            ("Small Premium Selling", "Sedikit IV premium = slight edge untuk theta strategies, tapi jangan oversize.", "#f4a261"),
            ("ATM Options OK", "Harga opsi wajar. Bisa beli untuk hedging atau directional trade tanpa overpaying.", "#00c896"),
            ("Monitor IVP Trend", "Perhatikan apakah IVP trending naik atau turun untuk antisipasi regime shift.", "#64748b"),
        ]
    else:
        regime = "TRANSITIONAL"
        color  = "#a78bfa"
        desc   = "Regime sedang dalam transisi — sinyal campuran antara IV level, RV, dan percentile. Butuh konfirmasi lebih sebelum sizing up."
        strats = [
            ("Wait for Clarity", "Jangan force trade di regime ambiguous. Tunggu IVP break di atas 60 atau di bawah 40.", "#a78bfa"),
            ("Small Size Only", "Kalau harus trade, pakai size minimal (25-50% normal) sampai regime clear.", "#64748b"),
            ("Watch Term Structure", "Cek apakah front-month vs back-month IV spread melebar atau menyempit — ini leading indicator regime.", "#7b8cde"),
            ("GEX Confirmation", "Konfirmasi dengan GEX: apakah dealer positioning konsisten dengan arah yang lu anticipate?", "#f4a261"),
        ]

    return {
        "regime":      regime,
        "color":       color,
        "description": desc,
        "strategies":  strats,
    }


# ══════════════════════════════════════════════════════════════════
#  FETCH DATA
# ══════════════════════════════════════════════════════════════════
def fetch_data(ticker: str, iv_ticker: str, lookback: int) -> dict | None:
    print(f"\n[FETCH] Mengambil price data {ticker}...")
    end   = datetime.date.today()
    # Butuh data untuk rolling window (252) + data untuk chart (252 hari) = minimal 504 trading days.
    # Kita mundur sekitar 3 tahun agar sangat aman.
    start = end - datetime.timedelta(days=(lookback * 2) + 365)

    price_data = yf.download(ticker, start=start, end=end, progress=False)
    if price_data.empty:
        print("  Gagal mengambil price data.")
        return None

    iv_data = yf.download(iv_ticker, start=start, end=end, progress=False)
    if iv_data.empty:
        print(f"  Gagal mengambil IV data ({iv_ticker}). Pakai ATM IV estimasi.")
        iv_series = None
    else:
        iv_series = iv_data["Close"].squeeze()
        print(f"  IV data ({iv_ticker}): {len(iv_series)} baris")

    close = price_data["Close"].squeeze()
    log_returns = np.log(close / close.shift(1)).dropna()

    print(f"  Price data {ticker}: {len(close)} baris")
    print(f"  Date range: {close.index[0].date()} -> {close.index[-1].date()}")

    return {
        "close":       close,
        "log_returns": log_returns,
        "iv_series":   iv_series,
    }


# ══════════════════════════════════════════════════════════════════
#  COMPUTE VOL METRICS
# ══════════════════════════════════════════════════════════════════
def compute_vol_metrics(data: dict, rv_windows: list) -> pd.DataFrame:
    print("\n[COMPUTE] Menghitung vol metrics...")

    close       = data["close"]
    log_returns = data["log_returns"]
    iv_series   = data["iv_series"]

    df = pd.DataFrame(index=close.index)
    df["close"] = close

    # Realized Vol per window
    for w in rv_windows:
        df[f"rv_{w}"] = realized_vol(log_returns, w)

    # IV — pakai VIX/VXN kalau tersedia, fallback ke RV21 + noise
    if iv_series is not None:
        iv_aligned = iv_series.reindex(df.index, method="ffill")
        df["iv"] = iv_aligned
    else:
        # Fallback: estimasi IV = RV21 × 1.1 (rata-rata vol risk premium historis)
        df["iv"] = df["rv_21"] * 1.10 if "rv_21" in df.columns else df[f"rv_{rv_windows[0]}"] * 1.10
        print("  [WARN] Menggunakan estimated IV (RV21 × 1.10)")

    # IV Percentile & IV Rank
    df["ivp"]  = iv_percentile(df["iv"], window=252)
    df["ivr"]  = iv_rank(df["iv"], window=252)

    # Vol Premium (IV - RV21)
    rv_mid = f"rv_{rv_windows[1]}" if len(rv_windows) > 1 else f"rv_{rv_windows[0]}"
    df["vol_premium"] = df["iv"] - df[rv_mid]

    # Regime per hari
    regimes = []
    for _, row in df.iterrows():
        if pd.isna(row["iv"]) or pd.isna(row.get(rv_mid, np.nan)) or pd.isna(row["ivp"]):
            regimes.append(None)
        else:
            r = classify_regime(row["iv"], row[rv_mid], row["ivp"])
            regimes.append(r["regime"])
    df["regime"] = regimes

    # Drop awal yang masih NaN (warmup period)
    df = df.dropna(subset=["iv", rv_mid, "ivp"]).copy()

    if len(df) == 0:
        raise ValueError("Data historis tidak mencukupi untuk menghitung rolling metrics (semuanya NaN setelah dropna).")

    print(f"  Rows setelah warmup: {len(df)}")
    print(f"  Current IV        : {df['iv'].iloc[-1]:.2f}%")
    print(f"  Current IVP       : {df['ivp'].iloc[-1]:.1f}th percentile")
    print(f"  Current IVR       : {df['ivr'].iloc[-1]:.1f}")
    print(f"  Current Regime    : {df['regime'].iloc[-1]}")

    return df


# ══════════════════════════════════════════════════════════════════
#  BUILD HTML
# ══════════════════════════════════════════════════════════════════
def build_html(df: pd.DataFrame, ticker: str, iv_ticker: str, rv_windows: list) -> str:
    today_str = datetime.date.today().strftime("%d %B %Y")

    # Last N rows buat chart (max 252 hari = 1 tahun)
    chart_df = df.tail(252).copy()
    rv_mid   = f"rv_{rv_windows[1]}" if len(rv_windows) > 1 else f"rv_{rv_windows[0]}"

    # Serialize untuk JS
    dates  = [d.strftime("%Y-%m-%d") for d in chart_df.index]
    closes = [round(float(v), 2) for v in chart_df["close"].tolist()]
    iv_vals = [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["iv"].tolist()]
    rv_data = {}
    for w in rv_windows:
        col = f"rv_{w}"
        rv_data[str(w)] = [round(float(v), 2) if not np.isnan(v) else None for v in chart_df[col].tolist()]
    ivp_vals = [round(float(v), 1) if not np.isnan(v) else None for v in chart_df["ivp"].tolist()]
    ivr_vals = [round(float(v), 1) if not np.isnan(v) else None for v in chart_df["ivr"].tolist()]
    vp_vals  = [round(float(v), 2) if not np.isnan(v) else None for v in chart_df["vol_premium"].tolist()]
    regime_vals = chart_df["regime"].tolist()

    # Current snapshot
    last      = df.iloc[-1]
    curr_iv   = float(last["iv"])
    curr_ivp  = float(last["ivp"])
    curr_ivr  = float(last["ivr"])
    curr_vp   = float(last["vol_premium"])
    curr_rv   = {w: float(last[f"rv_{w}"]) for w in rv_windows}
    curr_close = float(last["close"])
    curr_regime_info = classify_regime(curr_iv, float(last[rv_mid]), curr_ivp)

    # Regime distribution (last 63 hari)
    recent_regimes = df["regime"].tail(63).value_counts().to_dict()

    js_payload = {
        "dates":       dates,
        "closes":      closes,
        "iv":          iv_vals,
        "rv":          rv_data,
        "ivp":         ivp_vals,
        "ivr":         ivr_vals,
        "vp":          vp_vals,
        "regimes":     regime_vals,
        "rv_windows":  [str(w) for w in rv_windows],
        "current": {
            "iv":      curr_iv,
            "ivp":     curr_ivp,
            "ivr":     curr_ivr,
            "vp":      curr_vp,
            "rv":      curr_rv,
            "close":   curr_close,
            "regime":  curr_regime_info["regime"],
            "color":   curr_regime_info["color"],
            "desc":    curr_regime_info["description"],
            "strats":  curr_regime_info["strategies"],
        },
        "regime_dist": recent_regimes,
    }

    js_str = json.dumps(js_payload)

    regime_color_map = {
        "HIGH VOL / FEAR":        "#ff4060",
        "ELEVATED VOL / CAUTION": "#f4a261",
        "LOW VOL / COMPLACENCY":  "#00c896",
        "NORMAL VOL / BALANCED":  "#7b8cde",
        "TRANSITIONAL":           "#a78bfa",
    }
    regime_color_js = json.dumps(regime_color_map)

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{ticker} Vol Regime Detector</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {{
  --bg:      #07090e;
  --bg2:     #0c0f17;
  --bg3:     #111620;
  --border:  #1a2133;
  --text:    #dde6f5;
  --muted:   #4a607a;
  --accent:  #00d4aa;
  --gold:    #ffd166;
  --pos:     #00c896;
  --neg:     #ff4060;
  --vanna:   #7b8cde;
  --charm:   #f4a261;
  --vega:    #a78bfa;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  min-height: 100vh;
}}
body::before {{
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(0,212,170,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,170,0.02) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none; z-index: 0;
}}
.wrap {{ position: relative; z-index: 1; max-width: 1280px; margin: 0 auto; padding: 28px 20px; }}

/* HEADER */
header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 26px; flex-wrap: wrap; gap: 14px; }}
.h-left h1 {{ font-family: 'Space Mono', monospace; font-size: clamp(1.4rem, 3.5vw, 2rem); font-weight: 700; color: var(--accent); letter-spacing: -1px; }}
.h-left h1 span {{ color: var(--text); }}
.h-left p {{ color: var(--muted); font-size: 0.75rem; margin-top: 4px; font-family: 'Space Mono', monospace; }}
.price-badge {{ background: var(--bg3); border: 1px solid var(--border); border-left: 3px solid var(--gold); padding: 10px 16px; border-radius: 8px; text-align: right; }}
.price-badge .lbl {{ font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-family: 'Space Mono', monospace; }}
.price-badge .val {{ font-family: 'Space Mono', monospace; font-size: 1.3rem; font-weight: 700; color: var(--gold); margin-top: 2px; }}

/* REGIME BANNER */
.regime-banner {{
  border-radius: 14px; border: 1px solid; padding: 20px 24px;
  margin-bottom: 22px; display: flex; align-items: flex-start;
  justify-content: space-between; gap: 20px; flex-wrap: wrap;
}}
.regime-left {{ flex: 1; min-width: 220px; }}
.regime-label {{ font-family: 'Space Mono', monospace; font-size: clamp(1rem, 2.5vw, 1.4rem); font-weight: 700; letter-spacing: 1px; margin-bottom: 8px; }}
.regime-desc {{ font-size: 0.8rem; color: var(--muted); line-height: 1.65; max-width: 560px; }}
.regime-metrics {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.rm {{ background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px; min-width: 90px; text-align: center; }}
.rm .rl {{ font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-family: 'Space Mono', monospace; }}
.rm .rv {{ font-family: 'Space Mono', monospace; font-size: 1.05rem; font-weight: 700; margin-top: 4px; }}

/* STRATEGY GRID */
.section-title {{ font-family: 'Space Mono', monospace; font-size: 0.72rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; }}
.strat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin-bottom: 24px; }}
.strat-card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; border-left: 3px solid; }}
.strat-card .st {{ font-size: 0.78rem; font-weight: 600; margin-bottom: 5px; }}
.strat-card .sd {{ font-size: 0.72rem; color: var(--muted); line-height: 1.55; }}

/* STATS ROW */
.stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 9px; margin-bottom: 22px; }}
.stat-c {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; transition: border-color 0.2s; }}
.stat-c:hover {{ border-color: #2a3a52; }}
.stat-c .sl {{ font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-family: 'Space Mono', monospace; }}
.stat-c .sv {{ font-family: 'Space Mono', monospace; font-size: 0.95rem; font-weight: 700; margin-top: 5px; }}
.sv.pos {{ color: var(--pos); }} .sv.neg {{ color: var(--neg); }} .sv.gold {{ color: var(--gold); }}
.sv.vanna {{ color: var(--vanna); }} .sv.charm {{ color: var(--charm); }} .sv.vega {{ color: var(--vega); }}

/* CHART CARD */
.chart-card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 14px; padding: 20px; margin-bottom: 18px; }}
.chart-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px; }}
.chart-title {{ font-family: 'Space Mono', monospace; font-size: 0.78rem; font-weight: 700; color: var(--text); }}
.chart-controls {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.ctrl-btn {{ background: var(--bg3); border: 1px solid var(--border); color: var(--muted); font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 4px 10px; border-radius: 5px; cursor: pointer; transition: all 0.15s; }}
.ctrl-btn:hover {{ color: var(--text); }}
.ctrl-btn.active {{ color: var(--accent); border-color: var(--accent); background: rgba(0,212,170,0.07); }}
canvas {{ display: block; width: 100% !important; }}

/* REGIME DIST */
.dist-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; margin-bottom: 22px; }}
.dist-bar-wrap {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 9px; padding: 12px; }}
.dist-name {{ font-size: 0.65rem; font-family: 'Space Mono', monospace; font-weight: 700; margin-bottom: 7px; }}
.dist-bar {{ height: 6px; border-radius: 3px; background: var(--border); margin-bottom: 5px; overflow: hidden; }}
.dist-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.6s ease; }}
.dist-pct {{ font-family: 'Space Mono', monospace; font-size: 0.7rem; color: var(--muted); }}

footer {{ margin-top: 24px; text-align: center; font-size: 0.63rem; color: var(--muted); font-family: 'Space Mono', monospace; padding-bottom: 20px; }}
@media (max-width: 600px) {{ .wrap {{ padding: 12px; }} .chart-card {{ padding: 12px; }} }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="h-left">
    <h1><span>{ticker}</span> Vol Regime<span>.</span></h1>
    <p>Volatility Regime Detector &bull; IV vs RV &bull; {today_str}</p>
  </div>
  <div class="price-badge">
    <div class="lbl">Last Close</div>
    <div class="val" id="hdr-price">${curr_close:.2f}</div>
  </div>
</header>

<!-- REGIME BANNER -->
<div class="regime-banner" id="regime-banner">
  <div class="regime-left">
    <div class="regime-label" id="regime-label">—</div>
    <div class="regime-desc" id="regime-desc">—</div>
  </div>
  <div class="regime-metrics" id="regime-metrics"></div>
</div>

<!-- STRATEGY GUIDE -->
<div class="section-title">Strategy Guide — Current Regime</div>
<div class="strat-grid" id="strat-grid"></div>

<!-- STATS -->
<div class="stats-row">
  <div class="stat-c"><div class="sl">IV ({iv_ticker})</div><div class="sv gold" id="s-iv">—</div></div>
  <div class="stat-c"><div class="sl">IV Percentile</div><div class="sv" id="s-ivp">—</div></div>
  <div class="stat-c"><div class="sl">IV Rank</div><div class="sv" id="s-ivr">—</div></div>
  <div class="stat-c"><div class="sl">Vol Premium</div><div class="sv" id="s-vp">—</div></div>
  <div class="stat-c"><div class="sl">RV {rv_windows[0]}d</div><div class="sv vanna" id="s-rv0">—</div></div>
  <div class="stat-c"><div class="sl">RV {rv_windows[1]}d</div><div class="sv vanna" id="s-rv1">—</div></div>
  <div class="stat-c"><div class="sl">RV {rv_windows[2]}d</div><div class="sv vanna" id="s-rv2">—</div></div>
</div>

<!-- CHART 1: IV vs RV -->
<div class="chart-card">
  <div class="chart-header">
    <div class="chart-title">IV vs Realized Volatility (1Y)</div>
    <div class="chart-controls">
      <button class="ctrl-btn active" data-range="252">1Y</button>
      <button class="ctrl-btn" data-range="126">6M</button>
      <button class="ctrl-btn" data-range="63">3M</button>
      <button class="ctrl-btn" data-range="21">1M</button>
    </div>
  </div>
  <canvas id="chart-ivrv" height="220"></canvas>
</div>

<!-- CHART 2: IVP & Vol Premium -->
<div class="chart-card">
  <div class="chart-header">
    <div class="chart-title">IV Percentile & Vol Premium</div>
    <div class="chart-controls">
      <button class="ctrl-btn active" data-chart="ivp">IVP</button>
      <button class="ctrl-btn" data-chart="ivr">IVR</button>
      <button class="ctrl-btn" data-chart="vp">Vol Premium</button>
    </div>
  </div>
  <canvas id="chart-ivp" height="160"></canvas>
</div>

<!-- CHART 3: Regime Timeline -->
<div class="chart-card">
  <div class="chart-header">
    <div class="chart-title">Regime Timeline (1Y)</div>
  </div>
  <canvas id="chart-regime" height="90"></canvas>
</div>

<!-- REGIME DISTRIBUTION -->
<div class="section-title">Regime Distribution — Last 63 Days</div>
<div class="dist-grid" id="dist-grid"></div>

<footer>
  Data: Yahoo Finance via yfinance &bull; IV Proxy: {iv_ticker} &bull; RV Windows: {rv_windows[0]}d / {rv_windows[1]}d / {rv_windows[2]}d &bull; IVP lookback: 252 hari
</footer>
</div>

<script>
const D = {js_str};
const REGIME_COLORS = {regime_color_js};

// ── Populate static content ─────────────────────────────────────
function populateStats() {{
  const c = D.current;

  // Banner
  const banner = document.getElementById("regime-banner");
  banner.style.borderColor = c.color + "44";
  banner.style.background  = c.color + "0d";
  document.getElementById("regime-label").style.color = c.color;
  document.getElementById("regime-label").textContent = c.regime;
  document.getElementById("regime-desc").textContent  = c.desc;

  // Metrics in banner
  const rvMid = D.rv_windows[1] || D.rv_windows[0];
  document.getElementById("regime-metrics").innerHTML = `
    <div class="rm"><div class="rl">IV</div><div class="rv" style="color:${{c.color}}">${{c.iv.toFixed(1)}}%</div></div>
    <div class="rm"><div class="rl">IVP</div><div class="rv" style="color:${{c.color}}">${{c.ivp.toFixed(0)}}th</div></div>
    <div class="rm"><div class="rl">IVR</div><div class="rv" style="color:${{c.color}}">${{c.ivr.toFixed(0)}}</div></div>
    <div class="rm"><div class="rl">VP</div><div class="rv" style="color:${{c.vp >= 0 ? '#ff4060' : '#00c896'}}">${{c.vp >= 0 ? '+' : ''}}${{c.vp.toFixed(1)}}</div></div>
    <div class="rm"><div class="rl">RV${{rvMid}}d</div><div class="rv" style="color:#7b8cde">${{c.rv[rvMid].toFixed(1)}}%</div></div>
  `;

  // Strategy cards
  const sg = document.getElementById("strat-grid");
  sg.innerHTML = c.strats.map(([title, desc, col]) => `
    <div class="strat-card" style="border-left-color:${{col}}">
      <div class="st" style="color:${{col}}">${{title}}</div>
      <div class="sd">${{desc}}</div>
    </div>
  `).join("");

  // Stats row
  document.getElementById("s-iv").textContent  = c.iv.toFixed(2) + "%";
  const ivpEl = document.getElementById("s-ivp");
  ivpEl.textContent = c.ivp.toFixed(1) + "th";
  ivpEl.className = "sv " + (c.ivp >= 70 ? "neg" : c.ivp <= 30 ? "pos" : "vanna");
  document.getElementById("s-ivr").textContent = c.ivr.toFixed(1);
  const vpEl = document.getElementById("s-vp");
  vpEl.textContent = (c.vp >= 0 ? "+" : "") + c.vp.toFixed(2) + "%";
  vpEl.className = "sv " + (c.vp >= 0 ? "neg" : "pos");
  const rvKeys = Object.keys(c.rv);
  if (rvKeys[0]) document.getElementById("s-rv0").textContent = c.rv[rvKeys[0]].toFixed(2) + "%";
  if (rvKeys[1]) document.getElementById("s-rv1").textContent = c.rv[rvKeys[1]].toFixed(2) + "%";
  if (rvKeys[2]) document.getElementById("s-rv2").textContent = c.rv[rvKeys[2]].toFixed(2) + "%";

  // Regime distribution
  const total = Object.values(D.regime_dist).reduce((a, b) => a + b, 0);
  const dg = document.getElementById("dist-grid");
  dg.innerHTML = Object.entries(D.regime_dist)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => {{
      const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
      const col = REGIME_COLORS[name] || "#7b8cde";
      return `
        <div class="dist-bar-wrap">
          <div class="dist-name" style="color:${{col}}">${{name}}</div>
          <div class="dist-bar"><div class="dist-bar-fill" style="width:${{pct}}%;background:${{col}}"></div></div>
          <div class="dist-pct">${{pct}}% (${{count}} hari)</div>
        </div>`;
    }}).join("");
}}

// ── Canvas helpers ──────────────────────────────────────────────
function getCtx(id) {{
  const canvas = document.getElementById(id);
  canvas.width  = canvas.parentElement.clientWidth;
  return {{ canvas, ctx: canvas.getContext("2d") }};
}}

function sliceRange(range) {{
  const n = D.dates.length;
  const start = Math.max(0, n - range);
  return {{ start, end: n }};
}}

function drawLine(ctx, pts, color, lw = 1.5, dash = []) {{
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  ctx.setLineDash(dash);
  ctx.beginPath();
  let first = true;
  for (const [x, y] of pts) {{
    if (y === null) {{ first = true; continue; }}
    if (first) {{ ctx.moveTo(x, y); first = false; }} else ctx.lineTo(x, y);
  }}
  ctx.stroke();
  ctx.restore();
}}

function drawFill(ctx, pts, color) {{
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  let started = false;
  const valid = pts.filter(p => p[1] !== null);
  if (valid.length < 2) {{ ctx.restore(); return; }}
  ctx.moveTo(valid[0][0], valid[0][1]);
  for (let i = 1; i < valid.length; i++) ctx.lineTo(valid[i][0], valid[i][1]);
  ctx.lineTo(valid[valid.length-1][0], valid[valid.length-1][1]);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}}

// ── CHART 1: IV vs RV ──────────────────────────────────────────
function drawIVRV(range = 252) {{
  const {{ canvas, ctx }} = getCtx("chart-ivrv");
  const {{ start }} = sliceRange(range);
  const W = canvas.width, H = canvas.height;
  const padL = 44, padR = 12, padT = 14, padB = 28;
  const cW = W - padL - padR, cH = H - padT - padB;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0c0f17"; ctx.fillRect(0, 0, W, H);

  const ivSlice  = D.iv.slice(start);
  const rvSlices = {{}};
  for (const w of D.rv_windows) rvSlices[w] = D.rv[w].slice(start);

  const allVals = [...ivSlice, ...Object.values(rvSlices).flat()].filter(v => v !== null);
  const minV = Math.min(...allVals) * 0.92;
  const maxV = Math.max(...allVals) * 1.08;
  const n = ivSlice.length;

  const toX = i => padL + (i / (n - 1)) * cW;
  const toY = v => v === null ? null : padT + (1 - (v - minV) / (maxV - minV)) * cH;

  // Grid
  const ySteps = 5;
  ctx.fillStyle = "#2a3a52"; ctx.font = "8px 'Space Mono', monospace"; ctx.textAlign = "right";
  for (let i = 0; i <= ySteps; i++) {{
    const v = minV + (maxV - minV) * (i / ySteps);
    const y = padT + (1 - i / ySteps) * cH;
    ctx.strokeStyle = "#141e2e"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(1) + "%", padL - 4, y + 3);
  }}

  // RV lines
  const rvColors = ["#7b8cde", "#a78bfa", "#60a5fa"];
  D.rv_windows.forEach((w, i) => {{
    const pts = rvSlices[w].map((v, idx) => [toX(idx), toY(v)]);
    drawLine(ctx, pts, rvColors[i], 1.2, [4, 3]);
  }});

  // IV fill area
  const ivPts = ivSlice.map((v, i) => [toX(i), toY(v)]);
  const fillPts = [...ivPts.filter(p => p[1] !== null)];
  if (fillPts.length > 1) {{
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(fillPts[0][0], padT + cH);
    for (const [x, y] of fillPts) ctx.lineTo(x, y);
    ctx.lineTo(fillPts[fillPts.length-1][0], padT + cH);
    ctx.closePath();
    ctx.fillStyle = "rgba(255,209,102,0.07)";
    ctx.fill();
    ctx.restore();
  }}
  drawLine(ctx, ivPts, "#ffd166", 2);

  // X axis dates
  ctx.fillStyle = "#2a3a52"; ctx.font = "7.5px 'Space Mono', monospace"; ctx.textAlign = "center";
  const dateStep = Math.max(1, Math.floor(n / 6));
  for (let i = 0; i < n; i += dateStep) {{
    const x = toX(i);
    const d = D.dates[start + i];
    ctx.fillText(d ? d.slice(5) : "", x, H - 6);
  }}

  // Legend
  ctx.font = "7.5px 'Space Mono', monospace"; ctx.textAlign = "left";
  ctx.fillStyle = "#ffd166"; ctx.fillText("── IV", padL, padT - 2);
  D.rv_windows.forEach((w, i) => {{
    ctx.fillStyle = rvColors[i];
    ctx.fillText(`┄ RV${{w}}d`, padL + 50 + i * 55, padT - 2);
  }});
}}

// ── CHART 2: IVP / IVR / Vol Premium ──────────────────────────
let activeIVPChart = "ivp";
function drawIVPChart(type = "ivp") {{
  activeIVPChart = type;
  const {{ canvas, ctx }} = getCtx("chart-ivp");
  const W = canvas.width, H = canvas.height;
  const padL = 44, padR = 12, padT = 14, padB = 28;
  const cW = W - padL - padR, cH = H - padT - padB;
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = "#0c0f17"; ctx.fillRect(0, 0, W, H);

  const raw = type === "ivp" ? D.ivp : type === "ivr" ? D.ivr : D.vp;
  const vals = raw.filter(v => v !== null);
  let minV = Math.min(...vals), maxV = Math.max(...vals);

  // Add threshold lines
  let thresholds = [];
  if (type === "ivp" || type === "ivr") {{
    minV = 0; maxV = 100;
    thresholds = [{{v:70, c:"rgba(255,64,96,0.5)", l:"70"}}, {{v:30, c:"rgba(0,200,150,0.5)", l:"30"}}];
  }} else {{
    const pad = Math.max(Math.abs(minV), Math.abs(maxV)) * 0.15;
    minV -= pad; maxV += pad;
    thresholds = [{{v:0, c:"rgba(255,255,255,0.15)", l:"0"}}];
  }}

  const n = raw.length;
  const toX = i => padL + (i / (n - 1)) * cW;
  const toY = v => v === null ? null : padT + (1 - (v - minV) / (maxV - minV)) * cH;

  // Grid
  ctx.fillStyle = "#2a3a52"; ctx.font = "8px 'Space Mono', monospace"; ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {{
    const v = minV + (maxV - minV) * (i / 4);
    const y = padT + (1 - i / 4) * cH;
    ctx.strokeStyle = "#141e2e"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(type === "vp" ? 1 : 0), padL - 4, y + 3);
  }}

  // Threshold lines
  for (const t of thresholds) {{
    const y = toY(t.v);
    ctx.strokeStyle = t.c; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = t.c; ctx.textAlign = "left";
    ctx.fillText(t.l, padL + 3, y - 3);
    ctx.textAlign = "right";
  }}

  // Color fill based on value
  const color = type === "vp"
    ? (v => v >= 0 ? "rgba(255,64,96,0.12)" : "rgba(0,200,150,0.12)")
    : (v => v >= 70 ? "rgba(255,64,96,0.12)" : v <= 30 ? "rgba(0,200,150,0.12)" : "rgba(123,140,222,0.08)");

  // Simplified: just draw the line with color based on current regime
  const lineColor = type === "vp"
    ? (raw[raw.length-1] >= 0 ? "#ff4060" : "#00c896")
    : type === "ivp" ? "#ffd166" : "#7b8cde";

  const pts = raw.map((v, i) => [toX(i), toY(v)]);

  // Fill under/over
  if (type === "vp") {{
    const zeroY = toY(0);
    ctx.save();
    for (let i = 1; i < raw.length; i++) {{
      if (raw[i] === null || raw[i-1] === null) continue;
      const x0 = toX(i-1), x1 = toX(i);
      const y0 = toY(raw[i-1]), y1 = toY(raw[i]);
      ctx.beginPath();
      ctx.moveTo(x0, zeroY); ctx.lineTo(x0, y0); ctx.lineTo(x1, y1); ctx.lineTo(x1, zeroY);
      ctx.closePath();
      ctx.fillStyle = raw[i] >= 0 ? "rgba(255,64,96,0.15)" : "rgba(0,200,150,0.15)";
      ctx.fill();
    }}
    ctx.restore();
  }}

  drawLine(ctx, pts, lineColor, 1.8);

  // X axis
  ctx.fillStyle = "#2a3a52"; ctx.font = "7.5px 'Space Mono', monospace"; ctx.textAlign = "center";
  const dateStep = Math.max(1, Math.floor(n / 6));
  for (let i = 0; i < n; i += dateStep) {{
    ctx.fillText(D.dates[i] ? D.dates[i].slice(5) : "", toX(i), H - 6);
  }}
}}

// ── CHART 3: Regime Timeline ───────────────────────────────────
function drawRegimeTimeline() {{
  const {{ canvas, ctx }} = getCtx("chart-regime");
  const W = canvas.width, H = canvas.height;
  const padL = 44, padR = 12, padT = 8, padB = 20;
  const cW = W - padL - padR, cH = H - padT - padB;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0c0f17"; ctx.fillRect(0, 0, W, H);

  const n = D.regimes.length;
  const barW = cW / n;

  for (let i = 0; i < n; i++) {{
    const r = D.regimes[i];
    if (!r) continue;
    const col = REGIME_COLORS[r] || "#7b8cde";
    const x = padL + i * barW;
    ctx.fillStyle = col + "cc";
    ctx.fillRect(x, padT, Math.ceil(barW) + 0.5, cH);
  }}

  // Date labels
  ctx.fillStyle = "#2a3a52"; ctx.font = "7px 'Space Mono', monospace"; ctx.textAlign = "center";
  const step = Math.max(1, Math.floor(n / 6));
  for (let i = 0; i < n; i += step) {{
    ctx.fillText(D.dates[i] ? D.dates[i].slice(5) : "", padL + i * barW, H - 4);
  }}

  // Legend
  let lx = padL;
  ctx.font = "7px 'Space Mono', monospace"; ctx.textAlign = "left";
  for (const [name, col] of Object.entries(REGIME_COLORS)) {{
    ctx.fillStyle = col;
    const short = name.split(" / ")[0];
    ctx.fillText("■ " + short, lx, padT + cH + 14);
    lx += ctx.measureText("■ " + short).width + 14;
    if (lx > W - 60) break;
  }}
}}

// ── Wire controls ───────────────────────────────────────────────
document.querySelectorAll("[data-range]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("[data-range]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    drawIVRV(parseInt(btn.getAttribute("data-range")));
  }});
}});

document.querySelectorAll("[data-chart]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("[data-chart]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    drawIVPChart(btn.getAttribute("data-chart"));
  }});
}});

// ── Init ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {{
  populateStats();
  drawIVRV(252);
  drawIVPChart("ivp");
  drawRegimeTimeline();

  let rsz;
  window.addEventListener("resize", () => {{
    clearTimeout(rsz);
    rsz = setTimeout(() => {{
      drawIVRV(parseInt(document.querySelector("[data-range].active")?.getAttribute("data-range") || 252));
      drawIVPChart(activeIVPChart);
      drawRegimeTimeline();
    }}, 150);
  }});
}});
</script>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. Fetch
    data = fetch_data(TICKER, IV_TICKER, LOOKBACK_DAYS)
    if data is None:
        raise RuntimeError("Gagal fetch data.")

    # 2. Compute
    df = compute_vol_metrics(data, RV_WINDOWS)

    # 3. Build HTML
    print(f"\n[HTML] Membangun dashboard...")
    html = build_html(df, TICKER, IV_TICKER, RV_WINDOWS)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[DONE] Dashboard disimpan ke: {OUTPUT_HTML}")
    print(f"       Buka di browser untuk melihat Vol Regime Dashboard.")

    # Quick summary di terminal
    last = df.iloc[-1]
    rv_mid = f"rv_{RV_WINDOWS[1]}"
    r = classify_regime(float(last["iv"]), float(last[rv_mid]), float(last["ivp"]))
    print(f"\n{'='*55}")
    print(f"  CURRENT REGIME: {r['regime']}")
    print(f"  IV             : {float(last['iv']):.2f}%")
    print(f"  IVP            : {float(last['ivp']):.1f}th percentile")
    print(f"  IVR            : {float(last['ivr']):.1f}")
    print(f"  Vol Premium    : {float(last['vol_premium']):+.2f}%")
    for w in RV_WINDOWS:
        print(f"  RV {w:2d}d          : {float(last[f'rv_{w}']):.2f}%")
    print(f"{'='*55}")