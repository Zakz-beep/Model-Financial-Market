import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import json
import warnings
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES FULL GREEKS
#  Returns dict: delta, gamma, vanna, charm, vega
# ══════════════════════════════════════════════════════════════════
def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    """
    Hitung semua Greeks pakai Black-Scholes analytik.

    Vanna  = dDelta/dIV  = dVega/dS
           = -norm.pdf(d1) * d2 / sigma
           Interpretasi: seberapa besar delta berubah kalau IV berubah 1 unit
           → Penting waktu IV expanding/contracting (vol events)

    Charm  = dDelta/dT   (theta decay of delta)
           = -norm.pdf(d1) * (2rT - d2*sigma*sqrt(T)) / (2T*sigma*sqrt(T))
           Interpretasi: seberapa besar delta berubah seiring waktu berlalu
           → Krusial menjelang OPEX (Charm pinning effect)

    Vega   = dPrice/dIV
           = S * norm.pdf(d1) * sqrt(T)
    """
    zero = {"delta": 0.0, "gamma": 0.0, "vanna": 0.0, "charm": 0.0, "vega": 0.0}
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return zero
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)

        # Delta
        if option_type == "Call":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (sama untuk call dan put)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))

        # Vanna = -pdf(d1) * d2 / sigma
        vanna = -pdf_d1 * d2 / sigma

        # Charm = -pdf(d1) * [2rT - d2*sigma*sqrt(T)] / [2T*sigma*sqrt(T)]
        # Sign flip untuk put
        charm_raw = -pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        charm = charm_raw if option_type == "Call" else -charm_raw

        # Vega
        vega = S * pdf_d1 * np.sqrt(T)

        return {
            "delta": delta,
            "gamma": gamma,
            "vanna": vanna,
            "charm": charm,
            "vega":  vega,
        }
    except Exception:
        return zero


# ══════════════════════════════════════════════════════════════════
#  EXPIRY BUCKET
# ══════════════════════════════════════════════════════════════════
def get_expiry_bucket(days: int) -> str | None:
    if days == 0:   return "0DTE"
    if days == 1:   return "1DTE"
    if days <= 7:   return "7DTE"
    if days <= 14:  return "14DTE"
    if days <= 30:  return "30DTE"
    return None


# ══════════════════════════════════════════════════════════════════
#  FETCH & CLEAN OPTIONS
# ══════════════════════════════════════════════════════════════════
def fetch_and_clean_options(ticker_symbol: str = "SPY", days_forward: int = 30) -> pd.DataFrame | None:
    print(f"\n[FETCH] Mengambil data options {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    all_expirations = ticker.options

    if not all_expirations:
        print("  Tidak ada data options tersedia.")
        return None

    today = datetime.date.today()
    filtered_exps = []
    for e in all_expirations:
        exp_date = datetime.datetime.strptime(e, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if 0 <= dte <= days_forward:
            filtered_exps.append((e, dte))

    print(f"  Total expiration tersedia : {len(all_expirations)}")
    print(f"  Expiration dalam 0-{days_forward} DTE : {len(filtered_exps)}")

    records = []
    for exp_str, dte in filtered_exps:
        bucket = get_expiry_bucket(dte)
        if bucket is None:
            continue
        try:
            chain = ticker.option_chain(exp_str)
            T = max(dte / 365.0, 1 / 365)
            for df_side, tipe in [(chain.calls, "Call"), (chain.puts, "Put")]:
                df_side = df_side.copy()
                df_side["tipe"]               = tipe
                df_side["tanggal_kedaluwarsa"] = exp_str
                df_side["dte"]                = dte
                df_side["bucket"]             = bucket
                df_side["T"]                  = T
                records.append(df_side)
        except Exception as e:
            print(f"  [SKIP] {exp_str}: {e}")

    if not records:
        return None

    raw = pd.concat(records, ignore_index=True)
    needed = ["strike", "openInterest", "impliedVolatility",
              "tipe", "tanggal_kedaluwarsa", "dte", "bucket", "T"]
    raw = raw[[c for c in needed if c in raw.columns]].copy()
    raw = raw.dropna(subset=["strike", "openInterest", "impliedVolatility"])
    raw = raw[raw["openInterest"] > 0]
    raw = raw[raw["impliedVolatility"] > 0]
    raw = raw[raw["impliedVolatility"] < 5.0]
    raw["strike"] = raw["strike"].astype(float)

    print(f"  Baris bersih              : {len(raw)}")
    return raw.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  COMPUTE ALL GREEKS PER BUCKET
# ══════════════════════════════════════════════════════════════════
def compute_all_greeks_for_bucket(df: pd.DataFrame, spot: float, r: float = 0.05):
    """
    Hitung GEX, Vanna Exposure, Charm Exposure, Vega Exposure per strike.
    Juga kalkulasi Call Wall, Put Wall, dan Gamma Flip.

    Dollar scaling:
      GEX   : gamma  × OI × 100 × spot² × 0.01   ($ per 1% move)
      VEX   : vanna  × OI × 100 × spot  × 0.01   ($ per 1% IV move)
      ChEX  : charm  × OI × 100                  (delta units per day)
      VegaEX: vega   × OI × 100                  ($ per 1 vol point)
    """
    df = df.copy()

    greeks_list = []
    for _, row in df.iterrows():
        g = bs_greeks(
            S=spot, K=row["strike"], T=row["T"],
            r=r, sigma=row["impliedVolatility"],
            option_type=row["tipe"]
        )
        greeks_list.append(g)

    greeks_df = pd.DataFrame(greeks_list)
    for col in ["delta", "gamma", "vanna", "charm", "vega"]:
        df[col] = greeks_df[col].values

    oi   = df["openInterest"]
    mult = 100

    # ── GEX (dealer perspective: call = long gamma, put = short gamma)
    df["gex"] = np.where(
        df["tipe"] == "Call",
        +df["gamma"] * oi * mult * (spot**2) * 0.01,
        -df["gamma"] * oi * mult * (spot**2) * 0.01,
    )

    # ── Vanna Exposure (VEX)
    # Call dealer: short call → short vanna dari perspective mereka
    # Put dealer: short put → long vanna
    # Net VEX per strike = (call_vanna - put_vanna) × OI × 100 × spot × 0.01
    df["vex"] = np.where(
        df["tipe"] == "Call",
        +df["vanna"] * oi * mult * spot * 0.01,
        -df["vanna"] * oi * mult * spot * 0.01,
    )

    # ── Charm Exposure (ChEX) — dalam delta units per day
    # Call dealer: long charm (positive), Put dealer: short charm
    df["chex"] = np.where(
        df["tipe"] == "Call",
        +df["charm"] * oi * mult,
        -df["charm"] * oi * mult,
    )

    # ── Vega Exposure (VegaEX) — $ per 1 vol point
    df["vegaex"] = np.where(
        df["tipe"] == "Call",
        +df["vega"] * oi * mult,
        -df["vega"] * oi * mult,
    )

    # ── Filter ±10% dari spot ──────────────────
    lower = spot * 0.90
    upper = spot * 1.10
    df = df[(df["strike"] >= lower) & (df["strike"] <= upper)].copy()

    def agg(col, rename):
        return (
            df.groupby("strike", as_index=False)[col]
            .sum()
            .rename(columns={col: rename})
            .sort_values("strike")
            .reset_index(drop=True)
        )

    gex_df   = agg("gex",    "net_gex")
    vex_df   = agg("vex",    "net_vex")
    chex_df  = agg("chex",   "net_chex")
    vega_df  = agg("vegaex", "net_vega")

    # ── Gamma Flip (zero-cross dari GEX) ──────
    sorted_gex = gex_df.sort_values("strike")
    flip_candidates = sorted_gex[
        sorted_gex["net_gex"].shift(1) * sorted_gex["net_gex"] < 0
    ]
    flip_level = float(flip_candidates.iloc[0]["strike"]) if not flip_candidates.empty else None

    # ── Call Wall & Put Wall ───────────────────
    call_df = df[df["tipe"] == "Call"].copy()
    call_df["gex_raw"] = call_df["gamma"] * call_df["openInterest"] * mult * (spot**2) * 0.01
    call_wall = None
    if not call_df.empty:
        cw = call_df.groupby("strike")["gex_raw"].sum()
        call_wall = float(cw.idxmax())

    put_df = df[df["tipe"] == "Put"].copy()
    put_df["gex_raw"] = put_df["gamma"] * put_df["openInterest"] * mult * (spot**2) * 0.01
    put_wall = None
    if not put_df.empty:
        pw = put_df.groupby("strike")["gex_raw"].sum()
        put_wall = float(pw.idxmax())

    # ── High Vanna Strike (peak absolute VEX) ─
    high_vanna = None
    if not vex_df.empty:
        high_vanna = float(vex_df.loc[vex_df["net_vex"].abs().idxmax(), "strike"])

    # ── High Charm Strike (peak absolute ChEX) ─
    high_charm = None
    if not chex_df.empty:
        high_charm = float(chex_df.loc[chex_df["net_chex"].abs().idxmax(), "strike"])

    return {
        "gex_df":      gex_df,
        "vex_df":      vex_df,
        "chex_df":     chex_df,
        "vega_df":     vega_df,
        "flip":        flip_level,
        "call_wall":   call_wall,
        "put_wall":    put_wall,
        "high_vanna":  high_vanna,
        "high_charm":  high_charm,
    }


# ══════════════════════════════════════════════════════════════════
#  BUILD HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════
def build_html(bucket_data: dict, spot: float, ticker: str = "SPY") -> str:
    today_str = datetime.date.today().strftime("%d %B %Y")

    js_data = {}
    for bucket, info in bucket_data.items():
        gex_df   = info["gex_df"]
        vex_df   = info["vex_df"]
        chex_df  = info["chex_df"]
        vega_df  = info["vega_df"]

        def df_to_lists(df, val_col, scale=1e6):
            return {
                "strikes": df["strike"].tolist(),
                "vals":    [round(v / scale, 5) for v in df[val_col].tolist()],
            }

        js_data[bucket] = {
            "gex":        df_to_lists(gex_df,  "net_gex",  1e6),
            "vex":        df_to_lists(vex_df,  "net_vex",  1e6),
            "chex":       df_to_lists(chex_df, "net_chex", 1e3),   # ribuan delta units
            "vega":       df_to_lists(vega_df, "net_vega", 1e6),
            "flip":       info["flip"],
            "call_wall":  info["call_wall"],
            "put_wall":   info["put_wall"],
            "high_vanna": info["high_vanna"],
            "high_charm": info["high_charm"],
            "expirations": info["expirations"],
            "total_gex":   round(gex_df["net_gex"].sum() / 1e6, 4) if not gex_df.empty else 0,
            "total_vex":   round(vex_df["net_vex"].sum()  / 1e6, 4) if not vex_df.empty else 0,
            "dominant_strike": float(gex_df.loc[gex_df["net_gex"].abs().idxmax(), "strike"]) if not gex_df.empty else None,
        }

    js_data_str = json.dumps(js_data)
    bucket_order     = ["0DTE", "1DTE", "7DTE", "14DTE", "30DTE"]
    available_buckets = [b for b in bucket_order if b in bucket_data]

    tab_buttons_html = ""
    for i, b in enumerate(available_buckets):
        active = "active" if i == 0 else ""
        tab_buttons_html += f'<button class="tab-btn {active}" data-bucket="{b}">{b}</button>\n'

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{ticker} Greeks Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {{
  --bg:      #080b10;
  --bg2:     #0d1118;
  --bg3:     #131920;
  --border:  #1c2535;
  --accent:  #00d4aa;
  --gold:    #ffd166;
  --text:    #dde4f0;
  --muted:   #4e6080;
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
    linear-gradient(rgba(0,212,170,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,170,0.025) 1px, transparent 1px);
  background-size: 44px 44px;
  pointer-events: none; z-index: 0;
}}
.container {{
  position: relative; z-index: 1;
  max-width: 1240px; margin: 0 auto; padding: 28px 20px;
}}

/* ── HEADER ── */
header {{
  display: flex; justify-content: space-between; align-items: flex-start;
  margin-bottom: 28px; flex-wrap: wrap; gap: 14px;
}}
.header-left h1 {{
  font-family: 'Space Mono', monospace;
  font-size: clamp(1.5rem, 4vw, 2.2rem);
  font-weight: 700; color: var(--accent); letter-spacing: -1px; line-height: 1;
}}
.header-left h1 span {{ color: var(--text); }}
.header-left p {{ color: var(--muted); font-size: 0.78rem; margin-top: 5px; font-family: 'Space Mono', monospace; }}
.spot-badge {{
  background: var(--bg3); border: 1px solid var(--border);
  border-left: 3px solid var(--gold); padding: 10px 18px; border-radius: 8px; text-align: right;
}}
.spot-badge .lbl {{ font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-family: 'Space Mono', monospace; }}
.spot-badge .val {{ font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700; color: var(--gold); margin-top: 2px; }}

/* ── STATS ── */
.stats-row {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 22px;
}}
.stat-card {{
  background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
  transition: border-color 0.2s;
}}
.stat-card:hover {{ border-color: var(--accent); }}
.stat-card .s-lbl {{ font-size: 0.63rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.1px; font-family: 'Space Mono', monospace; }}
.stat-card .s-val {{ font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700; margin-top: 5px; }}
.s-val.pos {{ color: var(--pos); }}
.s-val.neg {{ color: var(--neg); }}
.s-val.gold {{ color: var(--gold); }}
.s-val.vanna {{ color: var(--vanna); }}
.s-val.charm {{ color: var(--charm); }}

/* ── OUTER TABS (bucket) ── */
.outer-tabs {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }}
.outer-tabs-header {{
  display: flex; border-bottom: 1px solid var(--border); overflow-x: auto; scrollbar-width: none;
}}
.outer-tabs-header::-webkit-scrollbar {{ display: none; }}
.tab-btn {{
  flex: 0 0 auto; background: none; border: none; color: var(--muted);
  font-family: 'Space Mono', monospace; font-size: 0.75rem; font-weight: 700;
  letter-spacing: 1px; padding: 13px 22px; cursor: pointer; transition: all 0.2s;
  border-bottom: 2px solid transparent; white-space: nowrap;
}}
.tab-btn:hover {{ color: var(--text); }}
.tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); background: rgba(0,212,170,0.04); }}
.outer-tabs-body {{ padding: 20px; }}
.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}

/* ── INNER TABS (greek type) ── */
.inner-tabs {{
  display: flex; gap: 6px; margin-bottom: 18px; flex-wrap: wrap;
}}
.greek-btn {{
  background: var(--bg3); border: 1px solid var(--border); color: var(--muted);
  font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700;
  padding: 6px 14px; border-radius: 6px; cursor: pointer; transition: all 0.18s;
  letter-spacing: 0.5px;
}}
.greek-btn:hover {{ color: var(--text); border-color: #2e3d52; }}
.greek-btn.active.gex   {{ color: var(--pos);   border-color: var(--pos);   background: rgba(0,200,150,0.08); }}
.greek-btn.active.vanna {{ color: var(--vanna); border-color: var(--vanna); background: rgba(123,140,222,0.08); }}
.greek-btn.active.charm {{ color: var(--charm); border-color: var(--charm); background: rgba(244,162,97,0.08); }}
.greek-btn.active.vega  {{ color: var(--vega);  border-color: var(--vega);  background: rgba(167,139,250,0.08); }}

/* ── PANEL META ── */
.panel-meta {{
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 14px; flex-wrap: wrap; gap: 8px;
}}
.exp-list {{ font-size: 0.68rem; color: var(--muted); font-family: 'Space Mono', monospace; }}
.exp-list strong {{ color: var(--accent); }}
.badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.badge {{
  display: inline-flex; align-items: center; gap: 5px;
  border-radius: 5px; padding: 3px 10px;
  font-size: 0.65rem; font-family: 'Space Mono', monospace;
  border: 1px solid;
}}
.badge.flip  {{ color: var(--gold);  border-color: rgba(255,209,102,0.35); background: rgba(255,209,102,0.06); }}
.badge.cwall {{ color: var(--pos);   border-color: rgba(0,200,150,0.35);   background: rgba(0,200,150,0.06); }}
.badge.pwall {{ color: var(--neg);   border-color: rgba(255,64,96,0.35);   background: rgba(255,64,96,0.06); }}
.badge.hvanna {{ color: var(--vanna); border-color: rgba(123,140,222,0.35); background: rgba(123,140,222,0.06); }}
.badge.hcharm {{ color: var(--charm); border-color: rgba(244,162,97,0.35);  background: rgba(244,162,97,0.06); }}

/* ── CHART ── */
.chart-scroll {{
  overflow-y: auto; overflow-x: hidden; max-height: 540px;
  scrollbar-width: thin; scrollbar-color: var(--border) transparent;
}}
.chart-scroll::-webkit-scrollbar {{ width: 3px; }}
.chart-scroll::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
canvas {{ display: block; }}

/* ── GREEK DESC BOX ── */
.greek-desc {{
  background: var(--bg3); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; margin-bottom: 16px;
  font-size: 0.72rem; color: var(--muted); line-height: 1.6;
}}
.greek-desc strong {{ color: var(--text); }}

/* ── LEGEND ── */
.legend {{ display: flex; gap: 16px; margin-top: 14px; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.68rem; color: var(--muted); font-family: 'Space Mono', monospace; }}
.legend-dot {{ width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }}

footer {{
  margin-top: 28px; text-align: center; font-size: 0.65rem;
  color: var(--muted); font-family: 'Space Mono', monospace; padding-bottom: 20px;
}}
@media (max-width: 600px) {{
  .container {{ padding: 12px; }}
  .outer-tabs-body {{ padding: 12px; }}
}}
</style>
</head>
<body>
<div class="container">

<header>
  <div class="header-left">
    <h1><span>{ticker}</span> Greeks<span>.</span></h1>
    <p>GEX · Vanna · Charm · Vega Dashboard &bull; {today_str}</p>
  </div>
  <div class="spot-badge">
    <div class="lbl">Spot Price</div>
    <div class="val">${spot:.2f}</div>
  </div>
</header>

<div class="stats-row">
  <div class="stat-card"><div class="s-lbl">Bucket Active</div><div class="s-val" id="st-bucket">—</div></div>
  <div class="stat-card"><div class="s-lbl">Net GEX</div><div class="s-val" id="st-gex">—</div></div>
  <div class="stat-card"><div class="s-lbl">Gamma Flip</div><div class="s-val gold" id="st-flip">—</div></div>
  <div class="stat-card"><div class="s-lbl">Call Wall</div><div class="s-val pos" id="st-cwall">—</div></div>
  <div class="stat-card"><div class="s-lbl">Put Wall</div><div class="s-val neg" id="st-pwall">—</div></div>
  <div class="stat-card"><div class="s-lbl">Peak Vanna</div><div class="s-val vanna" id="st-vanna">—</div></div>
  <div class="stat-card"><div class="s-lbl">Peak Charm</div><div class="s-val charm" id="st-charm">—</div></div>
</div>

<div class="outer-tabs">
  <div class="outer-tabs-header">
    {tab_buttons_html}
  </div>
  <div class="outer-tabs-body">
    <div id="panels-container"></div>
  </div>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:var(--pos)"></div>Positive</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--neg)"></div>Negative</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--gold)"></div>Gamma Flip</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--pos);opacity:.5"></div>Call Wall</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--neg);opacity:.5"></div>Put Wall</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--vanna)"></div>Peak Vanna</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--charm)"></div>Peak Charm</div>
</div>

<footer>
  Data: Yahoo Finance via yfinance &bull; Greeks dihitung dengan Black-Scholes Analytik &bull; Range: Spot ±10%
</footer>
</div>

<script>
const DATA    = {js_data_str};
const SPOT    = {spot};
const BUCKETS = {json.dumps(available_buckets)};

// ── Greek configs ─────────────────────────────
const GREEK_CONFIGS = {{
  gex: {{
    label: "GEX",
    key: "gex",
    unit: "M",
    posColor: "#00c896",
    negColor: "#ff4060",
    desc: "<strong>Gamma Exposure (GEX)</strong> — Seberapa besar dealer harus hedge delta mereka kalau spot bergerak 1%. Positive GEX = dealer long gamma = mereka jual saat naik, beli saat turun → <strong>pinning/resistance</strong>. Negative GEX = dealer short gamma = mereka kejar harga → <strong>volatilitas amplified</strong>. Call Wall & Put Wall = strike dengan konsentrasi GEX terbesar.",
    wallKeys: ["call_wall", "put_wall"],
    peakKey: null,
  }},
  vanna: {{
    label: "Vanna",
    key: "vex",
    unit: "M",
    posColor: "#7b8cde",
    negColor: "#e07070",
    desc: "<strong>Vanna Exposure (VEX)</strong> — dDelta/dIV. Mengukur seberapa besar delta dealer berubah kalau IV berubah 1 unit. Waktu IV turun (vol crush post-event), Vanna mendorong dealer unwind hedge → bisa jadi <strong>fuel untuk rally</strong>. Waktu IV naik, Vanna bisa amplifikasi sell-off. Peak Vanna Strike = magnet price saat vol bergerak.",
    wallKeys: [],
    peakKey: "high_vanna",
  }},
  charm: {{
    label: "Charm",
    key: "chex",
    unit: "K",
    posColor: "#f4a261",
    negColor: "#e07070",
    desc: "<strong>Charm Exposure (ChEX)</strong> — dDelta/dT (theta decay of delta). Seberapa besar delta dealer berubah seiring waktu berlalu tanpa pergerakan harga. Sangat kuat menjelang <strong>OPEX</strong> — Charm pinning effect bisa paksa spot mendekati strike dengan ChEX terbesar karena dealer unwind hedge secara gradual. Satuan: ribuan delta units per hari.",
    wallKeys: [],
    peakKey: "high_charm",
  }},
  vega: {{
    label: "Vega",
    key: "vega",
    unit: "M",
    posColor: "#a78bfa",
    negColor: "#e07070",
    desc: "<strong>Vega Exposure (VegaEX)</strong> — dPrice/dIV. Seberapa besar P&L dealer berubah kalau IV bergerak 1 vol point. Strike dengan Vega tinggi adalah level di mana pasar paling sensitif terhadap perubahan implied volatility. Berguna untuk mengidentifikasi zona di mana vol expansion atau compression paling impactful.",
    wallKeys: [],
    peakKey: null,
  }},
}};

// ── Canvas chart ───────────────────────────────
function drawGreekChart(canvasId, bucket, greekKey) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const d    = DATA[bucket];
  const cfg  = GREEK_CONFIGS[greekKey];
  const src  = d[cfg.key];
  if (!src || src.strikes.length === 0) return;

  const strikes = src.strikes;
  const vals    = src.vals;

  const rowH = 22;
  const padL = 72;
  const padR = 90;
  const padT = 20;
  const padB = 32;
  const W    = canvas.parentElement.clientWidth || 960;
  const H    = strikes.length * rowH + padT + padB;

  canvas.width  = W;
  canvas.height = H;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0d1118";
  ctx.fillRect(0, 0, W, H);

  const chartW = W - padL - padR;
  const maxAbs = Math.max(...vals.map(Math.abs), 0.0001);
  const zeroX  = padL + chartW / 2;

  // ── special marks ──
  const specialStrikes = {{}};
  if (d.flip)       specialStrikes[d.flip.toFixed(1)]       = {{ type: "flip",   color: "#ffd166", label: "FLIP"    }};
  if (d.call_wall)  specialStrikes[d.call_wall.toFixed(1)]  = {{ type: "cwall",  color: "#00c896", label: "C.WALL"  }};
  if (d.put_wall)   specialStrikes[d.put_wall.toFixed(1)]   = {{ type: "pwall",  color: "#ff4060", label: "P.WALL"  }};
  if (d.high_vanna) specialStrikes[d.high_vanna.toFixed(1)] = {{ type: "hvanna", color: "#7b8cde", label: "VANNA↑"  }};
  if (d.high_charm) specialStrikes[d.high_charm.toFixed(1)] = {{ type: "hcharm", color: "#f4a261", label: "CHARM↑"  }};

  for (let i = 0; i < strikes.length; i++) {{
    const val    = vals[i];
    const y      = padT + i * rowH;
    const barLen = Math.abs(val) / maxAbs * (chartW / 2);
    const isPos  = val >= 0;
    const barColor    = isPos ? cfg.posColor : cfg.negColor;
    const barColorDim = isPos
      ? (cfg.posColor === "#00c896" ? "rgba(0,200,150,0.15)"  : "rgba(123,140,222,0.15)")
      : "rgba(255,64,96,0.15)";

    // Dimmed fill
    ctx.fillStyle = barColorDim;
    if (isPos) ctx.fillRect(zeroX, y + 2, barLen, rowH - 5);
    else       ctx.fillRect(zeroX - barLen, y + 2, barLen, rowH - 5);

    // Accent top line
    ctx.fillStyle = barColor;
    if (isPos) {{
      ctx.fillRect(zeroX, y + 2, barLen, 2);
      ctx.fillRect(zeroX, y + 2, Math.min(barLen, 2.5), rowH - 5);
    }} else {{
      ctx.fillRect(zeroX - barLen, y + 2, barLen, 2);
      ctx.fillRect(zeroX - barLen, y + 2, Math.min(barLen, 2.5), rowH - 5);
    }}

    // Spot check
    const isSpot = Math.abs(strikes[i] - SPOT) < 0.51;
    const sKey   = strikes[i].toFixed(1);
    const sp     = specialStrikes[sKey];

    // Strike label
    ctx.fillStyle = isSpot ? "#ffd166" : (sp ? sp.color : "#6b7f99");
    ctx.font = (isSpot || sp) ? "bold 9px 'Space Mono', monospace" : "9px 'Space Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText("$" + strikes[i].toFixed(1), padL - 5, y + rowH / 2 + 3);

    // Value label
    if (Math.abs(val) > maxAbs * 0.04) {{
      ctx.fillStyle = barColor;
      ctx.font = "7.5px 'Space Mono', monospace";
      ctx.textAlign = isPos ? "left" : "right";
      const lx = isPos ? zeroX + barLen + 3 : zeroX - barLen - 3;
      ctx.fillText((val >= 0 ? "+" : "") + val.toFixed(2) + cfg.unit, lx, y + rowH / 2 + 3);
    }}

    // Spot line
    if (isSpot) {{
      ctx.strokeStyle = "rgba(255,209,102,0.55)";
      ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(padL, y + rowH/2); ctx.lineTo(W - padR, y + rowH/2); ctx.stroke();
      ctx.setLineDash([]);
    }}

    // Special mark lines
    if (sp && !isSpot) {{
      ctx.strokeStyle = sp.color + "66";
      ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(padL, y + rowH/2); ctx.lineTo(W - padR, y + rowH/2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = sp.color;
      ctx.font = "bold 7.5px 'Space Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(sp.label, W - padR - 2, y + rowH/2 - 3);
    }}
  }}

  // Zero line
  ctx.strokeStyle = "#2a3545"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(zeroX, 0); ctx.lineTo(zeroX, H); ctx.stroke();

  // X axis
  const steps = [-1.0, -0.5, 0, 0.5, 1.0];
  ctx.fillStyle = "#3d5070"; ctx.font = "7.5px 'Space Mono', monospace"; ctx.textAlign = "center";
  for (const s of steps) {{
    const px = zeroX + s * (chartW / 2);
    ctx.fillText((s * maxAbs).toFixed(2) + cfg.unit, px, H - 8);
    ctx.strokeStyle = "#1a2535"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, H - padB); ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

// ── Build panels ──────────────────────────────
function buildPanels() {{
  const container = document.getElementById("panels-container");
  container.innerHTML = "";

  for (let i = 0; i < BUCKETS.length; i++) {{
    const b = BUCKETS[i];
    const d = DATA[b];
    const panel = document.createElement("div");
    panel.className = "tab-panel" + (i === 0 ? " active" : "");
    panel.id = "panel-" + b;

    const expsText = d.expirations.join(", ") || "—";

    let badgesHtml = "";
    if (d.flip)       badgesHtml += `<span class="badge flip">⟳ Flip ${{d.flip.toFixed(2)}}</span>`;
    if (d.call_wall)  badgesHtml += `<span class="badge cwall">▲ C.Wall ${{d.call_wall.toFixed(2)}}</span>`;
    if (d.put_wall)   badgesHtml += `<span class="badge pwall">▼ P.Wall ${{d.put_wall.toFixed(2)}}</span>`;
    if (d.high_vanna) badgesHtml += `<span class="badge hvanna">∂ Vanna ${{d.high_vanna.toFixed(2)}}</span>`;
    if (d.high_charm) badgesHtml += `<span class="badge hcharm">θ Charm ${{d.high_charm.toFixed(2)}}</span>`;

    // Inner tabs per greek
    let innerTabsHtml = "";
    for (const [gk, gcfg] of Object.entries(GREEK_CONFIGS)) {{
      const isFirst = gk === "gex";
      innerTabsHtml += `<button class="greek-btn ${{gk}} ${{isFirst ? "active" : ""}}" data-greek="${{gk}}" data-bucket="${{b}}">${{gcfg.label}}</button>`;
    }}

    // Canvas per greek
    let canvasesHtml = "";
    for (const [gk, gcfg] of Object.entries(GREEK_CONFIGS)) {{
      canvasesHtml += `
        <div class="greek-panel ${{gk === "gex" ? "active" : ""}}" id="gpanel-${{b}}-${{gk}}" style="display:${{gk === "gex" ? "block" : "none"}}">
          <div class="greek-desc">${{gcfg.desc}}</div>
          <div class="chart-scroll"><canvas id="canvas-${{b}}-${{gk}}"></canvas></div>
        </div>`;
    }}

    panel.innerHTML = `
      <div class="panel-meta">
        <div class="exp-list">Expirations: <strong>${{expsText}}</strong></div>
        <div class="badges">${{badgesHtml}}</div>
      </div>
      <div class="inner-tabs">${{innerTabsHtml}}</div>
      ${{canvasesHtml}}
    `;
    container.appendChild(panel);
  }}

  // Wire inner tab clicks
  document.querySelectorAll(".greek-btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      const greek  = btn.getAttribute("data-greek");
      const bucket = btn.getAttribute("data-bucket");
      switchGreek(bucket, greek);
    }});
  }});
}}

function switchGreek(bucket, greek) {{
  // Update inner tab buttons
  document.querySelectorAll(`[data-bucket="${{bucket}}"].greek-btn`).forEach(b => b.classList.remove("active"));
  const activeBtn = document.querySelector(`[data-bucket="${{bucket}}"][data-greek="${{greek}}"]`);
  if (activeBtn) activeBtn.classList.add("active");

  // Show/hide greek panels
  for (const gk of Object.keys(GREEK_CONFIGS)) {{
    const gpanel = document.getElementById(`gpanel-${{bucket}}-${{gk}}`);
    if (gpanel) gpanel.style.display = gk === greek ? "block" : "none";
  }}

  setTimeout(() => drawGreekChart(`canvas-${{bucket}}-${{greek}}`, bucket, greek), 40);
}}

function updateStats(bucket) {{
  const d = DATA[bucket];
  document.getElementById("st-bucket").textContent  = bucket;
  const gexEl = document.getElementById("st-gex");
  gexEl.textContent = (d.total_gex >= 0 ? "+" : "") + d.total_gex.toFixed(2) + "M";
  gexEl.className = "s-val " + (d.total_gex >= 0 ? "pos" : "neg");
  document.getElementById("st-flip").textContent   = d.flip       ? "$" + d.flip.toFixed(2)       : "N/A";
  document.getElementById("st-cwall").textContent  = d.call_wall  ? "$" + d.call_wall.toFixed(2)  : "N/A";
  document.getElementById("st-pwall").textContent  = d.put_wall   ? "$" + d.put_wall.toFixed(2)   : "N/A";
  document.getElementById("st-vanna").textContent  = d.high_vanna ? "$" + d.high_vanna.toFixed(2) : "N/A";
  document.getElementById("st-charm").textContent  = d.high_charm ? "$" + d.high_charm.toFixed(2) : "N/A";
}}

function switchBucket(bucket) {{
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  const btn   = document.querySelector(`.tab-btn[data-bucket="${{bucket}}"]`);
  const panel = document.getElementById("panel-" + bucket);
  if (btn)   btn.classList.add("active");
  if (panel) panel.classList.add("active");
  updateStats(bucket);
  // Draw active greek for this bucket (default = gex)
  const activeGreek = document.querySelector(`[data-bucket="${{bucket}}"].greek-btn.active`);
  const gk = activeGreek ? activeGreek.getAttribute("data-greek") : "gex";
  setTimeout(() => drawGreekChart(`canvas-${{bucket}}-${{gk}}`, bucket, gk), 50);
}}

document.addEventListener("DOMContentLoaded", () => {{
  buildPanels();

  document.querySelectorAll(".tab-btn").forEach(btn => {{
    btn.addEventListener("click", () => switchBucket(btn.getAttribute("data-bucket")));
  }});

  if (BUCKETS.length > 0) switchBucket(BUCKETS[0]);

  let rsz;
  window.addEventListener("resize", () => {{
    clearTimeout(rsz);
    rsz = setTimeout(() => {{
      const ab = document.querySelector(".tab-btn.active");
      if (!ab) return;
      const bk = ab.getAttribute("data-bucket");
      const ag = document.querySelector(`[data-bucket="${{bk}}"].greek-btn.active`);
      const gk = ag ? ag.getAttribute("data-greek") : "gex";
      drawGreekChart(`canvas-${{bk}}-${{gk}}`, bk, gk);
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
    TICKER         = "SPY"
    RISK_FREE_RATE = 0.05
    OUTPUT_HTML    = "GLD_Greeks_dashboard.html"

    # 1. Spot price
    print(f"[INIT] Mengambil spot price {TICKER}...")
    spot_data = yf.Ticker(TICKER).history(period="1d")
    if spot_data.empty:
        raise RuntimeError("Gagal mengambil spot price SPY.")
    SPOT = float(spot_data["Close"].iloc[-1])
    print(f"       Spot price: ${SPOT:.2f}")

    # 2. Fetch & clean
    df_clean = fetch_and_clean_options(ticker_symbol=TICKER, days_forward=30)
    if df_clean is None:
        raise RuntimeError("Tidak ada data options yang valid.")

    # 3. Compute per bucket
    BUCKET_ORDER = ["0DTE", "1DTE", "7DTE", "14DTE", "30DTE"]
    bucket_data  = {}

    print("\n[GREEKS] Menghitung Full Greeks per expiry bucket...")
    for bucket in BUCKET_ORDER:
        subset = df_clean[df_clean["bucket"] == bucket].copy()
        if subset.empty:
            print(f"  [{bucket}] Tidak ada data — skip")
            continue

        expirations = sorted(subset["tanggal_kedaluwarsa"].unique().tolist())
        result = compute_all_greeks_for_bucket(subset, spot=SPOT, r=RISK_FREE_RATE)

        if result["gex_df"].empty:
            print(f"  [{bucket}] Kosong setelah filter ±10% — skip")
            continue

        result["expirations"] = expirations
        bucket_data[bucket]   = result

        print(f"  [{bucket}] {len(result['gex_df'])} strikes | "
              f"Flip: {'$'+str(round(result['flip'],2)) if result['flip'] else 'N/A'} | "
              f"CWall: {'$'+str(round(result['call_wall'],2)) if result['call_wall'] else 'N/A'} | "
              f"PWall: {'$'+str(round(result['put_wall'],2)) if result['put_wall'] else 'N/A'} | "
              f"PeakVanna: {'$'+str(round(result['high_vanna'],2)) if result['high_vanna'] else 'N/A'} | "
              f"PeakCharm: {'$'+str(round(result['high_charm'],2)) if result['high_charm'] else 'N/A'}")

    if not bucket_data:
        raise RuntimeError("Tidak ada bucket yang berhasil dihitung.")

    # 4. Build HTML
    print(f"\n[HTML] Membangun dashboard...")
    html_content = build_html(bucket_data, spot=SPOT, ticker=TICKER)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[DONE] Dashboard disimpan ke: {OUTPUT_HTML}")
    print(f"       Buka di browser untuk melihat GEX, Vanna, Charm, Vega per expiry bucket.")