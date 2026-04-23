import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import json
import warnings
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  BLACK-SCHOLES GAMMA
# ──────────────────────────────────────────────
def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
#  EXPIRY BUCKET HELPER
# ──────────────────────────────────────────────
def get_expiry_bucket(days: int) -> str:
    """
    Klasifikasikan DTE ke dalam bucket level expiry.
    """
    if days == 0:
        return "0DTE"
    elif days == 1:
        return "1DTE"
    elif days <= 7:
        return "7DTE"
    elif days <= 14:
        return "14DTE"
    elif days <= 30:
        return "30DTE"
    else:
        return None  # di luar range, skip


# ──────────────────────────────────────────────
#  FETCH & CLEAN OPTIONS DATA
# ──────────────────────────────────────────────
def fetch_and_clean_options(
    ticker_symbol: str = "SPY",
    days_forward: int = 30,
) -> pd.DataFrame | None:
    print(f"\n[FETCH] Mengambil data options {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    all_expirations = ticker.options

    if not all_expirations:
        print("  Tidak ada data options tersedia.")
        return None

    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=days_forward)

    # Filter expiration dalam range 0–30 hari
    filtered_exps = []
    for e in all_expirations:
        exp_date = datetime.datetime.strptime(e, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if 0 <= dte <= days_forward:
            filtered_exps.append((e, dte))

    print(f"  Total expiration tersedia : {len(all_expirations)}")
    print(f"  Expiration dalam 0-30 DTE : {len(filtered_exps)}")

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


# ──────────────────────────────────────────────
#  HITUNG GEX PER BUCKET
# ──────────────────────────────────────────────
def compute_gex_for_bucket(df: pd.DataFrame, spot: float, r: float = 0.05):
    gammas = [
        bs_gamma(S=spot, K=row["strike"], T=row["T"], r=r, sigma=row["impliedVolatility"])
        for _, row in df.iterrows()
    ]
    df = df.copy()
    df["gamma"] = gammas

    multiplier = 100 * (spot ** 2) * 0.01
    df["gex"] = np.where(
        df["tipe"] == "Call",
        +df["gamma"] * df["openInterest"] * multiplier,
        -df["gamma"] * df["openInterest"] * multiplier,
    )
    # ── Call Wall & Put Wall ──────────────────
    # Call wall: strike dengan total GEX Call terbesar (paling positif)
    call_df = df[df["tipe"] == "Call"].copy()
    call_df["gex_abs"] = call_df["gamma"] * call_df["openInterest"] * multiplier
    call_wall_strike = None
    if not call_df.empty:
        call_by_strike = call_df.groupby("strike")["gex_abs"].sum()
        call_wall_strike = float(call_by_strike.idxmax())

    # Put wall: strike dengan total GEX Put terbesar (absolut nilainya)
    put_df = df[df["tipe"] == "Put"].copy()
    put_df["gex_abs"] = put_df["gamma"] * put_df["openInterest"] * multiplier
    put_wall_strike = None
    if not put_df.empty:
        put_by_strike = put_df.groupby("strike")["gex_abs"].sum()
        put_wall_strike = float(put_by_strike.idxmax())

    gex_by_strike = (
        df.groupby("strike", as_index=False)["gex"]
        .sum()
        .rename(columns={"gex": "net_gex"})
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # Filter ±10% dari spot
    lower = spot * 0.90
    upper = spot * 1.10
    gex_by_strike = gex_by_strike[
        (gex_by_strike["strike"] >= lower) & (gex_by_strike["strike"] <= upper)
    ].copy()

    # Cari gamma flip
    sorted_df = gex_by_strike.sort_values("strike")
    flip_candidates = sorted_df[
        (sorted_df["net_gex"].shift(1) * sorted_df["net_gex"] < 0)
    ]
    flip_level = float(flip_candidates.iloc[0]["strike"]) if not flip_candidates.empty else None

    return gex_by_strike, flip_level, call_wall_strike, put_wall_strike


# ──────────────────────────────────────────────
#  BUILD HTML OUTPUT
# ──────────────────────────────────────────────
def build_html(bucket_data: dict, spot: float, ticker: str = "SPY") -> str:
    """
    bucket_data: dict {bucket_name: {"gex_df": df, "flip": float|None, "expirations": list}}
    """
    today_str = datetime.date.today().strftime("%d %B %Y")

    # Serialize data per bucket ke JSON untuk JavaScript
    js_data = {}
    for bucket, info in bucket_data.items():
        df = info["gex_df"]
        flip = info["flip"]
        exps = info["expirations"]
        js_data[bucket] = {
            "strikes": df["strike"].tolist(),
            "gex":     [round(v / 1e6, 4) for v in df["net_gex"].tolist()],
            "flip":    flip,
            "expirations": exps,
            "call_wall": info["call_wall"],   # ← tambah ini
            "put_wall":  info["put_wall"],    # ← tambah ini
            "total_gex": round(df["net_gex"].sum() / 1e6, 4),
            "dominant_strike": float(df.loc[df["net_gex"].abs().idxmax(), "strike"]) if not df.empty else None,
        }

    js_data_str = json.dumps(js_data)
    bucket_order = ["0DTE", "1DTE", "7DTE", "14DTE", "30DTE"]
    available_buckets = [b for b in bucket_order if b in bucket_data]

    # Buat tab buttons HTML
    tab_buttons_html = ""
    for i, b in enumerate(available_buckets):
        active = "active" if i == 0 else ""
        tab_buttons_html += f'<button class="tab-btn {active}" data-bucket="{b}">{b}</button>\n'

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{ticker} GEX Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {{
    --bg:        #0a0c10;
    --bg2:       #0f1117;
    --bg3:       #161b24;
    --border:    #1e2530;
    --accent:    #00d4aa;
    --accent2:   #ff6b6b;
    --gold:      #ffd166;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --pos:       #00c896;
    --neg:       #ff4d6d;
    --pos-dim:   rgba(0,200,150,0.15);
    --neg-dim:   rgba(255,77,109,0.15);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* GRID BG */
  body::before {{
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }}

  .container {{
    position: relative; z-index: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px;
  }}

  /* HEADER */
  header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 32px;
    flex-wrap: wrap;
    gap: 16px;
  }}

  .header-left h1 {{
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    line-height: 1;
  }}

  .header-left h1 span {{ color: var(--text); }}

  .header-left p {{
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
  }}

  .spot-badge {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    padding: 12px 20px;
    border-radius: 8px;
    text-align: right;
  }}

  .spot-badge .label {{
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
  }}

  .spot-badge .value {{
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--gold);
    margin-top: 2px;
  }}

  /* STATS ROW */
  .stats-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 28px;
  }}

  .stat-card {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    transition: border-color 0.2s;
  }}

  .stat-card:hover {{ border-color: var(--accent); }}

  .stat-card .s-label {{
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-family: 'Space Mono', monospace;
  }}

  .stat-card .s-value {{
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 6px;
    color: var(--text);
  }}

  .stat-card .s-value.pos {{ color: var(--pos); }}
  .stat-card .s-value.neg {{ color: var(--neg); }}
  .stat-card .s-value.gold {{ color: var(--gold); }}

  /* TABS */
  .tabs-wrapper {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
  }}

  .tabs-header {{
    display: flex;
    border-bottom: 1px solid var(--border);
    overflow-x: auto;
    scrollbar-width: none;
  }}

  .tabs-header::-webkit-scrollbar {{ display: none; }}

  .tab-btn {{
    flex: 0 0 auto;
    background: none;
    border: none;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 14px 24px;
    cursor: pointer;
    transition: all 0.2s;
    border-bottom: 2px solid transparent;
    position: relative;
    white-space: nowrap;
  }}

  .tab-btn:hover {{ color: var(--text); }}

  .tab-btn.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
    background: rgba(0,212,170,0.05);
  }}

  .tabs-body {{
    padding: 24px;
  }}

  /* TAB PANEL */
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  .panel-meta {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    flex-wrap: wrap;
    gap: 10px;
  }}

  .panel-meta .exp-list {{
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }}

  .panel-meta .exp-list strong {{ color: var(--accent); }}

  .flip-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,209,102,0.1);
    border: 1px solid rgba(255,209,102,0.3);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    color: var(--gold);
  }}

  .flip-badge::before {{
    content: '⟳';
    font-size: 0.9rem;
  }}

  /* CHART */
  .chart-wrapper {{
    position: relative;
    width: 100%;
    overflow: hidden;
  }}

  .chart-scroll {{
    overflow-y: auto;
    overflow-x: hidden;
    max-height: 520px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }}

  .chart-scroll::-webkit-scrollbar {{ width: 4px; }}
  .chart-scroll::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}

  canvas {{
    display: block;
  }}

  /* LEGEND */
  .legend {{
    display: flex;
    gap: 20px;
    margin-top: 16px;
    flex-wrap: wrap;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.75rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }}

  .legend-dot {{
    width: 10px; height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }}

  /* FOOTER */
  footer {{
    margin-top: 32px;
    text-align: center;
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    padding-bottom: 24px;
  }}

  /* RESPONSIVE */
  @media (max-width: 600px) {{
    .container {{ padding: 16px; }}
    .tabs-body {{ padding: 16px; }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <header>
    <div class="header-left">
      <h1><span>{ticker}</span> GEX<span>.</span></h1>
      <p>Gamma Exposure Dashboard &bull; {today_str}</p>
    </div>
    <div class="spot-badge">
      <div class="label">Spot Price</div>
      <div class="value">${spot:.2f}</div>
    </div>
  </header>

  <!-- STATS (updated by JS) -->
  <div class="stats-row" id="stats-row">
    <div class="stat-card">
      <div class="s-label">Bucket Active</div>
      <div class="s-value" id="stat-bucket">—</div>
    </div>
    <div class="stat-card">
      <div class="s-label">Total Net GEX</div>
      <div class="s-value" id="stat-total-gex">—</div>
    </div>
    <div class="stat-card">
      <div class="s-label">Gamma Flip</div>
      <div class="s-value gold" id="stat-flip">—</div>
    </div>
    <div class="stat-card">
      <div class="s-label">Dominant Strike</div>
      <div class="s-value" id="stat-dominant">—</div>
    </div>
  </div>

  <!-- TABS -->
  <div class="tabs-wrapper">
    <div class="tabs-header">
      {tab_buttons_html}
    </div>
    <div class="tabs-body">
      <!-- Panel placeholder per bucket, dibuat oleh JS -->
      <div id="panels-container"></div>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:var(--pos)"></div>Positive GEX (Dealer Long Γ — Resistance)</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--neg)"></div>Negative GEX (Dealer Short Γ — Accelerator)</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--gold)"></div>Gamma Flip Level</div>
    <div class="legend-item"><div class="legend-dot" style="background:#fff;opacity:.5"></div>Spot Price</div>
  </div>

  <footer>
    Data: Yahoo Finance via yfinance &bull; GEX dihitung menggunakan Black-Scholes &bull; Range: Spot ±10% &bull; Satuan: Juta USD
  </footer>
</div>

<script>
const DATA  = {js_data_str};
const SPOT  = {spot};
const BUCKETS = {json.dumps(available_buckets)};

// ── Canvas chart per bucket ──────────────────────
function drawChart(canvasId, bucket) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const d = DATA[bucket];
  if (!d || d.strikes.length === 0) return;

  const strikes = d.strikes;
  const gexVals = d.gex; // dalam juta
  const flip    = d.flip;

  const rowH   = 22;
  const padL   = 72;
  const padR   = 80;
  const padT   = 20;
  const padB   = 30;
  const W      = canvas.parentElement.clientWidth || 900;
  const H      = strikes.length * rowH + padT + padB;

  canvas.width  = W;
  canvas.height = H;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = "#0f1117";
  ctx.fillRect(0, 0, W, H);

  const chartW = W - padL - padR;
  const maxAbs = Math.max(...gexVals.map(Math.abs), 0.001);

  // ── Draw bars ──
  for (let i = 0; i < strikes.length; i++) {{
    const val    = gexVals[i];
    const y      = padT + i * rowH;
    const barLen = Math.abs(val) / maxAbs * (chartW / 2);
    const cx     = padL + chartW / 2;  // zero line x

    const isPos  = val >= 0;
    const isSpot = Math.abs(strikes[i] - SPOT) < 0.51;
    const isFlip = flip !== null && Math.abs(strikes[i] - flip) < 0.51;

    // Bar fill
    ctx.fillStyle = isPos ? "rgba(0,200,150,0.18)" : "rgba(255,77,109,0.18)";
    if (isPos) {{
      ctx.fillRect(cx, y + 2, barLen, rowH - 5);
    }} else {{
      ctx.fillRect(cx - barLen, y + 2, barLen, rowH - 5);
    }}

    // Bar accent line
    ctx.fillStyle = isPos ? "#00c896" : "#ff4d6d";
    if (isPos) {{
      ctx.fillRect(cx, y + 2, Math.min(barLen, 3), rowH - 5);
      ctx.fillRect(cx, y + 2, barLen, 2);
    }} else {{
      ctx.fillRect(cx - barLen, y + 2, Math.min(barLen, 3), rowH - 5);
      ctx.fillRect(cx - barLen, y + 2, barLen, 2);
    }}

    // Strike label
    ctx.fillStyle = isSpot ? "#ffd166" : (isFlip ? "#ff9f43" : "#94a3b8");
    ctx.font = isSpot || isFlip ? "bold 9px 'Space Mono', monospace" : "9px 'Space Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText("$" + strikes[i].toFixed(1), padL - 6, y + rowH / 2 + 3);

    // Value label on bar end
    if (Math.abs(val) > maxAbs * 0.05) {{
      ctx.fillStyle = isPos ? "#00c896" : "#ff4d6d";
      ctx.font = "8px 'Space Mono', monospace";
      ctx.textAlign = isPos ? "left" : "right";
      const lx = isPos ? cx + barLen + 3 : cx - barLen - 3;
      ctx.fillText((val >= 0 ? "+" : "") + val.toFixed(2) + "M", lx, y + rowH / 2 + 3);
    }}

    // Row highlight (spot / flip)
    if (isSpot) {{
      ctx.strokeStyle = "rgba(255,209,102,0.5)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y + rowH / 2);
      ctx.lineTo(W - padR, y + rowH / 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }}
    if (isFlip && !isSpot) {{
      ctx.strokeStyle = "rgba(255,159,67,0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y + rowH / 2);
      ctx.lineTo(W - padR, y + rowH / 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }}
  }}
  const isCallWall = d.call_wall !== null && Math.abs(strikes[i] - d.call_wall) < 0.51;
    const isPutWall  = d.put_wall  !== null && Math.abs(strikes[i] - d.put_wall)  < 0.51;

    if (isCallWall) {{
      ctx.strokeStyle = "rgba(0,200,150,0.6)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y + rowH / 2);
      ctx.lineTo(W - padR, y + rowH / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      // Label
      ctx.fillStyle = "#00c896";
      ctx.font = "bold 8px 'Space Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText("CALL WALL", W - padR - 2, y + rowH / 2 - 3);
    }}

    if (isPutWall) {{
      ctx.strokeStyle = "rgba(255,77,109,0.6)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y + rowH / 2);
      ctx.lineTo(W - padR, y + rowH / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ff4d6d";
      ctx.font = "bold 8px 'Space Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText("PUT WALL", W - padR - 2, y + rowH / 2 - 3);
    }}

  // ── Zero line ──
  const zeroX = padL + chartW / 2;
  ctx.strokeStyle = "#2e3748";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(zeroX, 0);
  ctx.lineTo(zeroX, H);
  ctx.stroke();

  // ── X axis labels ──
  const steps = [-1.0, -0.5, 0, 0.5, 1.0];
  ctx.fillStyle = "#475569";
  ctx.font = "8px 'Space Mono', monospace";
  ctx.textAlign = "center";
  for (const s of steps) {{
    const px = zeroX + s * (chartW / 2);
    ctx.fillText((s * maxAbs).toFixed(2) + "M", px, H - 8);
    ctx.strokeStyle = "#1e2530";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(px, padT);
    ctx.lineTo(px, H - padB);
    ctx.stroke();
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
    const canvasId = "canvas-" + b;

    const panel = document.createElement("div");
    panel.className = "tab-panel" + (i === 0 ? " active" : "");
    panel.id = "panel-" + b;

    const expsText = d.expirations.join(", ") || "—";
    const flipText = d.flip !== null ? `${{d.flip.toFixed(2)}}` : "N/A";

    panel.innerHTML = `
      <div class="panel-meta">
        <div class="exp-list">Expirations: <strong>${{expsText}}</strong></div>
        ${{d.call_wall !== null ? `<div class="flip-badge" style="border-color:rgba(0,200,150,0.4);color:#00c896">▲ Call Wall: $${{d.call_wall.toFixed(2)}}</div>` : ""}}
      ${{d.put_wall !== null ? `<div class="flip-badge" style="border-color:rgba(255,77,109,0.4);color:#ff4d6d">▼ Put Wall: $${{d.put_wall.toFixed(2)}}</div>` : ""}}
      </div>
      <div class="chart-wrapper">
        <div class="chart-scroll">
          <canvas id="${{canvasId}}"></canvas>
        </div>
      </div>
    `;
    container.appendChild(panel);
  }}
}}

// ── Update stats ──────────────────────────────
function updateStats(bucket) {{
  const d = DATA[bucket];
  document.getElementById("stat-bucket").textContent = bucket;
  const totalGex = d.total_gex;
  const gexEl = document.getElementById("stat-total-gex");
  gexEl.textContent = (totalGex >= 0 ? "+" : "") + totalGex.toFixed(2) + "M";
  gexEl.className = "s-value " + (totalGex >= 0 ? "pos" : "neg");
  document.getElementById("stat-flip").textContent = d.flip !== null ? "$" + d.flip.toFixed(2) : "N/A";
  document.getElementById("stat-dominant").textContent = d.dominant_strike !== null ? "$" + d.dominant_strike.toFixed(2) : "N/A";
}}

// ── Tab switching ──────────────────────────────
function switchTab(bucket) {{
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

  const btn = document.querySelector(`.tab-btn[data-bucket="${{bucket}}"]`);
  const panel = document.getElementById("panel-" + bucket);
  if (btn) btn.classList.add("active");
  if (panel) panel.classList.add("active");

  updateStats(bucket);
  setTimeout(() => drawChart("canvas-" + bucket, bucket), 50);
}}

// ── Init ──────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {{
  buildPanels();

  document.querySelectorAll(".tab-btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      const b = btn.getAttribute("data-bucket");
      switchTab(b);
    }});
  }});

  if (BUCKETS.length > 0) {{
    switchTab(BUCKETS[0]);
  }}

  // Redraw on resize
  let resizeTimer;
  window.addEventListener("resize", () => {{
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {{
      const active = document.querySelector(".tab-btn.active");
      if (active) drawChart("canvas-" + active.getAttribute("data-bucket"), active.getAttribute("data-bucket"));
    }}, 150);
  }});
}});
</script>
</body>
</html>"""
    return html


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    TICKER         = "SPY"
    RISK_FREE_RATE = 0.05
    OUTPUT_HTML    = "SPY_GEX_dashboard.html"

    # 1. Spot price
    print(f"[INIT] Mengambil spot price {TICKER}...")
    spot_data = yf.Ticker(TICKER).history(period="1d")
    if spot_data.empty:
        raise RuntimeError("Gagal mengambil spot price SPY.")
    SPOT = float(spot_data["Close"].iloc[-1])
    print(f"       Spot price: ${SPOT:.2f}")

    # 2. Fetch & clean options (0–30 DTE)
    df_clean = fetch_and_clean_options(ticker_symbol=TICKER, days_forward=30)
    if df_clean is None:
        raise RuntimeError("Tidak ada data options yang valid.")

    # 3. Hitung GEX per bucket
    BUCKET_ORDER = ["0DTE", "1DTE", "7DTE", "14DTE", "30DTE"]
    bucket_data  = {}

    print("\n[GEX] Menghitung GEX per expiry bucket...")
    for bucket in BUCKET_ORDER:
        subset = df_clean[df_clean["bucket"] == bucket].copy()
        if subset.empty:
            print(f"  [{bucket}] Tidak ada data — skip")
            continue

        expirations = sorted(subset["tanggal_kedaluwarsa"].unique().tolist())
        gex_df, flip, call_wall, put_wall = compute_gex_for_bucket(subset, spot=SPOT, r=RISK_FREE_RATE)

        if gex_df.empty:
            print(f"  [{bucket}] GEX kosong setelah filter ±10% — skip")
            continue

        bucket_data[bucket] = {
            "gex_df":      gex_df,
            "flip":        flip,
            "call_wall":   call_wall,   # ← tambah ini
            "put_wall":    put_wall,    # ← tambah ini
            "expirations": expirations,
        }
        print(f"  [{bucket}] {len(gex_df)} strikes | Flip: {'$' + str(round(flip, 2)) if flip else 'N/A'} | Total GEX: ${gex_df['net_gex'].sum():,.0f}")

    if not bucket_data:
        raise RuntimeError("Tidak ada bucket yang berhasil dihitung.")

    # 4. Build & simpan HTML
    print(f"\n[HTML] Membangun dashboard...")
    html_content = build_html(bucket_data, spot=SPOT, ticker=TICKER)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[DONE] Dashboard disimpan ke: {OUTPUT_HTML}")
    print(f"       Buka file ini di browser untuk melihat chart interaktif.")