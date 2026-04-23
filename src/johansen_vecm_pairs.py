"""
╔══════════════════════════════════════════════════════════╗
║   Johansen + VECM Pairs Trading — Full Pipeline          ║
║   by FLOW                                                ║
║   deps: yfinance statsmodels plotly pandas numpy scipy   ║
╚══════════════════════════════════════════════════════════╝

Install:
  pip install yfinance statsmodels plotly pandas numpy scipy

Jalankan:
  python johansen_vecm_pairs.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─── Dependency check ─────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["yfinance", "statsmodels", "plotly", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[!] Package belum terinstall: {', '.join(missing)}")
        print(f"    Jalankan: pip install {' '.join(missing)}\n")
        sys.exit(1)

check_deps()

import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEFAULT_TICKER_1  = "AAPL"
DEFAULT_TICKER_2  = "MSFT"
DEFAULT_START     = "2022-01-01"
DEFAULT_END       = "2024-12-31"
ENTRY_Z           = 2.0     # threshold entry (z-score ECT)
EXIT_Z            = 0.5     # threshold exit
SIGNIFICANCE      = 0.05    # 0.01 / 0.05 / 0.10
VECM_LAGS         = 2       # lag order untuk VECM (k_ar_diff)
# ──────────────────────────────────────────────────────────────────────────────

SEP = "─" * 60


def get_input(prompt, default):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data(t1: str, t2: str, start: str, end: str) -> pd.DataFrame:
    print(f"\n[*] Downloading {t1} & {t2} dari Yahoo Finance...")
    raw = yf.download([t1, t2], start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        print("[!] Data kosong — cek ticker / koneksi.")
        sys.exit(1)
    close = raw["Close"][[t1, t2]].dropna()
    print(f"    {len(close)} hari trading ({close.index[0].date()} → {close.index[-1].date()})")
    return close


# ══════════════════════════════════════════════════════════════════════════════
# 2. ADF UNIT ROOT
# ══════════════════════════════════════════════════════════════════════════════

def adf_test(series: pd.Series, name: str) -> dict:
    r = adfuller(series.dropna(), autolag="AIC")
    return {
        "name":   name,
        "stat":   round(r[0], 4),
        "pvalue": round(r[1], 4),
        "crit":   {k: round(v, 4) for k, v in r[4].items()},
        "I1":     r[1] > 0.05,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. JOHANSEN TEST
# ══════════════════════════════════════════════════════════════════════════════

def johansen_test(data: pd.DataFrame, sig: float = 0.05) -> dict:
    """
    det_order = 0  → restricted constant (paling umum untuk price series)
    k_ar_diff = 1  → VAR(1) sebelum differencing = VECM(1)
    """
    res = coint_johansen(data, det_order=0, k_ar_diff=1)
    sig_idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(sig, 1)

    trace_stats = res.lr1
    trace_crit  = res.cvt[:, sig_idx]
    eigen_stats = res.lr2
    eigen_crit  = res.cvm[:, sig_idx]

    cointegrated = (trace_stats[0] > trace_crit[0]) and (eigen_stats[0] > eigen_crit[0])

    # Cointegrating vector pertama → hedge ratio β
    evec = res.evec[:, 0]
    beta = -evec[1] / evec[0]

    return {
        "cointegrated": cointegrated,
        "trace_stats":  trace_stats,
        "trace_crit":   trace_crit,
        "eigen_stats":  eigen_stats,
        "eigen_crit":   eigen_crit,
        "beta":         round(float(beta), 4),
        "sig_pct":      int(sig * 100),
        "evec":         evec,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. VECM — INTI MODEL
# ══════════════════════════════════════════════════════════════════════════════

def fit_vecm(data: pd.DataFrame, k_ar_diff: int = 2) -> dict:
    """
    Fit VECM(k_ar_diff) dengan 1 cointegrating relation.

    Output utama:
    ─────────────
    alpha   : adjustment coefficients [α₁, α₂]
              → seberapa cepat tiap series koreksi ke equilibrium
              → sign negatif = mean-reverting (bagus)
              → magnitude besar = cepat revert

    beta    : cointegrating vector (normalized)
              → long-run equilibrium relation

    ECT     : Error Correction Term time series
              → spread yang "beneran" dari perspektif VECM
              → ini yang dipake buat trading signal, bukan simple spread

    gamma   : short-run dynamics coefficients (k_ar_diff × 2 × 2 matrix)
              → efek lagged differences

    resid   : residuals dari VECM equations
    """
    model  = VECM(data, k_ar_diff=k_ar_diff, coint_rank=1, deterministic="co")
    result = model.fit()

    # Alpha: shape (2, 1) → flatten
    alpha = result.alpha.flatten()

    # Beta: normalized cointegrating vector, shape (2, 1) → flatten
    beta_vec = result.beta.flatten()

    # ECT = data @ beta_vec  (dot product setiap row)
    ect = data.values @ beta_vec  # shape (T,)
    ect_series = pd.Series(ect, index=data.index, name="ECT")

    # Normalize ECT ke z-score untuk signal
    ect_z = (ect_series - ect_series.mean()) / ect_series.std()

    # Half-life of mean reversion (dari AR(1) fit pada ECT)
    ect_lag  = ect_series.shift(1).dropna()
    ect_diff = ect_series.diff().dropna()
    aligned  = pd.concat([ect_diff, ect_lag], axis=1).dropna()
    rho      = np.polyfit(aligned.iloc[:, 1], aligned.iloc[:, 0], 1)[0]
    half_life = -np.log(2) / rho if rho < 0 else np.inf

    # Granger-like check: apakah α signifikan?
    alpha_tstat = result.alpha / (result.sigma_u.diagonal()[:, None] ** 0.5 + 1e-10)

    return {
        "result":     result,
        "alpha":      alpha,
        "beta_vec":   beta_vec,
        "ect":        ect_series,
        "ect_z":      ect_z,
        "half_life":  round(float(half_life), 2),
        "rho":        round(float(rho), 6),
        "alpha_t":    alpha_tstat.flatten(),
        "resid":      result.resid,
        "k_ar_diff":  k_ar_diff,
        "summary":    result.summary(),
    }


def dominant_reverter(alpha: np.ndarray, t1: str, t2: str) -> str:
    """
    Series dengan |α| lebih besar = mean-reverter dominan.
    Untuk pairs trading: short the series yang adjust lebih lambat,
    long yang adjust lebih cepat (atau sebaliknya tergantung sign spread).
    """
    if abs(alpha[0]) > abs(alpha[1]):
        return t1
    return t2


# ══════════════════════════════════════════════════════════════════════════════
# 5. BACKTEST — ECT-BASED SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def backtest_ect(ect_z: pd.Series, entry_z: float, exit_z: float,
                 prices: pd.DataFrame, t1: str, t2: str,
                 beta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strategy:
      - ECT_z < -entry_z  → long spread (long t1, short β×t2)
      - ECT_z > +entry_z  → short spread (short t1, long β×t2)
      - |ECT_z| < exit_z  → flat / keluar

    P&L dihitung dari actual price returns, bukan z-score proxy.
    """
    pos      = 0
    trades   = []
    pnl_list = []
    cum_pnl  = 0.0
    entry_idx = None
    entry_p1  = None
    entry_p2  = None

    for i, date in enumerate(ect_z.index):
        z  = ect_z.iloc[i]
        p1 = prices[t1].iloc[i]
        p2 = prices[t2].iloc[i]

        daily_pnl = 0.0
        if pos != 0 and i > 0:
            ret1 = (p1 - prices[t1].iloc[i-1]) / prices[t1].iloc[i-1]
            ret2 = (p2 - prices[t2].iloc[i-1]) / prices[t2].iloc[i-1]
            # long spread = long t1, short β×t2
            daily_pnl = pos * (ret1 - beta * ret2) * 1000

        cum_pnl += daily_pnl

        if pos == 0:
            if z < -entry_z:
                pos = 1;  entry_idx = date; entry_p1 = p1; entry_p2 = p2
            elif z > entry_z:
                pos = -1; entry_idx = date; entry_p1 = p1; entry_p2 = p2
        elif pos == 1 and z > -exit_z:
            trades.append({
                "entry_date": entry_idx, "exit_date": date,
                "side": "long spread", "entry_z": round(ect_z.loc[entry_idx], 3),
                "exit_z": round(z, 3), "pnl": round(cum_pnl, 2)
            })
            pos = 0
        elif pos == -1 and z < exit_z:
            trades.append({
                "entry_date": entry_idx, "exit_date": date,
                "side": "short spread", "entry_z": round(ect_z.loc[entry_idx], 3),
                "exit_z": round(z, 3), "pnl": round(cum_pnl, 2)
            })
            pos = 0

        pnl_list.append({
            "date": date, "cum_pnl": round(cum_pnl, 2),
            "position": pos, "ect_z": round(z, 4)
        })

    pnl_df    = pd.DataFrame(pnl_list).set_index("date")
    trades_df = pd.DataFrame(trades)
    return pnl_df, trades_df


def compute_metrics(pnl_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    daily_ret  = pnl_df["cum_pnl"].diff().dropna()
    sharpe     = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)
    cum        = pnl_df["cum_pnl"]
    rolling_max = cum.cummax()
    drawdown   = cum - rolling_max
    max_dd     = drawdown.min()
    win_rate   = (trades_df["pnl"] > 0).mean() * 100
    avg_win    = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() if (trades_df["pnl"] > 0).any() else 0
    avg_loss   = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean() if (trades_df["pnl"] < 0).any() else 0
    pf         = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    return {
        "sharpe":    round(sharpe, 3),
        "max_dd":    round(max_dd, 2),
        "win_rate":  round(win_rate, 1),
        "avg_win":   round(avg_win, 2),
        "avg_loss":  round(avg_loss, 2),
        "pf":        round(pf, 3),
        "n_trades":  len(trades_df),
        "final_pnl": round(pnl_df["cum_pnl"].iloc[-1], 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def vecm_diagnostics(vecm_result: dict) -> dict:
    resid = vecm_result["resid"]
    dw    = [round(durbin_watson(resid[:, i]), 4) for i in range(resid.shape[1])]
    # Jarque-Bera normality test on residuals
    jb = [stats.jarque_bera(resid[:, i]) for i in range(resid.shape[1])]
    return {
        "durbin_watson": dw,
        "jb_stat":  [round(j.statistic, 3) for j in jb],
        "jb_pval":  [round(j.pvalue, 4) for j in jb],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_full_summary(t1, t2, adf1, adf2, joh, vecm, diag, metrics, entry_z, exit_z):
    print(f"\n{SEP}")
    print(f"  FULL PIPELINE SUMMARY  |  {t1} / {t2}")
    print(SEP)

    # ADF
    print(f"\n  [1] ADF Unit Root Test")
    for r in [adf1, adf2]:
        tag = "I(1) ✓ (non-stationary)" if r["I1"] else "I(0) — already stationary"
        print(f"      {r['name']:6s}  stat={r['stat']:8.4f}  p={r['pvalue']:.4f}  {tag}")

    # Johansen
    print(f"\n  [2] Johansen Test  ({joh['sig_pct']}% level)")
    print(f"      {'Hypothesis':<14}  {'Trace':>8}  {'Crit':>8}  {'Eigen':>8}  {'Crit':>8}  Result")
    hyps = ["r = 0", "r <= 1"]
    for i in range(2):
        tr = "REJECT" if joh["trace_stats"][i] > joh["trace_crit"][i] else "accept"
        print(f"      {hyps[i]:<14}  {joh['trace_stats'][i]:>8.3f}  {joh['trace_crit'][i]:>8.3f}"
              f"  {joh['eigen_stats'][i]:>8.3f}  {joh['eigen_crit'][i]:>8.3f}  {tr}")
    cstr = "COINTEGRATED ✓" if joh["cointegrated"] else "NOT cointegrated ✗"
    print(f"\n      → {cstr}  |  Hedge ratio β = {joh['beta']}")

    # VECM
    alpha  = vecm["alpha"]
    bvec   = vecm["beta_vec"]
    hl     = vecm["half_life"]
    rho    = vecm["rho"]
    print(f"\n  [3] VECM  (k_ar_diff = {vecm['k_ar_diff']}, coint_rank = 1)")
    print(f"      Cointegrating vector  β = [{bvec[0]:+.4f}, {bvec[1]:+.4f}]")
    print(f"      ECT = {bvec[0]:+.4f}·{t1} + {bvec[1]:+.4f}·{t2}")
    print(f"\n      Adjustment coefficients α:")
    for i, (name, a) in enumerate(zip([t1, t2], alpha)):
        sign_ok = "mean-reverting ✓" if a < 0 else "diverging ✗"
        print(f"        α_{i+1} ({name:6s}) = {a:+.6f}   {sign_ok}")
    print(f"\n      Half-life of mean reversion : {hl} days")
    print(f"      AR(1) persistence (ρ)       : {rho}")
    dom = dominant_reverter(alpha, t1, t2)
    print(f"      Dominant mean-reverter      : {dom}")
    print(f"\n      Interpretation:")
    print(f"        Kalau ECT naik jauh dari 0 → {t1} dan {t2} adjust balik")
    print(f"        α₁={alpha[0]:+.4f} artinya {t1} koreksi {abs(alpha[0])*100:.2f}% per hari")
    print(f"        α₂={alpha[1]:+.4f} artinya {t2} koreksi {abs(alpha[1])*100:.2f}% per hari")

    # Diagnostics
    print(f"\n  [4] Diagnostics")
    for i, name in enumerate([t1, t2]):
        dw  = diag["durbin_watson"][i]
        jbp = diag["jb_pval"][i]
        dw_ok = "ok" if 1.5 < dw < 2.5 else "autocorrelation?"
        jb_ok = "normal" if jbp > 0.05 else "non-normal"
        print(f"      {name:6s}  Durbin-Watson={dw} ({dw_ok})  JB p={jbp} ({jb_ok})")

    # Backtest metrics
    if metrics:
        print(f"\n  [5] Backtest (entry z=±{entry_z}, exit z=±{exit_z})")
        print(f"      Trades     : {metrics['n_trades']}")
        print(f"      Win rate   : {metrics['win_rate']}%")
        print(f"      Sharpe     : {metrics['sharpe']}")
        print(f"      Max DD     : {metrics['max_dd']}")
        print(f"      Avg win    : {metrics['avg_win']}")
        print(f"      Avg loss   : {metrics['avg_loss']}")
        print(f"      Profit fac : {metrics['pf']}")
        print(f"      Final P&L  : {metrics['final_pnl']}")
    else:
        print(f"\n  [5] Backtest — tidak ada completed trade.")

    print(f"\n{SEP}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(data, t1, t2, joh, vecm, pnl_df, trades_df, entry_z, exit_z, metrics):
    ect_z = vecm["ect_z"]
    alpha = vecm["alpha"]
    hl    = vecm["half_life"]

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[
            f"[1] Price Series (normalized, base=100)",
            f"[2] ECT Z-score  |  half-life ≈ {hl}d  |  β = {joh['beta']}",
            f"[3] Adjustment Speed  |  α₁({t1})={alpha[0]:+.4f}  α₂({t2})={alpha[1]:+.4f}",
            f"[4] Johansen Trace vs Critical Value ({joh['sig_pct']}%)",
            f"[5] Simulated Cumulative P&L  |  Sharpe={metrics.get('sharpe','—')}",
        ],
        row_heights=[0.22, 0.22, 0.16, 0.16, 0.24],
        vertical_spacing=0.07
    )

    # ── 1. Price ────────────────────────────────────────────────────────────
    s1n = data[t1] / data[t1].iloc[0] * 100
    s2n = data[t2] / data[t2].iloc[0] * 100
    fig.add_trace(go.Scatter(x=data.index, y=s1n, name=t1,
                             line=dict(color="#3266ad", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=s2n, name=t2,
                             line=dict(color="#d85a30", width=1.5)), row=1, col=1)

    # ── 2. ECT z-score ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=ect_z.index, y=ect_z, name="ECT z-score",
                             line=dict(color="#888780", width=1.2)), row=2, col=1)
    for val, color, dash, lbl in [
        ( entry_z, "#e24b4a", "dash",  f"+{entry_z}"),
        (-entry_z, "#3266ad", "dash",  f"-{entry_z}"),
        ( exit_z,  "#639922", "dot",   f"+{exit_z}"),
        (-exit_z,  "#639922", "dot",   f"-{exit_z}"),
        ( 0,       "#b4b2a9", "solid", "0"),
    ]:
        fig.add_hline(y=val, line=dict(color=color, dash=dash, width=1),
                      annotation_text=lbl, annotation_font_size=9, row=2, col=1)

    long_e  = ect_z[ect_z < -entry_z]
    short_e = ect_z[ect_z >  entry_z]
    fig.add_trace(go.Scatter(x=long_e.index,  y=long_e,  mode="markers",
                             name="Long entry",
                             marker=dict(color="#3266ad", size=5, symbol="triangle-up")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=short_e.index, y=short_e, mode="markers",
                             name="Short entry",
                             marker=dict(color="#e24b4a", size=5, symbol="triangle-down")),
                  row=2, col=1)

    # ── 3. Alpha bar — adjustment speed ─────────────────────────────────────
    alpha_colors = ["#1d9e75" if a < 0 else "#e24b4a" for a in alpha]
    fig.add_trace(go.Bar(
        x=[t1, t2], y=alpha,
        marker_color=alpha_colors,
        text=[f"{a:+.4f}" for a in alpha],
        textposition="outside",
        name="α (adjustment)"
    ), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="#b4b2a9", dash="dot", width=1), row=3, col=1)

    # ── 4. Johansen trace bar ────────────────────────────────────────────────
    hyps = ["r = 0", "r ≤ 1"]
    bar_col = ["#1d9e75" if joh["trace_stats"][i] > joh["trace_crit"][i] else "#e24b4a"
               for i in range(2)]
    fig.add_trace(go.Bar(x=hyps, y=joh["trace_stats"],
                         marker_color=bar_col,
                         text=[f"{v:.2f}" for v in joh["trace_stats"]],
                         textposition="outside", name="Trace stat"), row=4, col=1)
    fig.add_trace(go.Bar(x=hyps, y=joh["trace_crit"],
                         marker_color="#b4b2a9",
                         text=[f"{v:.2f}" for v in joh["trace_crit"]],
                         textposition="outside", name=f"Crit {joh['sig_pct']}%"), row=4, col=1)

    # ── 5. P&L ───────────────────────────────────────────────────────────────
    last_pnl  = pnl_df["cum_pnl"].iloc[-1]
    pnl_color = "#1d9e75" if last_pnl >= 0 else "#e24b4a"
    fill_rgba = "rgba(29,158,117,0.08)" if last_pnl >= 0 else "rgba(226,75,74,0.08)"
    fig.add_trace(go.Scatter(x=pnl_df.index, y=pnl_df["cum_pnl"],
                             name="Cum. P&L",
                             line=dict(color=pnl_color, width=1.5),
                             fill="tozeroy", fillcolor=fill_rgba), row=5, col=1)
    fig.add_hline(y=0, line=dict(color="#b4b2a9", dash="dot", width=1), row=5, col=1)

    # Trade markers on P&L
    if not trades_df.empty:
        for _, tr in trades_df.iterrows():
            pnl_at_exit = pnl_df.loc[tr["exit_date"], "cum_pnl"] if tr["exit_date"] in pnl_df.index else None
            if pnl_at_exit is not None:
                fig.add_trace(go.Scatter(
                    x=[tr["exit_date"]], y=[pnl_at_exit],
                    mode="markers",
                    marker=dict(color="#1d9e75" if tr["pnl"] > 0 else "#e24b4a",
                                size=6, symbol="circle"),
                    showlegend=False
                ), row=5, col=1)

    # ── Layout ──────────────────────────────────────────────────────────────
    cstr = "COINTEGRATED ✓" if joh["cointegrated"] else "NOT cointegrated ✗"
    title = (f"Johansen + VECM Pairs Trading  |  {t1}/{t2}  |  "
             f"{cstr}  |  β={joh['beta']}  |  half-life≈{hl}d")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=1050,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        barmode="group",
        font=dict(family="monospace", size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.show()
    print("\n[*] Chart dibuka di browser.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  JOHANSEN + VECM — PAIRS TRADING FULL PIPELINE")
    print("═" * 60)
    print("  (tekan Enter buat pakai default value)\n")

    t1    = get_input("  Ticker 1", DEFAULT_TICKER_1).upper()
    t2    = get_input("  Ticker 2", DEFAULT_TICKER_2).upper()
    start = get_input("  Start date (YYYY-MM-DD)", DEFAULT_START)
    end   = get_input("  End date   (YYYY-MM-DD)", DEFAULT_END)

    try:
        ez    = float(get_input("  Entry z-score", str(ENTRY_Z)))
        xz    = float(get_input("  Exit z-score ", str(EXIT_Z)))
        sg    = float(get_input("  Significance (0.01/0.05/0.10)", str(SIGNIFICANCE)))
        lags  = int(get_input("  VECM lags (k_ar_diff)", str(VECM_LAGS)))
    except ValueError:
        print("[!] Input harus angka. Pakai default.")
        ez, xz, sg, lags = ENTRY_Z, EXIT_Z, SIGNIFICANCE, VECM_LAGS

    # 1. Data
    data = fetch_data(t1, t2, start, end)

    # 2. ADF
    print("\n[*] Running ADF test...")
    adf1 = adf_test(data[t1], t1)
    adf2 = adf_test(data[t2], t2)

    # 3. Johansen
    print("[*] Running Johansen test...")
    joh = johansen_test(data[[t1, t2]], sig=sg)

    if not joh["cointegrated"]:
        print(f"\n  [!] Pair {t1}/{t2} TIDAK cointegrated pada level {int(sg*100)}%.")
        print(f"      VECM tetap difit tapi hasilnya mungkin tidak reliable.")
        print(f"      Coba pair lain atau periode yang berbeda.\n")

    # 4. VECM
    print(f"[*] Fitting VECM(k_ar_diff={lags})...")
    vecm = fit_vecm(data[[t1, t2]], k_ar_diff=lags)

    # 5. Diagnostics
    print("[*] Running diagnostics...")
    diag = vecm_diagnostics(vecm)

    # 6. Backtest
    print("[*] Running backtest...")
    pnl_df, trades_df = backtest_ect(
        vecm["ect_z"], ez, xz, data, t1, t2, joh["beta"]
    )
    metrics = compute_metrics(pnl_df, trades_df)

    # 7. Summary
    print_full_summary(t1, t2, adf1, adf2, joh, vecm, diag, metrics, ez, xz)

    # 8. Optional: print VECM statsmodels summary
    show_detail = get_input("\n  Tampilkan VECM statsmodels summary? (y/n)", "n")
    if show_detail.lower() == "y":
        print("\n" + str(vecm["summary"]))

    # 9. Plot
    print("\n[*] Generating chart...")
    plot_all(data, t1, t2, joh, vecm, pnl_df, trades_df, ez, xz, metrics)


if __name__ == "__main__":
    main()