"""
================================================
  Johansen Cointegration — Pairs Trading Model
  by FLOW | requires: yfinance statsmodels plotly
================================================
Install deps dulu:
  pip install yfinance statsmodels plotly pandas numpy
"""

import sys
import numpy as np
import pandas as pd

# ─── Dependency check ────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["yfinance", "statsmodels", "plotly"]:
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
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

# ─── CONFIG (edit sesuai kebutuhan lu) ───────────────────────────────────────
DEFAULT_TICKER_1  = "AAPL"
DEFAULT_TICKER_2  = "MSFT"
DEFAULT_START     = "2022-01-01"
DEFAULT_END       = "2024-12-31"
ENTRY_Z           = 2.0     # z-score threshold buat masuk trade
EXIT_Z            = 0.5     # z-score threshold buat keluar trade
SIGNIFICANCE      = 0.05    # level signifikansi (0.05 = 5%, 0.01 = 1%)
ROLLING_WINDOW    = 60      # window buat rolling hedge ratio (opsional)
# ─────────────────────────────────────────────────────────────────────────────


def get_input(prompt, default):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def fetch_data(t1: str, t2: str, start: str, end: str) -> pd.DataFrame:
    print(f"\n[*] Downloading {t1} & {t2} dari Yahoo Finance...")
    raw = yf.download([t1, t2], start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        print("[!] Data kosong — cek ticker atau koneksi internet.")
        sys.exit(1)
    close = raw["Close"][[t1, t2]].dropna()
    print(f"    {len(close)} trading days loaded ({close.index[0].date()} → {close.index[-1].date()})")
    return close


def adf_test(series: pd.Series, name: str) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "name":    name,
        "stat":    round(result[0], 4),
        "pvalue":  round(result[1], 4),
        "crit_1":  round(result[4]["1%"], 4),
        "crit_5":  round(result[4]["5%"], 4),
        "crit_10": round(result[4]["10%"], 4),
        "I1":      result[1] > 0.05,
    }


def johansen_test(data: pd.DataFrame, sig: float = 0.05) -> dict:
    """
    Johansen test untuk 2 series.
    det_order:  -1 = no const, 0 = const, 1 = linear trend
    k_ar_diff:  lag order (1 = VAR(1))
    """
    det_order = 0
    k_ar_diff = 1
    result = coint_johansen(data, det_order, k_ar_diff)

    # Significance level index: 0=10%, 1=5%, 2=1%
    sig_idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(sig, 1)

    trace_stats = result.lr1          # trace statistic
    trace_crit  = result.cvt[:, sig_idx]
    eigen_stats = result.lr2          # max-eigenvalue statistic
    eigen_crit  = result.cvm[:, sig_idx]

    # Hipotesis r=0: tidak ada cointegration vector
    # Hipotesis r<=1: maksimal 1 cointegration vector
    r0_trace  = trace_stats[0] > trace_crit[0]
    r0_eigen  = eigen_stats[0] > eigen_crit[0]
    cointegrated = r0_trace and r0_eigen

    # Eigenvector pertama = hedge ratio
    evec = result.evec[:, 0]
    beta = -evec[1] / evec[0]

    return {
        "cointegrated":   cointegrated,
        "r0_trace":       r0_trace,
        "r0_eigen":       r0_eigen,
        "trace_stats":    trace_stats,
        "trace_crit":     trace_crit,
        "eigen_stats":    eigen_stats,
        "eigen_crit":     eigen_crit,
        "beta":           round(float(beta), 4),
        "sig_pct":        int(sig * 100),
        "evec":           evec,
    }


def compute_spread_zscore(s1: pd.Series, s2: pd.Series, beta: float) -> pd.Series:
    spread = s1 - beta * s2
    mu  = spread.mean()
    std = spread.std()
    return (spread - mu) / std


def simulate_pnl(zscore: pd.Series, entry_z: float, exit_z: float) -> pd.DataFrame:
    pos, trades, pnl_list = 0, [], []
    cum_pnl, open_pnl = 0.0, 0.0
    entry_price, entry_idx = None, None

    for i, (idx, z) in enumerate(zscore.items()):
        daily_pnl = 0.0
        if pos == 1:
            daily_pnl = (zscore.iloc[i] - zscore.iloc[i-1]) * 100 if i > 0 else 0
        elif pos == -1:
            daily_pnl = -(zscore.iloc[i] - zscore.iloc[i-1]) * 100 if i > 0 else 0

        cum_pnl += daily_pnl

        if pos == 0:
            if z < -entry_z:
                pos = 1
                entry_price, entry_idx = z, idx
            elif z > entry_z:
                pos = -1
                entry_price, entry_idx = z, idx
        elif pos == 1 and z > -exit_z:
            trades.append({"entry": entry_idx, "exit": idx, "side": "long", "pnl": cum_pnl})
            pos = 0
        elif pos == -1 and z < exit_z:
            trades.append({"entry": entry_idx, "exit": idx, "side": "short", "pnl": cum_pnl})
            pos = 0

        pnl_list.append({"date": idx, "cum_pnl": round(cum_pnl, 2), "position": pos})

    return pd.DataFrame(pnl_list).set_index("date"), pd.DataFrame(trades)


def print_summary(t1, t2, adf1, adf2, joh, entry_z, exit_z):
    line = "─" * 56
    print(f"\n{line}")
    print(f"  JOHANSEN COINTEGRATION SUMMARY")
    print(f"  {t1} / {t2}")
    print(line)

    print(f"\n  ADF Unit Root Test (H0: series is stationary)")
    for r in [adf1, adf2]:
        tag = "I(1) ✓" if r["I1"] else "I(0) — mungkin sudah stasioner"
        print(f"  {r['name']:8s} | stat={r['stat']:8.4f} | p={r['pvalue']:.4f} | {tag}")

    print(f"\n  Johansen Test ({joh['sig_pct']}% significance)")
    print(f"  {'Hypothesis':<18} {'Trace':>8} {'Crit':>8} {'Result':>10}")
    hyps = ["r = 0", "r <= 1"]
    for i in range(2):
        res = "REJECT" if joh["trace_stats"][i] > joh["trace_crit"][i] else "fail to reject"
        print(f"  {hyps[i]:<18} {joh['trace_stats'][i]:>8.3f} {joh['trace_crit'][i]:>8.3f} {res:>10}")

    print(f"\n  {'Max-Eigenvalue':<18} {'Stat':>8} {'Crit':>8} {'Result':>10}")
    for i in range(2):
        res = "REJECT" if joh["eigen_stats"][i] > joh["eigen_crit"][i] else "fail to reject"
        print(f"  {hyps[i]:<18} {joh['eigen_stats'][i]:>8.3f} {joh['eigen_crit'][i]:>8.3f} {res:>10}")

    coint_str = "COINTEGRATED ✓" if joh["cointegrated"] else "NOT cointegrated ✗"
    print(f"\n  Kesimpulan  : {coint_str}")
    print(f"  Hedge ratio : β = {joh['beta']}")
    print(f"  Spread      : {t1} - {joh['beta']} × {t2}")
    print(f"  Entry z     : ±{entry_z}  |  Exit z: ±{exit_z}")
    print(line)


def plot_all(data, t1, t2, joh, zscore, pnl_df, trades_df, entry_z, exit_z):
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            f"Price Series: {t1} (blue) vs {t2} (orange)",
            f"Spread Z-score  |  β = {joh['beta']}",
            f"Johansen Trace Statistic vs Critical Value ({joh['sig_pct']}%)",
            "Simulated Cumulative P&L"
        ],
        row_heights=[0.28, 0.28, 0.20, 0.24],
        vertical_spacing=0.08
    )

    # ── 1. Price series ──────────────────────────────────────────────────────
    s1_norm = data[t1] / data[t1].iloc[0] * 100
    s2_norm = data[t2] / data[t2].iloc[0] * 100
    fig.add_trace(go.Scatter(x=data.index, y=s1_norm, name=t1,
                             line=dict(color="#3266ad", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=s2_norm, name=t2,
                             line=dict(color="#d85a30", width=1.5)), row=1, col=1)

    # ── 2. Z-score + bands ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score",
                             line=dict(color="#888780", width=1.2)), row=2, col=1)
    for val, color, dash, lbl in [
        ( entry_z, "#e24b4a", "dash",  f"+{entry_z} entry"),
        (-entry_z, "#3266ad", "dash",  f"-{entry_z} entry"),
        ( exit_z,  "#639922", "dot",   f"+{exit_z} exit"),
        (-exit_z,  "#639922", "dot",   f"-{exit_z} exit"),
        ( 0,       "#b4b2a9", "solid", "mean"),
    ]:
        fig.add_hline(y=val, line=dict(color=color, dash=dash, width=1),
                      annotation_text=lbl, annotation_font_size=10, row=2, col=1)

    # Mark entry signals on z-score
    long_entries  = zscore[zscore < -entry_z]
    short_entries = zscore[zscore >  entry_z]
    fig.add_trace(go.Scatter(x=long_entries.index, y=long_entries,
                             mode="markers", name="Long entry",
                             marker=dict(color="#3266ad", size=5, symbol="triangle-up")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=short_entries.index, y=short_entries,
                             mode="markers", name="Short entry",
                             marker=dict(color="#e24b4a", size=5, symbol="triangle-down")),
                  row=2, col=1)

    # ── 3. Johansen bar chart ─────────────────────────────────────────────────
    hyps = ["r = 0", "r ≤ 1"]
    bar_color = ["#1d9e75" if joh["trace_stats"][i] > joh["trace_crit"][i] else "#e24b4a"
                 for i in range(2)]
    fig.add_trace(go.Bar(x=hyps, y=joh["trace_stats"], name="Trace stat",
                         marker_color=bar_color, text=[f"{v:.2f}" for v in joh["trace_stats"]],
                         textposition="outside"), row=3, col=1)
    fig.add_trace(go.Bar(x=hyps, y=joh["trace_crit"], name=f"Crit val {joh['sig_pct']}%",
                         marker_color="#b4b2a9", text=[f"{v:.2f}" for v in joh["trace_crit"]],
                         textposition="outside"), row=3, col=1)

    # ── 4. P&L ───────────────────────────────────────────────────────────────
    last_pnl = pnl_df["cum_pnl"].iloc[-1]
    pnl_color = "#1d9e75" if last_pnl >= 0 else "#e24b4a"
    fig.add_trace(go.Scatter(x=pnl_df.index, y=pnl_df["cum_pnl"], name="Cum. P&L",
                             line=dict(color=pnl_color, width=1.5),
                             fill="tozeroy",
                             fillcolor=f"rgba({'29,158,117' if last_pnl>=0 else '226,75,74'},0.08)"),
                  row=4, col=1)
    fig.add_hline(y=0, line=dict(color="#b4b2a9", dash="dot", width=1), row=4, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    coint_str = "COINTEGRATED ✓" if joh["cointegrated"] else "NOT cointegrated ✗"
    fig.update_layout(
        title=dict(
            text=f"Johansen Pairs Trading | {t1} / {t2} | {coint_str} | β = {joh['beta']}",
            font=dict(size=15)
        ),
        height=900,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        barmode="group",
        showlegend=True,
        font=dict(family="monospace", size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    fig.show()
    print("\n[*] Chart dibuka di browser.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 56)
    print("  JOHANSEN COINTEGRATION — PAIRS TRADING MODEL")
    print("=" * 56)
    print("  (tekan Enter buat pakai default value)\n")

    t1    = get_input("  Ticker 1", DEFAULT_TICKER_1).upper()
    t2    = get_input("  Ticker 2", DEFAULT_TICKER_2).upper()
    start = get_input("  Start date (YYYY-MM-DD)", DEFAULT_START)
    end   = get_input("  End date   (YYYY-MM-DD)", DEFAULT_END)

    try:
        ez = float(get_input("  Entry z-score", str(ENTRY_Z)))
        xz = float(get_input("  Exit z-score ", str(EXIT_Z)))
        sg = float(get_input("  Significance (0.01 / 0.05 / 0.10)", str(SIGNIFICANCE)))
    except ValueError:
        print("[!] Input harus angka. Pakai default.")
        ez, xz, sg = ENTRY_Z, EXIT_Z, SIGNIFICANCE

    # 1. Fetch
    data = fetch_data(t1, t2, start, end)

    # 2. ADF
    print("\n[*] Running ADF unit root test...")
    adf1 = adf_test(data[t1], t1)
    adf2 = adf_test(data[t2], t2)

    # 3. Johansen
    print("[*] Running Johansen cointegration test...")
    joh = johansen_test(data[[t1, t2]], sig=sg)

    # 4. Spread & Z-score
    zscore = compute_spread_zscore(data[t1], data[t2], joh["beta"])

    # 5. P&L sim
    pnl_df, trades_df = simulate_pnl(zscore, ez, xz)

    # 6. Print summary
    print_summary(t1, t2, adf1, adf2, joh, ez, xz)

    if not trades_df.empty:
        print(f"\n  Total simulated trades : {len(trades_df)}")
        print(f"  Final cumulative P&L   : {pnl_df['cum_pnl'].iloc[-1]:.2f}")
        win_rate = (trades_df["pnl"] > 0).mean() * 100 if "pnl" in trades_df else 0
        print(f"  Win rate (approx)      : {win_rate:.1f}%")
    else:
        print("\n  Tidak ada completed trade dalam periode ini.")

    # 7. Plot
    print("\n[*] Generating interactive chart...")
    plot_all(data, t1, t2, joh, zscore, pnl_df, trades_df, ez, xz)


if __name__ == "__main__":
    main()