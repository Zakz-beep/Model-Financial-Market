"""
================================================================
  GARCH VOLATILITY MODEL — S&P 500 E-mini Futures (ES=F)
  Historical Analysis: April 6-10, 2026
  Output: Interactive HTML Dashboard (Plotly)
================================================================

Models:
  1. GARCH(1,1)     — Bollerslev (1986)
  2. EGARCH(1,1)    — Nelson (1991)
  3. GJR-GARCH(1,1) — Glosten-Jagannathan-Runkle (1993)
================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from arch import arch_model
import yfinance as yf
import webbrowser
import os
import sys
import io
import warnings

warnings.filterwarnings("ignore")

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# ──────────────────── Color Palette ────────────────────
CYAN   = "#00d2ff"
BLUE   = "#0099ff"
PURPLE = "#7c3aed"
PINK   = "#f472b6"
GREEN  = "#10b981"
ORANGE = "#f59e0b"
RED    = "#ef4444"
GOLD   = "#fbbf24"
TEAL   = "#14b8a6"
DIM    = "#5a6a7a"
TEXT   = "#c8d6e5"
BG     = "#0a0e17"
PANEL  = "#0f1520"
GRID   = "#1a2332"

MODEL_COLORS = {
    "GARCH(1,1)":     CYAN,
    "EGARCH(1,1)":    PURPLE,
    "GJR-GARCH(1,1)": ORANGE,
}

# ── Analysis window ──
ZOOM_START = "2026-04-06"
ZOOM_END   = "2026-04-10"
ZOOM_LABEL = "Apr 6-10, 2026"


# ================================================================
#  1. DATA
# ================================================================
def fetch_data():
    print("  >> Fetching ES=F data (1 year)...")
    tk = yf.Ticker("ES=F")
    hist = tk.history(period="1y")
    if hist.empty:
        raise ValueError("No data for ES=F")

    hist["LogReturn"] = np.log(hist["Close"] / hist["Close"].shift(1)) * 100
    hist.dropna(inplace=True)

    start = pd.Timestamp(ZOOM_START)
    end = pd.Timestamp(ZOOM_END)
    if hist.index.tz is not None:
        start = start.tz_localize(hist.index.tz)
        end = end.tz_localize(hist.index.tz)

    zoom = hist.loc[start:end]
    print(f"  >> Total obs: {len(hist)} | Zoom obs: {len(zoom)}")
    return hist, zoom


# ================================================================
#  2. FIT MODELS
# ================================================================
def fit_models(returns):
    results = {}
    configs = [
        ("GARCH(1,1)",     {"vol": "GARCH",  "p": 1, "q": 1}),
        ("EGARCH(1,1)",    {"vol": "EGARCH", "p": 1, "q": 1}),
        ("GJR-GARCH(1,1)", {"vol": "GARCH",  "p": 1, "o": 1, "q": 1}),
    ]
    for name, kwargs in configs:
        print(f"  >> Fitting {name}...")
        am = arch_model(returns, mean="Constant", **kwargs, dist="t")
        res = am.fit(disp="off", show_warning=False)
        results[name] = res
        print(f"     AIC={res.aic:.1f}  BIC={res.bic:.1f}")
    return results


# ================================================================
#  3. FORECAST
# ================================================================
def forecast_volatility(model_result, horizon=5):
    fcast = model_result.forecast(horizon=horizon)
    var_f = fcast.variance.iloc[-1].values
    return np.sqrt(var_f)


# ================================================================
#  4. INTERACTIVE HTML DASHBOARD
# ================================================================
def build_html_dashboard(hist, zoom, models, output_path):
    """Build a Plotly interactive HTML dashboard."""

    returns = hist["LogReturn"]
    best_name = min(models, key=lambda k: models[k].aic)
    best_model = models[best_name]

    cond_vols = {name: res.conditional_volatility for name, res in models.items()}
    forecast_vol = forecast_volatility(best_model, horizon=5)
    std_resid = best_model.std_resid.dropna()

    last_vol = cond_vols[best_name].iloc[-1]
    var_95 = -1.645 * last_vol
    var_99 = -2.326 * last_vol

    tail = hist.tail(60)
    vol_tail = {n: cv.tail(120) for n, cv in cond_vols.items()}

    # ── Build subplots ──
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Price & Log Returns (60 days)",
            f"Zoom: {ZOOM_LABEL}",
            "Conditional Volatility (120 days)",
            f"Volatility Forecast ({best_name})",
            "Q-Q Plot (Std. Residuals)",
            "Return Distribution & VaR",
        ],
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "table", "colspan": 3}, None, None],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
        row_heights=[0.38, 0.35, 0.27],
    )

    # ────────────────────────────────────────
    # Panel 1: Price + Returns (60 days)
    # ────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=tail.index, y=tail["Close"],
            name="Close Price", line=dict(color=CYAN, width=2),
            hovertemplate="%{x|%b %d}<br>Price: %{y:,.2f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False,
    )
    bar_colors = [GREEN if r >= 0 else RED for r in tail["LogReturn"]]
    fig.add_trace(
        go.Bar(
            x=tail.index, y=tail["LogReturn"],
            name="Log Return %", marker_color=bar_colors, opacity=0.5,
            hovertemplate="%{x|%b %d}<br>Return: %{y:.3f}%<extra></extra>",
        ),
        row=1, col=1, secondary_y=True,
    )
    # Highlight zoom window
    if not zoom.empty:
        fig.add_vrect(
            x0=zoom.index[0], x1=zoom.index[-1],
            fillcolor=GOLD, opacity=0.08, line_width=0,
            annotation_text=ZOOM_LABEL.split(",")[0],
            annotation_position="top left",
            annotation_font_color=GOLD, annotation_font_size=10,
            row=1, col=1,
        )

    # ────────────────────────────────────────
    # Panel 2: Zoom window detail
    # ────────────────────────────────────────
    if not zoom.empty:
        fig.add_trace(
            go.Scatter(
                x=zoom.index, y=zoom["Close"], name="Close (Zoom)",
                mode="lines+markers+text",
                line=dict(color=GOLD, width=3),
                marker=dict(size=10, color=BG, line=dict(color=GOLD, width=2.5)),
                text=[f"{v:,.0f}" for v in zoom["Close"]],
                textposition="top center",
                textfont=dict(color=GOLD, size=11),
                hovertemplate="%{x|%b %d}<br>Close: %{y:,.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2, secondary_y=False,
        )
        zoom_bar_c = [GREEN if r >= 0 else RED for r in zoom["LogReturn"]]
        fig.add_trace(
            go.Bar(
                x=zoom.index, y=zoom["LogReturn"],
                name="Return (Zoom)", marker_color=zoom_bar_c, opacity=0.45,
                hovertemplate="%{x|%b %d}<br>Return: %{y:.3f}%<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2, secondary_y=True,
        )
    else:
        fig.add_annotation(
            text=f"No data for {ZOOM_LABEL}",
            x=0.5, y=0.5, xref="x2", yref="y2",
            showarrow=False, font=dict(color=DIM, size=14),
        )

    # ────────────────────────────────────────
    # Panel 3: Conditional Volatility
    # ────────────────────────────────────────
    for name, cv in vol_tail.items():
        fig.add_trace(
            go.Scatter(
                x=cv.index, y=cv.values, name=name,
                line=dict(color=MODEL_COLORS[name], width=2),
                hovertemplate="%{x|%b %d}<br>Vol: %{y:.4f}%<extra>" + name + "</extra>",
            ),
            row=1, col=3,
        )
    if not zoom.empty:
        fig.add_vrect(
            x0=zoom.index[0], x1=zoom.index[-1],
            fillcolor=GOLD, opacity=0.08, line_width=0, row=1, col=3,
        )

    # ────────────────────────────────────────
    # Panel 4: Volatility Forecast
    # ────────────────────────────────────────
    days_ahead = [f"T+{i}" for i in range(1, 6)]
    gradient_colors = [
        f"rgba({int(0 + i*40)}, {int(210 - i*30)}, {int(255 - i*20)}, 0.85)"
        for i in range(5)
    ]
    fig.add_trace(
        go.Bar(
            x=days_ahead, y=forecast_vol,
            marker_color=gradient_colors,
            text=[f"{v:.4f}%" for v in forecast_vol],
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            hovertemplate="Day %{x}<br>Forecast Vol: %{y:.4f}%<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ────────────────────────────────────────
    # Panel 5: Q-Q Plot
    # ────────────────────────────────────────
    sorted_resid = np.sort(std_resid.values)
    n = len(sorted_resid)
    theoretical = norm.ppf(np.linspace(1 / (n + 1), n / (n + 1), n))

    fig.add_trace(
        go.Scattergl(
            x=theoretical, y=sorted_resid, mode="markers",
            marker=dict(
                size=3, color=np.abs(sorted_resid),
                colorscale="Bluered", opacity=0.7,
            ),
            name="Residuals",
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=2,
    )
    lim = max(abs(theoretical.min()), abs(theoretical.max()))
    fig.add_trace(
        go.Scatter(
            x=[-lim, lim], y=[-lim, lim], mode="lines",
            line=dict(color=RED, width=1.5, dash="dash"),
            name="45-deg line", showlegend=False,
        ),
        row=2, col=2,
    )

    # ────────────────────────────────────────
    # Panel 6: Return Distribution + VaR
    # ────────────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=returns, nbinsx=80, histnorm="probability density",
            marker_color=TEAL, opacity=0.7, name="Returns",
            hovertemplate="Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>",
        ),
        row=2, col=3,
    )
    x_n = np.linspace(returns.min(), returns.max(), 200)
    y_n = norm.pdf(x_n, returns.mean(), returns.std())
    fig.add_trace(
        go.Scatter(
            x=x_n, y=y_n, mode="lines", name="Normal Fit",
            line=dict(color=PINK, width=2),
        ),
        row=2, col=3,
    )
    fig.add_vline(
        x=var_95, line=dict(color=ORANGE, width=2, dash="dash"),
        annotation_text=f"VaR 95%: {var_95:.2f}%",
        annotation_font_color=ORANGE, annotation_font_size=10,
        row=2, col=3,
    )
    fig.add_vline(
        x=var_99, line=dict(color=RED, width=2, dash="dash"),
        annotation_text=f"VaR 99%: {var_99:.2f}%",
        annotation_font_color=RED, annotation_font_size=10,
        row=2, col=3,
    )

    # ────────────────────────────────────────
    # Panel 7: Model Comparison Table
    # ────────────────────────────────────────
    table_headers = ["Model", "AIC", "BIC", "Log-Lik", "Last Vol (%)", "Status"]
    table_values = [[], [], [], [], [], []]

    for name, res in models.items():
        lv = cond_vols[name].iloc[-1]
        is_best = name == best_name
        table_values[0].append(f"<b>{name}</b>" if is_best else name)
        table_values[1].append(f"{res.aic:.1f}")
        table_values[2].append(f"{res.bic:.1f}")
        table_values[3].append(f"{res.loglikelihood:.1f}")
        table_values[4].append(f"{lv:.4f}")
        table_values[5].append(
            f"<b style='color:{GREEN}'>BEST</b>" if is_best else "-"
        )

    # Separator
    for col in table_values:
        col.append("")

    table_values[0].append("<b>Risk Metric</b>")
    table_values[1].append("<b>Value</b>")
    for col in table_values[2:]:
        col.append("")

    table_values[0].append("VaR 95%")
    table_values[1].append(f"{var_95:.4f}%")
    for col in table_values[2:]:
        col.append("")

    table_values[0].append("VaR 99%")
    table_values[1].append(f"{var_99:.4f}%")
    for col in table_values[2:]:
        col.append("")

    table_values[0].append("5-Day Avg Forecast Vol")
    table_values[1].append(f"{forecast_vol.mean():.4f}%")
    for col in table_values[2:]:
        col.append("")

    # Separator
    for col in table_values:
        col.append("")

    table_values[0].append(f"<b>{best_name} Parameters</b>")
    table_values[1].append("<b>Value</b>")
    for col in table_values[2:]:
        col.append("")

    for param, val in best_model.params.items():
        table_values[0].append(f"  {param}")
        table_values[1].append(f"{val:.6f}")
        for col in table_values[2:]:
            col.append("")

    # Row colors
    n_rows = len(table_values[0])
    row_fills = []
    for i in range(n_rows):
        if table_values[0][i] == "":
            row_fills.append(BG)
        elif "<b>" in table_values[0][i] and "Parameters" not in table_values[0][i] and "Risk" not in table_values[0][i]:
            row_fills.append("#162030")
        else:
            row_fills.append(PANEL)

    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in table_headers],
                fill_color="#111827",
                font=dict(color=CYAN, size=12),
                align="left",
                line_color=GRID,
                height=32,
            ),
            cells=dict(
                values=table_values,
                fill_color=[row_fills],
                font=dict(color=TEXT, size=11),
                align="left",
                line_color=GRID,
                height=26,
            ),
        ),
        row=3, col=1,
    )

    # ── Global Layout ──
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT),
        title=dict(
            text=(
                "<b style='font-size:24px; color:#00d2ff'>"
                "GARCH VOLATILITY MODEL DASHBOARD"
                "</b><br>"
                f"<span style='font-size:13px; color:{DIM}'>"
                f"S&P 500 E-mini Futures (ES=F)  |  "
                f"Zoom: {ZOOM_LABEL}  |  "
                f"Best Model: {best_name} (AIC={best_model.aic:.1f})"
                "</span>"
            ),
            x=0.5,
            y=0.98,
        ),
        height=1200,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(15,21,32,0.8)",
            bordercolor=GRID,
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=40, t=100, b=40),
        hovermode="x unified",
    )

    # Style all axes
    axis_style = dict(
        gridcolor=GRID, gridwidth=0.5,
        linecolor=GRID, linewidth=0.5,
        zerolinecolor=GRID, zerolinewidth=0.5,
    )
    for i in range(1, 10):
        xa = f"xaxis{i}" if i > 1 else "xaxis"
        ya = f"yaxis{i}" if i > 1 else "yaxis"
        if xa in fig.layout:
            fig.layout[xa].update(**axis_style)
        if ya in fig.layout:
            fig.layout[ya].update(**axis_style)

    for ann in fig.layout.annotations:
        ann.font.color = TEXT
        ann.font.size = 13

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Daily Vol (%)", row=1, col=3)
    fig.update_yaxes(title_text="Forecast Vol (%)", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantile", row=2, col=2)
    fig.update_xaxes(title_text="Theoretical Quantile", row=2, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=3)
    fig.update_xaxes(title_text="Return (%)", row=2, col=3)

    # ── Save to HTML ──
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "toImageButtonOptions": {
                "format": "png", "width": 1920, "height": 1080,
                "filename": "garch_dashboard_apr6_10",
            },
        },
    )
    print(f"  >> Dashboard saved to: {output_path}")
    return output_path


# ================================================================
#  MAIN
# ================================================================
def main():
    print("=" * 56)
    print("   GARCH VOLATILITY MODEL — ES Futures (S&P 500)")
    print(f"   Analysis Window: {ZOOM_LABEL}")
    print("   Interactive HTML Dashboard")
    print("=" * 56)
    print()

    # 1. Fetch
    hist, zoom = fetch_data()

    # 2. Fit
    print()
    models = fit_models(hist["LogReturn"])

    # 3. Summary
    best = min(models, key=lambda k: models[k].aic)
    print(f"\n  >> Best model: {best}")

    if not zoom.empty:
        print(f"\n  >> {ZOOM_LABEL}:")
        print("  " + "-" * 50)
        for idx, row in zoom.iterrows():
            d = "+" if row["LogReturn"] >= 0 else ""
            print(
                f"  {idx.strftime('%Y-%m-%d')}  "
                f"Close: {row['Close']:>10,.2f}  "
                f"Return: {d}{row['LogReturn']:.3f}%"
            )
        print("  " + "-" * 50)
    else:
        print(f"\n  >> No trading data for {ZOOM_LABEL}")

    # 4. Forecast
    fvol = forecast_volatility(models[best])
    print(f"\n  >> 5-Day Forecast ({best}):")
    for d, v in enumerate(fvol, 1):
        print(f"     T+{d}: {v:.4f}% daily")

    # 5. Build HTML
    print()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        output_dir, "..", "output", "garch_dashboard_apr6_10.html"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_path = os.path.abspath(output_path)

    build_html_dashboard(hist, zoom, models, output_path)

    # 6. Auto-open
    print("  >> Opening in browser...")
    webbrowser.open(f"file:///{output_path.replace(os.sep, '/')}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
