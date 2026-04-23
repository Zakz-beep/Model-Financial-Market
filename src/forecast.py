"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           GARCH VOLATILITY FORECASTING PIPELINE · ES FUTURES (S&P 500)     ║
║                          OUTPUT: HTML INTERACTIVE DASHBOARD                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Models compared:
  · GARCH(1,1)    – baseline symmetric
  · EGARCH(1,1)   – asymmetric (leverage effect)
  · GJR-GARCH(1,1)– Glosten-Jagannathan-Runkle (news impact)
  · GARCH(1,1) t  – Student-t innovation (fat tails)
  · GARCH(2,2)    – higher-order lag structure

Distributions tested: Normal, Student-t, GED (Generalized Error Distribution)
Output: Fully interactive HTML dashboard (Plotly.js via CDN)
"""

import sys
import io
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, kurtosis, skew, norm, t as student_t
from sklearn.cluster import KMeans
import yfinance as yf
from arch import arch_model
from arch.unitroot import ADF, KPSS

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TRADING_DAYS_PER_YEAR = 252
WEEKS_AHEAD = [1, 2, 3, 4]
DAYS_PER_WEEK = 5


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 · DATA ACQUISITION
# ════════════════════════════════════════════════════════════════════════════
def fetch_data(ticker="ES=F", years=3, start_date=None, end_date=None):
    _banner("STEP 1 · DATA ACQUISITION")
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    print(f"  Ticker: {ticker}  |  {start_date} → {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"  Rows: {len(df):,}  |  Close range: {df['Close'].min():.2f} – {df['Close'].max():.2f}")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 · DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════
def clean_data(df):
    _banner("STEP 2 · DATA CLEANING")
    df_c = df.copy()
    n_raw = len(df_c)

    n_dup = df_c.index.duplicated().sum()
    df_c = df_c[~df_c.index.duplicated(keep="last")]
    print(f"  [2.1] Duplicates removed : {n_dup}")

    bad_mask = df_c["Close"].isna() | (df_c["Close"] <= 0)
    df_c = df_c[~bad_mask]
    print(f"  [2.2] Bad rows removed   : {bad_mask.sum()}")

    full_idx = pd.bdate_range(df_c.index.min(), df_c.index.max())
    n_before = len(df_c)
    df_c = df_c.reindex(full_idx).ffill(limit=3).dropna(subset=["Close"])
    print(f"  [2.3] Days forward-filled: {len(df_c) - n_before}")

    close_vals = df_c["Close"].values
    median_c = np.median(close_vals)
    mad_c = np.median(np.abs(close_vals - median_c))
    mod_z = 0.6745 * (close_vals - median_c) / (mad_c + 1e-9)
    outlier_mask = np.abs(mod_z) > 3.5
    if outlier_mask.sum() > 0:
        lo_cap = np.percentile(close_vals[~outlier_mask], 0.5)
        hi_cap = np.percentile(close_vals[~outlier_mask], 99.5)
        df_c["Close"] = df_c["Close"].clip(lower=lo_cap, upper=hi_cap)
    print(f"  [2.4] Price outliers capped: {outlier_mask.sum()}")

    log_ret = np.log(df_c["Close"] / df_c["Close"].shift(1)).dropna()
    lo_pct, hi_pct = log_ret.quantile(0.01), log_ret.quantile(0.99)
    n_ret_out = ((log_ret < lo_pct) | (log_ret > hi_pct)).sum()
    log_ret = log_ret.clip(lower=lo_pct, upper=hi_pct)
    print(f"  [2.5] Return outliers winsorized: {n_ret_out}")
    print(f"  Returns: {len(log_ret):,}  |  Ann.Vol: {log_ret.std()*np.sqrt(252)*100:.2f}%")
    return df_c, log_ret


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 · DATA VALIDATION
# ════════════════════════════════════════════════════════════════════════════
def validate_data(log_ret):
    _banner("STEP 3 · DATA VALIDATION")
    ret = log_ret.values
    n = len(ret)
    results = {}

    # Stationarity
    adf = ADF(log_ret, lags=20)
    adf_pval = float(adf.pvalue)
    kpss_res = KPSS(log_ret, trend="c")
    kpss_pval = float(kpss_res.pvalue)
    results["adf"] = {"stat": float(adf.stat), "pval": adf_pval, "stationary": adf_pval < 0.05}
    results["kpss"] = {"stat": float(kpss_res.stat), "pval": kpss_pval, "stationary": kpss_pval > 0.05}
    results["is_stationary"] = (adf_pval < 0.05) and (kpss_pval > 0.05)
    print(f"  [A] ADF p={adf_pval:.4f} | KPSS p={kpss_pval:.4f} | Stationary: {results['is_stationary']}")

    # Ljung-Box
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_sq_pval = float(acorr_ljungbox(ret**2, lags=20, return_df=True)["lb_pvalue"].iloc[-1])
    results["lb_sq_pval"] = lb_sq_pval
    results["arch_effects"] = lb_sq_pval < 0.05
    print(f"  [B] Ljung-Box sq. p={lb_sq_pval:.5f} | ARCH effects: {results['arch_effects']}")

    # Jarque-Bera
    jb_stat, jb_pval = jarque_bera(ret)
    results["jb_stat"] = jb_stat
    results["jb_pval"] = jb_pval
    results["is_normal"] = jb_pval > 0.05
    print(f"  [C] JB stat={jb_stat:.2f} p={jb_pval:.2e} | Normal: {results['is_normal']}")

    # ARCH-LM
    from statsmodels.stats.diagnostic import het_arch
    arch_lm_stat, arch_lm_pval, _, _ = het_arch(ret, nlags=10)
    results["arch_lm_stat"] = arch_lm_stat
    results["arch_lm_pval"] = arch_lm_pval
    results["arch_lm_sig"] = arch_lm_pval < 0.05
    print(f"  [D] ARCH-LM p={arch_lm_pval:.5f} | GARCH appropriate: {results['arch_lm_sig']}")

    # Fat tails
    exc_kurt = float(kurtosis(ret, fisher=True))
    skewness = float(skew(ret))
    sorted_ret = np.sort(np.abs(ret))[::-1]
    k_hill = max(int(n ** 0.5), 30)
    hill_idx = k_hill / np.sum(np.log(sorted_ret[:k_hill] / sorted_ret[k_hill]))
    nu_est = max(6.0 / exc_kurt + 4, 2.1) if exc_kurt > 0 else np.inf
    results["skewness"] = skewness
    results["exc_kurt"] = exc_kurt
    results["hill_idx"] = hill_idx
    results["nu_est"] = nu_est
    results["fat_tailed"] = exc_kurt > 1.0
    print(f"  [E] Kurt={exc_kurt:.4f} Skew={skewness:.4f} Hill={hill_idx:.4f} ν≈{nu_est:.1f}")

    # Regime clustering
    roll_std = pd.Series(ret).rolling(21).std().dropna().values
    roll_std_ann = roll_std * np.sqrt(252) * 100
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(roll_std_ann.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)
    regime_map = {old: new for new, old in enumerate(order)}
    labels_ordered = np.array([regime_map[l] for l in labels])
    results["regime_labels"] = labels_ordered
    results["regime_centers"] = centers[order]
    results["roll_std_ann"] = roll_std_ann
    results["regime_names"] = ["Low Vol", "Mid Vol", "High Vol"]
    results["current_regime"] = int(labels_ordered[-1])
    print(f"  [F] Current regime: {results['regime_names'][results['current_regime']]}")

    dist_rec = "t" if (results["fat_tailed"] and not results["is_normal"]) else \
               "ged" if exc_kurt > 0.5 else "normal"
    results["dist_rec"] = dist_rec
    print(f"  Recommended dist: {dist_rec.upper()}")
    return results


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 · MODEL SELECTION
# ════════════════════════════════════════════════════════════════════════════
def fit_best_garch(log_ret, diagnostics, scale=100.0):
    _banner("STEP 4 · MODEL SELECTION & FITTING")
    ret_scaled = log_ret.values * scale

    candidates = [
        ("GARCH(1,1)-normal",    "GARCH", 1, 0, 1, "normal"),
        ("GARCH(1,1)-t",         "GARCH", 1, 0, 1, "t"),
        ("GARCH(1,1)-ged",       "GARCH", 1, 0, 1, "ged"),
        ("EGARCH(1,1)-t",        "EGARCH",1, 0, 1, "t"),
        ("EGARCH(1,1)-normal",   "EGARCH",1, 0, 1, "normal"),
        ("GJR-GARCH(1,1)-t",    "GARCH", 1, 1, 1, "t"),
        ("GJR-GARCH(1,1)-normal","GARCH", 1, 1, 1, "normal"),
        ("GARCH(2,2)-t",         "GARCH", 2, 0, 2, "t"),
    ]

    rows = []
    fitted_models = {}

    for label, vol, p, o, q, dist in candidates:
        try:
            am = arch_model(ret_scaled, vol=vol, p=p, o=o, q=q, dist=dist,
                            mean="Zero", rescale=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", show_warning=False)
            rows.append({
                "Model": label, "vol": vol, "p": p, "o": o, "q": q, "dist": dist,
                "AIC": res.aic, "BIC": res.bic, "LogL": res.loglikelihood,
                "Converged": res.convergence_flag == 0
            })
            fitted_models[label] = res
            conv = "✓" if res.convergence_flag == 0 else "✗"
            print(f"  {label:<28} AIC={res.aic:>10.2f}  {conv}")
        except Exception as e:
            print(f"  {label:<28} FAILED: {e}")

    results_df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    conv_df = results_df[results_df["Converged"] == True]
    if conv_df.empty:
        conv_df = results_df

    best_row = conv_df.iloc[0]
    best_label = best_row["Model"]
    best_model = fitted_models[best_label]

    print(f"\n  Best: {best_label}  |  AIC={best_row['AIC']:.4f}  BIC={best_row['BIC']:.4f}")
    return best_model, best_row["vol"], best_row["dist"], results_df, fitted_models


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 · FORECAST
# ════════════════════════════════════════════════════════════════════════════
def generate_forecasts(best_model, best_vol, best_dist, log_ret, diagnostics, scale=100.0):
    _banner("STEP 5 · FORECAST  (1–4 WEEKS)")
    max_horizon = max(WEEKS_AHEAD) * DAYS_PER_WEEK
    fc = best_model.forecast(horizon=max_horizon, reindex=False)
    daily_var_scaled = fc.variance.iloc[-1].values
    daily_vol_true = np.sqrt(daily_var_scaled) / scale
    ann_vol_daily = daily_vol_true * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    weekly_rows = []
    for w in WEEKS_AHEAD:
        h_start = (w - 1) * DAYS_PER_WEEK
        h_end = w * DAYS_PER_WEEK
        weekly_var_true = daily_vol_true[h_start:h_end] ** 2
        weekly_vol_true = np.sqrt(weekly_var_true.sum())
        weekly_vol_pct = weekly_vol_true * 100
        weekly_vol_ann = weekly_vol_true * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

        if best_dist == "t":
            nu = min(max(diagnostics.get("nu_est", 8.0), 2.1), 30)
            alpha_95 = student_t.ppf(0.025, df=nu)
            alpha_99 = student_t.ppf(0.005, df=nu)
        else:
            alpha_95 = norm.ppf(0.025)
            alpha_99 = norm.ppf(0.005)

        var_95 = alpha_95 * weekly_vol_pct
        var_99 = alpha_99 * weekly_vol_pct

        if best_dist == "t":
            nu_c = min(max(diagnostics.get("nu_est", 8.0), 2.1), 30)
            pdf_t = student_t.pdf(student_t.ppf(0.05, df=nu_c), df=nu_c)
            cvar_95 = -pdf_t / 0.05 * (nu_c + student_t.ppf(0.05, df=nu_c)**2) / (nu_c - 1) * weekly_vol_pct
        else:
            cvar_95 = -norm.pdf(norm.ppf(0.05)) / 0.05 * weekly_vol_pct

        h = h_end - h_start
        se_vol = weekly_vol_pct / np.sqrt(2 * h)
        ci_lo_95 = max(0, weekly_vol_pct - 1.96 * se_vol)
        ci_hi_95 = weekly_vol_pct + 1.96 * se_vol

        weekly_rows.append({
            "week": w,
            "weekly_vol_pct": round(weekly_vol_pct, 5),
            "weekly_vol_ann": round(weekly_vol_ann, 4),
            "var_95": round(var_95, 5),
            "var_99": round(var_99, 5),
            "cvar_95": round(cvar_95, 5),
            "ci_lo_95": round(ci_lo_95, 5),
            "ci_hi_95": round(ci_hi_95, 5),
        })
        print(f"  W{w}: ann_vol={weekly_vol_ann:.2f}%  VaR95={var_95:.4f}%  VaR99={var_99:.4f}%")

    return {
        "daily_vol_pct": (daily_vol_true * 100).tolist(),
        "ann_vol_daily": ann_vol_daily.tolist(),
        "weekly_rows": weekly_rows,
    }


# ════════════════════════════════════════════════════════════════════════════
#  HTML DASHBOARD BUILDER
# ════════════════════════════════════════════════════════════════════════════
def build_html_dashboard(
    df, log_ret, diagnostics, best_model, best_vol, best_dist,
    results_df, fitted_models, forecasts, ticker="ES=F"
):
    _banner("BUILDING HTML INTERACTIVE DASHBOARD")

    ret = log_ret.values
    dates_str = [d.strftime("%Y-%m-%d") for d in log_ret.index]
    close_vals = df["Close"].iloc[-len(log_ret):].values.tolist()

    # Rolling 21-day vol
    roll_21_vol = (pd.Series(ret, index=log_ret.index).rolling(21).std() * np.sqrt(252) * 100)
    roll_vol_list = [round(v, 4) if not np.isnan(v) else None for v in roll_21_vol.values]

    # Conditional volatility
    cond_vol_raw = np.asarray(best_model.conditional_volatility)
    cond_vol_ann = cond_vol_raw / 100 * np.sqrt(252) * 100
    cond_vol_idx = log_ret.index[-len(cond_vol_raw):]
    cond_dates = [d.strftime("%Y-%m-%d") for d in cond_vol_idx]
    cond_vol_list = [round(v, 4) for v in cond_vol_ann]

    # Return distribution histogram
    ret_pct = (ret * 100).tolist()
    ret_hist_count, ret_hist_edges = np.histogram(ret_pct, bins=80, density=True)
    ret_hist_x = [round(0.5 * (ret_hist_edges[i] + ret_hist_edges[i+1]), 5)
                  for i in range(len(ret_hist_count))]
    ret_hist_y = [round(v, 6) for v in ret_hist_count]

    # Normal overlay for distribution
    x_ln = np.linspace(min(ret_pct), max(ret_pct), 200)
    mu_r, sd_r = np.mean(ret_pct), np.std(ret_pct)
    normal_pdf_y = norm.pdf(x_ln, mu_r, sd_r).tolist()
    normal_pdf_x = x_ln.tolist()

    # Student-t overlay
    nu_p = min(max(diagnostics.get("nu_est", 8), 2.1), 30)
    t_pdf_y = (student_t.pdf((x_ln - mu_r) / sd_r, df=nu_p) / sd_r).tolist()

    # Regime scatter
    roll_std = pd.Series(ret).rolling(21).std().dropna().values
    roll_std_ann_list = (roll_std * np.sqrt(252) * 100).tolist()
    regime_labels = diagnostics["regime_labels"]
    roll_regime_dates = [d.strftime("%Y-%m-%d") for d in log_ret.index[20:]]

    regime_scatters = []
    for i in range(3):
        mask = regime_labels == i
        regime_scatters.append({
            "dates": [roll_regime_dates[j] for j in range(len(mask)) if mask[j]],
            "vols":  [round(roll_std_ann_list[j], 4) for j in range(len(mask)) if mask[j]],
            "name":  diagnostics["regime_names"][i],
            "center": round(float(diagnostics["regime_centers"][i]), 4),
        })

    # QQ plot data
    std_resid = best_model.std_resid
    if best_dist == "t":
        (osm, osr), (slope, intercept, r2) = stats.probplot(std_resid, dist=student_t, sparams=(nu_p,))
    else:
        (osm, osr), (slope, intercept, r2) = stats.probplot(std_resid)
    qq_line_x = [round(float(osm.min()), 4), round(float(osm.max()), 4)]
    qq_line_y = [round(slope * osm.min() + intercept, 4), round(slope * osm.max() + intercept, 4)]

    # Model comparison
    model_comp = []
    aic_best = float(results_df["AIC"].min())
    for _, row in results_df.iterrows():
        model_comp.append({
            "name": row["Model"],
            "aic": round(row["AIC"], 2),
            "bic": round(row["BIC"], 2),
            "delta_aic": round(row["AIC"] - aic_best, 2),
            "converged": bool(row["Converged"]),
        })

    # Forecast data
    weekly_rows = forecasts["weekly_rows"]
    best_label = results_df.iloc[0]["Model"]
    current_ann_vol = round(float(cond_vol_ann[-1]), 2)

    # Stats summary
    stats_summary = {
        "ticker": ticker,
        "n_obs": len(ret),
        "date_start": dates_str[0],
        "date_end": dates_str[-1],
        "ann_vol": round(float(np.std(ret) * np.sqrt(252) * 100), 2),
        "skewness": round(float(diagnostics["skewness"]), 4),
        "exc_kurtosis": round(float(diagnostics["exc_kurt"]), 4),
        "nu_est": round(float(nu_p), 2),
        "hill_idx": round(float(diagnostics["hill_idx"]), 4),
        "is_stationary": diagnostics["is_stationary"],
        "arch_effects": diagnostics["arch_effects"],
        "fat_tailed": diagnostics["fat_tailed"],
        "dist_rec": best_dist,
        "best_model": best_label,
        "current_regime": diagnostics["regime_names"][diagnostics["current_regime"]],
        "current_ann_vol": current_ann_vol,
        "adf_pval": round(diagnostics["adf"]["pval"], 6),
        "kpss_pval": round(diagnostics["kpss"]["pval"], 6),
        "arch_lm_pval": round(float(diagnostics["arch_lm_pval"]), 6),
        "jb_pval": round(float(diagnostics["jb_pval"]), 4),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Inject all data as JSON into HTML ────────────────────────────────
    data_payload = {
        "dates": dates_str,
        "close": [round(v, 2) for v in close_vals],
        "roll_vol": roll_vol_list,
        "cond_dates": cond_dates,
        "cond_vol": cond_vol_list,
        "ret_hist_x": ret_hist_x,
        "ret_hist_y": ret_hist_y,
        "normal_pdf_x": normal_pdf_x,
        "normal_pdf_y": [round(v, 6) for v in normal_pdf_y],
        "t_pdf_x": normal_pdf_x,
        "t_pdf_y": [round(v, 6) for v in t_pdf_y],
        "regime_scatters": regime_scatters,
        "qq_x": [round(float(v), 4) for v in osm],
        "qq_y": [round(float(v), 4) for v in osr],
        "qq_line_x": qq_line_x,
        "qq_line_y": qq_line_y,
        "qq_r2": round(float(r2), 6),
        "model_comp": model_comp,
        "weekly_rows": weekly_rows,
        "stats": stats_summary,
    }

    html_content = _build_html_template(data_payload)
    return html_content


def _build_html_template(data):
    data_json = json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>GARCH Dashboard · {data['stats']['ticker']}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&family=Exo+2:ital,wght@0,200;0,400;0,600;1,300&display=swap" rel="stylesheet"/>
<style>
  :root {{
    --bg:       #050b14;
    --panel:    #080f1c;
    --card:     #0c1628;
    --border:   #162035;
    --cyan:     #00d4ff;
    --blue:     #0077ff;
    --purple:   #8b5cf6;
    --pink:     #f472b6;
    --green:    #10b981;
    --orange:   #f59e0b;
    --red:      #ef4444;
    --gold:     #fbbf24;
    --teal:     #14b8a6;
    --text:     #b8cce0;
    --textdim:  #3d5268;
    --mono:     'Share Tech Mono', monospace;
    --head:     'Rajdhani', sans-serif;
    --body:     'Exo 2', sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--body);
    font-weight: 400;
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── scanline overlay ── */
  body::before {{
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: repeating-linear-gradient(
      0deg,
      transparent, transparent 2px,
      rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px
    );
  }}

  /* ── noise grain ── */
  body::after {{
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.03;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }}

  /* ── layout ── */
  .wrapper {{ position: relative; z-index: 1; max-width: 1600px; margin: 0 auto; padding: 0 20px 40px; }}

  /* ── HEADER ── */
  .header {{
    display: flex; flex-direction: column; align-items: center;
    padding: 36px 0 24px;
    border-bottom: 1px solid var(--border);
    position: relative;
  }}
  .header-glow {{
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--purple), var(--pink), transparent);
    filter: blur(1px);
  }}
  .logo-tag {{
    font-family: var(--mono);
    font-size: 10px;
    color: var(--cyan);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 8px;
    opacity: 0.7;
  }}
  .header h1 {{
    font-family: var(--head);
    font-size: clamp(22px, 4vw, 38px);
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--purple) 50%, var(--pink) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
  }}
  .header-sub {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--textdim);
    letter-spacing: 1.5px;
  }}

  /* ── NAV TABS ── */
  .tabs {{
    display: flex; gap: 4px;
    margin: 20px 0 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
    overflow-x: auto;
  }}
  .tab {{
    font-family: var(--head);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 9px 20px;
    cursor: pointer;
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    color: var(--textdim);
    transition: all 0.2s;
    white-space: nowrap;
    user-select: none;
  }}
  .tab:hover {{ color: var(--cyan); border-color: var(--border); background: var(--card); }}
  .tab.active {{
    color: var(--cyan);
    border-color: var(--border);
    background: var(--panel);
    border-bottom-color: var(--panel);
    margin-bottom: -1px;
  }}

  /* ── PAGES ── */
  .page {{ display: none; animation: fadeIn 0.3s ease; }}
  .page.active {{ display: block; }}
  @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}

  /* ── STAT CARDS ── */
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 10px;
    margin: 20px 0;
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }}
  .stat-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent-color, var(--cyan));
  }}
  .stat-card:hover {{ border-color: var(--accent-color, var(--cyan)); }}
  .stat-label {{
    font-family: var(--mono);
    font-size: 9px;
    color: var(--textdim);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .stat-value {{
    font-family: var(--mono);
    font-size: 20px;
    font-weight: 700;
    color: var(--accent-color, var(--cyan));
    line-height: 1;
  }}
  .stat-sub {{
    font-family: var(--mono);
    font-size: 9px;
    color: var(--textdim);
    margin-top: 4px;
  }}

  /* ── BADGE ── */
  .badge {{
    display: inline-block;
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
    font-weight: 700;
  }}
  .badge-green  {{ background: rgba(16,185,129,0.15); color: var(--green);  border: 1px solid rgba(16,185,129,0.3); }}
  .badge-red    {{ background: rgba(239,68,68,0.15);  color: var(--red);   border: 1px solid rgba(239,68,68,0.3); }}
  .badge-orange {{ background: rgba(245,158,11,0.15); color: var(--orange);border: 1px solid rgba(245,158,11,0.3); }}
  .badge-cyan   {{ background: rgba(0,212,255,0.12);  color: var(--cyan);  border: 1px solid rgba(0,212,255,0.25); }}
  .badge-purple {{ background: rgba(139,92,246,0.15); color: var(--purple);border: 1px solid rgba(139,92,246,0.3); }}

  /* ── CHART PANELS ── */
  .chart-grid {{
    display: grid;
    gap: 14px;
    margin: 14px 0;
  }}
  .col-2 {{ grid-template-columns: 1fr 1fr; }}
  .col-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .col-full {{ grid-template-columns: 1fr; }}
  @media (max-width: 900px) {{ .col-2, .col-3 {{ grid-template-columns: 1fr; }} }}

  .chart-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .chart-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 16px 8px;
    border-bottom: 1px solid var(--border);
  }}
  .chart-title {{
    font-family: var(--head);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text);
  }}
  .chart-body {{ padding: 0; }}

  /* ── FORECAST TABLE ── */
  .fc-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 12px;
  }}
  .fc-table th {{
    background: rgba(0,212,255,0.07);
    color: var(--cyan);
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: right;
    border-bottom: 1px solid var(--border);
  }}
  .fc-table th:first-child {{ text-align: left; }}
  .fc-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid rgba(22,32,53,0.6);
    text-align: right;
    color: var(--text);
  }}
  .fc-table td:first-child {{ text-align: left; color: var(--cyan); font-weight: 700; }}
  .fc-table tr:hover td {{ background: rgba(255,255,255,0.02); }}

  /* ── MODEL TABLE ── */
  .model-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 11px;
  }}
  .model-table th {{
    background: rgba(139,92,246,0.08);
    color: var(--purple);
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    text-align: right;
  }}
  .model-table th:first-child {{ text-align: left; }}
  .model-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid rgba(22,32,53,0.4);
    text-align: right;
    color: var(--text);
  }}
  .model-table td:first-child {{ text-align: left; }}
  .model-table tr:first-child td {{ color: var(--gold); }}
  .model-table tr:hover td {{ background: rgba(255,255,255,0.015); }}

  /* ── DIAG GRID ── */
  .diag-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 10px;
    margin: 14px 0;
  }}
  .diag-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
  }}
  .diag-title {{
    font-family: var(--head);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--textdim);
    margin-bottom: 10px;
  }}
  .diag-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid rgba(22,32,53,0.6);
    font-family: var(--mono);
    font-size: 11px;
  }}
  .diag-row:last-child {{ border-bottom: none; }}
  .diag-key {{ color: var(--textdim); }}
  .diag-val {{ color: var(--text); }}

  /* ── FOOTER ── */
  .footer {{
    text-align: center;
    padding: 20px 0 10px;
    font-family: var(--mono);
    font-size: 9px;
    color: var(--textdim);
    letter-spacing: 1px;
    border-top: 1px solid var(--border);
    margin-top: 30px;
  }}

  /* ── Section label ── */
  .section-label {{
    font-family: var(--head);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--textdim);
    margin: 22px 0 10px;
    display: flex; align-items: center; gap: 10px;
  }}
  .section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  /* ── Plotly config ── */
  .plotly-graph-div {{ border-radius: 0 0 8px 8px; }}
</style>
</head>
<body>
<div class="wrapper">

  <!-- HEADER -->
  <header class="header">
    <div class="header-glow"></div>
    <div class="logo-tag">Quantitative Volatility Analysis</div>
    <h1>GARCH Forecast Dashboard</h1>
    <div class="header-sub" id="headerSub"></div>
  </header>

  <!-- TABS -->
  <nav class="tabs">
    <div class="tab active" onclick="switchTab('overview')">Overview</div>
    <div class="tab" onclick="switchTab('forecast')">Forecast</div>
    <div class="tab" onclick="switchTab('diagnostics')">Diagnostics</div>
    <div class="tab" onclick="switchTab('models')">Model Selection</div>
  </nav>

  <!-- ╔══════════════════════════════════════╗ -->
  <!-- ║  PAGE: OVERVIEW                       ║ -->
  <!-- ╚══════════════════════════════════════╝ -->
  <div class="page active" id="page-overview">
    <div class="stat-grid" id="statCards"></div>

    <div class="section-label">Price History & Volatility</div>
    <div class="chart-grid col-full">
      <div class="chart-panel">
        <div class="chart-header">
          <span class="chart-title">Price History · Rolling 21-Day Realised Vol</span>
        </div>
        <div class="chart-body" id="chartPrice"></div>
      </div>
    </div>

    <div class="section-label">Volatility Structure</div>
    <div class="chart-grid col-2">
      <div class="chart-panel">
        <div class="chart-header"><span class="chart-title">GARCH Conditional Volatility</span></div>
        <div class="chart-body" id="chartCondVol"></div>
      </div>
      <div class="chart-panel">
        <div class="chart-header"><span class="chart-title">Return Distribution</span></div>
        <div class="chart-body" id="chartDist"></div>
      </div>
    </div>
  </div>

  <!-- ╔══════════════════════════════════════╗ -->
  <!-- ║  PAGE: FORECAST                       ║ -->
  <!-- ╚══════════════════════════════════════╝ -->
  <div class="page" id="page-forecast">
    <div class="section-label">1–4 Week Forward Volatility</div>
    <div class="chart-grid col-full">
      <div class="chart-panel">
        <div class="chart-header">
          <span class="chart-title">Annualised Vol Forecast with 95% CI</span>
          <span id="currentVolBadge"></span>
        </div>
        <div class="chart-body" id="chartForecast"></div>
      </div>
    </div>

    <div class="section-label">Forecast Details</div>
    <div class="chart-panel">
      <div class="chart-header"><span class="chart-title">Weekly Risk Metrics</span></div>
      <div class="chart-body" style="padding:0">
        <table class="fc-table" id="forecastTable"></table>
      </div>
    </div>

    <div class="section-label">VaR / CVaR Comparison</div>
    <div class="chart-panel">
      <div class="chart-header"><span class="chart-title">Value-at-Risk Across Horizons</span></div>
      <div class="chart-body" id="chartVaR"></div>
    </div>
  </div>

  <!-- ╔══════════════════════════════════════╗ -->
  <!-- ║  PAGE: DIAGNOSTICS                    ║ -->
  <!-- ╚══════════════════════════════════════╝ -->
  <div class="page" id="page-diagnostics">
    <div class="section-label">Statistical Tests</div>
    <div class="diag-grid" id="diagCards"></div>

    <div class="section-label">Regime Analysis & Residuals</div>
    <div class="chart-grid col-2">
      <div class="chart-panel">
        <div class="chart-header"><span class="chart-title">Volatility Regime Clustering</span></div>
        <div class="chart-body" id="chartRegime"></div>
      </div>
      <div class="chart-panel">
        <div class="chart-header"><span class="chart-title">Q-Q Plot · Standardised Residuals</span></div>
        <div class="chart-body" id="chartQQ"></div>
      </div>
    </div>
  </div>

  <!-- ╔══════════════════════════════════════╗ -->
  <!-- ║  PAGE: MODELS                         ║ -->
  <!-- ╚══════════════════════════════════════╝ -->
  <div class="page" id="page-models">
    <div class="section-label">Model Comparison</div>
    <div class="chart-grid col-2">
      <div class="chart-panel">
        <div class="chart-header"><span class="chart-title">AIC Comparison (ΔAIC from best)</span></div>
        <div class="chart-body" id="chartAIC"></div>
      </div>
      <div class="chart-panel">
        <div class="chart-header"><span class="chart-title">BIC Comparison</span></div>
        <div class="chart-body" id="chartBIC"></div>
      </div>
    </div>
    <div class="section-label">Full Model Ranking</div>
    <div class="chart-panel">
      <div class="chart-header"><span class="chart-title">All Models · AIC/BIC/LogL</span></div>
      <div class="chart-body" style="padding: 0 0 8px">
        <table class="model-table" id="modelTable"></table>
      </div>
    </div>
  </div>

  <footer class="footer" id="footer"></footer>
</div>

<script>
// ─────────────────────────────────────────────────────────────────────────────
//  DATA PAYLOAD
// ─────────────────────────────────────────────────────────────────────────────
const D = {data_json};
const S = D.stats;

// ─────────────────────────────────────────────────────────────────────────────
//  TAB SWITCHING
// ─────────────────────────────────────────────────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t, i) => {{
    const ids = ['overview','forecast','diagnostics','models'];
    t.classList.toggle('active', ids[i] === name);
  }});
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  // Lazy render charts on tab switch
  if (name === 'forecast' && !window._fcRendered)   {{ renderForecastCharts(); window._fcRendered = true; }}
  if (name === 'diagnostics' && !window._dxRendered) {{ renderDiagCharts(); window._dxRendered = true; }}
  if (name === 'models' && !window._mdRendered)      {{ renderModelCharts(); window._mdRendered = true; }}
}}

// ─────────────────────────────────────────────────────────────────────────────
//  PLOTLY BASE LAYOUT
// ─────────────────────────────────────────────────────────────────────────────
const PLOT_CFG = {{ displayModeBar: true, displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ['select2d','lasso2d','toggleSpikelines'] }};

function baseLayout(overrides = {{}}) {{
  return Object.assign({{
    paper_bgcolor: 'transparent',
    plot_bgcolor:  '#080f1c',
    font: {{ family: "'Share Tech Mono', monospace", color: '#7a99b8', size: 10 }},
    margin: {{ l: 55, r: 20, t: 20, b: 45 }},
    height: 320,
    xaxis: {{
      gridcolor: '#162035', gridwidth: 0.5, zeroline: false,
      tickfont: {{ size: 9 }}, showspikes: true, spikecolor: '#00d4ff', spikethickness: 1,
    }},
    yaxis: {{
      gridcolor: '#162035', gridwidth: 0.5, zeroline: false,
      tickfont: {{ size: 9 }},
    }},
    legend: {{
      bgcolor: 'rgba(8,15,28,0.85)', bordercolor: '#162035', borderwidth: 1,
      font: {{ size: 9 }},
    }},
    hovermode: 'x unified',
    hoverlabel: {{ bgcolor: '#0c1628', bordercolor: '#162035', font: {{ size: 10, family: "'Share Tech Mono'" }} }},
  }}, overrides);
}}

// ─────────────────────────────────────────────────────────────────────────────
//  HEADER + STAT CARDS
// ─────────────────────────────────────────────────────────────────────────────
document.getElementById('headerSub').textContent =
  S.ticker + ' · ' + S.date_start + ' → ' + S.date_end +
  ' · Best: ' + S.best_model + ' · N=' + S.n_obs.toLocaleString() + ' obs';

document.getElementById('footer').textContent =
  'GARCH Volatility Dashboard · ' + S.ticker + ' · Best: ' + S.best_model +
  ' · Dist: ' + S.dist_rec.toUpperCase() + ' · Generated: ' + S.generated;

const CARDS = [
  {{ label:'Current Vol', value: S.current_ann_vol + '%', sub: 'Annualised GARCH σ', color: '--cyan' }},
  {{ label:'Ann. Vol (hist)', value: S.ann_vol + '%', sub: 'Historical realised', color: '--teal' }},
  {{ label:'Excess Kurtosis', value: S.exc_kurtosis, sub: 'Fat-tailed: ' + (S.fat_tailed ? 'YES' : 'NO'), color: '--orange' }},
  {{ label:'Skewness', value: S.skewness, sub: 'Return asymmetry', color: '--pink' }},
  {{ label:'Regime', value: S.current_regime, sub: 'K-Means (k=3)', color: S.current_regime==='Low Vol' ? '--green' : S.current_regime==='High Vol' ? '--red' : '--orange' }},
  {{ label:'Tail Index α', value: S.hill_idx, sub: 'Hill estimator', color: '--purple' }},
  {{ label:'Student-t ν', value: S.nu_est, sub: 'Implied d.o.f.', color: '--gold' }},
  {{ label:'N Observations', value: S.n_obs.toLocaleString(), sub: S.date_start + ' –', color: '--blue' }},
];

const cardHtml = CARDS.map(c => `
  <div class="stat-card" style="--accent-color: var(${{c.color}})">
    <div class="stat-label">${{c.label}}</div>
    <div class="stat-value">${{c.value}}</div>
    <div class="stat-sub">${{c.sub}}</div>
  </div>`).join('');
document.getElementById('statCards').innerHTML = cardHtml;

// ─────────────────────────────────────────────────────────────────────────────
//  OVERVIEW CHARTS
// ─────────────────────────────────────────────────────────────────────────────
(function renderOverview() {{
  // Price + Rolling Vol (dual axis)
  const tracePrice = {{
    x: D.dates, y: D.close,
    name: S.ticker + ' Close', type: 'scatter', mode: 'lines',
    line: {{ color: '#00d4ff', width: 1.2 }},
    fill: 'tozeroy', fillcolor: 'rgba(0,212,255,0.04)',
    yaxis: 'y',
  }};
  const traceRV = {{
    x: D.dates, y: D.roll_vol,
    name: '21d Realised Vol (%)', type: 'scatter', mode: 'lines',
    line: {{ color: '#f59e0b', width: 1, dash: 'dot' }},
    yaxis: 'y2',
    connectgaps: true,
  }};
  const layoutPrice = baseLayout({{
    height: 360,
    yaxis:  {{ title: 'Price', gridcolor: '#162035', zeroline: false }},
    yaxis2: {{ title: 'Vol (%)', overlaying: 'y', side: 'right', showgrid: false,
               tickfont: {{ size: 9 }}, titlefont: {{ size: 9 }} }},
  }});
  Plotly.newPlot('chartPrice', [tracePrice, traceRV], layoutPrice, PLOT_CFG);

  // Conditional volatility
  const traceCond = {{
    x: D.cond_dates, y: D.cond_vol,
    name: 'GARCH σ (ann%)', type: 'scatter', mode: 'lines',
    line: {{ color: '#8b5cf6', width: 1.5 }},
    fill: 'tozeroy', fillcolor: 'rgba(139,92,246,0.08)',
  }};
  const traceRV2 = {{
    x: D.dates, y: D.roll_vol,
    name: '21d RV', type: 'scatter', mode: 'lines',
    line: {{ color: '#f59e0b', width: 0.8, dash: 'dot' }},
    connectgaps: true,
  }};
  const lastVol = D.cond_vol[D.cond_vol.length - 1];
  const shapeLine = {{
    type: 'line',
    x0: D.cond_dates[0], x1: D.cond_dates[D.cond_dates.length-1],
    y0: lastVol, y1: lastVol,
    line: {{ color: '#fbbf24', width: 1, dash: 'dash' }},
  }};
  Plotly.newPlot('chartCondVol', [traceCond, traceRV2],
    baseLayout({{ shapes: [shapeLine],
      annotations: [{{ x: D.cond_dates[D.cond_dates.length-1], y: lastVol,
        text: lastVol.toFixed(1) + '%', showarrow: false, xanchor: 'right',
        font: {{ color: '#fbbf24', size: 9 }}, yshift: 10 }}] }}),
    PLOT_CFG);

  // Return distribution
  const traceDist = {{
    x: D.ret_hist_x, y: D.ret_hist_y,
    name: 'Actual', type: 'bar',
    marker: {{
      color: D.ret_hist_x.map(v => {{
        const t = (v - Math.min(...D.ret_hist_x)) / (Math.max(...D.ret_hist_x) - Math.min(...D.ret_hist_x));
        return `hsla(${{Math.round(180 + t * 120)}}, 80%, 60%, 0.7)`;
      }})
    }},
  }};
  const traceNorm = {{
    x: D.normal_pdf_x, y: D.normal_pdf_y,
    name: 'Normal', type: 'scatter', mode: 'lines',
    line: {{ color: '#f472b6', width: 2 }},
  }};
  const traceT = {{
    x: D.t_pdf_x, y: D.t_pdf_y,
    name: `Student-t (ν≈${{S.nu_est}})`, type: 'scatter', mode: 'lines',
    line: {{ color: '#fbbf24', width: 1.8, dash: 'dash' }},
  }};
  Plotly.newPlot('chartDist', [traceDist, traceNorm, traceT],
    baseLayout({{ yaxis: {{ title: 'Density' }}, xaxis: {{ title: 'Log Return (%)' }},
      annotations: [{{
        x: 0.98, y: 0.97, xref: 'paper', yref: 'paper', showarrow: false,
        text: `Kurt=${{S.exc_kurtosis}}<br>Skew=${{S.skewness}}`,
        align: 'right', font: {{ color: '#fbbf24', size: 9, family: "'Share Tech Mono'" }},
        bgcolor: 'rgba(12,22,40,0.8)', bordercolor: '#162035', borderwidth: 1, borderpad: 4,
      }}] }}),
    PLOT_CFG);
}})();

// ─────────────────────────────────────────────────────────────────────────────
//  FORECAST CHARTS (lazy)
// ─────────────────────────────────────────────────────────────────────────────
function renderForecastCharts() {{
  const weeks = D.weekly_rows.map(r => 'Week ' + r.week);
  const vols  = D.weekly_rows.map(r => r.weekly_vol_ann);
  const ciLo  = D.weekly_rows.map(r => r.ci_lo_95 * Math.sqrt(252/5));
  const ciHi  = D.weekly_rows.map(r => r.ci_hi_95 * Math.sqrt(252/5));
  const var95 = D.weekly_rows.map(r => Math.abs(r.var_95) * Math.sqrt(252/5));
  const var99 = D.weekly_rows.map(r => Math.abs(r.var_99) * Math.sqrt(252/5));
  const cvar  = D.weekly_rows.map(r => Math.abs(r.cvar_95) * Math.sqrt(252/5));
  const curVol = S.current_ann_vol;

  // Forecast ribbon
  const traceHi = {{
    x: weeks, y: ciHi,
    fill: 'tonexty', fillcolor: 'rgba(139,92,246,0.12)',
    line: {{ color: 'transparent' }}, showlegend: false, name: '95% CI Hi',
    type: 'scatter', mode: 'lines',
  }};
  const traceLo = {{
    x: weeks, y: ciLo,
    line: {{ color: 'transparent' }}, showlegend: false, name: '95% CI Lo',
    type: 'scatter', mode: 'lines',
  }};
  const traceFc = {{
    x: weeks, y: vols,
    name: 'Forecast Ann.Vol', type: 'scatter', mode: 'lines+markers',
    line: {{ color: '#00d4ff', width: 2.5 }},
    marker: {{ size: 9, color: '#00d4ff', symbol: 'circle',
      line: {{ color: '#0c1628', width: 2 }} }},
    text: vols.map(v => v.toFixed(2) + '%'),
    textposition: 'top center',
    textfont: {{ size: 9, color: '#00d4ff' }},
    mode: 'lines+markers+text',
  }};
  const traceBase = {{
    x: [weeks[0], weeks[weeks.length-1]],
    y: [curVol, curVol],
    name: 'Current: ' + curVol + '%',
    type: 'scatter', mode: 'lines',
    line: {{ color: '#fbbf24', width: 1, dash: 'dot' }},
  }};

  document.getElementById('currentVolBadge').innerHTML =
    `<span class="badge badge-cyan">Current: ${{curVol}}%</span>`;

  Plotly.newPlot('chartForecast', [traceLo, traceHi, traceFc, traceBase],
    baseLayout({{ height: 340, yaxis: {{ title: 'Ann. Vol (%)' }},
      xaxis: {{ title: 'Horizon' }} }}),
    PLOT_CFG);

  // VaR chart
  const traceVaR95 = {{
    x: weeks, y: var95,
    name: 'VaR 95%', type: 'bar',
    marker: {{ color: 'rgba(245,158,11,0.75)' }},
  }};
  const traceVaR99 = {{
    x: weeks, y: var99,
    name: 'VaR 99%', type: 'bar',
    marker: {{ color: 'rgba(239,68,68,0.75)' }},
  }};
  const traceCVaR = {{
    x: weeks, y: cvar,
    name: 'CVaR 95%', type: 'scatter', mode: 'lines+markers',
    line: {{ color: '#f472b6', width: 2 }},
    marker: {{ size: 7, color: '#f472b6' }},
  }};
  Plotly.newPlot('chartVaR', [traceVaR95, traceVaR99, traceCVaR],
    baseLayout({{ height: 300, barmode: 'group',
      yaxis: {{ title: 'Risk (ann.% equiv.)' }} }}),
    PLOT_CFG);

  // Forecast table
  let th = `<tr>
    <th>Horizon</th><th>Weekly Vol</th><th>Ann. Vol</th>
    <th>VaR 95%</th><th>VaR 99%</th><th>CVaR 95%</th>
  </tr>`;
  let tbody = D.weekly_rows.map(r => `
    <tr>
      <td>Week ${{r.week}} (${{r.week*5}}d)</td>
      <td>${{r.weekly_vol_pct.toFixed(4)}}%</td>
      <td>${{r.weekly_vol_ann.toFixed(2)}}%</td>
      <td style="color:#f59e0b">${{r.var_95.toFixed(4)}}%</td>
      <td style="color:#ef4444">${{r.var_99.toFixed(4)}}%</td>
      <td style="color:#f472b6">${{r.cvar_95.toFixed(4)}}%</td>
    </tr>`).join('');
  document.getElementById('forecastTable').innerHTML =
    `<thead>${{th}}</thead><tbody>${{tbody}}</tbody>`;
}}

// ─────────────────────────────────────────────────────────────────────────────
//  DIAGNOSTICS CHARTS (lazy)
// ─────────────────────────────────────────────────────────────────────────────
function renderDiagCharts() {{
  // Diag cards
  const tests = [
    {{
      title: 'Stationarity Tests',
      rows: [
        ['ADF p-value', S.adf_pval, S.adf_pval < 0.05 ? 'green' : 'red', S.adf_pval < 0.05 ? '✓ STATIONARY' : '✗ NONSTAT'],
        ['KPSS p-value', S.kpss_pval, S.kpss_pval > 0.05 ? 'green' : 'red', S.kpss_pval > 0.05 ? '✓ STATIONARY' : '✗ NONSTAT'],
        ['Overall', '', S.is_stationary ? 'green' : 'red', S.is_stationary ? '✓ PASS' : '✗ FAIL'],
      ]
    }},
    {{
      title: 'ARCH & Normality',
      rows: [
        ['ARCH-LM p-val', S.arch_lm_pval, S.arch_effects ? 'green' : 'orange', S.arch_effects ? '✓ ARCH PRESENT' : '✗ NO ARCH'],
        ['Jarque-Bera p', S.jb_pval < 0.0001 ? '<0.0001' : S.jb_pval, 'red', '✗ NON-NORMAL'],
        ['Dist. Rec.', S.dist_rec.toUpperCase(), 'cyan', 'best innovation dist'],
      ]
    }},
    {{
      title: 'Tail Analysis',
      rows: [
        ['Excess Kurtosis', S.exc_kurtosis, S.fat_tailed ? 'red' : 'green', S.fat_tailed ? '✓ FAT TAILS' : '✗ THIN TAILS'],
        ['Hill Index α', S.hill_idx, 'orange', 'lower = heavier tail'],
        ['Implied ν (t)', S.nu_est, 'purple', 'Student-t d.o.f.'],
      ]
    }},
    {{
      title: 'Distribution Shape',
      rows: [
        ['Skewness', S.skewness, Math.abs(S.skewness) > 0.5 ? 'orange' : 'green', Math.abs(S.skewness) > 0.5 ? 'ASYMMETRIC' : 'APPROX SYM'],
        ['Kurtosis', S.exc_kurtosis, 'text', 'excess (Fisher)'],
        ['Q-Q R²', D.qq_r2, D.qq_r2 > 0.99 ? 'green' : 'orange', 'fit quality'],
      ]
    }},
  ];

  const colorMap = {{ green:'--green', red:'--red', orange:'--orange', cyan:'--cyan', purple:'--purple', text:'--text' }};
  document.getElementById('diagCards').innerHTML = tests.map(t => `
    <div class="diag-card">
      <div class="diag-title">${{t.title}}</div>
      ${{t.rows.map(r => `
        <div class="diag-row">
          <span class="diag-key">${{r[0]}}</span>
          <span style="display:flex;align-items:center;gap:6px">
            <span class="diag-val">${{r[1]}}</span>
            <span class="badge badge-${{r[2]}}">${{r[3]}}</span>
          </span>
        </div>`).join('')}}
    </div>`).join('');

  // Regime chart
  const regColors = ['#10b981','#f59e0b','#ef4444'];
  const regTraces = D.regime_scatters.map((r, i) => ({{
    x: r.dates, y: r.vols,
    name: r.name + ' (center: ' + r.center + '%)',
    type: 'scatter', mode: 'markers',
    marker: {{ color: regColors[i], size: 4, opacity: 0.7 }},
  }}));
  const centerLines = D.regime_scatters.map((r, i) => ({{
    type: 'line',
    x0: D.regime_scatters[0].dates[0],
    x1: D.regime_scatters[D.regime_scatters.length-1].dates.slice(-1)[0] || '',
    y0: r.center, y1: r.center,
    line: {{ color: regColors[i], width: 1, dash: 'dash' }},
  }}));
  Plotly.newPlot('chartRegime', regTraces,
    baseLayout({{ shapes: centerLines, yaxis: {{ title: 'Ann. Vol (%)' }} }}),
    PLOT_CFG);

  // QQ Plot
  const traceQQ = {{
    x: D.qq_x, y: D.qq_y,
    name: 'Residuals', type: 'scatter', mode: 'markers',
    marker: {{ color: '#00d4ff', size: 3, opacity: 0.5 }},
  }};
  const traceQQLine = {{
    x: D.qq_line_x, y: D.qq_line_y,
    name: 'Fit line (R²=' + D.qq_r2 + ')',
    type: 'scatter', mode: 'lines',
    line: {{ color: '#f472b6', width: 1.5 }},
  }};
  const diagLine = {{
    x: [Math.min(...D.qq_x), Math.max(...D.qq_x)],
    y: [Math.min(...D.qq_x), Math.max(...D.qq_x)],
    name: 'Diagonal', type: 'scatter', mode: 'lines',
    line: {{ color: '#3d5268', width: 1, dash: 'dot' }},
  }};
  Plotly.newPlot('chartQQ', [traceQQ, traceQQLine, diagLine],
    baseLayout({{ xaxis: {{ title: 'Theoretical Quantiles' }},
                  yaxis: {{ title: 'Sample Quantiles' }} }}),
    PLOT_CFG);
}}

// ─────────────────────────────────────────────────────────────────────────────
//  MODEL CHARTS (lazy)
// ─────────────────────────────────────────────────────────────────────────────
function renderModelCharts() {{
  const mc = D.model_comp;
  const names = mc.map(m => m.name);
  const aicVals = mc.map(m => m.delta_aic);
  const bicVals = mc.map(m => m.bic);
  const bicMin = Math.min(...bicVals);

  // AIC bar
  Plotly.newPlot('chartAIC',
    [{{
      x: aicVals.slice().reverse(), y: names.slice().reverse(),
      type: 'bar', orientation: 'h',
      marker: {{ color: names.slice().reverse().map((_, i) =>
        i === names.length-1 ? '#00d4ff' : 'rgba(99,102,241,0.6)'
      )}},
      text: aicVals.slice().reverse().map((v, i) =>
        mc[mc.length-1-i].aic.toFixed(1)),
      textposition: 'outside',
      textfont: {{ size: 9, color: '#7a99b8' }},
    }}],
    baseLayout({{ height: 340, margin: {{ l: 160, r: 60, t: 20, b: 40 }},
      xaxis: {{ title: 'ΔAIC from best' }},
      yaxis: {{ tickfont: {{ size: 9 }} }} }}),
    PLOT_CFG);

  // BIC bar
  Plotly.newPlot('chartBIC',
    [{{
      x: bicVals.map(v => v - bicMin).slice().reverse(),
      y: names.slice().reverse(),
      type: 'bar', orientation: 'h',
      marker: {{ color: names.slice().reverse().map((_, i) =>
        i === names.length-1 ? '#8b5cf6' : 'rgba(244,114,182,0.55)'
      )}},
      text: bicVals.slice().reverse().map(v => v.toFixed(1)),
      textposition: 'outside',
      textfont: {{ size: 9, color: '#7a99b8' }},
    }}],
    baseLayout({{ height: 340, margin: {{ l: 160, r: 60, t: 20, b: 40 }},
      xaxis: {{ title: 'ΔBIC from best' }},
      yaxis: {{ tickfont: {{ size: 9 }} }} }}),
    PLOT_CFG);

  // Full model table
  const thead = `<tr>
    <th>#</th><th style="text-align:left">Model</th>
    <th>AIC</th><th>BIC</th><th>ΔAIC</th><th>Conv.</th>
  </tr>`;
  const tbody = mc.map((m, i) => `
    <tr>
      <td>${{i+1}}</td>
      <td style="text-align:left;color:${{i===0?'#fbbf24':'#b8cce0'}}">${{m.name}}</td>
      <td>${{m.aic.toFixed(2)}}</td>
      <td>${{m.bic.toFixed(2)}}</td>
      <td style="color:${{i===0?'#10b981':'#f59e0b'}}">${{m.delta_aic.toFixed(2)}}</td>
      <td><span class="badge ${{m.converged?'badge-green':'badge-red'}}">${{m.converged?'✓':'✗'}}</span></td>
    </tr>`).join('');
  document.getElementById('modelTable').innerHTML =
    `<thead>${{thead}}</thead><tbody>${{tbody}}</tbody>`;
}}
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _banner(title):
    line = "═" * (len(title) + 6)
    print(f"\n╔{line}╗\n║   {title}   ║\n╚{line}╝")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main(ticker="ES=F", years=3, output_path=None):
    """
    Full pipeline: fetch → clean → validate → fit → forecast → HTML dashboard.

    Parameters
    ----------
    ticker      : Yahoo Finance symbol (default 'ES=F')
    years       : Historical look-back in years (2–5)
    output_path : Where to save HTML (default: ./garch_dashboard.html)
    """
    print("\n" + "█" * 60)
    print(f"  GARCH FORECASTING PIPELINE  ·  {ticker}  ·  {years}y")
    print("█" * 60)

    df          = fetch_data(ticker=ticker, years=years)
    df_clean, log_ret = clean_data(df)
    diagnostics = validate_data(log_ret)
    best_model, best_vol, best_dist, results_df, fitted_models = fit_best_garch(log_ret, diagnostics)
    forecasts   = generate_forecasts(best_model, best_vol, best_dist, log_ret, diagnostics)
    html        = build_html_dashboard(
        df_clean, log_ret, diagnostics,
        best_model, best_vol, best_dist,
        results_df, fitted_models, forecasts,
        ticker=ticker,
    )

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "garch_dashboard.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    _banner(f"DONE · Dashboard saved → {output_path}")
    print(f"\n  Open in browser: file://{os.path.abspath(output_path)}")
    print(f"  Or: python -m http.server  then navigate to the file\n")

    return {
        "df": df_clean,
        "log_ret": log_ret,
        "diagnostics": diagnostics,
        "best_model": best_model,
        "results_df": results_df,
        "forecasts": forecasts,
        "html_path": output_path,
    }


if __name__ == "__main__":
    TICKER = "GC=F"   # E-mini S&P 500 Futures (ganti ke ^SPX, QQQ, NQ=F, dll.)
    YEARS  = 3        # 2–5 tahun data historis

    results = main(ticker=TICKER, years=YEARS)