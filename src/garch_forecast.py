"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           GARCH VOLATILITY FORECASTING PIPELINE · ES FUTURES (S&P 500)     ║
║                                                                              ║
║  STEP 1 · STEP 2 · STEP 3 · STEP 4 · STEP 5                                ║
║  ──────   ──────   ──────   ──────   ──────                                 ║
║  Kumpulkan → Bersihkan → Validasi → Fit Model → Generate                   ║
║  Data        Data         Data        (GARCH)     Forecast                  ║
║  (2-5 tahun) (missing,    (Stasioner, terbaik     1-4 minggu                ║
║              outlier,     clustering, (AIC/BIC)   ke depan                  ║
║              adjust)      fat tails)                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Models compared:
  · GARCH(1,1)    – baseline symmetric
  · EGARCH(1,1)   – asymmetric (leverage effect)
  · GJR-GARCH(1,1)– Glosten-Jagannathan-Runkle (news impact)
  · GARCH(1,1) t  – Student-t innovation (fat tails)
  · GARCH(2,2)    – higher-order lag structure

Distributions tested: Normal, Student-t, GED (Generalized Error Distribution)

Diagnostics:
  · ADF / KPSS stationarity tests
  · Ljung-Box autocorrelation test
  · Jarque-Bera normality test
  · Engle ARCH-LM test
  · Hill tail index (fat tail detection)
  · Volatility regime clustering (K-Means)

Output:
  · Forecast σ (weekly: 1, 2, 3, 4 weeks ahead)
  · 95% / 99% Confidence intervals
  · VaR / CVaR estimates
  · Full dashboard (matplotlib)
"""

import sys
import io
import warnings
import itertools
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.stats import jarque_bera, kurtosis, skew, norm, t as student_t
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from arch import arch_model
from arch.unitroot import ADF, KPSS

warnings.filterwarnings("ignore")

# ── Windows console fix ─────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════════════════
#  COLOR PALETTE (dark theme, consistent with project style)
# ════════════════════════════════════════════════════════════════════════════
C = {
    "bg_dark":       "#0a0e17",
    "bg_panel":      "#0f1520",
    "bg_card":       "#131d2e",
    "grid":          "#1a2332",
    "text":          "#c8d6e5",
    "text_dim":      "#5a6a7a",
    "accent_cyan":   "#00d2ff",
    "accent_blue":   "#0099ff",
    "accent_purple": "#7c3aed",
    "accent_pink":   "#f472b6",
    "accent_green":  "#10b981",
    "accent_orange": "#f59e0b",
    "accent_red":    "#ef4444",
    "gold":          "#fbbf24",
    "teal":          "#14b8a6",
    "indigo":        "#6366f1",
}

GRAD_MAIN   = LinearSegmentedColormap.from_list("main",  ["#00d2ff", "#7c3aed", "#f472b6"])
GRAD_FIRE   = LinearSegmentedColormap.from_list("fire",  ["#10b981", "#f59e0b", "#ef4444"])
GRAD_REGIME = LinearSegmentedColormap.from_list("regime",["#00d2ff", "#f59e0b", "#ef4444"])

TRADING_DAYS_PER_YEAR = 252
WEEKS_AHEAD = [1, 2, 3, 4]          # forecast horizons
DAYS_PER_WEEK = 5


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 · DATA ACQUISITION
# ════════════════════════════════════════════════════════════════════════════
def fetch_data(
    ticker: str = "ES=F",
    years: int = 3,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker     : Yahoo Finance symbol  (default: ES=F = E-mini S&P 500)
    years      : Look-back window in years (2–5 recommended)
    start_date : Override start date  (YYYY-MM-DD)
    end_date   : Override end date    (YYYY-MM-DD)

    Returns
    -------
    DataFrame with columns: Open, High, Low, Close, Volume
    """
    _banner("STEP 1 · DATA ACQUISITION")

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_dt = datetime.today() - timedelta(days=years * 365)
        start_date = start_dt.strftime("%Y-%m-%d")

    print(f"  Ticker     : {ticker}")
    print(f"  Start date : {start_date}")
    print(f"  End date   : {end_date}")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"[!] No data returned for ticker '{ticker}'. Check connectivity.")

    # Flatten MultiIndex columns if present (yfinance ≥0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    print(f"  Raw rows   : {len(df):,}")
    print(f"  Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Close range: {df['Close'].min():.2f} – {df['Close'].max():.2f}")

    return df


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 · DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean price data and compute log returns.

    Methods applied:
      1. Remove rows with NaN / zero / negative Close
      2. Detect & cap price outliers  (modified Z-score > 3.5)
      3. Detect & winsorize return outliers (IQR method, 1st/99th pct)
      4. Forward-fill isolated missing trading days
      5. Drop duplicate index entries

    Returns
    -------
    df_clean   : cleaned OHLCV DataFrame
    log_ret    : daily log returns (already cleaned)
    """
    _banner("STEP 2 · DATA CLEANING")

    df_c = df.copy()
    n_raw = len(df_c)

    # ── 2.1  Remove duplicates ───────────────────────────────────────────
    n_dup = df_c.index.duplicated().sum()
    if n_dup > 0:
        df_c = df_c[~df_c.index.duplicated(keep="last")]
        print(f"  [2.1] Removed {n_dup} duplicate rows")
    else:
        print("  [2.1] No duplicates found")

    # ── 2.2  Remove NaN / zero / negative prices ────────────────────────
    bad_mask = df_c["Close"].isna() | (df_c["Close"] <= 0)
    n_bad = bad_mask.sum()
    df_c = df_c[~bad_mask]
    print(f"  [2.2] Removed {n_bad} rows with NaN/zero/negative Close")

    # ── 2.3  Forward-fill missing weekday gaps (≤3 consecutive days) ─────
    #  Reindex to full business-day calendar, then ffill gaps ≤3 days
    full_idx = pd.bdate_range(df_c.index.min(), df_c.index.max())
    n_before = len(df_c)
    df_c = df_c.reindex(full_idx).ffill(limit=3)
    df_c = df_c.dropna(subset=["Close"])   # drop remaining NaN (long gaps)
    n_filled = len(df_c) - n_before
    print(f"  [2.3] Forward-filled {n_filled} missing business days (limit=3)")

    # ── 2.4  Price outlier detection – Modified Z-score (Iglewicz & Hoaglin) ──
    close_vals = df_c["Close"].values
    median_c   = np.median(close_vals)
    mad_c      = np.median(np.abs(close_vals - median_c))
    mod_z      = 0.6745 * (close_vals - median_c) / (mad_c + 1e-9)
    outlier_mask = np.abs(mod_z) > 3.5
    n_price_out  = outlier_mask.sum()

    if n_price_out > 0:
        # Cap at percentile 0.5 / 99.5 instead of dropping
        lo_cap = np.percentile(close_vals[~outlier_mask], 0.5)
        hi_cap = np.percentile(close_vals[~outlier_mask], 99.5)
        df_c["Close"] = df_c["Close"].clip(lower=lo_cap, upper=hi_cap)
        print(f"  [2.4] Capped {n_price_out} price outliers  "
              f"[{lo_cap:.1f} – {hi_cap:.1f}]")
    else:
        print("  [2.4] No price outliers detected")

    # ── 2.5  Compute log returns ─────────────────────────────────────────
    log_ret = np.log(df_c["Close"] / df_c["Close"].shift(1)).dropna()
    n_ret_raw = len(log_ret)

    # ── 2.6  Return outlier winsorization (1st / 99th percentile) ────────
    lo_pct = log_ret.quantile(0.01)
    hi_pct = log_ret.quantile(0.99)
    n_ret_out = ((log_ret < lo_pct) | (log_ret > hi_pct)).sum()
    log_ret_w = log_ret.clip(lower=lo_pct, upper=hi_pct)
    print(f"  [2.5] Log returns computed  → {n_ret_raw:,} observations")
    print(f"  [2.6] Winsorized {n_ret_out} return outliers  "
          f"[{lo_pct:.5f} – {hi_pct:.5f}]")

    # ── Summary ──────────────────────────────────────────────────────────
    n_final = len(df_c)
    print(f"\n  Rows   : {n_raw:,} → {n_final:,}  (net change: {n_final - n_raw:+d})")
    print(f"  Returns: {len(log_ret_w):,} clean observations")
    print(f"  Mean   : {log_ret_w.mean()*100:.4f}%  |  "
          f"Std: {log_ret_w.std()*100:.4f}%  |  "
          f"Ann.Vol: {log_ret_w.std()*np.sqrt(252)*100:.2f}%")

    return df_c, log_ret_w


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 · DATA VALIDATION
# ════════════════════════════════════════════════════════════════════════════
def validate_data(log_ret: pd.Series) -> dict:
    """
    Run statistical diagnostics to validate return series properties.

    Tests:
      A. Stationarity   : ADF (Augmented Dickey-Fuller) + KPSS
      B. Autocorrelation: Ljung-Box on returns and squared returns
      C. Normality      : Jarque-Bera test
      D. ARCH effects   : Engle LM test  (confirms GARCH is appropriate)
      E. Fat tails      : Excess kurtosis + Hill estimator (tail index α)
      F. Regime cluster : K-Means on rolling volatility (k=3: low/mid/high)

    Returns
    -------
    results dict with all test statistics and conclusions
    """
    _banner("STEP 3 · DATA VALIDATION")
    ret   = log_ret.values
    n     = len(ret)
    results = {}

    # ── A. Stationarity ──────────────────────────────────────────────────
    print("  [A] Stationarity Tests")
    adf = ADF(log_ret, lags=20)
    adf_res = adf.summary()
    adf_stat  = float(adf.stat)
    adf_pval  = float(adf.pvalue)
    adf_crit  = adf.critical_values

    kpss_res  = KPSS(log_ret, trend="c")
    kpss_stat = float(kpss_res.stat)
    kpss_pval = float(kpss_res.pvalue)

    adf_stationary  = adf_pval  < 0.05          # reject H0 (unit root)  → stationary
    kpss_stationary = kpss_pval > 0.05          # fail to reject H0      → stationary
    is_stationary   = adf_stationary and kpss_stationary

    print(f"     ADF  : stat={adf_stat:.4f}  p={adf_pval:.4f}  "
          f"→ {'STATIONARY ✓' if adf_stationary else 'NON-STATIONARY ✗'}")
    print(f"     KPSS : stat={kpss_stat:.4f}  p={kpss_pval:.4f}  "
          f"→ {'STATIONARY ✓' if kpss_stationary else 'NON-STATIONARY ✗'}")
    print(f"     Conclusion: {'Series IS stationary ✓' if is_stationary else 'Series may NOT be stationary ✗'}")

    results["adf"]  = {"stat": adf_stat,  "pval": adf_pval,  "stationary": adf_stationary}
    results["kpss"] = {"stat": kpss_stat, "pval": kpss_pval, "stationary": kpss_stationary}
    results["is_stationary"] = is_stationary

    # ── B. Autocorrelation (Ljung-Box) ───────────────────────────────────
    print("  [B] Autocorrelation – Ljung-Box")
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_ret  = acorr_ljungbox(ret,   lags=20, return_df=True)
    lb_sq   = acorr_ljungbox(ret**2, lags=20, return_df=True)
    lb_ret_pval  = float(lb_ret["lb_pvalue"].iloc[-1])
    lb_sq_pval   = float(lb_sq["lb_pvalue"].iloc[-1])
    arch_effect  = lb_sq_pval < 0.05

    print(f"     Returns       p(lag=20) = {lb_ret_pval:.5f}  "
          f"→ {'serial corr. ✓' if lb_ret_pval < 0.05 else 'no serial corr.'}")
    print(f"     Squared ret.  p(lag=20) = {lb_sq_pval:.5f}  "
          f"→ {'ARCH effects PRESENT ✓' if arch_effect else 'no ARCH effects'}")

    results["lb_ret_pval"] = lb_ret_pval
    results["lb_sq_pval"]  = lb_sq_pval
    results["arch_effects"] = arch_effect

    # ── C. Normality (Jarque-Bera) ───────────────────────────────────────
    print("  [C] Normality – Jarque-Bera")
    jb_stat, jb_pval = jarque_bera(ret)
    jb_normal = jb_pval > 0.05
    print(f"     JB stat = {jb_stat:.2f}  p = {jb_pval:.2e}  "
          f"→ {'Normal' if jb_normal else 'NON-NORMAL (fat tails likely)'}")
    results["jb_stat"] = jb_stat
    results["jb_pval"] = jb_pval
    results["is_normal"] = jb_normal

    # ── D. ARCH-LM test (Engle 1982) ────────────────────────────────────
    print("  [D] ARCH-LM Test (Engle)")
    from statsmodels.stats.diagnostic import het_arch
    arch_lm_stat, arch_lm_pval, _, _ = het_arch(ret, nlags=10)
    arch_lm_sig = arch_lm_pval < 0.05
    print(f"     LM stat = {arch_lm_stat:.4f}  p = {arch_lm_pval:.5f}  "
          f"→ {'ARCH effects confirmed ✓ (GARCH appropriate)' if arch_lm_sig else 'No ARCH effects'}")
    results["arch_lm_stat"] = arch_lm_stat
    results["arch_lm_pval"] = arch_lm_pval
    results["arch_lm_sig"]  = arch_lm_sig

    # ── E. Fat Tails ────────────────────────────────────────────────────
    print("  [E] Fat Tail Analysis")
    exc_kurt   = float(kurtosis(ret, fisher=True))       # excess kurtosis
    skewness   = float(skew(ret))

    # Hill estimator – tail index α (for upper tail)
    sorted_ret   = np.sort(np.abs(ret))[::-1]            # descending |ret|
    k_hill       = max(int(n ** 0.5), 30)                # rule of thumb: √n
    hill_idx     = k_hill / np.sum(np.log(sorted_ret[:k_hill] / sorted_ret[k_hill]))

    # Degrees of freedom estimate by moment matching: kurt = 6/(ν-4) → ν = 6/kurt+4
    if exc_kurt > 0:
        nu_est = max(6.0 / exc_kurt + 4, 2.1)
    else:
        nu_est = np.inf

    fat_tailed = exc_kurt > 1.0   # practical threshold
    print(f"     Skewness     = {skewness:.4f}")
    print(f"     Excess Kurt  = {exc_kurt:.4f}  "
          f"→ {'Fat-tailed ✓' if fat_tailed else 'Near-normal'}")
    print(f"     Hill index α = {hill_idx:.4f}  (lower → heavier tail)")
    print(f"     Implied df   ≈ {nu_est:.2f}  (Student-t)")

    results["skewness"]  = skewness
    results["exc_kurt"]  = exc_kurt
    results["hill_idx"]  = hill_idx
    results["nu_est"]    = nu_est
    results["fat_tailed"]= fat_tailed

    # ── F. Volatility Regime Clustering (K-Means) ───────────────────────
    print("  [F] Volatility Regime Clustering")
    roll_std  = pd.Series(ret).rolling(21).std().dropna().values   # 21-day rolling vol
    roll_std_ann = roll_std * np.sqrt(252) * 100                   # annualised %

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(roll_std_ann.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    order   = np.argsort(centers)
    regime_names = ["Low Vol", "Mid Vol", "High Vol"]

    regime_map = {old: new for new, old in enumerate(order)}
    labels_ordered = np.array([regime_map[l] for l in labels])
    pct_each = [100 * (labels_ordered == i).mean() for i in range(n_clusters)]

    print(f"     Cluster centers (ann.vol %):")
    for i, orig in enumerate(order):
        print(f"       {regime_names[i]:10s}: {centers[orig]:.2f}%  "
              f"({pct_each[i]:.1f}% of time)")

    current_regime_idx = labels_ordered[-1]
    print(f"     Current regime : {regime_names[current_regime_idx]}")

    results["regime_labels"]  = labels_ordered
    results["regime_centers"] = centers[order]
    results["roll_std_ann"]   = roll_std_ann
    results["regime_names"]   = regime_names
    results["current_regime"] = current_regime_idx

    # ── Recommended distribution ─────────────────────────────────────────
    if fat_tailed and not jb_normal:
        dist_rec = "t"       # Student-t (handles fat tails)
    elif exc_kurt > 0.5:
        dist_rec = "ged"     # Generalized Error Distribution
    else:
        dist_rec = "normal"

    results["dist_rec"] = dist_rec
    print(f"\n  Recommended innovation distribution: '{dist_rec.upper()}'")

    return results


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 · MODEL SELECTION (GARCH family comparison)
# ════════════════════════════════════════════════════════════════════════════
def fit_best_garch(
    log_ret: pd.Series,
    diagnostics: dict,
    scale: float = 100.0,
) -> tuple[object, str, str, pd.DataFrame]:
    """
    Fit multiple GARCH-family models, select best by AIC.

    Candidates:
      · GARCH(1,1)    – normal, t, GED
      · EGARCH(1,1)   – t
      · GJR-GARCH(1,1)– t
      · GARCH(2,2)    – t

    Parameters
    ----------
    log_ret     : cleaned log return series
    diagnostics : output from validate_data()
    scale       : multiply returns (arch lib stability; default 100)

    Returns
    -------
    best_model  : fitted arch model result object
    best_vol    : best volatility model name string
    best_dist   : best distribution name string
    results_df  : comparison DataFrame sorted by AIC
    """
    _banner("STEP 4 · MODEL SELECTION & FITTING")

    ret_scaled = log_ret.values * scale
    dist_rec   = diagnostics.get("dist_rec", "t")

    # Candidate configurations: (label, vol_model, p, o, q, dist)
    # GJR-GARCH = GARCH with o=1 (asymmetry / leverage term)
    candidates = [
        ("GARCH(1,1)-normal",   "GARCH", 1, 0, 1, "normal"),
        ("GARCH(1,1)-t",        "GARCH", 1, 0, 1, "t"),
        ("GARCH(1,1)-ged",      "GARCH", 1, 0, 1, "ged"),
        ("EGARCH(1,1)-t",       "EGARCH",1, 0, 1, "t"),
        ("EGARCH(1,1)-normal",  "EGARCH",1, 0, 1, "normal"),
        ("GJR-GARCH(1,1)-t",   "GARCH", 1, 1, 1, "t"),
        ("GJR-GARCH(1,1)-normal","GARCH",1, 1, 1, "normal"),
        ("GARCH(2,2)-t",        "GARCH", 2, 0, 2, "t"),
    ]

    rows = []
    fitted_models = {}

    print(f"  {'Model':<26} {'Dist':<8} {'AIC':>10} {'BIC':>10} {'LogL':>12}  Conv?")
    print("  " + "─" * 76)

    for label, vol, p, o, q, dist in candidates:
        try:
            am = arch_model(
                ret_scaled,
                vol=vol,
                p=p, o=o, q=q,
                dist=dist,
                mean="Zero",
                rescale=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", show_warning=False)

            aic, bic, logl = res.aic, res.bic, res.loglikelihood
            converged = res.convergence_flag == 0

            rows.append({
                "Model": label, "vol": vol, "p": p, "o": o, "q": q, "dist": dist,
                "AIC": aic, "BIC": bic, "LogL": logl,
                "Converged": converged
            })
            fitted_models[label] = res

            conv_str = "✓" if converged else "✗"
            print(f"  {label:<26} {dist:<8} {aic:>10.2f} {bic:>10.2f} "
                  f"{logl:>12.4f}  {conv_str}")

        except Exception as e:
            print(f"  {label:<26} FAILED: {e}")

    results_df = (
        pd.DataFrame(rows)
        .sort_values("AIC")
        .reset_index(drop=True)
    )

    # pick best converged model
    conv_df = results_df[results_df["Converged"] == True]
    if conv_df.empty:
        conv_df = results_df                # fallback if all failed convergence

    best_row  = conv_df.iloc[0]
    best_label = best_row["Model"]
    best_vol   = best_row["vol"]
    best_dist  = best_row["dist"]

    print(f"\n  Best model (min AIC): {best_label}")
    print(f"  AIC = {best_row['AIC']:.4f}  |  BIC = {best_row['BIC']:.4f}")

    best_model = fitted_models[best_label]

    # Print parameter summary
    print("\n  Parameter Estimates:")
    params = best_model.params
    pvalues = best_model.pvalues
    for name in params.index:
        sig = "***" if pvalues[name] < 0.001 else (
              "**"  if pvalues[name] < 0.01  else (
              "*"   if pvalues[name] < 0.05  else ""))
        print(f"     {name:<18}: {params[name]:>10.6f}  "
              f"(p={pvalues[name]:.4f}) {sig}")

    # Persistence check: α + β for GARCH
    if best_vol == "GARCH":
        try:
            alpha = params[params.index.str.startswith("alpha")].sum()
            beta  = params[params.index.str.startswith("beta")].sum()
            pers  = alpha + beta
            print(f"\n  Persistence (α+β) = {pers:.6f}  "
                  f"{'(near unit-root – high persistence)' if pers > 0.98 else ''}")
        except Exception:
            pass

    return best_model, best_vol, best_dist, results_df, fitted_models


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 · FORECAST (1–4 WEEKS AHEAD)
# ════════════════════════════════════════════════════════════════════════════
def generate_forecasts(
    best_model,
    best_vol: str,
    best_dist: str,
    log_ret: pd.Series,
    diagnostics: dict,
    scale: float = 100.0,
) -> dict:
    """
    Generate volatility forecasts for 1–4 weeks ahead.

    Approach:
      · Use arch model's built-in forecast() for h-step ahead variance
      · Convert from scaled daily vol → annualised % vol
      · Build 95% / 99% confidence intervals (analytic for normal, bootstrap for t)
      · Compute VaR and CVaR at each horizon

    Returns
    -------
    dict with keys: daily_vols, weekly_summary, var_table, conf_intervals
    """
    _banner("STEP 5 · FORECAST  (1 – 4 WEEKS)")

    max_horizon = max(WEEKS_AHEAD) * DAYS_PER_WEEK   # 20 trading days

    # Rolling forecast over the full horizon
    fc = best_model.forecast(horizon=max_horizon, reindex=False)

    # fc.variance.iloc[-1] → daily variance for h = 1 … max_horizon
    daily_var_scaled = fc.variance.iloc[-1].values          # len = max_horizon
    daily_vol_scaled = np.sqrt(daily_var_scaled)            # scaled daily vol %
    daily_vol_true   = daily_vol_scaled / scale             # actual daily vol

    # Annualised vol
    ann_vol_daily = daily_vol_true * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    # ── Weekly aggregation ───────────────────────────────────────────────
    # Last known price (placeholder; not used downstream)
    last_price = None

    weekly_rows = []
    ci_rows     = []

    ret_vals = log_ret.values

    for w in WEEKS_AHEAD:
        h_start = (w - 1) * DAYS_PER_WEEK
        h_end   = w * DAYS_PER_WEEK

        # Weekly variance = sum of daily variances over the week
        weekly_var_true = daily_vol_true[h_start:h_end] ** 2
        weekly_vol_true = np.sqrt(weekly_var_true.sum())    # weekly vol (daily units)
        weekly_vol_pct  = weekly_vol_true * 100
        weekly_vol_ann  = weekly_vol_true * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

        # Mean daily vol over the week (annualised)
        mean_day_ann = ann_vol_daily[h_start:h_end].mean()

        # ── Confidence intervals (parametric) ────────────────────────────
        if best_dist == "t":
            # Use estimated degrees of freedom
            nu = diagnostics.get("nu_est", 8.0)
            nu = min(max(nu, 2.1), 30)        # clamp to reasonable range
            alpha_95 = student_t.ppf(0.025, df=nu)
            alpha_99 = student_t.ppf(0.005, df=nu)
        else:
            alpha_95 = norm.ppf(0.025)        # –1.96
            alpha_99 = norm.ppf(0.005)        # –2.576

        # VaR as a return level (expressed as % of value)
        var_95 = alpha_95 * weekly_vol_pct
        var_99 = alpha_99 * weekly_vol_pct

        # CVaR (Expected Shortfall) – analytical for t and normal
        if best_dist == "t":
            nu_c  = min(max(diagnostics.get("nu_est", 8.0), 2.1), 30)
            pdf_t = student_t.pdf(student_t.ppf(0.05, df=nu_c), df=nu_c)
            cvar_95_z = -pdf_t / 0.05 * (nu_c + student_t.ppf(0.05, df=nu_c)**2) / (nu_c - 1)
            cvar_95 = cvar_95_z * weekly_vol_pct
        else:
            cvar_95 = -norm.pdf(norm.ppf(0.05)) / 0.05 * weekly_vol_pct

        # 95% CI for the vol forecast itself (±1.96 * se_vol)
        # se_vol approximation: vol / sqrt(2*h) [Fisher information]
        h = h_end - h_start
        se_vol = weekly_vol_pct / np.sqrt(2 * h)
        ci_lo_95 = max(0, weekly_vol_pct - 1.96 * se_vol)
        ci_hi_95 = weekly_vol_pct + 1.96 * se_vol

        weekly_rows.append({
            "Week": f"Week {w}",
            "Horizon (days)": f"{h_end}",
            "Weekly Vol %":   f"{weekly_vol_pct:.4f}%",
            "Ann. Vol %":     f"{weekly_vol_ann:.2f}%",
            "VaR 95%":        f"{var_95:.4f}%",
            "VaR 99%":        f"{var_99:.4f}%",
            "CVaR 95%":       f"{cvar_95:.4f}%",
            # raw values for plotting
            "_weekly_vol_pct": weekly_vol_pct,
            "_weekly_vol_ann": weekly_vol_ann,
            "_var_95":         var_95,
            "_var_99":         var_99,
            "_cvar_95":        cvar_95,
            "_ci_lo_95":       ci_lo_95,
            "_ci_hi_95":       ci_hi_95,
        })

    weekly_df = pd.DataFrame(weekly_rows)

    # Print table
    print(f"\n  {'Week':<8} {'Horizon':>9}  {'Weekly Vol':>12}  "
          f"{'Ann.Vol':>10}  {'VaR 95%':>10}  {'VaR 99%':>10}  {'CVaR 95%':>10}")
    print("  " + "─" * 80)
    for r in weekly_rows:
        print(f"  {r['Week']:<8} {r['Horizon (days)']:>9}  "
              f"{r['Weekly Vol %']:>12}  {r['Ann. Vol %']:>10}  "
              f"{r['VaR 95%']:>10}  {r['VaR 99%']:>10}  {r['CVaR 95%']:>10}")

    return {
        "daily_vol_pct":  daily_vol_true * 100,
        "ann_vol_daily":  ann_vol_daily,
        "weekly_rows":    weekly_rows,
        "weekly_df":      weekly_df,
    }


# ════════════════════════════════════════════════════════════════════════════
#  DASHBOARD BUILDER
# ════════════════════════════════════════════════════════════════════════════
def build_dashboard(
    df: pd.DataFrame,
    log_ret: pd.Series,
    diagnostics: dict,
    best_model,
    best_vol: str,
    best_dist: str,
    results_df: pd.DataFrame,
    fitted_models: dict,
    forecasts: dict,
):
    """Build and display a comprehensive 9-panel dashboard."""
    _banner("RENDERING DASHBOARD")

    ret = log_ret.values
    n   = len(ret)

    # ── Rolling stats for panels ─────────────────────────────────────────
    roll_21_vol  = pd.Series(ret, index=log_ret.index).rolling(21).std() * np.sqrt(252) * 100
    # conditional_volatility is a ndarray → wrap as Series aligned to log_ret.index
    cond_vol_raw = np.asarray(best_model.conditional_volatility)
    conditional_vol = pd.Series(
        cond_vol_raw / 100 * np.sqrt(252) * 100,
        index=log_ret.index[-len(cond_vol_raw):]
    )  # annualised %

    # ── Figure setup ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 15), facecolor=C["bg_dark"])
    fig.subplots_adjust(top=0.91, bottom=0.05, left=0.05, right=0.97,
                        hspace=0.52, wspace=0.30)

    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1.0, 1.0])

    # ── Master title ─────────────────────────────────────────────────────
    best_label = results_df.iloc[0]["Model"]
    ticker_str = "ES=F (E-mini S&P 500 Futures)"
    date_rng   = f"{log_ret.index[0].date()} → {log_ret.index[-1].date()}"

    fig.suptitle(
        "GARCH VOLATILITY FORECASTING DASHBOARD",
        fontsize=22, fontweight="bold",
        color=C["accent_cyan"], y=0.97
    )
    fig.text(
        0.5, 0.935,
        f"{ticker_str}  ·  {date_rng}  ·  "
        f"Best model: {best_label}  ·  N={n:,} obs",
        ha="center", fontsize=9.5, color=C["text_dim"], fontstyle="italic"
    )

    # ═══════════════════════════════════════════════════════════════════
    # Panel 1 · Price History + Rolling Volatility
    # ═══════════════════════════════════════════════════════════════════
    ax1a = fig.add_subplot(gs[0, :2])
    _style(ax1a, "PRICE HISTORY  &  ROLLING 21-DAY VOLATILITY",
           "Date", "ES Close Price")

    price_idx = df.index[-len(log_ret):]
    price_vals = df["Close"].iloc[-len(log_ret):].values

    ax1a.plot(log_ret.index, price_vals, color=C["accent_cyan"], lw=1.2, alpha=0.9)
    ax1a.fill_between(log_ret.index, price_vals, price_vals.min(),
                      color=C["accent_cyan"], alpha=0.05)

    ax1b = ax1a.twinx()
    ax1b.plot(log_ret.index, roll_21_vol, color=C["accent_orange"],
              lw=1.0, alpha=0.7, linestyle="--", label="21d Vol (ann%)")
    ax1b.set_ylabel("Realised Vol (%)", color=C["text_dim"], fontsize=9)
    ax1b.tick_params(colors=C["text_dim"], labelsize=8)
    ax1b.spines["right"].set_color(C["grid"])

    lines1 = [
        plt.Line2D([0], [0], color=C["accent_cyan"], lw=1.5),
        plt.Line2D([0], [0], color=C["accent_orange"], lw=1.2, ls="--"),
    ]
    ax1a.legend(lines1, ["ES Close", "21d RV (ann%)"],
                loc="upper left", fontsize=8,
                facecolor=C["bg_panel"], edgecolor=C["grid"], labelcolor=C["text"])

    # ═══════════════════════════════════════════════════════════════════
    # Panel 2 · Return Distribution + Normal overlay
    # ═══════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2, "RETURN DISTRIBUTION", "Daily Log Return", "Density")

    n_bins = min(80, int(n ** 0.5) * 4)
    ret_pct = ret * 100
    bin_edges = np.linspace(ret_pct.min(), ret_pct.max(), n_bins)
    hist_n, edges, patches = ax2.hist(ret_pct, bins=bin_edges, density=True,
                                      edgecolor="none", alpha=0.8)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    norm_bc = (bin_centres - bin_centres.min()) / (bin_centres.max() - bin_centres.min() + 1e-9)
    for patch, nv in zip(patches, norm_bc):
        patch.set_facecolor(GRAD_MAIN(nv))

    x_ln = np.linspace(ret_pct.min(), ret_pct.max(), 300)
    mu_r, sd_r = ret_pct.mean(), ret_pct.std()
    ax2.plot(x_ln, norm.pdf(x_ln, mu_r, sd_r), color=C["accent_pink"],
             lw=2, label="Normal")

    if best_dist == "t":
        nu_p = diagnostics.get("nu_est", 8)
        ax2.plot(x_ln, student_t.pdf((x_ln - mu_r) / sd_r, df=nu_p) / sd_r,
                 color=C["gold"], lw=1.8, ls="--", label=f"Student-t (ν≈{nu_p:.1f})")

    ax2.axvline(0, color=C["text_dim"], lw=0.8, ls=":")
    ax2.legend(fontsize=8, facecolor=C["bg_panel"], edgecolor=C["grid"],
               labelcolor=C["text"])

    # Kurtosis annotation
    ax2.text(0.97, 0.95, f"Kurt={diagnostics['exc_kurt']:.2f}\nSkew={diagnostics['skewness']:.2f}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=8, color=C["gold"], fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C["bg_card"], alpha=0.8))

    # ═══════════════════════════════════════════════════════════════════
    # Panel 3 · Conditional Volatility (GARCH fitted)
    # ═══════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, :2])
    _style(ax3, f"CONDITIONAL VOLATILITY  [{best_label}]",
           "Date", "Ann. Volatility (%)")

    ax3.fill_between(log_ret.index, conditional_vol,
                     color=C["accent_purple"], alpha=0.15)
    ax3.plot(log_ret.index, conditional_vol, color=C["accent_purple"], lw=1.4)
    ax3.plot(log_ret.index, roll_21_vol, color=C["accent_orange"],
             lw=0.8, alpha=0.6, linestyle="--", label="21d Realised Vol")

    # Shade regimes
    regime_labels = diagnostics.get("regime_labels", None)
    if regime_labels is not None:
        regime_colors = [C["accent_green"], C["accent_orange"], C["accent_red"]]
        roll_idx = log_ret.index[20:]   # rolling starts at 21
        rl = regime_labels[-len(roll_idx):]
        for i, rc in enumerate(regime_colors):
            mask = rl == i
            if mask.any():
                ax3.fill_between(roll_idx, 0, conditional_vol[-len(roll_idx):].max() * 1.1,
                                 where=mask, color=rc, alpha=0.04, label=None)

    last_cond_vol = float(conditional_vol.iloc[-1])
    ax3.axhline(last_cond_vol, color=C["gold"], lw=1, ls=":",
                label=f"Latest: {last_cond_vol:.1f}%")
    ax3.legend(loc="upper right", fontsize=8, facecolor=C["bg_panel"],
               edgecolor=C["grid"], labelcolor=C["text"])

    # ═══════════════════════════════════════════════════════════════════
    # Panel 4 · Model Comparison (AIC / BIC bar chart)
    # ═══════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 2])
    _style(ax4, "MODEL COMPARISON (AIC)", "AIC", "Model")

    top_n = min(8, len(results_df))
    sub_df = results_df.head(top_n).iloc[::-1]    # best at top (reversed → best on top)
    colors_bar = [C["accent_cyan"] if i == top_n - 1 else C["indigo"]
                  for i in range(top_n)]

    # AIC relative to best (so bars are all positive, best = 0)
    aic_vals = sub_df["AIC"].values          # already sorted worst→best (reversed)
    aic_best = aic_vals.min()
    aic_rel  = aic_vals - aic_best
    ax4.barh(sub_df["Model"].values, aic_rel, color=colors_bar[::-1], alpha=0.85, height=0.6)
    ax4.set_xlabel("ΔAIC from best", color=C["text_dim"], fontsize=8)
    ax4.tick_params(labelleft=True, labelsize=7.5)

    for i, (delta, raw) in enumerate(zip(aic_rel, aic_vals)):
        ax4.text(delta + 0.1, i, f"{raw:.1f}",
                 va="center", fontsize=7, color=C["text"])

    # ═══════════════════════════════════════════════════════════════════
    # Panel 5 · Regime Clustering
    # ═══════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 0])
    _style(ax5, "VOLATILITY REGIME CLUSTERING", "Date", "Ann. Vol (%)")

    roll_vol_daily = pd.Series(ret).rolling(21).std() * np.sqrt(252) * 100
    roll_idx       = log_ret.index[20:]
    rl_vals        = roll_vol_daily.dropna().values
    r_labs         = regime_labels[-len(rl_vals):] if regime_labels is not None else np.zeros(len(rl_vals), dtype=int)

    regime_colors_full = [C["accent_green"], C["accent_orange"], C["accent_red"]]
    for i, (rc, rn) in enumerate(zip(regime_colors_full, ["Low", "Mid", "High"])):
        mask = r_labs == i
        if mask.any():
            ax5.scatter(roll_idx[mask], rl_vals[mask],
                        color=rc, s=6, alpha=0.6, label=rn, zorder=4)

    centers = diagnostics.get("regime_centers", [])
    for i, (cen, rc) in enumerate(zip(centers, regime_colors_full)):
        ax5.axhline(cen, color=rc, lw=1, ls="--", alpha=0.7)

    ax5.legend(fontsize=8, facecolor=C["bg_panel"], edgecolor=C["grid"],
               labelcolor=C["text"])

    # ═══════════════════════════════════════════════════════════════════
    # Panel 6 · QQ Plot (residuals vs t-distribution)
    # ═══════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[2, 1])
    _style(ax6, "Q-Q PLOT (Standardised Residuals)", "Theoretical Quantiles",
           "Sample Quantiles")

    std_resid = best_model.std_resid
    nu_qq = diagnostics.get("nu_est", 8)
    nu_qq = min(max(nu_qq, 2.1), 30)

    if best_dist == "t":
        (osm, osr), (slope, intercept, r2) = stats.probplot(std_resid, dist=student_t, sparams=(nu_qq,))
    else:
        (osm, osr), (slope, intercept, r2) = stats.probplot(std_resid)

    ax6.scatter(osm, osr, color=C["accent_cyan"], s=4, alpha=0.5, zorder=5)
    line_x = np.linspace(osm.min(), osm.max(), 200)
    ax6.plot(line_x, slope * line_x + intercept, color=C["accent_pink"],
             lw=1.5, label=f"R²={r2:.4f}")
    ax6.plot(line_x, line_x, color=C["text_dim"], lw=0.8, ls=":", label="Diagonal")
    ax6.legend(fontsize=8, facecolor=C["bg_panel"], edgecolor=C["grid"],
               labelcolor=C["text"])

    # ═══════════════════════════════════════════════════════════════════
    # Panel 7 (wide bottom) · FORECAST CHART
    # ═══════════════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[2, 2])
    _style(ax7, "VOLATILITY FORECAST  (1–4 WEEKS AHEAD)",
           "Forecast Horizon", "Ann. Volatility (%)")

    weekly_rows = forecasts["weekly_rows"]
    weeks       = [1, 2, 3, 4]
    w_vol_ann   = [r["_weekly_vol_ann"] for r in weekly_rows]
    w_ci_lo     = [r["_ci_lo_95"] * np.sqrt(252 / 5) for r in weekly_rows]  # crude annualise
    w_ci_hi     = [r["_ci_hi_95"] * np.sqrt(252 / 5) for r in weekly_rows]
    w_var95     = [-r["_var_95"] * np.sqrt(252 / 5) for r in weekly_rows]
    w_var99     = [-r["_var_99"] * np.sqrt(252 / 5) for r in weekly_rows]

    ax7.fill_between(weeks, w_ci_lo, w_ci_hi, color=C["accent_purple"],
                     alpha=0.15, label="95% CI (vol)")
    ax7.plot(weeks, w_vol_ann, color=C["accent_cyan"], lw=2.5,
             marker="o", ms=7, zorder=5)

    ax7.axhline(float(conditional_vol.iloc[-1]), color=C["gold"], lw=1,
                ls=":", label=f"Current: {float(conditional_vol.iloc[-1]):.1f}%")

    for w, va, vl in zip(weeks, w_vol_ann, w_ci_lo):
        ax7.text(w, va + 0.3, f"{va:.1f}%", ha="center", va="bottom",
                 fontsize=8, color=C["accent_cyan"], fontfamily="monospace")

    ax7.set_xticks(weeks)
    ax7.set_xticklabels([f"W{w}" for w in weeks])
    ax7.legend(fontsize=8, facecolor=C["bg_panel"], edgecolor=C["grid"],
               labelcolor=C["text"])

    # ── Footer ──────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.013,
        f"GARCH Volatility Forecast  ·  Best model: {best_label}  ·  "
        f"Dist: {best_dist.upper()}  ·  Data: Yahoo Finance (yfinance)  ·  "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ha="center", fontsize=7.5, color=C["text_dim"], fontstyle="italic"
    )

    # ── Save + show ──────────────────────────────────────────────────────
    import os
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "garch_dashboard.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=C["bg_dark"])
    print(f"\n  Dashboard saved → {outpath}")
    plt.show()
    print("  ✓ Dashboard rendered successfully.")


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _banner(title: str):
    line = "═" * (len(title) + 6)
    print(f"\n╔{line}╗")
    print(f"║   {title}   ║")
    print(f"╚{line}╝")


def _style(ax, title="", xlabel="", ylabel="", title_size=12):
    ax.set_facecolor(C["bg_panel"])
    ax.set_title(title, color=C["text"], fontsize=title_size,
                 fontweight="bold", pad=10, loc="left")
    ax.set_xlabel(xlabel, color=C["text_dim"], fontsize=8.5)
    ax.set_ylabel(ylabel, color=C["text_dim"], fontsize=8.5)
    ax.tick_params(colors=C["text_dim"], labelsize=8)
    ax.grid(True, color=C["grid"], alpha=0.45, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])
        spine.set_linewidth(0.5)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def main(
    ticker: str = "ES=F",
    years:  int = 3,
):
    """
    Full pipeline: fetch → clean → validate → fit → forecast → dashboard.

    Parameters
    ----------
    ticker : Yahoo Finance symbol  (default 'ES=F' = E-mini S&P 500 Futures)
    years  : Historical look-back window in years (2–5 recommended)
    """
    print("\n" + "█" * 60)
    print("  GARCH FORECASTING PIPELINE")
    print(f"  Ticker : {ticker}  |  Look-back : {years} years")
    print("█" * 60)

    # ── 1. Data Acquisition ─────────────────────────────────────────────
    df = fetch_data(ticker=ticker, years=years)

    # ── 2. Data Cleaning ────────────────────────────────────────────────
    df_clean, log_ret = clean_data(df)

    # ── 3. Validation ───────────────────────────────────────────────────
    diagnostics = validate_data(log_ret)

    # ── 4. Model Selection ──────────────────────────────────────────────
    best_model, best_vol, best_dist, results_df, fitted_models = fit_best_garch(
        log_ret, diagnostics
    )

    # ── 5. Forecast ─────────────────────────────────────────────────────
    forecasts = generate_forecasts(
        best_model, best_vol, best_dist, log_ret, diagnostics
    )

    # ── Dashboard ───────────────────────────────────────────────────────
    build_dashboard(
        df_clean, log_ret, diagnostics,
        best_model, best_vol, best_dist,
        results_df, fitted_models, forecasts
    )

    return {
        "df": df_clean,
        "log_ret": log_ret,
        "diagnostics": diagnostics,
        "best_model": best_model,
        "results_df": results_df,
        "forecasts": forecasts,
    }


if __name__ == "__main__":
    # ── Configuration ────────────────────────────────────────────────────
    TICKER = "ES=F"   # E-mini S&P 500 Futures
    YEARS  = 3        # 3-year historical window (adjust 2–5)

    results = main(ticker=TICKER, years=YEARS)
