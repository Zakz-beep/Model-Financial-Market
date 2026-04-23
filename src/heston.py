"""
╔══════════════════════════════════════════════════════════════════╗
║          HESTON STOCHASTIC VOLATILITY MODEL DASHBOARD          ║
║         Monte Carlo Simulation · ES Futures (E-mini S&P)       ║
╚══════════════════════════════════════════════════════════════════╝

Simulates asset price dynamics under the Heston (1993) model:
    dS = (r - 0.5v)S dt + √v S dW₁
    dv = κ(θ - v) dt + σ √v dW₂
    Corr(dW₁, dW₂) = ρ

Dashboard panels:
    1. Monte Carlo Price Paths with Confidence Envelope
    2. Stochastic Variance Paths
    3. Terminal Price Distribution vs Log-Normal
    4. Implied Volatility Smile (Heston vs BSM)
    5. Variance of Variance over Time
    6. Summary Statistics Panel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, lognorm
import yfinance as yf
import warnings
import sys
import io
warnings.filterwarnings("ignore")

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ─────────────────────────── Color Palette ────────────────────────────
COLORS = {
    "bg_dark":       "#0a0e17",
    "bg_panel":      "#0f1520",
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
    "band_fill":     "#0099ff",
    "gold":          "#fbbf24",
}

# Gradient colormaps
CYAN_PURPLE = LinearSegmentedColormap.from_list(
    "cyan_purple", ["#00d2ff", "#7c3aed", "#f472b6"]
)
FIRE = LinearSegmentedColormap.from_list(
    "fire", ["#10b981", "#f59e0b", "#ef4444"]
)


# ════════════════════════════════════════════════════════════════════
#  1. DATA ACQUISITION
# ════════════════════════════════════════════════════════════════════
def fetch_market_data(ticker: str = "ES=F", period: str = "3mo"):
    """Fetch real market data and derive initial parameters."""
    print(f"  ▸ Fetching {ticker} data ({period})...")
    tk = yf.Ticker(ticker)
    hist = tk.history(period=period)

    if hist.empty:
        raise ValueError(f"No data returned for {ticker}")

    s0 = float(hist["Close"].iloc[-1])
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    realized_vol = float(log_returns.std()) * np.sqrt(252)
    v0 = realized_vol ** 2
    daily_ret = float(log_returns.mean()) * 252

    print(f"  ▸ Spot Price   : {s0:>10,.2f}")
    print(f"  ▸ Realized Vol : {realized_vol*100:>9.2f}%")
    print(f"  ▸ Annual Return: {daily_ret*100:>9.2f}%")

    return s0, v0, realized_vol, hist


# ════════════════════════════════════════════════════════════════════
#  2. HESTON ENGINE (Euler-Maruyama with full truncation)
# ════════════════════════════════════════════════════════════════════
def simulate_heston(
    S0, v0, r, kappa, theta, sigma_v, rho,
    T, N, n_paths, seed=42
):
    """
    Monte Carlo simulation of the Heston model with full truncation scheme.

    Returns:
        S : (n_paths, N+1) array of price paths
        v : (n_paths, N+1) array of variance paths
        t : (N+1,) time grid
    """
    np.random.seed(seed)
    dt = T / N
    sqrt_dt = np.sqrt(dt)

    S = np.zeros((n_paths, N + 1))
    v = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for i in range(1, N + 1):
        z1 = np.random.standard_normal(n_paths)
        z2 = np.random.standard_normal(n_paths)
        w_s = z1
        w_v = rho * z1 + np.sqrt(1 - rho ** 2) * z2

        v_pos = np.maximum(v[:, i - 1], 0)
        sqrt_v = np.sqrt(v_pos)

        # Variance process (full truncation)
        v[:, i] = (
            v_pos
            + kappa * (theta - v_pos) * dt
            + sigma_v * sqrt_v * sqrt_dt * w_v
        )

        # Log-price process
        S[:, i] = S[:, i - 1] * np.exp(
            (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w_s
        )

    t = np.linspace(0, T, N + 1)
    return S, v, t


# ════════════════════════════════════════════════════════════════════
#  3. HESTON SEMI-CLOSED FORM (Characteristic Function)
# ════════════════════════════════════════════════════════════════════
def heston_char_func(phi, S0, v0, r, kappa, theta, sigma_v, rho, T):
    """Heston characteristic function (Albrecher et al. formulation)."""
    xi = kappa - sigma_v * rho * 1j * phi
    d = np.sqrt(xi ** 2 + sigma_v ** 2 * (1j * phi + phi ** 2))
    g = (xi - d) / (xi + d)
    exp_dT = np.exp(-d * T)

    C = (r * 1j * phi * T
         + (kappa * theta / sigma_v ** 2)
         * ((xi - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))))
    D = ((xi - d) / sigma_v ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))

    return np.exp(C + D * v0 + 1j * phi * np.log(S0))


def heston_call_price(S0, K, v0, r, kappa, theta, sigma_v, rho, T, n_points=256):
    """Heston European call price via numerical integration of the char func."""
    dphi = 0.01
    phi_max = n_points * dphi
    phi = np.arange(dphi, phi_max, dphi)

    def integrand(phi_val):
        numer = np.exp(-1j * phi_val * np.log(K)) * heston_char_func(
            phi_val - 1j, S0, v0, r, kappa, theta, sigma_v, rho, T
        )
        denom = 1j * phi_val * S0 * np.exp(r * T)
        return (numer / denom).real

    integral_vals = np.array([integrand(p) for p in phi])
    P1 = 0.5 + (1 / np.pi) * np.trapz(integral_vals, phi)

    def integrand2(phi_val):
        numer = np.exp(-1j * phi_val * np.log(K)) * heston_char_func(
            phi_val, S0, v0, r, kappa, theta, sigma_v, rho, T
        )
        denom = 1j * phi_val
        return (numer / denom).real

    integral_vals2 = np.array([integrand2(p) for p in phi])
    P2 = 0.5 + (1 / np.pi) * np.trapz(integral_vals2, phi)

    return S0 * P1 - K * np.exp(-r * T) * P2


def bs_implied_vol(call_price, S0, K, r, T, tol=1e-8, max_iter=200):
    """Newton-Raphson implied volatility solver."""
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S0 * norm.pdf(d1) * np.sqrt(T)
        if vega < 1e-12:
            break
        sigma -= (bs_price - call_price) / vega
        if abs(bs_price - call_price) < tol:
            break
    return max(sigma, 0.01)


def compute_vol_smile(S0, v0, r, kappa, theta, sigma_v, rho, T, n_strikes=30):
    """Compute Heston implied vol smile across strikes."""
    moneyness = np.linspace(0.85, 1.15, n_strikes)
    strikes = S0 * moneyness
    ivs = []
    for K in strikes:
        try:
            price = heston_call_price(S0, K, v0, r, kappa, theta, sigma_v, rho, T)
            iv = bs_implied_vol(price, S0, K, r, T)
            ivs.append(iv)
        except Exception:
            ivs.append(np.nan)
    return moneyness, np.array(ivs)


# ════════════════════════════════════════════════════════════════════
#  4. PLOTTING HELPERS
# ════════════════════════════════════════════════════════════════════
def style_axis(ax, title="", xlabel="", ylabel="", title_size=13):
    """Apply consistent dark-theme styling to an axis."""
    ax.set_facecolor(COLORS["bg_panel"])
    ax.set_title(title, color=COLORS["text"], fontsize=title_size,
                 fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(xlabel, color=COLORS["text_dim"], fontsize=9)
    ax.set_ylabel(ylabel, color=COLORS["text_dim"], fontsize=9)
    ax.tick_params(colors=COLORS["text_dim"], labelsize=8)
    ax.grid(True, color=COLORS["grid"], alpha=0.5, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(0.5)


# ════════════════════════════════════════════════════════════════════
#  5. MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════
def build_dashboard():
    """Build the full Heston Model dashboard."""

    print("=" * 52)
    print("   HESTON STOCHASTIC VOLATILITY MODEL DASHBOARD")
    print("=" * 52)

    # ── Fetch data ──────────────────────────────────────────
    S0, v0, realized_vol, hist = fetch_market_data("ES=F", "3mo")

    # ── Model parameters ────────────────────────────────────
    r       = 0.045          # risk-free rate
    kappa   = 3.0            # mean-reversion speed
    theta   = v0             # long-run variance ≈ current
    sigma_v = 0.4            # vol of vol
    rho     = -0.70          # leverage effect
    T       = 30 / 252       # ~1 month
    N       = 252            # steps (daily granularity)
    n_paths = 500            # number of MC paths

    # ── Run simulation ──────────────────────────────────────
    print("\n  ▸ Running Monte Carlo simulation...")
    S, v, t_grid = simulate_heston(
        S0, v0, r, kappa, theta, sigma_v, rho, T, N, n_paths
    )
    t_days = t_grid * 252  # convert to trading days

    # ── Compute statistics ──────────────────────────────────
    mean_path = S.mean(axis=0)
    median_path = np.median(S, axis=0)
    p5  = np.percentile(S, 5, axis=0)
    p25 = np.percentile(S, 25, axis=0)
    p75 = np.percentile(S, 75, axis=0)
    p95 = np.percentile(S, 95, axis=0)

    terminal = S[:, -1]
    terminal_ret = (terminal / S0 - 1) * 100

    # ── Compute vol smile ───────────────────────────────────
    print("  ▸ Computing implied volatility smile...")
    moneyness, iv_smile = compute_vol_smile(
        S0, v0, r, kappa, theta, sigma_v, rho, T=60/252
    )
    bsm_flat_vol = np.sqrt(v0)

    # ── Variance stats ──────────────────────────────────────
    vol_paths = np.sqrt(np.maximum(v, 0)) * 100  # to % annualized
    mean_vol = vol_paths.mean(axis=0)
    p10_vol = np.percentile(vol_paths, 10, axis=0)
    p90_vol = np.percentile(vol_paths, 90, axis=0)

    var_of_var = v.var(axis=0)

    # ════════════════════════════════════════════════════════
    #   BUILD FIGURE
    # ════════════════════════════════════════════════════════
    print("  ▸ Rendering dashboard...\n")

    fig = plt.figure(figsize=(20, 13), facecolor=COLORS["bg_dark"])
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.05, right=0.97,
                        hspace=0.45, wspace=0.30)

    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.3, 1, 0.85])

    # ── Title ───────────────────────────────────────────────
    fig.suptitle(
        "HESTON STOCHASTIC VOLATILITY MODEL",
        fontsize=22, fontweight="bold", color=COLORS["accent_cyan"],
        y=0.97
    )
    fig.text(
        0.5, 0.935,
        f"ES Futures  ·  S₀ = {S0:,.2f}  ·  σ₀ = {realized_vol*100:.1f}%  "
        f"·  κ={kappa}  θ={theta:.4f}  ξ={sigma_v}  ρ={rho}  "
        f"·  {n_paths} paths  ·  T = {T*252:.0f} days",
        ha="center", fontsize=10, color=COLORS["text_dim"],
        fontstyle="italic"
    )

    # ────────────────────────────────────────────────────────
    #   Panel 1: Monte Carlo Price Paths (wide)
    # ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_axis(ax1, "MONTE CARLO PRICE PATHS", "Trading Days", "Price")

    # Individual paths (subtle)
    n_show = min(80, n_paths)
    path_colors = CYAN_PURPLE(np.linspace(0, 1, n_show))
    for i in range(n_show):
        ax1.plot(t_days, S[i], color=path_colors[i], alpha=0.08, lw=0.4)

    # Confidence bands
    ax1.fill_between(t_days, p5, p95, color=COLORS["band_fill"],
                     alpha=0.08, label="90% CI")
    ax1.fill_between(t_days, p25, p75, color=COLORS["band_fill"],
                     alpha=0.15, label="50% CI")

    # Key lines
    ax1.plot(t_days, mean_path, color=COLORS["accent_cyan"], lw=2.0,
             label=f"Mean: {mean_path[-1]:,.1f}", zorder=5)
    ax1.plot(t_days, median_path, color=COLORS["accent_purple"], lw=1.5,
             ls="--", label=f"Median: {median_path[-1]:,.1f}", zorder=5)
    ax1.axhline(S0, color=COLORS["gold"], lw=1, ls=":", alpha=0.7,
                label=f"S₀ = {S0:,.2f}")

    ax1.legend(loc="upper left", fontsize=8, facecolor=COLORS["bg_panel"],
               edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ────────────────────────────────────────────────────────
    #   Panel 2: Terminal Distribution
    # ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_axis(ax2, "TERMINAL DISTRIBUTION (T)", "Price", "Density")

    bins = np.linspace(terminal.min(), terminal.max(), 60)
    n_hist, bin_edges, patches = ax2.hist(
        terminal, bins=bins, density=True,
        edgecolor="none", alpha=0.85
    )
    # Color bins by gradient
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    norm_vals = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
    for patch, nv in zip(patches, norm_vals):
        patch.set_facecolor(CYAN_PURPLE(nv))

    # Fit log-normal overlay
    shape, loc, scale = lognorm.fit(terminal, floc=0)
    x_ln = np.linspace(terminal.min(), terminal.max(), 200)
    ax2.plot(x_ln, lognorm.pdf(x_ln, shape, loc, scale),
             color=COLORS["accent_pink"], lw=2, label="Log-Normal Fit")

    ax2.axvline(S0, color=COLORS["gold"], lw=1.2, ls=":", label=f"S₀")
    ax2.axvline(terminal.mean(), color=COLORS["accent_cyan"], lw=1.2,
                ls="--", label=f"μ = {terminal.mean():,.1f}")

    ax2.legend(fontsize=7.5, facecolor=COLORS["bg_panel"],
               edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ────────────────────────────────────────────────────────
    #   Panel 3: Volatility Paths
    # ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    style_axis(ax3, "STOCHASTIC VOLATILITY PATHS (σ annualized %)",
               "Trading Days", "Volatility (%)")

    n_vol_show = min(60, n_paths)
    vol_colors = FIRE(np.linspace(0, 1, n_vol_show))
    for i in range(n_vol_show):
        ax3.plot(t_days, vol_paths[i], color=vol_colors[i], alpha=0.1, lw=0.4)

    ax3.fill_between(t_days, p10_vol, p90_vol,
                     color=COLORS["accent_orange"], alpha=0.1, label="80% CI")
    ax3.plot(t_days, mean_vol, color=COLORS["accent_orange"], lw=2,
             label=f"Mean Vol: {mean_vol[-1]:.1f}%")
    ax3.axhline(np.sqrt(theta) * 100, color=COLORS["accent_green"], lw=1.2,
                ls=":", label=f"√θ = {np.sqrt(theta)*100:.1f}%")

    ax3.legend(loc="upper right", fontsize=8, facecolor=COLORS["bg_panel"],
               edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ────────────────────────────────────────────────────────
    #   Panel 4: Implied Vol Smile
    # ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_axis(ax4, "IMPLIED VOLATILITY SMILE", "Moneyness (K/S₀)", "IV (%)")

    valid = ~np.isnan(iv_smile)
    ax4.fill_between(moneyness[valid], iv_smile[valid] * 100,
                     bsm_flat_vol * 100, color=COLORS["accent_purple"],
                     alpha=0.12)
    ax4.plot(moneyness[valid], iv_smile[valid] * 100,
             color=COLORS["accent_purple"], lw=2.5, marker="o", ms=3,
             label="Heston IV", zorder=5)
    ax4.axhline(bsm_flat_vol * 100, color=COLORS["text_dim"], lw=1.2,
                ls="--", label=f"BSM Flat σ = {bsm_flat_vol*100:.1f}%")
    ax4.axvline(1.0, color=COLORS["gold"], lw=0.8, ls=":", alpha=0.6)

    ax4.legend(fontsize=8, facecolor=COLORS["bg_panel"],
               edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ────────────────────────────────────────────────────────
    #   Panel 5: Variance of Variance
    # ────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_axis(ax5, "VARIANCE OF VARIANCE", "Trading Days", "Var(v)")

    ax5.fill_between(t_days, 0, var_of_var, color=COLORS["accent_cyan"],
                     alpha=0.15)
    ax5.plot(t_days, var_of_var, color=COLORS["accent_cyan"], lw=1.8)

    # ────────────────────────────────────────────────────────
    #   Panel 6: Return Distribution
    # ────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_axis(ax6, "TERMINAL RETURN DISTRIBUTION", "Return (%)", "Density")

    bins_r = np.linspace(terminal_ret.min(), terminal_ret.max(), 50)
    n_r, edges_r, patches_r = ax6.hist(
        terminal_ret, bins=bins_r, density=True,
        edgecolor="none", alpha=0.85
    )
    for patch, edge in zip(patches_r, edges_r):
        patch.set_facecolor(COLORS["accent_green"] if edge >= 0 else COLORS["accent_red"])

    # Normal overlay
    mu_r, std_r = terminal_ret.mean(), terminal_ret.std()
    x_n = np.linspace(terminal_ret.min(), terminal_ret.max(), 200)
    ax6.plot(x_n, norm.pdf(x_n, mu_r, std_r),
             color=COLORS["accent_pink"], lw=2, label="Normal Fit")
    ax6.axvline(0, color=COLORS["gold"], lw=1, ls=":")
    ax6.legend(fontsize=8, facecolor=COLORS["bg_panel"],
               edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ────────────────────────────────────────────────────────
    #   Panel 7: Summary Statistics
    # ────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(COLORS["bg_panel"])
    ax7.axis("off")

    prob_profit = (terminal > S0).mean() * 100
    expected_ret = terminal_ret.mean()
    vol_term = terminal_ret.std()
    skew = float(((terminal_ret - mu_r) ** 3).mean() / std_r ** 3)
    kurt = float(((terminal_ret - mu_r) ** 4).mean() / std_r ** 4 - 3)
    max_dd = float(((S.min(axis=1) / S0 - 1) * 100).min())
    sharpe = expected_ret / vol_term if vol_term > 0 else 0

    stats_lines = [
        ("SUMMARY STATISTICS", "", COLORS["accent_cyan"], 13, "bold"),
        ("", "", COLORS["text"], 6, "normal"),
        ("Expected Return", f"{expected_ret:+.2f}%", COLORS["accent_green"] if expected_ret > 0 else COLORS["accent_red"], 10.5, "normal"),
        ("Volatility (σ)", f"{vol_term:.2f}%", COLORS["text"], 10.5, "normal"),
        ("Sharpe Ratio", f"{sharpe:.3f}", COLORS["text"], 10.5, "normal"),
        ("Skewness", f"{skew:.3f}", COLORS["text"], 10.5, "normal"),
        ("Excess Kurtosis", f"{kurt:.3f}", COLORS["text"], 10.5, "normal"),
        ("Max Drawdown", f"{max_dd:.2f}%", COLORS["accent_red"], 10.5, "normal"),
        ("P(Profit)", f"{prob_profit:.1f}%", COLORS["accent_green"], 10.5, "normal"),
        ("", "", COLORS["text"], 4, "normal"),
        ("VaR (5%)", f"{np.percentile(terminal_ret, 5):.2f}%", COLORS["accent_orange"], 10.5, "normal"),
        ("CVaR (5%)", f"{terminal_ret[terminal_ret <= np.percentile(terminal_ret, 5)].mean():.2f}%", COLORS["accent_orange"], 10.5, "normal"),
    ]

    y_pos = 0.95
    for label, value, color, size, weight in stats_lines:
        if value:
            ax7.text(0.08, y_pos, label, transform=ax7.transAxes,
                     fontsize=size, color=COLORS["text_dim"], fontweight=weight,
                     va="top", fontfamily="monospace")
            ax7.text(0.92, y_pos, value, transform=ax7.transAxes,
                     fontsize=size, color=color, fontweight="bold",
                     va="top", ha="right", fontfamily="monospace")
        else:
            ax7.text(0.08, y_pos, label, transform=ax7.transAxes,
                     fontsize=size, color=color, fontweight=weight,
                     va="top", fontfamily="monospace")
        y_pos -= 0.075

    # Border for stats panel
    for spine in ax7.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(0.5)
        spine.set_visible(True)

    # ── Footer ──────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Heston (1993) · Euler-Maruyama Full Truncation · Monte Carlo Simulation · "
        "Data: Yahoo Finance",
        ha="center", fontsize=8, color=COLORS["text_dim"],
        fontstyle="italic"
    )

    plt.show()
    print("  ✓ Dashboard rendered successfully.")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_dashboard()