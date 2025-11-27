"""Plotting helpers for the VIX project."""

from __future__ import annotations

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from . import config


sns.set_style(config.PLOT_STYLE)


def _save(fig: plt.Figure, name: str | None) -> None:
    if not name:
        return
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved figure to {path}")


def plot_vix_series(
    data: pd.DataFrame,
    shock_indicator: pd.Series | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    data["vix"].plot(ax=ax, color="#1f77b4", label="VIX")
    ax.set_ylabel("Level")
    ax.set_title("VIX level with identified shocks")
    if shock_indicator is not None:
        spike_dates = shock_indicator[shock_indicator == 1].index
        ax.scatter(spike_dates, data.loc[spike_dates, "vix"], color="red", s=15, label="Shock")
    ax.legend()
    _save(fig, save_as)
    return fig


def plot_residual_diagnostics(std_resid: pd.Series, save_as: str | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(std_resid, ax=axes[0], kde=True, stat="density")
    axes[0].set_title("Std residual density")
    smoothed = (std_resid**2).rolling(20).mean()
    axes[1].plot(smoothed)
    axes[1].set_title("Rolling variance of std residuals")
    _save(fig, save_as)
    return fig


def plot_shock_arrivals(counts: pd.DataFrame, save_as: str | None = None) -> plt.Figure:
    """Render monthly shock counts with readable date axis."""

    monthly = counts.copy().sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(monthly.index, monthly["shocks"], color="#ff7f0e", width=20, align="center")
    ax.set_title("Monthly shock counts")
    ax.set_ylabel("Count")
    ax.set_xlabel("Date")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_pit(pit_series: pd.Series, save_as: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(pit_series, bins=20, stat="density", ax=ax)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_title("PIT histogram")
    _save(fig, save_as)
    return fig


def plot_news_impact_curve(model_result, save_as: str | None = None) -> plt.Figure:
    """Visualize how shocks of different signs affect next-day variance."""

    z = np.linspace(-3, 3, 200)
    params = model_result.params
    vol_name = model_result.model.volatility.__class__.__name__.lower()

    if "egarch" in vol_name:
        omega = float(params.get("omega", 0.0))
        alpha = float(params.get("alpha[1]", 0.0))
        gamma = float(params.get("gamma[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        expected_abs = np.sqrt(2 / np.pi)
        long_run = np.log(np.maximum(1e-8, np.mean(model_result.conditional_volatility**2)))
        log_sigma2 = omega + beta * long_run + alpha * (np.abs(z) - expected_abs) + gamma * z
        sigma2 = np.exp(log_sigma2)
    else:
        omega = float(params.get("omega", 0.0))
        alpha = float(params.get("alpha[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        denom = max(1e-6, 1 - alpha - beta)
        long_run = omega / denom
        sigma2 = omega + alpha * (z**2 * long_run) + beta * long_run

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, sigma2, color="#2ca02c", linewidth=2)
    ax.set_xlabel("Shock size ($z_{t-1}$)")
    ax.set_ylabel("Next-day variance ($\\sigma_t^2$)")
    ax.set_title("News Impact Curve")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_cumulative_loss(loss_a: pd.Series, loss_b: pd.Series, labels: tuple[str, str], save_as: str | None = None) -> plt.Figure:
    """Plot cumulative difference between two log-score series."""

    diff = (loss_a - loss_b).dropna()
    cumulative = diff.cumsum()
    fig, ax = plt.subplots(figsize=(9, 4))
    cumulative.plot(ax=ax, color="#1f77b4", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"Cumulative Log-Score Difference ({labels[0]} - {labels[1]})")
    ax.set_ylabel("Cumulative difference")
    ax.set_xlabel("Date")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_qq_std_resid(std_resid: pd.Series, dist: str, dof: float | None = None, save_as: str | None = None) -> plt.Figure:
    """Q-Q plot comparing standardized residuals to theoretical distribution."""

    cleaned = std_resid.dropna().sort_values()
    if cleaned.empty:
        raise ValueError("Standardized residuals are empty; cannot plot Q-Q chart.")
    n = len(cleaned)
    probs = (np.arange(1, n + 1) - 0.5) / n
    if dist.lower() == "t" and dof is not None:
        theoretical = stats.t.ppf(probs, df=dof)
    elif dist.lower() == "ged" and dof is not None:
        theoretical = stats.gennorm.ppf(probs, beta=dof)
    else:
        theoretical = stats.norm.ppf(probs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(theoretical, cleaned, s=15, alpha=0.7)
    limits = [min(theoretical.min(), cleaned.min()), max(theoretical.max(), cleaned.max())]
    ax.plot(limits, limits, color="red", linestyle="--", linewidth=1)
    ax.set_title("Standardized Residual Q-Q Plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_interarrival_hist(interarrivals: pd.Series, rate_per_day: float, save_as: str | None = None) -> plt.Figure:
    """Overlay empirical inter-arrival histogram with exponential density."""

    cleaned = interarrivals.dropna()
    if cleaned.empty:
        raise ValueError("Need inter-arrival observations to plot distribution.")

    x = np.linspace(0, cleaned.max() * 1.2, 200)
    density = stats.expon.pdf(x, scale=1 / max(rate_per_day, 1e-9))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(cleaned, bins=20, stat="density", ax=ax, color="#9edae5", edgecolor="black")
    ax.plot(x, density, color="#d62728", linewidth=2, label="Exponential fit")
    ax.set_title("Inter-arrival Time Distribution")
    ax.set_xlabel("Days between shocks")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_acf_comparison(
    raw_squared: pd.Series,
    resid_squared: pd.Series,
    lags: int = 40,
    save_as: str | None = None,
) -> plt.Figure:
    """Compare autocorrelation of squared returns before/after filtering."""

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(raw_squared.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("ACF of squared returns")
    plot_acf(resid_squared.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title("ACF of squared standardized residuals")
    for ax in axes:
        ax.set_xlabel("Lag")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_hawkes_intensity(
    shock_indicator: pd.Series,
    intensity: pd.Series,
    save_as: str | None = None,
) -> plt.Figure:
    """Plot Hawkes self-exciting intensity with shock events."""

    fig, ax = plt.subplots(figsize=(11, 4))

    # Plot intensity
    ax.plot(intensity.index, intensity, color="#1f77b4", linewidth=1, label="Hawkes Intensity")

    # Mark shock events
    shock_dates = shock_indicator[shock_indicator == 1].index
    shock_intensities = intensity.loc[shock_dates]
    ax.scatter(shock_dates, shock_intensities, color="red", s=20, alpha=0.7, label="Shocks", zorder=5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Intensity (λ)")
    ax.set_title("Hawkes Self-Exciting Process Intensity")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_regime_comparison(
    regime_df: pd.DataFrame,
    save_as: str | None = None,
) -> plt.Figure:
    """Bar chart comparing shock rates across market regimes."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    regimes = regime_df["Regime"].tolist()
    x = np.arange(len(regimes))

    # Shock rate comparison
    axes[0].bar(x, regime_df["Rate/Year"], color="#2ca02c", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regimes, rotation=30, ha="right")
    axes[0].set_ylabel("Shocks per Year")
    axes[0].set_title("Shock Rate by Regime")

    # Volatility comparison
    axes[1].bar(x, regime_df["Ann. Vol"], color="#ff7f0e", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regimes, rotation=30, ha="right")
    axes[1].set_ylabel("Annualized Volatility")
    axes[1].set_title("Volatility by Regime")

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_model_comparison(
    model_summary: pd.DataFrame,
    save_as: str | None = None,
) -> plt.Figure:
    """Bar chart comparing AIC/BIC across volatility models."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    models = model_summary["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    # AIC comparison
    axes[0].bar(x, model_summary["aic"], width, color="#1f77b4", label="AIC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("AIC")
    axes[0].set_title("Model Comparison - AIC (lower is better)")

    # BIC comparison
    axes[1].bar(x, model_summary["bic"], width, color="#ff7f0e", label="BIC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("BIC")
    axes[1].set_title("Model Comparison - BIC (lower is better)")

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_jump_distribution(
    shock_magnitudes: pd.Series | np.ndarray,
    fitted_dist: str,
    fitted_params: dict,
    save_as: str | None = None,
) -> plt.Figure:
    """Plot histogram of shock magnitudes with fitted distribution overlay."""
    
    magnitudes = np.array(shock_magnitudes)
    magnitudes = magnitudes[magnitudes > 0]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histogram
    sns.histplot(magnitudes, bins=30, stat="density", ax=ax, color="#9edae5", 
                 edgecolor="black", alpha=0.7, label="Observed")
    
    # Fitted distribution
    x = np.linspace(0, np.percentile(magnitudes, 99), 200)
    
    if fitted_dist == "exponential":
        y = stats.expon.pdf(x, scale=fitted_params["scale"])
    elif fitted_dist == "gamma":
        y = stats.gamma.pdf(x, fitted_params["shape"], scale=fitted_params["scale"])
    elif fitted_dist == "lognormal":
        y = stats.lognorm.pdf(x, s=fitted_params["sigma"], scale=np.exp(fitted_params["mu"]))
    elif fitted_dist == "pareto":
        y = stats.pareto.pdf(x, fitted_params["alpha"], scale=fitted_params["xmin"])
    elif fitted_dist == "weibull":
        y = stats.weibull_min.pdf(x, fitted_params["shape"], scale=fitted_params["scale"])
    else:
        y = np.zeros_like(x)
    
    ax.plot(x, y, color="#d62728", linewidth=2.5, label=f"Fitted {fitted_dist.title()}")
    
    ax.set_xlabel("Shock Magnitude (|Δlog VIX|)")
    ax.set_ylabel("Density")
    ax.set_title("Jump Size Distribution (Compound Poisson)")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_cpp_simulation_paths(
    paths: np.ndarray,
    percentile_df: pd.DataFrame = None,
    n_sample_paths: int = 50,
    save_as: str | None = None,
) -> plt.Figure:
    """Plot simulated Compound Poisson Process paths with confidence bands."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    n_paths, n_steps = paths.shape
    time = np.arange(n_steps)
    
    # Plot sample paths (light gray)
    sample_idx = np.random.choice(n_paths, min(n_sample_paths, n_paths), replace=False)
    for idx in sample_idx:
        ax.plot(time, paths[idx], color="gray", alpha=0.2, linewidth=0.5)
    
    # Plot percentile bands
    if percentile_df is not None:
        ax.fill_between(time, percentile_df["p5"], percentile_df["p95"], 
                       alpha=0.3, color="#1f77b4", label="5-95% CI")
        ax.fill_between(time, percentile_df["p25"], percentile_df["p75"], 
                       alpha=0.4, color="#1f77b4", label="25-75% CI")
        ax.plot(time, percentile_df["p50"], color="#d62728", linewidth=2, label="Median")
    
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative Shock Impact")
    ax.set_title("Compound Poisson Process: Simulated Paths (1 Year)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_cpp_var_distribution(
    annual_impacts: np.ndarray,
    var_95: float,
    cvar_95: float,
    save_as: str | None = None,
) -> plt.Figure:
    """Plot distribution of annual impacts with VaR/CVaR markers."""
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Histogram
    sns.histplot(annual_impacts, bins=50, stat="density", ax=ax, 
                 color="#2ca02c", alpha=0.6, edgecolor="black")
    
    # VaR line
    ax.axvline(var_95, color="#d62728", linewidth=2, linestyle="--", 
               label=f"VaR 95% = {var_95:.3f}")
    
    # CVaR line
    ax.axvline(cvar_95, color="#9467bd", linewidth=2, linestyle="-.", 
               label=f"CVaR 95% = {cvar_95:.3f}")
    
    # Shade the tail
    x_tail = annual_impacts[annual_impacts >= var_95]
    if len(x_tail) > 0:
        ax.axvspan(var_95, annual_impacts.max(), alpha=0.2, color="#d62728", 
                   label="95% Tail")
    
    ax.set_xlabel("Annual Cumulative Shock Impact")
    ax.set_ylabel("Density")
    ax.set_title("Annual Impact Distribution (Compound Poisson)")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_cpp_regime_comparison(
    regime_df: pd.DataFrame,
    save_as: str | None = None,
) -> plt.Figure:
    """Compare Compound Poisson parameters across regimes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    regimes = regime_df["Regime"].tolist()
    x = np.arange(len(regimes))
    
    # Arrival rate
    axes[0, 0].bar(x, regime_df["λ/Year"], color="#1f77b4", alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(regimes, rotation=20, ha="right")
    axes[0, 0].set_ylabel("Shocks per Year")
    axes[0, 0].set_title("Arrival Rate (λ)")
    
    # Mean jump
    axes[0, 1].bar(x, regime_df["E[J]"], color="#ff7f0e", alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(regimes, rotation=20, ha="right")
    axes[0, 1].set_ylabel("Mean Jump Size")
    axes[0, 1].set_title("Expected Jump E[J]")
    
    # Expected annual impact
    axes[1, 0].bar(x, regime_df["E[S]/Year"], color="#2ca02c", alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(regimes, rotation=20, ha="right")
    axes[1, 0].set_ylabel("Expected Annual Impact")
    axes[1, 0].set_title("E[S] = λ × E[J] × 252")
    
    # VaR comparison
    axes[1, 1].bar(x, regime_df["VaR 95%"], color="#d62728", alpha=0.7, label="VaR 95%")
    axes[1, 1].bar(x, regime_df["CVaR 95%"], color="#9467bd", alpha=0.4, label="CVaR 95%")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(regimes, rotation=20, ha="right")
    axes[1, 1].set_ylabel("Risk Measure")
    axes[1, 1].set_title("VaR & CVaR (95%)")
    axes[1, 1].legend()
    
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_shock_magnitude_over_time(
    returns: pd.Series,
    shock_indicator: pd.Series,
    save_as: str | None = None,
) -> plt.Figure:
    """Plot shock magnitudes over time with rolling statistics."""
    
    shock_dates = shock_indicator[shock_indicator == 1].index
    shock_magnitudes = np.abs(returns.loc[shock_dates])
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    # Top: Shock magnitudes scatter
    axes[0].scatter(shock_dates, shock_magnitudes, s=30, alpha=0.7, 
                    c=shock_magnitudes, cmap="Reds", edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Shock Magnitude")
    axes[0].set_title("Shock Magnitudes Over Time")
    
    # Add rolling mean of shock magnitudes
    shock_series = pd.Series(shock_magnitudes.values, index=shock_dates)
    rolling_mean = shock_series.rolling(window=20, min_periods=5).mean()
    axes[0].plot(rolling_mean.index, rolling_mean.values, color="#1f77b4", 
                 linewidth=2, label="20-shock rolling mean")
    axes[0].legend()
    
    # Bottom: Cumulative shock count
    cumulative = shock_indicator.cumsum()
    axes[1].plot(cumulative.index, cumulative.values, color="#2ca02c", linewidth=1.5)
    axes[1].set_ylabel("Cumulative Shocks")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Cumulative Shock Count")
    
    fig.tight_layout()
    _save(fig, save_as)
    return fig
