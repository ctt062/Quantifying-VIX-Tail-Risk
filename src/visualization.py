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
