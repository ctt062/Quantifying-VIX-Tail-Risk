"""Plotting helpers for the VIX project."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config


sns.set_style(config.PLOT_STYLE)


def plot_vix_series(data: pd.DataFrame, shock_indicator: pd.Series | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    data["vix"].plot(ax=ax, color="#1f77b4", label="VIX")
    ax.set_ylabel("Level")
    ax.set_title("VIX level with identified shocks")
    if shock_indicator is not None:
        spike_dates = shock_indicator[shock_indicator == 1].index
        ax.scatter(spike_dates, data.loc[spike_dates, "vix"], color="red", s=15, label="Shock")
    ax.legend()
    return fig


def plot_residual_diagnostics(std_resid: pd.Series) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(std_resid, ax=axes[0], kde=True, stat="density")
    axes[0].set_title("Std residual density")
    smoothed = (std_resid**2).rolling(20).mean()
    axes[1].plot(smoothed)
    axes[1].set_title("Rolling variance of std residuals")
    return fig


def plot_shock_arrivals(counts: pd.DataFrame) -> plt.Figure:
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
    return fig


def plot_pit(pit_series: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(pit_series, bins=20, stat="density", ax=ax)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_title("PIT histogram")
    return fig
