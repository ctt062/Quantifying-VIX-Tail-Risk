"""Forecast generation and evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acovf


def _ensure_series(obj, index, name):
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return pd.Series(obj, index=index, name=name)


def arch_forecast(result, horizon: int = 1) -> pd.DataFrame:
    """Extract one-step-ahead mean and variance forecasts from an ARCH fit."""

    fc = result.forecast(horizon=horizon, reindex=True)
    mean = fc.mean.iloc[:, 0]
    variance = fc.variance.iloc[:, 0]
    return pd.DataFrame({"mean": mean, "variance": variance}).dropna()


def ewma_variance(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Exponentially weighted variance estimate."""

    squared = returns**2
    ewma = squared.ewm(alpha=1 - lam).mean()
    return ewma


def rolling_variance(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling sample variance baseline."""

    return returns.rolling(window).var()


def log_score(actual: pd.Series, mean: pd.Series, variance: pd.Series) -> float:
    """Compute average predictive log score under Normal assumption."""

    mean = _ensure_series(mean, actual.index, "mean")
    variance = _ensure_series(variance, actual.index, "var")
    aligned = pd.concat([actual, mean, variance], axis=1).dropna()
    aligned.columns = ["actual", "mean", "var"]
    ll = stats.norm.logpdf(aligned["actual"], loc=aligned["mean"], scale=np.sqrt(aligned["var"]))
    return float(ll.mean())


def coverage_rate(actual: pd.Series, mean: pd.Series, variance: pd.Series, level: float = 0.95) -> float:
    """Check empirical coverage of symmetric prediction intervals."""

    mean = _ensure_series(mean, actual.index, "mean")
    variance = _ensure_series(variance, actual.index, "var")
    z = stats.norm.ppf((1 + level) / 2)
    upper = mean + z * np.sqrt(variance)
    lower = mean - z * np.sqrt(variance)
    inside = (actual >= lower) & (actual <= upper)
    return inside.mean()


def pit_values(actual: pd.Series, mean: pd.Series, variance: pd.Series) -> pd.Series:
    """Probability integral transform values assuming Normal predictive density."""

    mean = _ensure_series(mean, actual.index, "mean")
    variance = _ensure_series(variance, actual.index, "var")
    return stats.norm.cdf(actual, loc=mean, scale=np.sqrt(variance))


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, h: int = 1) -> float:
    """Return Diebold-Mariano p-value for equal predictive accuracy."""

    d = (loss_a - loss_b).dropna()
    if d.empty:
        raise ValueError("Loss differential is empty.")
    mean_d = d.mean()
    gamma = acovf(d, nlag=h - 1, fft=False)
    var_d = gamma[0] + 2 * np.sum(gamma[1:]) if h > 1 else gamma[0]
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=len(d) - 1))
    return float(p_value)
