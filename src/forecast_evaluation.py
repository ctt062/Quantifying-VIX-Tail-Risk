"""Forecast generation and evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acovf

from . import config


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
    aligned["var"] = aligned["var"].clip(lower=config.PIT_CLIP_EPS)
    ll = stats.norm.logpdf(
        aligned["actual"],
        loc=aligned["mean"],
        scale=np.sqrt(aligned["var"]),
    )
    return float(ll.mean())


def pointwise_log_scores(actual: pd.Series, mean: pd.Series, variance: pd.Series) -> pd.Series:
    """Return per-date log scores to support cumulative loss visuals."""

    mean = _ensure_series(mean, actual.index, "mean")
    variance = _ensure_series(variance, actual.index, "var")
    aligned = pd.concat([actual, mean, variance], axis=1).dropna()
    aligned.columns = ["actual", "mean", "var"]
    aligned["var"] = aligned["var"].clip(lower=config.PIT_CLIP_EPS)
    ll = stats.norm.logpdf(
        aligned["actual"],
        loc=aligned["mean"],
        scale=np.sqrt(aligned["var"]),
    )
    return pd.Series(ll, index=aligned.index, name="log_score")


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
    pit = stats.norm.cdf(actual, loc=mean, scale=np.sqrt(variance))
    return pd.Series(pit, index=actual.index, name="pit")


def pit_log_loss(pit_values: pd.Series, clip: float = config.PIT_CLIP_EPS) -> pd.Series:
    """Return negative log PIT values with clipping for numerical safety."""

    series = _ensure_series(
        pit_values,
        pit_values.index if isinstance(pit_values, pd.Series) else None,
        "pit",
    )
    clipped = series.clip(lower=clip, upper=1 - clip)
    return -np.log(clipped)


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
