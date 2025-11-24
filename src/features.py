"""Feature engineering helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def compute_shock_thresholds(
    returns: pd.Series, quantiles: Iterable[float]
) -> pd.Series:
    """Return quantile-based thresholds for defining shocks."""

    q_values = {f"q_{int(q*100)}": returns.quantile(q) for q in quantiles}
    return pd.Series(q_values)


def label_shocks(returns: pd.Series, threshold: float) -> pd.Series:
    """Create binary shock indicator based on threshold exceedance."""

    return (returns >= threshold).astype(int)


def summarize_shocks(labels: pd.Series) -> Tuple[int, float]:
    """Return number of shocks and annualized frequency."""

    if labels.index.freq is None:
        step_days = np.median(np.diff(labels.index.to_numpy()).astype("timedelta64[D]"))
        freq_per_year = 365.25 / max(step_days.astype(float), 1.0)
    else:
        freq_per_year = labels.index.freq.n / 365.25

    total = int(labels.sum())
    rate = total / len(labels) * 252  # approx trading days per year
    return total, rate
