"""Data acquisition and preprocessing utilities for the VIX study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from . import config


@dataclass
class VIXData:
    """Container holding prepared VIX time series."""

    frame: pd.DataFrame
    metadata: dict


def _winsorize(series: pd.Series, alpha: float) -> pd.Series:
    lower = series.quantile(alpha / 2)
    upper = series.quantile(1 - alpha / 2)
    return series.clip(lower, upper)


def download_vix(
    ticker: str = config.DEFAULT_TICKER,
    start: str = config.DEFAULT_START,
    end: Optional[str] = config.DEFAULT_END,
    force: bool = False,
) -> pd.DataFrame:
    """Download VIX data from Yahoo Finance and cache to disk."""

    if config.CACHE_FILE.exists() and not force:
        return pd.read_parquet(config.CACHE_FILE)

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("No data returned by yfinance; check ticker or dates.")

    if isinstance(raw.columns, pd.MultiIndex):
        adj_close = raw["Adj Close"].iloc[:, 0]
    else:
        adj_close = raw["Adj Close"]

    frame = adj_close.to_frame(name="vix")
    frame = frame.asfreq(config.BUSINESS_FREQ).ffill().dropna()
    frame.to_parquet(config.CACHE_FILE)
    return frame


def prepare_series(
    force_download: bool = False,
    winsor_alpha: float = config.WINSOR_ALPHA,
    start: Optional[str] = None,
) -> VIXData:
    """Load, clean, and transform VIX data into log levels and returns."""

    frame = download_vix(start=start or config.DEFAULT_START, force=force_download)

    if winsor_alpha is not None and 0 < winsor_alpha < 1:
        frame["vix_w"] = _winsorize(frame["vix"], alpha=winsor_alpha)
    else:
        frame["vix_w"] = frame["vix"]

    frame["log_vix"] = np.log(frame["vix_w"])
    frame["dlog_vix"] = frame["log_vix"].diff()
    frame = frame.dropna().copy()
    frame.index.name = "date"

    metadata = {
        "start": frame.index.min(),
        "end": frame.index.max(),
        "rows": len(frame),
        "winsor_alpha": winsor_alpha,
    }

    return VIXData(frame=frame, metadata=metadata)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add helper features used in downstream modeling."""

    enriched = data.copy()
    enriched["abs_dlog_vix"] = enriched["dlog_vix"].abs()
    enriched["month"] = enriched.index.month
    enriched["year"] = enriched.index.year
    enriched["quarter"] = enriched.index.quarter
    return enriched
