"""Shock identification and arrival-rate modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass
class ShockDefinition:
    threshold: float
    indicator: pd.Series


@dataclass
class HPPResult:
    rate_per_day: float
    rate_per_year: float
    ci_95: tuple


@dataclass
class NHPPResult:
    model: sm.GLM
    result: sm.GLMResultsWrapper
    design: pd.DataFrame


def define_shocks(
    returns: pd.Series,
    quantile: float,
) -> ShockDefinition:
    threshold = returns.quantile(quantile)
    indicator = (returns >= threshold).astype(int)
    return ShockDefinition(threshold=threshold, indicator=indicator)


def fit_gpd(
    returns: pd.Series,
    threshold: Optional[float] = None,
) -> stats._distn_infrastructure.rv_frozen:
    """Fit a Generalized Pareto distribution to exceedances."""

    threshold = threshold or returns.quantile(0.95)
    exceedances = returns[returns > threshold] - threshold
    if exceedances.empty:
        raise ValueError("No exceedances above the specified threshold.")
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    return stats.genpareto(c=shape, loc=0, scale=scale)


def interarrival_series(shock_indicator: pd.Series) -> pd.Series:
    """Compute inter-arrival times (in days) between shocks."""

    shock_dates = shock_indicator[shock_indicator == 1].index
    if len(shock_dates) < 2:
        raise ValueError("Need at least two shocks to compute inter-arrivals.")
    diffs = shock_dates.to_series().diff().dropna().dt.days
    return diffs


def fit_hpp(interarrivals: pd.Series) -> HPPResult:
    """Estimate homogeneous Poisson rate with confidence interval."""

    avg_days = interarrivals.mean()
    rate_per_day = 1.0 / avg_days
    rate_per_year = rate_per_day * 252
    n = len(interarrivals)
    ci_low = stats.chi2.ppf(0.025, 2 * n) / (2 * interarrivals.sum())
    ci_high = stats.chi2.ppf(0.975, 2 * (n + 1)) / (2 * interarrivals.sum())
    return HPPResult(rate_per_day, rate_per_year, (ci_low * 252, ci_high * 252))


def monthly_counts(shock_indicator: pd.Series) -> pd.DataFrame:
    """Aggregate shocks by calendar month for NHPP modeling."""

    df = shock_indicator.to_frame(name="shocks")
    monthly = df.resample("ME").sum()
    monthly["exposure"] = shock_indicator.resample("ME").size()
    monthly["time"] = np.arange(len(monthly))
    monthly["avg_log_vix"] = monthly["shocks"].rolling(3, min_periods=1).mean()
    monthly["month"] = monthly.index.month
    return monthly


def fit_nhpp(counts: pd.DataFrame) -> NHPPResult:
    """Fit Poisson GLM with exposure to allow time-varying rates."""

    design = pd.get_dummies(counts[["time", "avg_log_vix", "month"]], columns=["month"], drop_first=True)
    design = sm.add_constant(design)
    design = design.astype(float)
    model = sm.GLM(
        counts["shocks"],
        design,
        family=sm.families.Poisson(),
        offset=np.log(counts["exposure"].clip(lower=1)),
    )
    result = model.fit()
    return NHPPResult(model=model, result=result, design=design)
