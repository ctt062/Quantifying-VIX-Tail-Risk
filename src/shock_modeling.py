"""Shock identification and arrival-rate modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize


@dataclass
class ShockDefinition:
    threshold: float
    indicator: pd.Series
    method: str = "quantile"  # "quantile" or "volatility_relative"


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


@dataclass
class HawkesResult:
    """Results from Hawkes self-exciting point process."""
    mu: float           # Baseline intensity
    alpha: float        # Excitation magnitude
    beta: float         # Decay rate
    branching_ratio: float  # alpha/beta - measures clustering
    log_likelihood: float
    half_life: float    # ln(2)/beta - time for excitation to halve


def define_shocks(
    returns: pd.Series,
    quantile: float,
) -> ShockDefinition:
    threshold = returns.quantile(quantile)
    indicator = (returns >= threshold).astype(int)
    return ShockDefinition(threshold=threshold, indicator=indicator, method="quantile")


def define_shocks_volatility_relative(
    returns: pd.Series,
    conditional_vol: pd.Series,
    multiplier: float = 2.0,
) -> ShockDefinition:
    """Define shocks as |r_t| > k * sigma_t (relative to conditional volatility).

    This is more economically meaningful as a 'shock' is something unexpected
    given current volatility expectations, not just a large absolute return.

    Parameters
    ----------
    returns : pd.Series
        Return series (log returns or simple returns)
    conditional_vol : pd.Series
        Conditional volatility from GARCH or similar model
    multiplier : float
        Number of standard deviations to define a shock (default 2.0)

    Returns
    -------
    ShockDefinition with volatility-relative threshold
    """
    aligned = pd.concat([returns, conditional_vol], axis=1).dropna()
    aligned.columns = ["returns", "vol"]

    # Shock occurs when |return| exceeds multiplier * conditional vol
    threshold_series = multiplier * aligned["vol"]
    indicator = (np.abs(aligned["returns"]) > threshold_series).astype(int)

    # Store the average threshold for reporting
    avg_threshold = threshold_series.mean()

    return ShockDefinition(
        threshold=avg_threshold,
        indicator=indicator,
        method=f"volatility_relative_k={multiplier}",
    )


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


# ---------------------------------------------------------------------------
# Hawkes Self-Exciting Point Process
# ---------------------------------------------------------------------------


def _hawkes_intensity(times: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Compute Hawkes intensity at each event time.

    λ(t) = μ + Σ_{t_i < t} α * exp(-β * (t - t_i))
    """
    n = len(times)
    intensities = np.zeros(n)

    for i in range(n):
        intensities[i] = mu
        for j in range(i):
            intensities[i] += alpha * np.exp(-beta * (times[i] - times[j]))

    return intensities


def _hawkes_log_likelihood(params: np.ndarray, times: np.ndarray, T: float) -> float:
    """Negative log-likelihood for Hawkes process (to minimize)."""
    mu, alpha, beta = params

    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return 1e10  # Invalid parameters

    n = len(times)
    if n == 0:
        return 1e10

    # Sum of log intensities at event times
    intensities = _hawkes_intensity(times, mu, alpha, beta)
    ll_events = np.sum(np.log(np.maximum(intensities, 1e-10)))

    # Integral of intensity over [0, T]
    integral = mu * T
    for ti in times:
        integral += (alpha / beta) * (1 - np.exp(-beta * (T - ti)))

    return -(ll_events - integral)


def fit_hawkes(shock_indicator: pd.Series) -> HawkesResult:
    """Fit a Hawkes self-exciting point process to shock arrivals.

    The Hawkes process models how past shocks increase the probability of
    future shocks, capturing the clustering behavior observed in financial markets.

    λ(t) = μ + Σ_{t_i < t} α * exp(-β * (t - t_i))

    where:
    - μ: baseline intensity (shocks per day without excitation)
    - α: excitation magnitude (how much each shock increases intensity)
    - β: decay rate (how quickly excitation dies down)
    - α/β: branching ratio (average number of 'child' events per 'parent')
    """
    shock_dates = shock_indicator[shock_indicator == 1].index
    if len(shock_dates) < 10:
        raise ValueError("Need at least 10 shocks to fit Hawkes process reliably.")

    # Convert to numeric times (days from start)
    t0 = shock_indicator.index[0]
    times = np.array([(t - t0).days for t in shock_dates], dtype=float)
    T = (shock_indicator.index[-1] - t0).days

    # Initial guess based on HPP rate
    n_events = len(times)
    mu_init = n_events / T * 0.5  # Assume half from baseline
    alpha_init = 0.3
    beta_init = 0.5

    # Optimize
    result = minimize(
        _hawkes_log_likelihood,
        x0=[mu_init, alpha_init, beta_init],
        args=(times, T),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
    )

    mu, alpha, beta = result.x
    branching_ratio = alpha / beta if beta > 0 else np.inf
    half_life = np.log(2) / beta if beta > 0 else np.inf
    log_likelihood = -result.fun

    return HawkesResult(
        mu=mu,
        alpha=alpha,
        beta=beta,
        branching_ratio=branching_ratio,
        log_likelihood=log_likelihood,
        half_life=half_life,
    )


def hawkes_simulate_intensity(
    shock_indicator: pd.Series,
    hawkes_result: HawkesResult,
) -> pd.Series:
    """Simulate the Hawkes intensity over time given fitted parameters."""
    shock_dates = shock_indicator[shock_indicator == 1].index
    t0 = shock_indicator.index[0]
    times = np.array([(t - t0).days for t in shock_dates], dtype=float)

    # Compute intensity at each day
    all_days = np.array([(t - t0).days for t in shock_indicator.index], dtype=float)
    mu, alpha, beta = hawkes_result.mu, hawkes_result.alpha, hawkes_result.beta

    intensities = np.full(len(all_days), mu)
    for ti in times:
        mask = all_days > ti
        intensities[mask] += alpha * np.exp(-beta * (all_days[mask] - ti))

    return pd.Series(intensities, index=shock_indicator.index, name="hawkes_intensity")


def monthly_counts(
    shock_indicator: pd.Series,
    log_vix: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Aggregate shocks by calendar month for NHPP modeling.

    Parameters
    ----------
    shock_indicator:
        Daily indicator (1/0) denoting whether a shock occurred.
    log_vix:
        Daily log VIX levels used to build *lagged* covariates that avoid
        incorporating information from the same month as the response.
    """

    if log_vix is None:
        raise ValueError("log_vix series is required to form lagged NHPP covariates.")

    df = shock_indicator.to_frame(name="shocks")
    monthly = df.resample("ME").sum()
    monthly["exposure"] = shock_indicator.resample("ME").size()
    monthly["time"] = np.arange(len(monthly))
    monthly["month"] = monthly.index.month

    avg_log_vix = log_vix.resample("ME").mean()
    monthly["lag_avg_log_vix"] = avg_log_vix.shift(1)
    monthly = monthly.dropna(subset=["lag_avg_log_vix"])
    return monthly


def fit_nhpp(counts: pd.DataFrame) -> NHPPResult:
    """Fit Poisson GLM with exposure to allow time-varying rates."""

    design = pd.get_dummies(
        counts[["time", "lag_avg_log_vix", "month"]],
        columns=["month"],
        drop_first=True,
    )
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


# ---------------------------------------------------------------------------
# Regime Analysis
# ---------------------------------------------------------------------------


@dataclass
class RegimeAnalysis:
    """Results from subsample/regime analysis."""
    regime_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_obs: int
    n_shocks: int
    shock_rate: float  # per year
    avg_return: float
    volatility: float
    hpp_rate: float


def analyze_regime(
    returns: pd.Series,
    shock_indicator: pd.Series,
    regime_name: str,
    start_date: str,
    end_date: str,
) -> RegimeAnalysis:
    """Analyze shock characteristics within a specific time regime."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    mask = (returns.index >= start) & (returns.index <= end)
    regime_returns = returns[mask]
    regime_shocks = shock_indicator[mask]

    n_obs = len(regime_returns)
    n_shocks = int(regime_shocks.sum())
    years = n_obs / 252
    shock_rate = n_shocks / years if years > 0 else 0

    # Try to fit HPP if enough shocks
    try:
        interarrivals = interarrival_series(regime_shocks)
        hpp = fit_hpp(interarrivals)
        hpp_rate = hpp.rate_per_year
    except ValueError:
        hpp_rate = shock_rate

    return RegimeAnalysis(
        regime_name=regime_name,
        start_date=start,
        end_date=end,
        n_obs=n_obs,
        n_shocks=n_shocks,
        shock_rate=shock_rate,
        avg_return=float(regime_returns.mean()),
        volatility=float(regime_returns.std() * np.sqrt(252)),
        hpp_rate=hpp_rate,
    )


def run_regime_analysis(
    returns: pd.Series,
    shock_indicator: pd.Series,
) -> pd.DataFrame:
    """Analyze multiple market regimes for structural differences."""
    regimes = [
        ("Pre-Crisis", "2010-01-01", "2019-12-31"),
        ("COVID Crisis", "2020-01-01", "2020-12-31"),
        ("Post-COVID", "2021-01-01", "2023-12-31"),
        ("Recent", "2024-01-01", "2025-12-31"),
        ("Full Sample", returns.index.min().strftime("%Y-%m-%d"),
         returns.index.max().strftime("%Y-%m-%d")),
    ]

    results = []
    for name, start, end in regimes:
        try:
            analysis = analyze_regime(returns, shock_indicator, name, start, end)
            if analysis.n_obs > 0:
                results.append({
                    "Regime": analysis.regime_name,
                    "Start": analysis.start_date.strftime("%Y-%m-%d"),
                    "End": analysis.end_date.strftime("%Y-%m-%d"),
                    "Obs": analysis.n_obs,
                    "Shocks": analysis.n_shocks,
                    "Rate/Year": analysis.shock_rate,
                    "Ann. Vol": analysis.volatility,
                    "HPP Rate": analysis.hpp_rate,
                })
        except Exception:
            continue

    return pd.DataFrame(results)
