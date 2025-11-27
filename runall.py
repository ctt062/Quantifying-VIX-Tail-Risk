"""Run the full VIX shock workflow sequentially."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src import (
    config,
    data_pipeline,
    volatility_models,
    shock_modeling,
    forecast_evaluation,
    visualization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the VIX shock workflow.")
    parser.add_argument(
        "--shock-quantile",
        type=float,
        default=config.DEFAULT_SHOCK_QUANTILE,
        help="Quantile threshold used to define shocks (default: %(default)s).",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD date to begin the out-of-sample period.",
    )
    parser.add_argument(
        "--split-fraction",
        type=float,
        default=config.FORECAST_SPLIT_FRACTION,
        help="Training fraction when split-date is unspecified (default: %(default)s).",
    )
    parser.add_argument(
        "--refit-frequency",
        type=str,
        default=config.DEFAULT_REFIT_FREQUENCY,
        help="Pandas offset alias for how often to re-estimate (use 'none' for single fit).",
    )
    parser.add_argument(
        "--shock-quantile-grid",
        type=float,
        nargs="+",
        help="Optional list of quantiles for shock sensitivity sweeps.",
    )
    parser.add_argument(
        "--split-date-grid",
        type=str,
        nargs="+",
        help="Optional YYYY-MM-DD dates for OOS split sensitivity sweeps.",
    )
    return parser.parse_args()


def _select_h1(frame: pd.DataFrame) -> pd.Series:
    """Return the h.1 column from ARCH forecast outputs regardless of index shape."""

    if isinstance(frame.columns, pd.MultiIndex):
        return frame.xs("h.1", axis=1, level=0)
    return frame.iloc[:, 0]


def build_comparison_frame(
    returns: pd.Series,
    garch_oos: pd.DataFrame,
    ewma_var: pd.Series,
    roll_var: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    comparison = pd.concat(
        [
            garch_oos.rename(columns={"mean": "garch_mean", "variance": "garch_var"}),
            ewma_var.rename("ewma_var"),
            roll_var.rename("roll_var"),
        ],
        axis=1,
    ).dropna()
    actual = returns.loc[comparison.index]
    zero_mean = pd.Series(0.0, index=comparison.index)
    return comparison, actual, zero_mean


def evaluate_forecasts(
    actual: pd.Series,
    comparison: pd.DataFrame,
    zero_mean: pd.Series,
) -> dict:
    scores = {
        "garch_log": forecast_evaluation.log_score(
            actual, comparison["garch_mean"], comparison["garch_var"]
        ),
        "ewma_log": forecast_evaluation.log_score(
            actual, zero_mean, comparison["ewma_var"]
        ),
        "roll_log": forecast_evaluation.log_score(
            actual, zero_mean, comparison["roll_var"]
        ),
    }
    coverage = forecast_evaluation.coverage_rate(
        actual, comparison["garch_mean"], comparison["garch_var"]
    )
    pit_garch = forecast_evaluation.pit_values(
        actual, comparison["garch_mean"], comparison["garch_var"]
    )
    pit_ewma = forecast_evaluation.pit_values(
        actual, zero_mean, comparison["ewma_var"]
    )
    return {
        "scores": scores,
        "coverage": coverage,
        "pit_garch": pit_garch,
        "pit_ewma": pit_ewma,
    }


def summarize_shocks(
    returns: pd.Series,
    log_vix: pd.Series,
    quantile: float,
    conditional_vol: pd.Series | None = None,
    use_vol_relative: bool = False,
    vol_multiplier: float = 2.0,
) -> tuple[
    shock_modeling.ShockDefinition,
    pd.DataFrame,
    shock_modeling.NHPPResult,
    dict,
    pd.Series,
    shock_modeling.HPPResult,
]:
    # Choose shock definition method
    if use_vol_relative and conditional_vol is not None:
        shock_def = shock_modeling.define_shocks_volatility_relative(
            returns, conditional_vol, multiplier=vol_multiplier
        )
    else:
        shock_def = shock_modeling.define_shocks(returns, quantile=quantile)

    interarrival = shock_modeling.interarrival_series(shock_def.indicator)
    hpp = shock_modeling.fit_hpp(interarrival)
    monthly = shock_modeling.monthly_counts(shock_def.indicator, log_vix)
    nhpp = shock_modeling.fit_nhpp(monthly)
    summary = {
        "quantile": quantile,
        "threshold": shock_def.threshold,
        "count": int(shock_def.indicator.sum()),
        "hpp_rate": hpp.rate_per_year,
        "hpp_ci_low": hpp.ci_95[0],
        "hpp_ci_high": hpp.ci_95[1],
        "lag_coef": nhpp.result.params.get("lag_avg_log_vix", float("nan")),
        "method": shock_def.method,
    }
    return shock_def, monthly, nhpp, summary, interarrival, hpp


def run_shock_sensitivity(
    returns: pd.Series,
    log_vix: pd.Series,
    quantiles: list[float],
) -> None:
    rows = []
    for q in quantiles:
        try:
            _shock_def, _monthly, _nhpp, summary, _interarrival, _hpp = summarize_shocks(returns, log_vix, q)
        except ValueError as exc:
            print(f"Skipping quantile {q:.3f}: {exc}")
            continue
        rows.append(summary)
    table = pd.DataFrame(rows)
    if not table.empty:
        print("\nShock quantile sensitivity sweep:")
        print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


def run_split_sensitivity(
    returns: pd.Series,
    ewma_var: pd.Series,
    roll_var: pd.Series,
    distribution: str,
    split_dates: list[str],
    split_fraction: float,
    refit_frequency: str | None,
) -> None:
    rows = []
    for split in split_dates:
        try:
            garch_oos = run_out_of_sample_garch(
                returns,
                split_date=split,
                split_fraction=split_fraction,
                distribution=distribution,
                refit_frequency=refit_frequency,
            )
        except ValueError as exc:
            print(f"Skipping split {split}: {exc}")
            continue
        if garch_oos.empty:
            continue
        comparison, actual, zero_mean = build_comparison_frame(
            returns, garch_oos, ewma_var, roll_var
        )
        metrics = evaluate_forecasts(actual, comparison, zero_mean)
        rows.append(
            {
                "split_date": split,
                "oos_obs": len(actual),
                "garch_log": metrics["scores"]["garch_log"],
                "ewma_log": metrics["scores"]["ewma_log"],
                "roll_log": metrics["scores"]["roll_log"],
                "coverage": metrics["coverage"],
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        print("\nSplit-date sensitivity sweep:")
        print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
def run_out_of_sample_garch(
    returns: pd.Series,
    split_date: str | None = None,
    split_fraction: float = config.FORECAST_SPLIT_FRACTION,
    distribution: str = "t",
    refit_frequency: str | None = config.DEFAULT_REFIT_FREQUENCY,
) -> pd.DataFrame:
    """Fit GARCH on the past and forecast into the future without leakage."""

    if split_date:
        cutoff = pd.Timestamp(split_date)
        train = returns[returns.index < cutoff]
        test = returns[returns.index >= cutoff]
    else:
        fraction = min(max(split_fraction, 0.05), 0.95)
        cut = max(1, min(int(len(returns) * fraction), len(returns) - 1))
        train = returns.iloc[:cut]
        test = returns.iloc[cut:]

    if train.empty or test.empty:
        raise ValueError("Need non-empty train and test splits for OOS evaluation.")

    forecast_start = test.index[0]
    print(
        "Training GARCH on"
        f" {len(train)} obs (through {train.index[-1].date()}) and forecasting"
        f" {len(test)} obs from {forecast_start.date()} onward..."
    )

    def _roll_forward(
        result,
        history_series: pd.Series,
        target_dates: pd.Index,
    ) -> pd.DataFrame:
        params = result.params
        const = float(params.get("Const", 0.0))
        ar1 = float(params.get("ar[1]", 0.0))
        omega = float(params.get("omega", 0.0))
        alpha = float(params.get("alpha[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        last_return_scaled = float(history_series.iloc[-1] * 100)
        last_sigma = float(result.conditional_volatility.iloc[-1])
        prev_sigma2 = last_sigma**2
        last_resid = float(result.resid.iloc[-1])
        rows = []
        for target_date in target_dates:
            mean_scaled = const + ar1 * last_return_scaled
            var_scaled = omega + alpha * (last_resid**2) + beta * prev_sigma2
            rows.append(
                {
                    "date": target_date,
                    "mean": mean_scaled / 100.0,
                    "variance": var_scaled / (100.0 ** 2),
                }
            )
            actual_scaled = float(returns.loc[target_date] * 100)
            last_resid = actual_scaled - mean_scaled
            last_return_scaled = actual_scaled
            prev_sigma2 = var_scaled
        return pd.DataFrame(rows).set_index("date")

    freq = (refit_frequency or "").lower()
    if freq in ("", "none"):
        fit = volatility_models.fit_garch(train, distribution=distribution)
        result = fit["result"]
        return _roll_forward(result, train, test.index)

    try:
        period_index = test.index.to_period(refit_frequency)
    except ValueError as exc:
        raise ValueError(f"Invalid refit frequency '{refit_frequency}': {exc}") from exc

    unique_periods = period_index.unique()
    print(
        f"Rolling refit frequency={refit_frequency} across {len(unique_periods)} windows"
    )
    history_end = train.index[-1]
    history = returns.loc[:history_end]
    forecasts_list: list[pd.DataFrame] = []

    for period in unique_periods:
        period_mask = period_index == period
        period_dates = test.index[period_mask]
        if period_dates.empty:
            continue
        fit = volatility_models.fit_garch(history, distribution=distribution)
        result = fit["result"]
        period_forecast = _roll_forward(result, history, period_dates)
        forecasts_list.append(period_forecast)
        history = returns.loc[:period_dates[-1]]

    if not forecasts_list:
        return pd.DataFrame(columns=["mean", "variance"])
    return pd.concat(forecasts_list).sort_index()


def main(
    shock_quantile: float = config.DEFAULT_SHOCK_QUANTILE,
    split_date: str | None = None,
    split_fraction: float = config.FORECAST_SPLIT_FRACTION,
    refit_frequency: str | None = config.DEFAULT_REFIT_FREQUENCY,
    shock_quantile_grid: list[float] | None = None,
    split_date_grid: list[str] | None = None,
) -> None:
    if not 0 < shock_quantile < 1:
        raise ValueError("Shock quantile must lie in (0, 1).")

    print("=" * 70)
    print("VIX SHOCK PERSISTENCE & FREQUENCY ANALYSIS")
    print("=" * 70)

    print("\n[1/8] Preparing VIX series...")
    vix_data = data_pipeline.prepare_series()
    df = data_pipeline.engineer_features(vix_data.frame)
    returns = df["dlog_vix"].dropna()
    print(
        f"Loaded {len(df):,} rows from {df.index.min().date()}"
        f" to {df.index.max().date()}"
    )

    # =========================================================================
    # VOLATILITY MODELS COMPARISON (GARCH, EGARCH, GJR-GARCH)
    # =========================================================================
    print("\n[2/8] Fitting volatility models...")
    print("-" * 50)

    print("Selecting optimal distribution via PIT diagnostics...")
    selected_dist, garch_fit = volatility_models.select_garch_distribution(returns)
    print(
        f"  Chosen distribution: {selected_dist}"
        f" (PIT KS={garch_fit.get('pit_stat', float('nan')):.4f})"
    )

    egarch_fit = volatility_models.fit_egarch(returns, distribution=selected_dist)
    gjr_fit = volatility_models.fit_gjr_garch(returns, distribution=selected_dist)

    print(f"\n  GJR-GARCH leverage effect (γ): {gjr_fit['leverage_effect']:.4f}")
    if gjr_fit['leverage_effect'] > 0:
        print("    → Negative shocks increase volatility more than positive shocks")

    # HAR-RV model
    print("\nFitting HAR-RV model...")
    har_fit = volatility_models.fit_har_rv(returns)
    print(f"  HAR-RV R²: {har_fit['r_squared']:.4f}")
    print(f"  Coefficients: β_daily={har_fit['coefficients'].get('rv_d_lag', 0):.4f}, "
          f"β_weekly={har_fit['coefficients'].get('rv_w_lag', 0):.4f}, "
          f"β_monthly={har_fit['coefficients'].get('rv_m_lag', 0):.4f}")

    # Model comparison summary
    summary = volatility_models.summarize_fits(garch_fit, egarch_fit, gjr_fit)
    print("\nVolatility Model Comparison:")
    print(summary.to_string(index=False))

    visualization.plot_news_impact_curve(egarch_fit["result"], save_as="news_impact.png")
    visualization.plot_qq_std_resid(
        garch_fit["result"].std_resid,
        dist=selected_dist,
        dof=garch_fit["result"].params.get("nu"),
        save_as="qq.png",
    )
    visualization.plot_acf_comparison(
        returns.pow(2),
        garch_fit["result"].std_resid.pow(2),
        save_as="acf.png",
    )
    visualization.plot_model_comparison(summary, save_as="model_comparison.png")

    # =========================================================================
    # SHOCK IDENTIFICATION (Quantile-based AND Volatility-relative)
    # =========================================================================
    quantile_pct = shock_quantile * 100
    print(f"\n[3/8] Shock identification...")
    print("-" * 50)

    # Standard quantile-based shocks
    print(f"Method 1: Quantile-based ({quantile_pct:.1f}th percentile)")
    shock_def, monthly, nhpp, shock_summary, interarrival, hpp = summarize_shocks(
        returns, df["log_vix"], shock_quantile
    )
    print(
        f"  Threshold={shock_summary['threshold']:.4f}, shocks={shock_summary['count']},"
        f" HPP rate/year={shock_summary['hpp_rate']:.2f}"
        f" (95% CI {shock_summary['hpp_ci_low']:.2f}-{shock_summary['hpp_ci_high']:.2f})"
    )

    # Volatility-relative shocks
    conditional_vol = garch_fit["result"].conditional_volatility / 100.0
    print(f"\nMethod 2: Volatility-relative (|r_t| > 2σ_t)")
    shock_def_vol, monthly_vol, nhpp_vol, shock_summary_vol, interarrival_vol, hpp_vol = summarize_shocks(
        returns, df["log_vix"], shock_quantile,
        conditional_vol=conditional_vol,
        use_vol_relative=True,
        vol_multiplier=2.0,
    )
    print(
        f"  Avg threshold={shock_summary_vol['threshold']:.4f}, shocks={shock_summary_vol['count']},"
        f" HPP rate/year={shock_summary_vol['hpp_rate']:.2f}"
    )

    print(f"\nNHPP lagged log VIX coefficient: {shock_summary['lag_coef']:.4f}")

    visualization.plot_interarrival_hist(
        interarrival,
        rate_per_day=hpp.rate_per_day,
        save_as="interarrival.png",
    )

    # =========================================================================
    # HAWKES SELF-EXCITING PROCESS
    # =========================================================================
    print(f"\n[4/9] Fitting Hawkes self-exciting process...")
    print("-" * 50)

    try:
        hawkes = shock_modeling.fit_hawkes(shock_def.indicator)
        print(f"  Baseline intensity (μ): {hawkes.mu:.4f} shocks/day")
        print(f"  Excitation magnitude (α): {hawkes.alpha:.4f}")
        print(f"  Decay rate (β): {hawkes.beta:.4f}")
        print(f"  Branching ratio (α/β): {hawkes.branching_ratio:.3f}")
        print(f"  Excitation half-life: {hawkes.half_life:.1f} days")

        if hawkes.branching_ratio < 1:
            print("    → Process is stationary (clustering decays over time)")
        else:
            print("    → WARNING: Branching ratio ≥ 1 suggests explosive clustering")

        hawkes_intensity = shock_modeling.hawkes_simulate_intensity(shock_def.indicator, hawkes)
        visualization.plot_hawkes_intensity(shock_def.indicator, hawkes_intensity, save_as="hawkes_intensity.png")
    except ValueError as e:
        print(f"  Could not fit Hawkes process: {e}")

    # =========================================================================
    # COMPOUND POISSON PROCESS
    # =========================================================================
    print(f"\n[5/9] Fitting Compound Poisson Process...")
    print("-" * 50)

    try:
        cpp = shock_modeling.fit_compound_poisson(returns, shock_def.indicator)
        print(f"  Best jump distribution: {cpp.jump_distribution}")
        print(f"  Jump distribution parameters: {cpp.jump_params}")
        print(f"  KS test: statistic={cpp.ks_statistic:.4f}, p-value={cpp.ks_pvalue:.4f}")
        print(f"\n  Arrival rate (λ): {cpp.arrival_rate_annual:.2f} shocks/year")
        print(f"  Mean jump size E[J]: {cpp.mean_jump:.4f} ({cpp.mean_jump*100:.2f}% log-move)")
        print(f"  Std jump size Std[J]: {cpp.std_jump:.4f}")
        print(f"  Expected annual impact E[S] = λ × E[J]: {cpp.expected_annual_impact:.3f}")
        print(f"\n  Risk metrics (10,000 simulations):")
        print(f"    VaR (95%): {cpp.var_95:.3f}")
        print(f"    CVaR (95%): {cpp.cvar_95:.3f}")

        # Plot jump distribution
        shock_dates = shock_def.indicator[shock_def.indicator == 1].index
        shock_magnitudes = np.abs(returns.loc[shock_dates])
        visualization.plot_jump_distribution(
            shock_magnitudes,
            cpp.jump_distribution,
            cpp.jump_params,
            save_as="jump_distribution.png"
        )

        # Simulate and plot CPP paths
        cpp_paths = shock_modeling.simulate_compound_poisson_paths(cpp, n_paths=1000, horizon_days=252)
        percentile_df = shock_modeling.compute_cpp_percentiles(cpp_paths)
        visualization.plot_cpp_simulation_paths(cpp_paths, percentile_df, save_as="cpp_paths.png")

        # Plot VaR distribution
        # Run more simulations for VaR plot
        np.random.seed(42)
        annual_impacts = []
        for _ in range(10000):
            n_shocks = np.random.poisson(cpp.arrival_rate_annual)
            if n_shocks > 0:
                jumps = shock_modeling._sample_from_distribution(
                    cpp.jump_distribution, cpp.jump_params, n_shocks
                )
                annual_impacts.append(np.sum(jumps))
            else:
                annual_impacts.append(0)
        annual_impacts = np.array(annual_impacts)
        visualization.plot_cpp_var_distribution(
            annual_impacts, cpp.var_95, cpp.cvar_95, save_as="cpp_var.png"
        )

        # Shock magnitudes over time
        visualization.plot_shock_magnitude_over_time(
            returns, shock_def.indicator, save_as="shock_magnitudes.png"
        )

        # CPP by regime
        print("\n  Compound Poisson by Market Regime:")
        cpp_regime_df = shock_modeling.compound_poisson_by_regime(returns, shock_def.indicator)
        if not cpp_regime_df.empty:
            print(cpp_regime_df.to_string(index=False))
            visualization.plot_cpp_regime_comparison(cpp_regime_df, save_as="cpp_regime.png")

    except ValueError as e:
        print(f"  Could not fit Compound Poisson: {e}")

    # =========================================================================
    # REGIME ANALYSIS
    # =========================================================================
    print(f"\n[6/9] Regime analysis...")
    print("-" * 50)

    regime_df = shock_modeling.run_regime_analysis(returns, shock_def.indicator)
    if not regime_df.empty:
        print("\nShock Characteristics by Market Regime:")
        print(regime_df.to_string(index=False))
        visualization.plot_regime_comparison(regime_df, save_as="regime_comparison.png")

    # =========================================================================
    # OUT-OF-SAMPLE FORECAST EVALUATION
    # =========================================================================
    print("\n[7/9] Out-of-sample forecast evaluation...")
    print("-" * 50)

    ewma_full = forecast_evaluation.ewma_variance(returns)
    roll_full = forecast_evaluation.rolling_variance(returns)
    har_var = volatility_models.har_rv_forecast(returns, har_fit)

    garch_oos = run_out_of_sample_garch(
        returns,
        split_date=split_date,
        split_fraction=split_fraction,
        distribution=selected_dist,
        refit_frequency=refit_frequency,
    )

    comparison, actual, zero_mean = build_comparison_frame(
        returns, garch_oos, ewma_full, roll_full
    )
    metrics = evaluate_forecasts(actual, comparison, zero_mean)

    # Add HAR-RV scores
    har_aligned = har_var.reindex(comparison.index).dropna()
    if len(har_aligned) > 0:
        har_log_score = forecast_evaluation.log_score(
            actual.loc[har_aligned.index], zero_mean.loc[har_aligned.index], har_aligned
        )
        metrics["scores"]["har_log"] = har_log_score

    print("\nLog-score comparison (OOS) - higher is better:")
    for model, score in metrics["scores"].items():
        print(f"  {model}: {score:.4f}")

    print(f"\n95% coverage (OOS GARCH): {metrics['coverage']:.3f}")

    visualization.plot_vix_series(df, shock_indicator=shock_def.indicator, save_as="vix_series.png")
    visualization.plot_shock_arrivals(monthly, save_as="shock_counts.png")
    visualization.plot_pit(metrics["pit_garch"], save_as="pit.png")

    garch_log_series = forecast_evaluation.pointwise_log_scores(
        actual, comparison["garch_mean"], comparison["garch_var"]
    )
    ewma_log_series = forecast_evaluation.pointwise_log_scores(
        actual, zero_mean, comparison["ewma_var"]
    )
    visualization.plot_cumulative_loss(
        garch_log_series, ewma_log_series,
        labels=("GARCH", "EWMA"),
        save_as="cum_loss_diff.png",
    )

    # =========================================================================
    # STATISTICAL TESTS
    # =========================================================================
    print(f"\n[8/9] Statistical tests...")
    print("-" * 50)

    loss_garch = forecast_evaluation.pit_log_loss(metrics["pit_garch"])
    loss_ewma = forecast_evaluation.pit_log_loss(metrics["pit_ewma"]).reindex(loss_garch.index)
    dm_p = forecast_evaluation.diebold_mariano(loss_garch, loss_ewma)
    print(f"Diebold-Mariano p-value (GARCH vs EWMA): {dm_p:.4f}")
    if dm_p < 0.05:
        print("  → GARCH significantly outperforms EWMA at 5% level")
    else:
        print("  → No significant difference between models")

    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================
    print(f"\n[9/9] Sensitivity analysis...")
    print("-" * 50)

    if shock_quantile_grid:
        sweep_values = [q for q in shock_quantile_grid if 0 < q < 1]
        if sweep_values:
            run_shock_sensitivity(returns, df["log_vix"], sweep_values)

    if split_date_grid:
        run_split_sensitivity(
            returns, ewma_full, roll_full, selected_dist,
            split_date_grid, split_fraction, refit_frequency,
        )

    # =========================================================================
    # ECONOMIC INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("ECONOMIC INTERPRETATION")
    print("=" * 70)

    print("""
Key Findings:
-------------
1. VOLATILITY PERSISTENCE:
   - GARCH persistence measures how long volatility shocks take to decay
   - High persistence (>0.95) indicates long memory in volatility
   - Half-life shows median reversion time to long-run volatility

2. LEVERAGE EFFECT (GJR-GARCH):
   - Positive gamma indicates negative returns increase volatility more
   - This asymmetry is crucial for risk management and option pricing

3. SHOCK CLUSTERING (Hawkes Process):
   - Branching ratio < 1: shocks trigger more shocks but decay over time
   - High alpha: strong initial excitation after each shock
   - Low beta: slow decay of excitation (prolonged clustering)

4. COMPOUND POISSON PROCESS:
   - Models BOTH shock timing (Poisson arrivals) AND magnitude (jump sizes)
   - E[S] = λ × E[J]: expected cumulative annual impact
   - VaR/CVaR quantify tail risk from shock accumulation
   - Jump distribution choice (gamma, lognormal, etc.) captures fat tails

5. REGIME DEPENDENCE:
   - Crisis periods show elevated shock rates and volatility
   - Risk models should account for regime-switching behavior

6. PRACTICAL IMPLICATIONS:
   - Use EGARCH/GJR-GARCH for asymmetric volatility modeling
   - Hawkes process captures self-excitation in shock arrivals
   - Compound Poisson provides VaR/CVaR for aggregate shock risk
   - Volatility-relative shocks provide cleaner 'surprise' identification
""")

    print("\n" + "=" * 70)
    print("Analysis complete. Figures saved to figures/")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    main(
        shock_quantile=args.shock_quantile,
        split_date=args.split_date,
        split_fraction=args.split_fraction,
        refit_frequency=args.refit_frequency,
        shock_quantile_grid=args.shock_quantile_grid,
        split_date_grid=args.split_date_grid,
    )
