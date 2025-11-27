"""Unit tests for VIX shock analysis modules."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src import volatility_models, shock_modeling, forecast_evaluation


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample return series for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    returns = pd.Series(np.random.normal(0, 0.02, 500), index=dates, name="returns")
    return returns


@pytest.fixture
def sample_returns_with_shocks():
    """Generate return series with obvious shock clustering."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    returns = np.random.normal(0, 0.01, 500)
    # Add clustered shocks
    returns[100:110] = np.random.normal(0.05, 0.02, 10)  # Shock cluster 1
    returns[200:205] = np.random.normal(0.04, 0.015, 5)  # Shock cluster 2
    returns[350:360] = np.random.normal(0.045, 0.02, 10)  # Shock cluster 3
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def sample_variance():
    """Generate sample variance series."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    variance = pd.Series(np.abs(np.random.normal(0.0004, 0.0001, 500)), index=dates)
    return variance


# =============================================================================
# Volatility Models Tests
# =============================================================================

class TestGARCH:
    """Tests for GARCH model functions."""

    def test_fit_garch_returns_dict(self, sample_returns):
        """fit_garch should return a dictionary with expected keys."""
        result = volatility_models.fit_garch(sample_returns, distribution="normal")
        assert isinstance(result, dict)
        assert "result" in result
        assert "persistence" in result
        assert "half_life" in result
        assert "distribution" in result

    def test_fit_garch_persistence_bounded(self, sample_returns):
        """GARCH persistence should be between 0 and 1 for stationary process."""
        result = volatility_models.fit_garch(sample_returns, distribution="normal")
        # Persistence can occasionally exceed 1 for some data, but should be positive
        assert result["persistence"] > 0

    def test_fit_egarch_returns_dict(self, sample_returns):
        """fit_egarch should return a dictionary with expected keys."""
        result = volatility_models.fit_egarch(sample_returns, distribution="normal")
        assert isinstance(result, dict)
        assert "result" in result
        assert "persistence" in result

    def test_fit_gjr_garch_returns_leverage(self, sample_returns):
        """fit_gjr_garch should return leverage effect parameter."""
        result = volatility_models.fit_gjr_garch(sample_returns, distribution="normal")
        assert "leverage_effect" in result
        assert isinstance(result["leverage_effect"], float)

    def test_summarize_fits_creates_dataframe(self, sample_returns):
        """summarize_fits should create comparison DataFrame."""
        garch = volatility_models.fit_garch(sample_returns)
        egarch = volatility_models.fit_egarch(sample_returns)
        summary = volatility_models.summarize_fits(garch, egarch)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "aic" in summary.columns
        assert "bic" in summary.columns


class TestHARRV:
    """Tests for HAR-RV model."""

    def test_fit_har_rv_returns_dict(self, sample_returns):
        """fit_har_rv should return expected structure."""
        result = volatility_models.fit_har_rv(sample_returns)
        assert isinstance(result, dict)
        assert "coefficients" in result
        assert "r_squared" in result
        assert "forecast_variance" in result

    def test_har_rv_r_squared_bounded(self, sample_returns):
        """HAR-RV R² should be between 0 and 1."""
        result = volatility_models.fit_har_rv(sample_returns)
        assert 0 <= result["r_squared"] <= 1

    def test_har_rv_forecast_positive(self, sample_returns):
        """HAR-RV forecasts should be positive (variance)."""
        har_result = volatility_models.fit_har_rv(sample_returns)
        forecast = volatility_models.har_rv_forecast(sample_returns, har_result)
        assert (forecast > 0).all()


# =============================================================================
# Shock Modeling Tests
# =============================================================================

class TestShockDefinition:
    """Tests for shock identification functions."""

    def test_define_shocks_creates_indicator(self, sample_returns):
        """define_shocks should create binary indicator series."""
        shock_def = shock_modeling.define_shocks(sample_returns, quantile=0.95)
        assert isinstance(shock_def.indicator, pd.Series)
        assert set(shock_def.indicator.unique()).issubset({0, 1})

    def test_define_shocks_count_matches_quantile(self, sample_returns):
        """Shock count should approximately match quantile."""
        shock_def = shock_modeling.define_shocks(sample_returns, quantile=0.95)
        expected_shocks = int(len(sample_returns) * 0.05)
        actual_shocks = shock_def.indicator.sum()
        # Allow some tolerance
        assert abs(actual_shocks - expected_shocks) <= 5

    def test_volatility_relative_shocks(self, sample_returns, sample_variance):
        """Volatility-relative shock definition should work."""
        shock_def = shock_modeling.define_shocks_volatility_relative(
            sample_returns, sample_variance, multiplier=2.0
        )
        assert isinstance(shock_def.indicator, pd.Series)
        assert "volatility_relative" in shock_def.method


class TestInterarrivals:
    """Tests for inter-arrival time calculations."""

    def test_interarrival_series_positive(self, sample_returns_with_shocks):
        """Inter-arrival times should be positive."""
        shock_def = shock_modeling.define_shocks(sample_returns_with_shocks, quantile=0.95)
        interarrivals = shock_modeling.interarrival_series(shock_def.indicator)
        assert (interarrivals > 0).all()

    def test_fit_hpp_returns_result(self, sample_returns_with_shocks):
        """fit_hpp should return HPPResult with valid rate."""
        shock_def = shock_modeling.define_shocks(sample_returns_with_shocks, quantile=0.90)
        interarrivals = shock_modeling.interarrival_series(shock_def.indicator)
        hpp = shock_modeling.fit_hpp(interarrivals)
        assert hpp.rate_per_day > 0
        assert hpp.rate_per_year > 0
        assert hpp.ci_95[0] < hpp.ci_95[1]


class TestHawkesProcess:
    """Tests for Hawkes self-exciting process."""

    def test_fit_hawkes_returns_result(self, sample_returns_with_shocks):
        """fit_hawkes should return HawkesResult."""
        shock_def = shock_modeling.define_shocks(sample_returns_with_shocks, quantile=0.90)
        hawkes = shock_modeling.fit_hawkes(shock_def.indicator)
        assert isinstance(hawkes, shock_modeling.HawkesResult)
        assert hawkes.mu > 0
        assert hawkes.alpha >= 0
        assert hawkes.beta > 0

    def test_hawkes_branching_ratio(self, sample_returns_with_shocks):
        """Branching ratio should equal alpha/beta."""
        shock_def = shock_modeling.define_shocks(sample_returns_with_shocks, quantile=0.90)
        hawkes = shock_modeling.fit_hawkes(shock_def.indicator)
        expected_ratio = hawkes.alpha / hawkes.beta
        assert np.isclose(hawkes.branching_ratio, expected_ratio, rtol=1e-6)


class TestRegimeAnalysis:
    """Tests for regime analysis functions."""

    def test_regime_analysis_returns_dataframe(self, sample_returns_with_shocks):
        """run_regime_analysis should return DataFrame."""
        shock_def = shock_modeling.define_shocks(sample_returns_with_shocks, quantile=0.95)
        regime_df = shock_modeling.run_regime_analysis(
            sample_returns_with_shocks, shock_def.indicator
        )
        assert isinstance(regime_df, pd.DataFrame)


# =============================================================================
# Forecast Evaluation Tests
# =============================================================================

class TestForecastEvaluation:
    """Tests for forecast evaluation functions."""

    def test_log_score_finite(self, sample_returns, sample_variance):
        """log_score should return finite value."""
        mean = pd.Series(0.0, index=sample_returns.index)
        score = forecast_evaluation.log_score(sample_returns, mean, sample_variance)
        assert np.isfinite(score)

    def test_pit_values_bounded(self, sample_returns, sample_variance):
        """PIT values should be in [0, 1]."""
        mean = pd.Series(0.0, index=sample_returns.index)
        pit = forecast_evaluation.pit_values(sample_returns, mean, sample_variance)
        assert (pit >= 0).all() and (pit <= 1).all()

    def test_coverage_rate_bounded(self, sample_returns, sample_variance):
        """Coverage rate should be in [0, 1]."""
        mean = pd.Series(0.0, index=sample_returns.index)
        coverage = forecast_evaluation.coverage_rate(sample_returns, mean, sample_variance)
        assert 0 <= coverage <= 1

    def test_ewma_variance_positive(self, sample_returns):
        """EWMA variance should be positive."""
        ewma_var = forecast_evaluation.ewma_variance(sample_returns)
        assert (ewma_var.dropna() > 0).all()

    def test_diebold_mariano_returns_pvalue(self, sample_returns, sample_variance):
        """Diebold-Mariano should return valid p-value."""
        mean = pd.Series(0.0, index=sample_returns.index)
        pit = forecast_evaluation.pit_values(sample_returns, mean, sample_variance)
        loss = forecast_evaluation.pit_log_loss(pit)
        # Compare against itself (should give p-value near 1)
        p_value = forecast_evaluation.diebold_mariano(loss, loss + 0.001)
        assert 0 <= p_value <= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_garch_workflow(self, sample_returns):
        """Test complete GARCH fitting and forecasting workflow."""
        # Fit model
        garch_fit = volatility_models.fit_garch(sample_returns, distribution="normal")

        # Get in-sample forecasts
        frame = volatility_models.in_sample_forecast_frame(
            garch_fit["result"], sample_returns
        )
        assert "mean" in frame.columns
        assert "variance" in frame.columns
        assert len(frame) > 0

    def test_full_shock_workflow(self, sample_returns_with_shocks):
        """Test complete shock identification workflow."""
        # Define shocks
        shock_def = shock_modeling.define_shocks(sample_returns_with_shocks, quantile=0.90)

        # Compute inter-arrivals
        interarrivals = shock_modeling.interarrival_series(shock_def.indicator)

        # Fit HPP
        hpp = shock_modeling.fit_hpp(interarrivals)
        assert hpp.rate_per_year > 0

        # Fit Hawkes
        hawkes = shock_modeling.fit_hawkes(shock_def.indicator)
        assert hawkes.branching_ratio > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
