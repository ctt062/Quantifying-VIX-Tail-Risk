# The Dynamics of Market Fear: Shock Persistence and Shock Frequency in VIX

This repository provides a comprehensive, reproducible research pipeline for analyzing VIX volatility dynamics, shock arrivals, and forecast evaluation. It implements state-of-the-art volatility models, self-exciting point processes, and rigorous out-of-sample evaluation.

## Key Features

- **Multiple Volatility Models**: GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), and HAR-RV
- **Self-Exciting Shock Modeling**: Hawkes process for capturing shock clustering
- **Two Shock Definitions**: Fixed quantile and volatility-relative (surprise-based)
- **Regime Analysis**: Pre-crisis, COVID, post-COVID comparison
- **Rigorous Evaluation**: Rolling OOS forecasts, PIT diagnostics, Diebold–Mariano tests
- **Comprehensive Test Suite**: 23 unit tests covering all modules

## Project Structure

```
Shock-Persistence-and-Shock-Frequency-in-VIX/
├── runall.py                  # Main pipeline driver with CLI
├── src/                       # Reusable project modules
│   ├── config.py              # Shared paths and tunings
│   ├── data_pipeline.py       # Download and cleaning
│   ├── features.py            # Helper transforms
│   ├── volatility_models.py   # GARCH/EGARCH/GJR-GARCH/HAR-RV
│   ├── shock_modeling.py      # Shock definitions + HPP/NHPP/Hawkes
│   ├── forecast_evaluation.py # Forecasting + scoring
│   └── visualization.py       # Plotting utilities
├── tests/                     # Unit test suite
│   └── test_models.py         # 23 comprehensive tests
├── notebooks/                 # Stepwise analysis notebooks
│   ├── 01_data_and_volatility.ipynb
│   ├── 02_shock_arrivals.ipynb
│   └── 03_forecast_evaluation.ipynb
├── figures/                   # Generated diagnostic plots
├── data/                      # Cached downloads
│   └── raw/
├── zz slides/                 # LaTeX presentation
├── requirements.txt
└── README.md
```

## Quick Start

1. **Environment Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Full Pipeline**
   ```bash
   python runall.py
   ```

3. **Run with Custom Parameters**
   ```bash
   python runall.py --shock-quantile 0.95 --split-date 2022-01-01 --refit-freq M
   ```

4. **Run Unit Tests**
   ```bash
   pytest tests/test_models.py -v
   ```

## Methodology

### Volatility Models

| Model | Description | Key Feature |
|-------|-------------|-------------|
| **GARCH(1,1)** | Symmetric volatility clustering | Baseline model |
| **EGARCH(1,1)** | Log-variance with leverage effect | Asymmetric response via γ |
| **GJR-GARCH(1,1)** | Threshold GARCH | Explicit leverage coefficient |
| **HAR-RV** | Heterogeneous Autoregressive | Daily/weekly/monthly components |

Distribution selection (Normal, Student-t, GED) is automatic via PIT uniformity diagnostics.

### Shock Identification

1. **Quantile-based**: Shock if Δlog(VIX) exceeds the q-th percentile (default 95%)
2. **Volatility-relative**: Shock if |rₜ| > k·σₜ (default k=2), capturing "surprises" relative to expected volatility

### Point Process Models

| Model | Intensity | Use Case |
|-------|-----------|----------|
| **HPP** | λ = constant | Baseline arrival rate |
| **NHPP** | λₜ = f(covariates) via Poisson GLM | Time-varying intensity |
| **Hawkes** | λₜ = μ + Σ α·exp(-β(t-tᵢ)) | Self-exciting clustering |

The **Hawkes process** captures shock clustering: each shock temporarily increases the probability of subsequent shocks, with intensity decaying exponentially.

### Forecast Evaluation

- **Rolling OOS**: Monthly re-estimation, 1-step-ahead forecasts
- **Metrics**: Log-score, 95% coverage, PIT histogram
- **Baselines**: EWMA (λ=0.94), 63-day rolling variance, HAR-RV
- **Statistical Tests**: Diebold–Mariano for significance

### Regime Analysis

Compares shock characteristics across market regimes:
- Pre-Crisis (2010–2019)
- COVID Crisis (2020)
- Post-COVID (2021–2023)
- Recent (2024+)

## Generated Figures

| Figure | Description |
|--------|-------------|
| `vix_series.png` | VIX time series with shock markers |
| `shock_counts.png` | Monthly shock counts |
| `news_impact.png` | GARCH vs EGARCH asymmetry |
| `qq.png` | Q-Q plot of standardized residuals |
| `acf.png` | ACF before/after GARCH filtering |
| `pit.png` | PIT histogram (calibration) |
| `cum_loss_diff.png` | Cumulative log-score differential |
| `interarrival.png` | Inter-arrival time distribution |
| `hawkes_intensity.png` | Hawkes intensity over time |
| `regime_comparison.png` | Shock rates by regime |
| `model_comparison.png` | AIC/BIC across models |

## Key Findings

1. **Volatility Persistence**: EGARCH captures longer memory (half-life ~10 days vs ~4 days for GARCH)
2. **Leverage Effect**: GJR-GARCH γ coefficient confirms asymmetric volatility response
3. **Shock Clustering**: Hawkes branching ratio ~0.23 indicates moderate self-excitation
4. **Regime Dependence**: COVID period shows elevated shock rates (17/year vs 12/year baseline)
5. **Forecast Performance**: EWMA remains a tough benchmark; GARCH provides better calibration

## Reproducibility

- All randomness controlled via `config.RANDOM_STATE`
- Yahoo Finance data cached to `data/raw/vix_history.parquet`
- Use `--force-download` flag to refresh data
- CI-ready: `pytest tests/test_models.py` validates all components

## References

- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *J. Econometrics*.
- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns. *Econometrica*.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the Relation between Expected Value and Volatility. *J. Finance*.
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *J. Financial Econometrics*.
- Hawkes, A. G. (1971). Spectra of Some Self-Exciting and Mutually Exciting Point Processes. *Biometrika*.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *JBES*.
