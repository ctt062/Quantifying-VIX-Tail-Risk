# The Dynamics of Market Fear: Shock Persistence and Shock Frequency in VIX

This repository operationalizes the research agenda described in the project proposal. It offers reproducible data ingestion, volatility modeling, shock-arrival analysis, and forecast calibration workflows built in Python.

## Project Structure

```
Shock-Persistence-and-Shock-Frequency-in-VIX/
├── data/                      # Cached downloads and intermediate data
│   └── raw/
├── notebooks/                 # Stepwise analysis notebooks
│   ├── 01_data_and_volatility.ipynb
│   ├── 02_shock_arrivals.ipynb
│   └── 03_forecast_evaluation.ipynb
├── src/                       # Reusable project modules
│   ├── config.py
│   ├── data_pipeline.py
│   ├── features.py
│   ├── volatility_models.py
│   ├── shock_modeling.py
│   ├── forecast_evaluation.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## Quick Start

1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the notebooks** in order:
   - `notebooks/01_data_and_volatility.ipynb`: data ingestion, AR-GARCH/EGARCH estimation, half-life diagnostics.
   - `notebooks/02_shock_arrivals.ipynb`: shock labeling, inter-arrival analysis, Poisson GLM (NHPP) fits.
   - `notebooks/03_forecast_evaluation.ipynb`: forecast generation, calibration checks, Diebold–Mariano tests.

   Each notebook caches Yahoo Finance pulls (`data/raw/vix_history.parquet`) for reproducibility. Set `force_download=True` in `data_pipeline.prepare_series` to refresh.

## Methodological Highlights

- **Volatility Dynamics**: AR(1)-mean with both GARCH(1,1) and EGARCH(1,1) variance structures under Normal vs Student-t errors. Persistence (`α+β` or `β`) and half-life diagnostics per proposal.
- **Shock Detection**: Quantile-based thresholds (configurable via `config.SHOCK_QUANTILES`) with optional EVT refinement through Peaks-Over-Threshold and GPD fits.
- **Arrival Modeling**: Homogeneous Poisson (inter-arrival MLE & CI) vs non-homogeneous Poisson via Poisson GLM with monthly seasonality and level covariates.
- **Forecast Evaluation**: ARCH-based one-step forecasts vs EWMA/rolling baselines, predictive log scores, PIT histograms, coverage checks, and Diebold–Mariano comparisons.

## Reproducibility Tips

- Set environment variables for proxies or store Yahoo cookies if necessary; otherwise `yfinance` handles requests anonymously.
- All randomness uses the global `RANDOM_STATE` from `src/config.py` where stochastic components are needed.
- To integrate with CI, convert notebooks to scripts via `jupyter nbconvert --to script notebooks/*.ipynb`.

## Next Steps

- Extend to intraday or alternative implied-volatility indices.
- Add bootstrap confidence intervals for half-life estimates.
- Integrate PIT and backtest plots into a lightweight report (e.g., Quarto or LaTeX).
