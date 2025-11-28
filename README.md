# Modelling VIX Dynamics: GARCH vs Compound Poisson

**HKUST IEDA4000E — Statistical Modelling for Financial Engineering**

*Course Project by: Vittorio Prana CHANDREAN, CHONG Tin Tak, CHOI Man Hou*

---

## Objective

This project compares two complementary frameworks for modeling VIX dynamics:

1. **GARCH-family models** (GARCH, EGARCH) — capturing volatility clustering and persistence
2. **Compound Poisson Process (CPP)** — modeling shock timing AND magnitude for risk quantification

We analyze 15+ years of daily VIX data spanning calm and crisis regimes, demonstrating how these approaches enable VaR/CVaR computation for risk management.

## Key Features

- **Volatility Models**: GARCH(1,1) and EGARCH(1,1) with automatic distribution selection
- **Compound Poisson Process**: Models shock arrivals (Poisson) and magnitudes (Pareto) for VaR/CVaR
- **Regime Analysis**: Pre-crisis, COVID, post-COVID, and recent period comparison
- **Out-of-Sample Evaluation**: Train/test split with CPP forecast validation
- **Comprehensive Visualization**: Jump distributions, simulated paths, regime comparisons

## Project Structure

```
Shock-Persistence-and-Shock-Frequency-in-VIX/
├── runall.py                  # Main pipeline driver
├── src/                       # Core modules
│   ├── config.py              # Shared parameters
│   ├── data_pipeline.py       # Data loading and cleaning
│   ├── volatility_models.py   # GARCH/EGARCH implementation
│   ├── shock_modeling.py      # CPP and shock definitions
│   ├── forecast_evaluation.py # OOS evaluation utilities
│   └── visualization.py       # Plotting functions
├── notebooks/                 # Analysis notebooks
│   ├── 01_data_and_volatility.ipynb
│   ├── 02_shock_arrivals.ipynb
│   └── 03_forecast_evaluation.ipynb
├── figures/                   # Generated plots
├── zz slides/                 # LaTeX presentation and CPP report
├── tests/                     # Unit tests
└── requirements.txt
```

## Quick Start

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
python runall.py

# Run tests
pytest tests/test_models.py -v
```

## Methodology

### GARCH Models

**GARCH(1,1):**
$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

- Captures **volatility clustering**: large shocks increase future variance
- **Persistence** governed by α + β (values near 1 imply long memory)

**EGARCH(1,1):**
$$\ln \sigma_t^2 = \omega + \beta \ln \sigma_{t-1}^2 + \alpha(|\epsilon_{t-1}| - \mathbb{E}|\epsilon|) + \gamma \epsilon_{t-1}$$

- Captures **leverage effect**: negative shocks impact volatility more than positive
- Ensures positive variance without parameter constraints

### Compound Poisson Process

$$S(T) = \sum_{i=1}^{N(T)} J_i, \quad N(T) \sim \text{Poisson}(\lambda T)$$

- **N(t)**: Shock count (arrival rate λ)
- **Jᵢ**: Shock magnitude (fitted to Pareto distribution)
- **S(T)**: Cumulative shock impact for VaR/CVaR computation

### Regime Analysis

| Regime | Period | Key Characteristic |
|--------|--------|-------------------|
| Pre-Crisis | 2010–2019 | Baseline volatility |
| COVID | 2020 | 76% higher expected annual impact |
| Post-COVID | 2021–2023 | Recovery period |
| Recent | 2024–2025 | Current conditions |

## Key Results

### Volatility Models

| Model | Distribution | AIC | Persistence | Half-life |
|-------|--------------|-----|-------------|-----------|
| GARCH(1,1) | GED | 27,531 | 0.852 | 4.3 days |
| **EGARCH(1,1)** | GED | **27,395** | 0.934 | **10.2 days** |

- EGARCH provides best fit (lowest AIC)
- Longer memory: shocks decay over ~10 days

### Compound Poisson Process

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| λ | 12.64/year | Shock arrival rate |
| E[J] | 0.211 | Mean jump size (21% log-move) |
| VaR (95%) | 4.24 | 95th percentile annual impact |
| CVaR (95%) | 5.01 | Expected Shortfall |

### CPP Out-of-Sample (2022–2025)

| Metric | Value |
|--------|-------|
| Predicted Shocks | 51.8 |
| Actual Shocks | 63 |
| Forecast Error | -17.8% |
| VaR Exceeded? | No ✓ |

## Generated Figures

| Figure | Description |
|--------|-------------|
| `vix_series.png` | VIX time series with shock markers |
| `news_impact.png` | EGARCH asymmetric response |
| `qq.png` | Q-Q plot (GED fit validation) |
| `jump_distribution.png` | Pareto fit to shock magnitudes |
| `cpp_paths.png` | Monte Carlo simulated paths |
| `cpp_var.png` | VaR/CVaR distribution |
| `cpp_regime.png` | Regime-specific CPP parameters |
| `cpp_forecast.png` | Out-of-sample evaluation |

## References

- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *J. Econometrics*.
- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns. *Econometrica*.
- Cont, R., & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*. Princeton University Press.
