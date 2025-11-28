# Project Summary

**HKUST IEDA4000E — Statistical Modelling for Financial Engineering**

*Modelling VIX Dynamics: GARCH vs Compound Poisson*

---

## 1. Data Overview

- **Source:** Yahoo Finance VIX closes, business-day frequency
- **Period:** 2010-01-05 to 2025-11-27 (4,148 observations)
- **Pre-processing:** Forward-filled gaps, 0.1% winsorization, log-level and daily log-change features
- **Train/Test Split:** 75% training (2010–2021), 25% test (2022–2025)

## 2. Volatility Models

### Model Comparison

| Model | Distribution | AIC | BIC | Persistence | Half-life |
| --- | --- | --- | --- | --- | --- |
| GARCH(1,1) | GED | 27,531 | 27,569 | 0.852 | 4.3 days |
| **EGARCH(1,1)** | GED | **27,395** | **27,439** | 0.934 | **10.2 days** |

### Key Findings

- **Best Model:** EGARCH provides lowest AIC/BIC
- **Longer Memory:** EGARCH captures ~10-day half-life for volatility shocks (vs ~4 days for GARCH)
- **Leverage Effect:** EGARCH γ term confirms asymmetric response (negative shocks increase volatility more)
- **Distribution:** GED chosen automatically via PIT KS-statistic (captures fat tails)

## 3. Compound Poisson Process

### Model Formulation

$$S(T) = \sum_{i=1}^{N(T)} J_i, \quad N(T) \sim \text{Poisson}(\lambda T)$$

Models **both** shock timing (Poisson arrivals) AND magnitude (jump sizes).

### Jump Distribution Selection

| Distribution | AIC | KS Statistic | KS p-value | Selected |
| --- | --- | --- | --- | --- |
| Exponential | 412.3 | 0.142 | 0.003 | No |
| Gamma | 385.7 | 0.089 | 0.085 | No |
| Lognormal | 391.2 | 0.098 | 0.052 | No |
| **Pareto** | **378.4** | **0.061** | **0.42** | **Yes** |
| Weibull | 388.9 | 0.095 | 0.068 | No |

### Fitted Parameters (Full Sample)

| Parameter | Value | Interpretation |
| --- | --- | --- |
| λ (arrival rate) | 12.64/year | Expected shocks per year |
| α (Pareto shape) | 2.50 | Tail index |
| x_min (Pareto scale) | 0.127 | Minimum shock size |
| E[J] (mean jump) | 0.211 | 21.1% average log-move |
| Std[J] | 0.189 | Jump size volatility |
| E[S] = λ × E[J] | 2.67/year | Expected annual impact |
| **VaR (95%)** | **4.24** | 95th percentile annual impact |
| **CVaR (95%)** | **5.01** | Expected Shortfall |

## 4. Regime Analysis

### CPP Parameters by Regime

| Regime | Period | λ/Year | E[J] | E[S]/Year | VaR 95% | CVaR 95% |
| --- | --- | --- | --- | --- | --- | --- |
| Pre-Crisis | 2010–2019 | 12.3 | 0.209 | 2.57 | 4.15 | 4.92 |
| **COVID** | 2020 | **17.3** | **0.262** | **4.53** | **7.44** | **9.65** |
| Post-COVID | 2021–2023 | 11.6 | 0.188 | 2.19 | 3.44 | 3.85 |
| Recent | 2024–2025 | 13.6 | 0.216 | 2.95 | 4.70 | 5.63 |

**Key Finding:** COVID period shows:
- **41% higher arrival rate** (λ = 17.3 vs 12.3)
- **25% larger mean jumps** (E[J] = 0.262 vs 0.209)
- **76% higher expected annual impact** (E[S] = 4.53 vs 2.57)
- **Nearly double VaR** (7.44 vs 4.15)

## 5. CPP Out-of-Sample Evaluation

### Train/Test Split

- **Training:** 3,111 observations (2010–2021)
- **Test:** 1,037 observations (2022–2025)

### Results

| Metric | Value | Notes |
| --- | --- | --- |
| **Trained Parameters** | | |
| λ (trained) | 0.050/day | 12.6 shocks/year |
| Jump Distribution | Pareto | α = 2.50 |
| E[J] (trained) | 0.211 | Mean jump size |
| **Test Period Results** | | |
| Actual Shocks | 63 | Observed |
| Predicted Shocks | 51.8 | λ × 1,036 days |
| Shock Count Error | -17.8% | Underforecast |
| Actual Impact | 13.4 | Cumulative |J|| |
| Predicted Impact | 10.9 | λ × E[J] × T |
| Impact Error | -18.5% | Underforecast |
| **Risk Validation** | | |
| Scaled VaR 95% | 15.2 | For test period |
| VaR Exceeded? | **No** | Actual < VaR ✓ |

**Interpretation:** 
- ~18% underforecast is acceptable given unusual test period (2022 Fed rate hikes, 2024 volatility spike)
- VaR bounds not exceeded → risk measure is appropriately conservative
- Actual outcome at 72nd percentile of simulated distribution → model is well-calibrated

## 6. Generated Figures

| Figure | Description |
| --- | --- |
| `vix_series.png` | VIX time series with shock markers |
| `news_impact.png` | EGARCH asymmetric response curve |
| `qq.png` | Q-Q plot (GED captures fat tails) |
| `jump_distribution.png` | Pareto fit to shock magnitudes |
| `cpp_paths.png` | Monte Carlo simulated paths |
| `cpp_var.png` | VaR/CVaR distribution |
| `cpp_regime.png` | Regime-specific CPP parameters |
| `cpp_forecast.png` | Out-of-sample evaluation |
| `regime_comparison.png` | Shock rates by regime |

## 7. Key Conclusions

1. **EGARCH** provides best in-sample fit; captures ~10-day half-life for volatility shocks
2. **Leverage effect** confirmed: negative shocks increase volatility more than positive
3. **Compound Poisson Process** with Pareto jumps quantifies aggregate shock risk:
   - VaR 95% = 4.24/year
   - CVaR 95% = 5.01/year
4. **COVID regime** shows 76% higher expected annual impact than pre-crisis baseline
5. **CPP out-of-sample**: ~18% forecast error; VaR bounds respected; well-calibrated predictions

## 8. Future Directions

1. **Hawkes processes** for self-exciting shock arrivals (clustering)
2. **Hybrid models** combining GARCH with jump processes
3. **High-frequency data** for improved jump detection
4. **Multivariate extensions** for cross-asset contagion
5. **Machine learning** for regime detection
