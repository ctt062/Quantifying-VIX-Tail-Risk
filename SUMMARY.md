# Project Verification Summary

_Date generated: 2025-11-27_

## 1. Data Integrity
- **Source / span:** Yahoo Finance VIX closes, business-day frequency, 2010-01-05 to 2025-11-27 (4,148 obs).
- **Pre-processing:** Forward-filled gaps, 0.1% winsorization, log-level and daily log-change features validated via `data_pipeline.prepare_series()`.
- **Check:** No missing values downstream; engineered features align with notebook tables.

## 2. Volatility Modeling Diagnostics

### Model Comparison Table
| Model | Dist | AIC | BIC | Persistence | Half-life (days) |
| --- | --- | --- | --- | --- | --- |
| GARCH(1,1) | GED | 27,531 | 27,569 | 0.852 | 4.3 |
| EGARCH(1,1) | GED | **27,395** | **27,439** | 0.934 | 10.2 |
| GJR-GARCH(1,1) | GED | 27,447 | 27,491 | 0.866 | 4.8 |

### Key Findings
- **Best Model:** EGARCH provides lowest AIC/BIC and captures longer shock memory (~10 days half-life).
- **Leverage Effect:** GJR-GARCH γ = -0.27 confirms asymmetric volatility response.
- **Distribution Selection:** GED chosen automatically via PIT KS-statistic (0.061).
- **Persistence:** All models show persistence < 1, indicating mean reversion is intact.

### HAR-RV Baseline
- **R²:** 0.049 (explains ~5% of variance in squared returns)
- **Coefficients:** β_daily=0.117, β_weekly=0.231, β_monthly=0.042
- **Interpretation:** Weekly component dominates; HAR-RV provides benchmark for comparison.

## 3. Shock Statistics

### Quantile-Based Shocks (95th Percentile)
- **Threshold:** 0.1267
- **Total Shocks:** 208 events over 15 years
- **HPP Rate:** ~9 shocks/year (95% CI: [7.84, 10.35])
- **NHPP Coefficient:** Lagged log(VIX) = -0.16 (lower VIX predicts fewer shocks)

### Volatility-Relative Shocks (|rₜ| > 2σₜ)
- **Average Threshold:** 0.147
- **Total Shocks:** 207 events
- **HPP Rate:** ~9 shocks/year
- **Interpretation:** Captures "surprises" relative to expected volatility.

## 4. Hawkes Self-Exciting Process

| Parameter | Value | Interpretation |
| --- | --- | --- |
| Baseline μ | 0.027 shocks/day | Background intensity |
| Excitation α | 0.043 | Jump in intensity after shock |
| Decay β | 0.183 | Rate of intensity decay |
| Branching Ratio | 0.23 | < 1 ⇒ stationary (clustering decays) |
| Half-life | 3.8 days | Time for excitation to halve |

**Interpretation:** Shocks trigger more shocks but the process is stationary; ~23% of shocks are "triggered" by previous shocks.

## 5. Compound Poisson Process

### Model Overview
Models **both** shock timing (Poisson arrivals) AND magnitude (jump sizes):
$$S(T) = \sum_{i=1}^{N(T)} J_i, \quad N(T) \sim \text{Poisson}(\lambda T)$$

### Jump Distribution Selection
| Distribution | AIC | KS Stat | Selected |
| --- | --- | --- | --- |
| Exponential | - | - | No |
| Gamma | - | - | No |
| Lognormal | - | - | No |
| **Pareto** | Best | 0.061 | **Yes** |
| Weibull | - | - | No |

### Fitted Parameters (Full Sample)
| Parameter | Value | Interpretation |
| --- | --- | --- |
| λ (arrival rate) | 12.64/year | Expected shocks per year |
| E[J] (mean jump) | 0.211 | 21.1% average log-move |
| Std[J] | 0.189 | Jump size volatility |
| E[S] = λ × E[J] | 2.67/year | Expected annual impact |
| VaR (95%) | 4.24 | 95th percentile annual impact |
| CVaR (95%) | 5.01 | Expected Shortfall |

### CPP by Regime
| Regime | λ/Year | E[J] | E[S]/Year | VaR 95% | CVaR 95% |
| --- | --- | --- | --- | --- | --- |
| Pre-Crisis | 12.3 | 0.209 | 2.57 | 4.15 | 4.92 |
| **COVID** | **17.3** | **0.262** | **4.53** | **7.44** | **9.65** |
| Post-COVID | 11.6 | 0.188 | 2.19 | 3.44 | 3.85 |
| Recent | 13.6 | 0.216 | 2.95 | 4.70 | 5.63 |

**Key Finding:** COVID period shows 76% higher expected annual impact (4.53 vs 2.57) due to both higher arrival rate AND larger jumps.

## 6. Regime Analysis

| Regime | Period | Obs | Shocks | Rate/Year | Ann. Vol |
| --- | --- | --- | --- | --- | --- |
| Pre-Crisis | 2010–2019 | 2,606 | 127 | 12.3 | 1.21 |
| COVID Crisis | 2020 | 262 | 18 | **17.3** | **1.36** |
| Post-COVID | 2021–2023 | 781 | 36 | 11.6 | 1.12 |
| Recent | 2024–2025 | 499 | 27 | 13.6 | 1.34 |
| Full Sample | 2010–2025 | 4,148 | 208 | 12.6 | 1.22 |

**Key Finding:** COVID period shows 41% higher shock rate than baseline (17.3 vs 12.3/year).

## 7. Forecast Evaluation (Out-of-Sample)

### Training/Test Split
- **Training:** 3,111 obs through 2021-12-07
- **Test:** 1,037 obs from 2021-12-08 onward
- **Refit:** Monthly rolling re-estimation (48 windows)

### Log-Score Comparison (Higher = Better)
| Model | Log-Score |
| --- | --- |
| GARCH | 1.275 |
| EWMA | **1.376** |
| Rolling Var | 1.276 |
| HAR-RV | 1.271 |

### Calibration Metrics
- **95% Coverage:** 95.0% (at nominal)
- **PIT Mean:** ~0.51 (ideal: 0.50)
- **PIT Std:** ~0.26 (ideal: 0.29)
- **Diebold–Mariano p-value:** <0.001 (GARCH vs EWMA significant)

## 8. Visual Verification
| Figure | Status | Notes |
| --- | --- | --- |
| `vix_series.png` | ✓ | VIX path with shock markers |
| `shock_counts.png` | ✓ | Monthly arrivals; clustering visible |
| `news_impact.png` | ✓ | GARCH/EGARCH asymmetry comparison |
| `qq.png` | ✓ | GED captures tails well |
| `acf.png` | ✓ | Residuals approximately white noise |
| `pit.png` | ✓ | Near-uniform; well calibrated |
| `cum_loss_diff.png` | ✓ | EWMA consistently outperforms |
| `interarrival.png` | ✓ | Exponential fit reasonable |
| `hawkes_intensity.png` | ✓ | Intensity spikes around shocks |
| `regime_comparison.png` | ✓ | COVID clearly elevated |
| `model_comparison.png` | ✓ | EGARCH best by AIC/BIC |
| `jump_distribution.png` | ✓ | Pareto fit to shock magnitudes |
| `cpp_paths.png` | ✓ | Monte Carlo simulation paths |
| `cpp_var.png` | ✓ | VaR/CVaR distribution |
| `cpp_regime.png` | ✓ | Regime-specific CPP comparison |
| `shock_magnitudes.png` | ✓ | Shock magnitudes over time |

## 9. Unit Test Suite
- **Location:** `tests/test_models.py`
- **Tests:** 28 tests across 9 test classes
- **Coverage:** GARCH, HAR-RV, shock definitions, Hawkes, CPP, regime analysis, forecasting
- **Status:** All passing

## 10. Key Conclusions
1. **EGARCH** provides best in-sample fit; captures ~10-day half-life for volatility shocks.
2. **GJR-GARCH** confirms leverage effect (negative γ).
3. **Hawkes process** reveals shock clustering with ~23% branching ratio.
4. **Compound Poisson Process** shows Pareto-distributed jumps with VaR 95% = 4.24/year.
5. **COVID regime** shows 76% higher expected annual impact (E[S]=4.53 vs 2.57).
6. **EWMA beats GARCH** on log-score OOS, but calibration is acceptable for both.

## 10. Recommendations
1. Consider regime-switching GARCH for adaptive modeling.
2. Extend NHPP covariates with macro indicators (credit spreads, VIX term structure).
3. Explore higher-frequency data for improved variance proxies.
4. Deploy real-time monitoring dashboard with Hawkes intensity tracking.
