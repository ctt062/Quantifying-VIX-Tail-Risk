"""AR-GARCH style modeling utilities."""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox


Distribution = Literal["normal", "t"]


def fit_garch(
    returns: pd.Series,
    distribution: Distribution = "normal",
) -> Dict[str, object]:
    """Fit AR(1)-GARCH(1,1) and extract diagnostics."""

    model = arch_model(
        returns * 100,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist=distribution,
    )
    res = model.fit(disp="off", show_warning=False)
    persistence = float(res.params.get("alpha[1]", 0.0) + res.params.get("beta[1]", 0.0))
    half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

    std_resid = res.std_resid
    lb_resid = acorr_ljungbox(std_resid, lags=[12], return_df=True)
    lb_sq = acorr_ljungbox(std_resid**2, lags=[12], return_df=True)

    return {
        "result": res,
        "persistence": persistence,
        "half_life": half_life,
        "lb_pvalue": lb_resid["lb_pvalue"].iloc[0],
        "lb_sq_pvalue": lb_sq["lb_pvalue"].iloc[0],
        "distribution": distribution,
    }


def fit_egarch(
    returns: pd.Series,
    distribution: Distribution = "normal",
) -> Dict[str, object]:
    """Fit AR(1)-EGARCH(1,1) and extract diagnostics."""

    model = arch_model(
        returns * 100,
        mean="AR",
        lags=1,
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist=distribution,
    )
    res = model.fit(disp="off", show_warning=False)
    beta = float(res.params.get("beta[1]", np.nan))
    half_life = np.log(0.5) / np.log(beta) if beta < 1 else np.inf

    std_resid = res.std_resid
    lb_resid = acorr_ljungbox(std_resid, lags=[12], return_df=True)
    lb_sq = acorr_ljungbox(std_resid**2, lags=[12], return_df=True)

    return {
        "result": res,
        "persistence": beta,
        "half_life": half_life,
        "lb_pvalue": lb_resid["lb_pvalue"].iloc[0],
        "lb_sq_pvalue": lb_sq["lb_pvalue"].iloc[0],
        "distribution": distribution,
    }


def summarize_fits(*fits: Dict[str, object]) -> pd.DataFrame:
    """Create a tidy summary table from multiple fit outputs."""

    rows = []
    for item in fits:
        res = item["result"]
        rows.append(
            {
                "model": res.model.volatility.__class__.__name__,
                "distribution": item["distribution"],
                "aic": res.aic,
                "bic": res.bic,
                "persistence": item["persistence"],
                "half_life_days": item["half_life"],
                "lb_pvalue": item["lb_pvalue"],
                "lb_sq_pvalue": item["lb_sq_pvalue"],
            }
        )
    return pd.DataFrame(rows)
