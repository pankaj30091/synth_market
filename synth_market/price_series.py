"""
price_series.py
---------------
Combines the fundamental moment-matched return series with the AR(1) jitter
layer to produce a synthetic per-second tick price series.

Price construction:
    log_return(t) = fundamental_return(t) + jitter(t)
    price(t)      = price(t-1) * exp(log_return(t))

The fundamental return drives the long-run drift and distributional shape.
The jitter introduces microstructure noise with controllable autocorrelation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from .distribution import DistributionSpec, MomentMatchedSampler
from .jitter import JitterSpec, JitterProcess


@dataclass(frozen=True)
class PriceSeriesConfig:
    """
    Full configuration for synthetic price series generation.

    Parameters
    ----------
    fundamental : DistributionSpec
        Distribution of the underlying log-returns (drift, volatility, shape).
    jitter : JitterSpec
        AR(1) noise overlay specification.
    n_ticks : int
        Number of ticks (seconds) to generate. Must be >= 2.
    start_price : float
        Initial price level. Must be > 0.
    start_time : str | pd.Timestamp
        Starting timestamp for the tick index (ISO 8601 string or Timestamp).
    freq : str
        Pandas frequency string for tick spacing. Default "1s" (one second).
    seed : int | None
        Master random seed. Fundamental and jitter use seed and seed+1.
    """
    fundamental: DistributionSpec
    jitter: JitterSpec
    n_ticks: int
    start_price: float = 100.0
    start_time: str = "2024-01-01 09:00:00"
    freq: str = "1s"
    seed: int | None = 42

    def __post_init__(self):
        if self.n_ticks < 2:
            raise ValueError(f"n_ticks must be >= 2, got {self.n_ticks}")
        if self.start_price <= 0:
            raise ValueError(f"start_price must be > 0, got {self.start_price}")


def generate_price_series(config: PriceSeriesConfig) -> pd.DataFrame:
    """
    Generate a synthetic tick price series.

    Parameters
    ----------
    config : PriceSeriesConfig

    Returns
    -------
    pd.DataFrame with columns:
        timestamp           : DatetimeTZNaive index at `freq` intervals
        fundamental_return  : Raw sampled log-return from fundamental dist
        jitter              : AR(1) noise value
        log_return          : fundamental_return + jitter
        price               : Reconstructed price level (exp of cumsum)

    Raises
    ------
    Any exception from distribution fitting or sampling propagates directly.
    No exceptions are caught or suppressed.
    """
    n = config.n_ticks
    seed = config.seed

    # --- Fundamental returns ---
    fundamental_sampler = MomentMatchedSampler(
        config.fundamental,
        seed=seed,
    )
    fundamental_returns = fundamental_sampler.sample(n)

    # --- Jitter ---
    jitter_process = JitterProcess(
        config.jitter,
        seed=(seed + 1) if seed is not None else None,
    )
    jitter_values = jitter_process.generate(n)

    # --- Combine ---
    log_returns = fundamental_returns + jitter_values

    # Price series: P(0) = start_price, P(t) = P(t-1) * exp(r(t))
    log_prices = np.empty(n, dtype=np.float64)
    log_prices[0] = np.log(config.start_price)
    for t in range(1, n):
        log_prices[t] = log_prices[t - 1] + log_returns[t]
    prices = np.exp(log_prices)
    # t=0 return is undefined (it's the starting tick); set to NaN for clarity
    log_returns[0] = np.nan

    # --- Build DataFrame ---
    index = pd.date_range(
        start=config.start_time,
        periods=n,
        freq=config.freq,
    )
    df = pd.DataFrame(
        {
            "fundamental_return": fundamental_returns,
            "jitter":             jitter_values,
            "log_return":         log_returns,
            "price":              prices,
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df