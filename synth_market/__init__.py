"""
synth_market
============
Synthetic tick-level market simulator for strategy testing.

Pipeline
--------
1. Define a DistributionSpec for the fundamental return process.
2. Define a JitterSpec (AR(1) overlay) for microstructure noise.
3. Bundle into PriceSeriesConfig and call generate_price_series().
4. Instantiate a Strategy and call .run(df) → BacktestResult.
5. Inspect result.summary() or result.to_dataframe().

Quick start
-----------
>>> from synth_market import (
...     DistributionSpec, JitterSpec, PriceSeriesConfig,
...     generate_price_series, MeanReversionStrategy
... )
>>> fund = DistributionSpec(mean=0.0, sd=0.0003, skewness=-0.5, kurtosis=2.0)
>>> jitter = JitterSpec(
...     dist=DistributionSpec(mean=0.0, sd=0.0001, skewness=0.0, kurtosis=1.0),
...     rho=-0.4,
... )
>>> cfg = PriceSeriesConfig(fundamental=fund, jitter=jitter, n_ticks=3600)
>>> df = generate_price_series(cfg)
>>> result = MeanReversionStrategy(lookback=3).run(df)
>>> print(result.summary())
"""

from .distribution import DistributionSpec, MomentMatchedSampler
from .jitter import JitterSpec, JitterProcess
from .price_series import PriceSeriesConfig, generate_price_series
from .strategy import (
    Strategy,
    BacktestResult,
    MeanReversionStrategy,
    MomentumStrategy,
)
from .fit import (
    estimate_moments,
    fit_distribution,
    decompose_returns,
    decompose_segments,
)
from .regimes import RegimeModel, calibrate, LABELS, market_regime_frame, explain_regime
from .generate import GeneratedSeries, simulate

__all__ = [
    "DistributionSpec",
    "MomentMatchedSampler",
    "JitterSpec",
    "JitterProcess",
    "PriceSeriesConfig",
    "generate_price_series",
    "Strategy",
    "BacktestResult",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "estimate_moments",
    "fit_distribution",
    "decompose_returns",
    "decompose_segments",
    "RegimeModel",
    "calibrate",
    "LABELS",
    "market_regime_frame",
    "explain_regime",
    "GeneratedSeries",
    "simulate",
]