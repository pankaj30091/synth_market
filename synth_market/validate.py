"""
validate.py
-----------
Runs the full pipeline end-to-end and prints diagnostics.
Validates that:
  1. Realised moments match target moments (within sampling noise)
  2. Jitter autocorrelation matches target rho
  3. Mean reversion strategy profits on negative-rho jitter
  4. Momentum strategy profits on positive-rho jitter
  5. Both strategies underperform on zero-rho (white noise) jitter

Run with: python validate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from synth_market import (
    DistributionSpec,
    JitterSpec,
    PriceSeriesConfig,
    generate_price_series,
    MeanReversionStrategy,
    MomentumStrategy,
    JitterProcess,
    MomentMatchedSampler,
)


SEPARATOR = "=" * 60


def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# 1. Moment matching validation
# ---------------------------------------------------------------------------
section("1. Moment Matching — Fundamental Distribution")

target = DistributionSpec(mean=0.0001, sd=0.0005, skewness=-0.8, kurtosis=3.0)
sampler = MomentMatchedSampler(target, seed=0)
realised = sampler.realised_moments(n=200_000)

print(f"{'Moment':<20} {'Target':>12} {'Realised':>12} {'Error':>10}")
print("-" * 56)
for key, tgt in [("mean", target.mean), ("sd", target.sd),
                  ("skewness", target.skewness), ("ex_kurtosis", target.kurtosis)]:
    r = realised[key]
    err = abs(r - tgt)
    print(f"{key:<20} {tgt:>12.6f} {r:>12.6f} {err:>10.6f}")


# ---------------------------------------------------------------------------
# 2. Jitter autocorrelation validation
# ---------------------------------------------------------------------------
section("2. Jitter Autocorrelation — rho = -0.4")

jitter_spec = JitterSpec(
    dist=DistributionSpec(mean=0.0, sd=0.0002, skewness=0.0, kurtosis=1.0),
    rho=-0.4,
)
jp = JitterProcess(jitter_spec, seed=1)
jitter_series = jp.generate(100_000)
acf = jp.autocorrelation(jitter_series, max_lag=5)

print(f"{'Lag':<8} {'Theoretical AR(1)':>20} {'Realised ACF':>15}")
print("-" * 45)
for lag in range(1, 6):
    theoretical = (-0.4) ** lag
    print(f"{lag:<8} {theoretical:>20.4f} {acf[lag]:>15.4f}")


# ---------------------------------------------------------------------------
# 3. Full pipeline + strategy backtest
# ---------------------------------------------------------------------------
section("3. Price Series Generation — 1 Hour of Ticks")

fund = DistributionSpec(mean=0.0, sd=0.0003, skewness=-0.3, kurtosis=2.0)


def run_scenario(label: str, rho: float, strategy_cls, strategy_kwargs: dict):
    jspec = JitterSpec(
        dist=DistributionSpec(mean=0.0, sd=0.00015, skewness=0.0, kurtosis=1.0),
        rho=rho,
    )
    cfg = PriceSeriesConfig(
        fundamental=fund,
        jitter=jspec,
        n_ticks=3_600,
        seed=42,
    )
    df = generate_price_series(cfg)
    strategy = strategy_cls(**strategy_kwargs)
    result = strategy.run(df)
    s = result.summary()
    print(f"\n  Scenario : {label}")
    print(f"  Strategy : {strategy_cls.__name__}({strategy_kwargs})")
    print(f"  rho      : {rho}")
    for k, v in s.items():
        print(f"    {k:<28} {v}")


run_scenario(
    "Mean-reverting jitter (rho=-0.4) → MeanReversion",
    rho=-0.4,
    strategy_cls=MeanReversionStrategy,
    strategy_kwargs={"lookback": 3, "cost_bps": 0.1},
)

run_scenario(
    "Momentum jitter (rho=+0.4) → Momentum",
    rho=0.4,
    strategy_cls=MomentumStrategy,
    strategy_kwargs={"lookback": 3, "cost_bps": 0.1},
)

run_scenario(
    "White noise (rho=0.0) → MeanReversion (should not profit)",
    rho=0.0,
    strategy_cls=MeanReversionStrategy,
    strategy_kwargs={"lookback": 3, "cost_bps": 0.1},
)

run_scenario(
    "White noise (rho=0.0) → Momentum (should not profit)",
    rho=0.0,
    strategy_cls=MomentumStrategy,
    strategy_kwargs={"lookback": 3, "cost_bps": 0.1},
)


# ---------------------------------------------------------------------------
# 4. Error propagation test (no hidden failures)
# ---------------------------------------------------------------------------
section("4. Error Propagation — Invalid Inputs Raise Immediately")

tests = [
    ("Negative sd",         lambda: DistributionSpec(mean=0, sd=-0.1, skewness=0, kurtosis=0)),
    ("Infeasible moments",  lambda: DistributionSpec(mean=0, sd=0.1, skewness=5.0, kurtosis=0)),
    ("|rho| >= 1",          lambda: JitterSpec(
        dist=DistributionSpec(mean=0, sd=0.001, skewness=0, kurtosis=0), rho=1.0
    )),
    ("n_ticks < 2",         lambda: PriceSeriesConfig(
        fundamental=DistributionSpec(mean=0, sd=0.001, skewness=0, kurtosis=0),
        jitter=JitterSpec(
            dist=DistributionSpec(mean=0, sd=0.0001, skewness=0, kurtosis=0), rho=0.0
        ),
        n_ticks=1,
    )),
]

for name, thunk in tests:
    try:
        thunk()
        print(f"  FAIL (no error raised): {name}")
    except (ValueError, TypeError) as e:
        print(f"  OK  {name}: {type(e).__name__}: {e}")

print(f"\n{SEPARATOR}")
print("  Validation complete.")
print(SEPARATOR)