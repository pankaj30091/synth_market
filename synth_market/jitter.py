"""
jitter.py
---------
Colored noise jitter layer added on top of the fundamental price series.

The jitter process is:
    J(t) = rho * J(t-1) + epsilon(t)

where epsilon(t) ~ MomentMatchedSampler(jitter_spec), scaled so that
the stationary variance of J equals jitter_spec.sd^2.

rho > 0  → positive autocorrelation → short-term momentum
rho < 0  → negative autocorrelation → mean reversion / bid-ask bounce
rho = 0  → white noise

The jitter is ADDITIVE to the fundamental log-return series before
exponentiation, so it perturbs price paths without altering the
long-run drift of the fundamental series.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .distribution import DistributionSpec, MomentMatchedSampler


@dataclass(frozen=True)
class JitterSpec:
    """
    Specification for the AR(1) jitter process.

    Parameters
    ----------
    dist : DistributionSpec
        Moment specification for the innovation epsilon(t).
        Only its sd / skewness / kurtosis are used for the jitter *shape*;
        the sd here controls the *stationary* sd of the jitter process.
        dist.mean is NOT used for the process mean (see `mean` below).
    rho : float
        AR(1) autocorrelation coefficient. Must be in (-1, 1).
        rho > 0  → momentum
        rho < 0  → mean reversion
        rho = 0  → white noise (jitter reduces to i.i.d. noise)
    mean : float
        Desired *stationary* mean of the jitter process. Default 0.0.
        The innovation mean is set internally to mean * (1 - rho) so that
        the realised stationary mean equals this value regardless of rho.
        Keep at 0.0 when reverse-engineering specs from data: the
        fundamental/jitter mean split is otherwise unidentifiable.
    """
    dist: DistributionSpec
    rho: float
    mean: float = 0.0

    def __post_init__(self):
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(
                f"rho must be in the open interval (-1, 1) for stationarity, got {self.rho}"
            )


class JitterProcess:
    """
    Generates a stationary AR(1) jitter series.

    The innovation variance is rescaled so that Var(J) = spec.dist.sd^2
    regardless of rho.  This keeps the jitter's amplitude interpretable
    directly from spec.dist.sd.

    Parameters
    ----------
    spec : JitterSpec
    seed : int | None
    """

    def __init__(self, spec: JitterSpec, seed: int | None = None):
        self.spec = spec
        # Rescale innovation sd so stationary sd matches spec.dist.sd
        # Var(J) = Var(eps) / (1 - rho^2)  =>  sd_eps = sd_J * sqrt(1 - rho^2)
        innovation_sd = spec.dist.sd * np.sqrt(1.0 - spec.rho ** 2)
        # Set innovation mean so stationary mean E[J] = mean_eps/(1-rho) == spec.mean
        innovation_mean = spec.mean * (1.0 - spec.rho)
        innovation_spec = DistributionSpec(
            mean=innovation_mean,
            sd=innovation_sd,
            skewness=spec.dist.skewness,
            kurtosis=spec.dist.kurtosis,
        )
        self._sampler = MomentMatchedSampler(innovation_spec, seed=seed)
        self._rho = spec.rho

    def generate(self, n: int) -> np.ndarray:
        """
        Generate `n` ticks of jitter.

        Returns
        -------
        np.ndarray of shape (n,), dtype float64
            Stationary AR(1) series with the specified moments and autocorrelation.
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

        innovations = self._sampler.sample(n)
        jitter = np.empty(n, dtype=np.float64)

        # Initialise at stationary distribution: J(0) ~ N(0, sd^2)
        # We use the first innovation as the burn-in start
        jitter[0] = innovations[0]
        for t in range(1, n):
            jitter[t] = self._rho * jitter[t - 1] + innovations[t]

        return jitter

    def autocorrelation(self, series: np.ndarray, max_lag: int = 10) -> np.ndarray:
        """
        Compute sample autocorrelations of `series` up to `max_lag`.
        Useful for validating that rho is being reproduced.
        """
        n = len(series)
        mean = series.mean()
        var = ((series - mean) ** 2).mean()
        if var == 0:
            raise ValueError("Series has zero variance; autocorrelation undefined.")
        acf = np.empty(max_lag + 1)
        acf[0] = 1.0
        for lag in range(1, max_lag + 1):
            acf[lag] = ((series[:n - lag] - mean) * (series[lag:] - mean)).mean() / var
        return acf