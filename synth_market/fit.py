"""
fit.py
------
Reverse-engineering: estimate DistributionSpec / JitterSpec from observed returns.

Model (per the simulator):
    r(t) = F(t) + J(t)
        F  : i.i.d. fundamental return (Johnson SU moments)
        J  : AR(1) jitter,  J(t) = rho*J(t-1) + eps(t),  E[J]=0

Because F is i.i.d. and J is AR(1), the sum is ARMA(1,1)-like and the
second-order structure is identifiable from autocovariances:

    gamma(k) = sigma_J^2 * rho^k        for k >= 1   (F contributes nothing)
    gamma(0) = sigma_F^2 + sigma_J^2

So:
    rho      from the decay of gamma(k) over lags (sign from gamma(1)),
    sigma_J^2 = gamma(1)/rho,
    sigma_F^2 = gamma(0) - sigma_J^2.

Higher moments: cumulants of independent variables add. Rather than forcing a
Gaussian jitter (which dumps all shape onto the small-variance fundamental and
inflates its kurtosis), we let BOTH components share the same standardised
skewness `s` and excess kurtosis `e`, choosing them so the total cumulants are
preserved exactly:

    s = kappa3_total / (sigma_F^3 + sigma_J^3)
    e = kappa4_total / (sigma_F^4 + sigma_J^4)

so that  s*(sigma_F^3+sigma_J^3) = kappa3_total  and  e*(sigma_F^4+sigma_J^4) =
kappa4_total. The fundamental's excess kurtosis is thereby tempered by a factor
sigma_F^4/(sigma_F^4+sigma_J^4) versus the all-on-fundamental convention, and
the jitter carries genuine (non-Gaussian) shape.

The jitter spec describes the AR(1) *innovation*; its standardised moments are
back-calculated from the desired jitter *marginal* (s, e) via the AR(1) cumulant
relations  kappa_m(J) = kappa_m(eps)/(1-rho^m).

No hidden fallbacks: infeasible / under-identified inputs raise.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from .distribution import DistributionSpec, project_pearson_feasible
from .jitter import JitterSpec

_INNOV_EXKURT_FLOOR = 1e-3          # Johnson SN/SU/SB degenerate-at-(0,0) floor


def estimate_moments(x: np.ndarray) -> dict[str, float]:
    """
    Sample mean, sd and the standardised central moments (skewness, excess
    kurtosis) of a 1-D array, using population central moments.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4:
        raise ValueError(f"Need at least 4 observations, got {x.size}")
    mu = x.mean()
    d = x - mu
    m2 = np.mean(d ** 2)
    m3 = np.mean(d ** 3)
    m4 = np.mean(d ** 4)
    if m2 <= 0:
        raise ValueError("Zero variance; cannot estimate moments.")
    return {
        "mean": float(mu),
        "sd": float(np.sqrt(m2)),
        "skewness": float(m3 / m2 ** 1.5),
        "kurtosis": float(m4 / m2 ** 2 - 3.0),
    }


def fit_distribution(x: np.ndarray) -> DistributionSpec:
    """
    Fit a DistributionSpec to the four sample moments of `x`.

    Used for i.i.d. quantities such as overnight gaps. Raises if the empirical
    moments are infeasible for the Johnson SU family.
    """
    m = estimate_moments(x)
    return DistributionSpec(
        mean=m["mean"], sd=m["sd"], skewness=m["skewness"], kurtosis=m["kurtosis"]
    )


def _pooled_central_moments(segments: Sequence[np.ndarray]) -> tuple[float, float, float, float, int]:
    """Pooled mean and central moments (m2, m3, m4) over all segment values."""
    allvals = np.concatenate([np.asarray(s, dtype=np.float64) for s in segments])
    n = allvals.size
    if n < 4:
        raise ValueError(f"Need at least 4 observations across segments, got {n}")
    mu = allvals.mean()
    d = allvals - mu
    m2 = np.mean(d ** 2)
    m3 = np.mean(d ** 3)
    m4 = np.mean(d ** 4)
    if m2 <= 0:
        raise ValueError("Zero pooled variance; cannot decompose.")
    return float(mu), float(m2), float(m3), float(m4), n


def _segment_autocovariances(
    segments: Sequence[np.ndarray], mu: float, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Autocovariances gamma(0..max_lag) estimated only from *within-segment*
    lagged pairs (never across segment boundaries), normalised by the total
    number of pairs at each lag. Uses the shared pooled mean `mu`.

    Returns (gammas, counts). Lags with no available pairs have gamma=0,
    count=0; the caller decides which lags are usable.
    """
    gammas = np.zeros(max_lag + 1, dtype=np.float64)
    counts = np.zeros(max_lag + 1, dtype=np.float64)
    for seg in segments:
        s = np.asarray(seg, dtype=np.float64) - mu
        m = s.size
        for k in range(0, max_lag + 1):
            if m > k:
                gammas[k] += np.dot(s[: m - k], s[k:])
                counts[k] += (m - k)
    if counts[0] == 0:
        raise ValueError("No observations available to estimate variance.")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, gammas / np.where(counts > 0, counts, 1.0), 0.0)
    return out, counts


def decompose_segments(
    segments: Sequence[np.ndarray],
    max_lag: int = 6,
    max_abs_rho: float = 0.95,
    jitter_var_frac: float = 0.5,
) -> tuple[DistributionSpec, JitterSpec]:
    """
    Decompose contiguous return segments into a fundamental DistributionSpec
    (i.i.d.) and a JitterSpec (AR(1)).

    Parameters
    ----------
    segments : sequence of 1-D arrays
        Each array is a CONTIGUOUS run of returns (e.g. one within-session
        episode). Autocorrelation is estimated within segments only.
    max_lag : int
        Highest lag for which autocovariances are computed (rho uses lags 1-2).
    max_abs_rho : float
        Stationarity clamp for the estimated AR(1) coefficient.
    jitter_var_frac : float
        Fallback jitter variance share when lag-2 is inconsistent with a
        positive-variance AR(1) (MA(1)-type bounce). The fallback reproduces the
        lag-1 autocorrelation exactly and assigns ~this fraction of variance to
        the jitter.

    Returns
    -------
    (fundamental, jitter)
    """
    mu, m2, m3, m4, n = _pooled_central_moments(segments)
    gamma, counts = _segment_autocovariances(segments, mu, max_lag)
    gamma0 = gamma[0]

    if counts[1] == 0 or counts[2] == 0:
        raise RuntimeError(
            "Need lag-1 and lag-2 pairs to identify the AR(1) jitter; segments "
            "are too short. Aggregate to a coarser periodicity or merge episodes."
        )

    # --- AR(1) coefficient via two-lag method of moments ---
    # iid + AR(1) is ARMA(1,1): gamma(k) = rho*gamma(k-1) for k >= 2, so
    #   rho = gamma(2)/gamma(1),   sigma_J^2 = gamma(1)/rho.
    acf1 = gamma[1] / gamma0
    signif = 1.96 / np.sqrt(n)              # ~95% band for white-noise ACF
    floor = (1e-3 ** 2) * gamma0

    if abs(acf1) < signif:
        # No detectable autocorrelation: jitter is unidentifiable from i.i.d.
        # fundamental. Assign negligible jitter, all variance to fundamental.
        rho = 0.0
        sigma_J2 = floor
    else:
        rho_mom = gamma[2] / gamma[1]
        sigma_J2_mom = gamma[1] / rho_mom if np.isfinite(rho_mom) and rho_mom != 0 else -1.0
        if np.isfinite(rho_mom) and abs(rho_mom) < max_abs_rho and sigma_J2_mom > 0:
            # Exact iid+AR(1) identification (lag-2 consistent with AR(1)).
            rho, sigma_J2 = float(rho_mom), float(sigma_J2_mom)
        else:
            # Lag-2 is inconsistent with a positive-variance AR(1) jitter
            # (e.g. MA(1)-type bid-ask bounce: gamma2 wrong sign). Reproduce the
            # dominant lag-1 autocorrelation with a target variance split
            # (sigma_J^2 ~= jitter_var_frac * gamma0); rho follows from gamma1.
            rho = float(np.clip(acf1 / jitter_var_frac, -max_abs_rho, max_abs_rho))
            sigma_J2 = gamma[1] / rho
        sigma_J2 = float(np.clip(sigma_J2, floor, 0.999 * gamma0))

    sigma_F2 = gamma0 - sigma_J2
    if sigma_F2 <= 0:
        raise RuntimeError(
            f"Non-positive fundamental variance (sigma_F^2={sigma_F2:.4g}); "
            "jitter variance estimate exceeds total. Check input segments."
        )

    sigma_F = np.sqrt(sigma_F2)
    sigma_J = np.sqrt(sigma_J2)

    # --- Shape: shared standardised (s, e), preserving total cumulants exactly ---
    kappa3 = m3                              # total 3rd cumulant
    kappa4 = m4 - 3.0 * m2 ** 2              # total 4th cumulant (excess)
    s = kappa3 / (sigma_F ** 3 + sigma_J ** 3)
    e = kappa4 / (sigma_F ** 4 + sigma_J ** 4)
    s, e = project_pearson_feasible(s, e)

    fundamental = DistributionSpec(
        mean=mu, sd=float(sigma_F), skewness=float(s), kurtosis=float(e)
    )

    # Jitter marginal (s, e); back out AR(1) innovation moments. When jitter
    # variance is negligible, shape is unidentifiable — use a near-Gaussian innov.
    if sigma_J2 <= floor * (1.0 + 1e-3):
        skew_eps, exkurt_eps = 0.0, _INNOV_EXKURT_FLOOR
    elif rho == 0.0:
        skew_eps, exkurt_eps = s, e
    else:
        one_m_r2 = 1.0 - rho ** 2
        skew_eps = s * (1.0 - rho ** 3) / one_m_r2 ** 1.5
        exkurt_eps = e * (1.0 - rho ** 4) / one_m_r2 ** 2
        skew_eps, exkurt_eps = project_pearson_feasible(skew_eps, exkurt_eps)

    jitter = JitterSpec(
        dist=DistributionSpec(
            mean=0.0, sd=float(sigma_J),
            skewness=float(skew_eps), kurtosis=float(exkurt_eps),
        ),
        rho=rho,
        mean=0.0,
    )
    return fundamental, jitter


def decompose_returns(
    r: np.ndarray,
    max_lag: int = 6,
    max_abs_rho: float = 0.95,
    jitter_var_frac: float = 0.5,
) -> tuple[DistributionSpec, JitterSpec]:
    """Convenience wrapper for a single contiguous return series."""
    return decompose_segments(
        [np.asarray(r, dtype=np.float64)], max_lag, max_abs_rho, jitter_var_frac,
    )
