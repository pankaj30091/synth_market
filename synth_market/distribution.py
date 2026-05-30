"""
distribution.py
---------------
Parametric distribution specification and moment-matched sampling.

Uses the Johnson SU family, which supports arbitrary (mean, sd, skewness, kurtosis)
and has a closed-form quantile function — no approximation fallbacks.

References:
  Johnson (1949) - Systems of frequency curves
  Wheeler (1980) - Quantiles of the Johnson system
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import johnsonsu, johnsonsb, norm
from scipy.optimize import fsolve, least_squares
from scipy.special import expit

# Gauss-Hermite nodes/weights for E[g(Z)], Z~N(0,1) (used for Johnson SB moments).
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(96)
_GH_W = _GH_WEIGHTS / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class DistributionSpec:
    """
    Four-moment specification for a return distribution.

    Parameters
    ----------
    mean : float
        Expected return per tick (e.g. 0.0 for zero-drift).
    sd : float
        Standard deviation of returns. Must be > 0.
    skewness : float
        Fisher skewness (third standardised moment). 0 = symmetric.
    kurtosis : float
        Excess kurtosis (fourth standardised moment minus 3).
        0 = normal, > 0 = fat-tailed.
    """
    mean: float
    sd: float
    skewness: float
    kurtosis: float

    def __post_init__(self):
        if self.sd <= 0:
            raise ValueError(f"sd must be > 0, got {self.sd}")
        total_kurtosis = self.kurtosis + 3          # Fisher → Pearson
        lower_bound = self.skewness ** 2 + 1        # Pearson feasibility constraint
        if total_kurtosis < lower_bound:
            raise ValueError(
                f"Infeasible moments: Pearson kurtosis ({total_kurtosis:.4f}) must be "
                f">= skewness^2 + 1 ({lower_bound:.4f}). "
                f"Reduce |skewness| or increase kurtosis."
            )


def project_pearson_feasible(
    skewness: float,
    excess_kurtosis: float,
    *,
    johnson_headroom: float = 0.08,
) -> tuple[float, float]:
    """
    Ensure moments satisfy Pearson kurtosis >= skewness^2 + 1 and sit inside a
    range the Johnson SU/SB fitters can match (small margin above the algebraic bound).
    """
    min_exkurt = skewness ** 2 - 2.0 + johnson_headroom
    if excess_kurtosis < min_exkurt:
        excess_kurtosis = min_exkurt
    return float(skewness), float(excess_kurtosis)


def _su_standard_moments(z: float, m: float) -> tuple[float, float, float, float]:
    """
    Exact moments of the *standardised* Johnson SU variable

        Y = sinh((Z - a) / b),   Z ~ N(0, 1)

    expressed through the reparameterisation
        z = 1 / b**2   (= ln w)      with w = exp(1/b**2)
        m = a / b

    Derived analytically from E[sinh(k(Z/b - m))] = -w**(k**2/2) sinh(k m)
    and E[cosh(k(Z/b - m))] = w**(k**2/2) cosh(k m).

    Returns
    -------
    (variance, mean, skewness, excess_kurtosis) of Y.
    """
    w = np.exp(z)
    sqrt_w = np.exp(z / 2.0)

    # Raw moments of Y
    mu1 = -sqrt_w * np.sinh(m)
    mu2 = (w ** 2 * np.cosh(2 * m) - 1.0) / 2.0
    mu3 = (-(w ** 4.5) * np.sinh(3 * m) + 3.0 * sqrt_w * np.sinh(m)) / 4.0
    mu4 = (w ** 8 * np.cosh(4 * m) - 4.0 * w ** 2 * np.cosh(2 * m) + 3.0) / 8.0

    # Central moments
    var = mu2 - mu1 ** 2
    c3 = mu3 - 3 * mu1 * mu2 + 2 * mu1 ** 3
    c4 = mu4 - 4 * mu1 * mu3 + 6 * mu1 ** 2 * mu2 - 3 * mu1 ** 4

    skewness = c3 / var ** 1.5
    excess_kurtosis = c4 / var ** 2 - 3.0
    return var, mu1, skewness, excess_kurtosis


def _fit_johnson_su(spec: DistributionSpec) -> tuple[float, float, float, float]:
    """
    Fit Johnson SU parameters (a, b, loc, scale) to match the four moments
    in `spec` *analytically* (no MLE, no approximation).

    scipy parameterisation: X = loc + scale * sinh((Z - a) / b), Z ~ N(0,1).

    1. Solve the closed-form (skewness, excess_kurtosis) = f(z, m) system for the
       shape pair (z = 1/b**2, m = a/b) via a Newton solver on exact moments.
    2. Recover b, a from (z, m).
    3. scale = sd / std(Y);  loc = mean - scale * E[Y].
    """
    target_skew = spec.skewness
    target_kurt = spec.kurtosis

    def residual(params):
        z, m = params
        if z <= 0:
            # 1/b**2 must be positive; signal infeasibility to the solver.
            return [1e6, 1e6]
        _, _, sk, ek = _su_standard_moments(z, m)
        return [sk - target_skew, ek - target_kurt]

    # Multi-start: z (=1/b^2) grows with target kurtosis; m carries the skew
    # (opposite sign). Larger kurtosis needs a larger initial z to converge.
    m_sign = -0.1 if target_skew >= 0 else 0.1
    z_seeds = (0.05, 0.2, 0.5, np.log1p(max(target_kurt, 0.0)) + 0.3, 1.5)
    z = m = None
    for z0 in z_seeds:
        sol, info, ier, msg = fsolve(residual, x0=[z0, m_sign], full_output=True)
        zc, mc = sol
        if ier == 1 and zc > 0:
            vs, ms, sk, ek = _su_standard_moments(zc, mc)
            if np.isfinite(sk) and np.isfinite(ek) \
                    and abs(sk - target_skew) <= 1e-6 and abs(ek - target_kurt) <= 1e-6:
                z, m = zc, mc
                break
    if z is None:
        raise RuntimeError(
            f"Johnson SU moment-matching did not converge for {spec}"
        )

    var_std, mean_std, _sk, _ek = _su_standard_moments(z, m)
    b = 1.0 / np.sqrt(z)        # delta
    a = m * b                    # gamma
    scale = spec.sd / np.sqrt(var_std)
    loc = spec.mean - scale * mean_std
    return float(a), float(b), float(loc), float(scale)


def _sb_standard_moments(a: float, b: float) -> tuple[float, float, float, float]:
    """
    Moments of the standardised Johnson SB variable on (0,1)

        W = 1 / (1 + exp(-(Z - a) / b)),   Z ~ N(0,1)

    via Gauss-Hermite quadrature. Returns (variance, mean, skewness, excess_kurtosis).
    """
    w = expit((_GH_NODES - a) / b)          # overflow-safe logistic
    m1 = np.sum(_GH_W * w)
    d = w - m1
    m2 = np.sum(_GH_W * d ** 2)
    m3 = np.sum(_GH_W * d ** 3)
    m4 = np.sum(_GH_W * d ** 4)
    if m2 <= 0:
        return m2, m1, np.nan, np.nan
    return m2, m1, m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0


def _fit_johnson_sb(spec: DistributionSpec) -> tuple[float, float, float, float]:
    """
    Fit Johnson SB params (a, b, loc, scale) to match the four moments in `spec`.

    SB covers the platykurtic / bounded region of the (skew, kurtosis) plane that
    Johnson SU cannot reach. Moments have no closed form, so (a, b) are solved
    numerically against quadrature-based moments.
    """
    target_skew = spec.skewness
    target_kurt = spec.kurtosis

    def residual(params):
        a, b = params
        _, _, sk, ek = _sb_standard_moments(a, b)
        if not (np.isfinite(sk) and np.isfinite(ek)):
            return [1e3, 1e3]
        return [sk - target_skew, ek - target_kurt]

    # Bounded multi-start least-squares (robust near the lognormal boundary).
    # SB skewness has the opposite sign to `a`, and |a| grows with |skew|;
    # b spans the platykurtic (small b) to peaked (large b) range.
    sign = -1.0 if target_skew >= 0 else 1.0
    a_seeds = [sign * v for v in (0.2, 0.5, 1.0, 1.8, 2.6, 4.0)]
    b_seeds = (0.4, 0.7, 1.0, 1.5, 2.5)
    a = b = var_std = mean_std = None
    for b0 in b_seeds:
        for a0 in a_seeds:
            res = least_squares(
                residual, x0=[a0, b0],
                bounds=([-20.0, 1e-3], [20.0, 50.0]), xtol=1e-12, ftol=1e-12,
            )
            ac, bc = res.x
            vs, ms, sk, ek = _sb_standard_moments(ac, bc)
            if np.isfinite(sk) and np.isfinite(ek) \
                    and abs(sk - target_skew) <= 1e-4 and abs(ek - target_kurt) <= 1e-4:
                a, b, var_std, mean_std = ac, bc, vs, ms
                break
        if a is not None:
            break
    if a is None:
        raise RuntimeError(f"Johnson SB moment-matching did not converge for {spec}")

    scale = spec.sd / np.sqrt(var_std)
    loc = spec.mean - scale * mean_std
    return float(a), float(b), float(loc), float(scale)


def _fit_johnson(spec: DistributionSpec) -> tuple[str, tuple[float, float, float, float]]:
    """
    Fit a Johnson-system distribution to `spec`, selecting the family:

      - SU (unbounded, leptokurtic) when feasible;
      - SB (bounded, platykurtic) otherwise;
      - SN (normal) when moments are near-Gaussian or Johnson fitters cannot
        converge even after projecting into the Pearson-feasible region.

    Returns (family, params) where family is "sn", "su" or "sb".
    """
    if abs(spec.skewness) < 1e-6 and abs(spec.kurtosis) < 1e-6:
        return "sn", (0.0, 0.0, spec.mean, spec.sd)

    for headroom in (0.0, 0.08, 0.15, 0.25, 0.4):
        sk, ek = project_pearson_feasible(
            spec.skewness, spec.kurtosis, johnson_headroom=headroom
        )
        trial = DistributionSpec(spec.mean, spec.sd, sk, ek)
        for fitter in (_fit_johnson_su, _fit_johnson_sb):
            try:
                family = "su" if fitter is _fit_johnson_su else "sb"
                return family, fitter(trial)
            except RuntimeError:
                continue

    # Johnson cannot represent this (skew, kurtosis) pair; match mean/sd only.
    return "sn", (0.0, 0.0, spec.mean, spec.sd)


class MomentMatchedSampler:
    """
    Draws i.i.d. samples from a Johnson-system distribution fitted to `spec`.

    Uses Johnson SU for leptokurtic targets and Johnson SB for the platykurtic /
    bounded region, selected automatically.

    Parameters
    ----------
    spec : DistributionSpec
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, spec: DistributionSpec, seed: int | None = None):
        self.spec = spec
        self.family, (self._a, self._b, self._loc, self._scale) = _fit_johnson(spec)
        if self.family == "sn":
            self._dist = norm(loc=self._loc, scale=self._scale)
        elif self.family == "su":
            self._dist = johnsonsu(self._a, self._b, loc=self._loc, scale=self._scale)
        else:
            self._dist = johnsonsb(self._a, self._b, loc=self._loc, scale=self._scale)
        self._rng = np.random.default_rng(seed)

    def sample(self, n: int) -> np.ndarray:
        """
        Draw `n` i.i.d. return samples.

        Returns
        -------
        np.ndarray of shape (n,), dtype float64
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        # Use the inverse CDF (PPF) on uniform draws for best moment fidelity
        u = self._rng.uniform(0.0, 1.0, size=n)
        return self._dist.ppf(u)

    def realised_moments(self, n: int = 100_000) -> dict[str, float]:
        """
        Sample `n` points and compute realised moments for validation.
        """
        from scipy.stats import skew, kurtosis as kurt
        samples = self.sample(n)
        return {
            "mean":      float(np.mean(samples)),
            "sd":        float(np.std(samples, ddof=1)),
            "skewness":  float(skew(samples)),
            "ex_kurtosis": float(kurt(samples, fisher=True)),
        }