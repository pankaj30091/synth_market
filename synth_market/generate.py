"""
generate.py
-----------
Synthesize OHLC data from a calibrated RegimeModel using a semi-Markov regime
process.

Pipeline
--------
1. Build a trading timeline at the model's (tick) periodicity over [start, end].
2. Walk regimes semi-Markov style: dwell times bootstrapped from the empirical
   per-regime episode durations; next regime from the transition matrix.
3. Per tick, log-return = fundamental(regime) + jitter(regime), where jitter is
   a continuous AR(1) carried across ticks/regimes. At each new intraday session
   an overnight gap (regime-conditional) is injected.
4. Reconstruct the tick price path and resample to the requested OHLC frequency
   (>= tick), so high/low emerge from the intra-bar path.

No hidden fallbacks: invalid output frequency / empty calendar raise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.signal import lfilter
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import (
    Week, MonthEnd, MonthBegin, QuarterEnd, QuarterBegin, YearEnd, YearBegin, Day,
)

import ohlcutils
from ohlcutils.enums import Periodicity

from .distribution import MomentMatchedSampler
from .regimes import RegimeModel, LABELS, market_regime_frame

# Default intraday session (NSE), matching ohlcutils load_symbol defaults.
SESSION_OPEN = "09:15"
SESSION_BARS_DEFAULT = 375          # 09:15..15:29 inclusive at 1-min


# Trend score: +1 per positive-family label, -1 per negative-family, 0 neutral.
REGIME_SCORE = {
    "strongly_negative": -1,
    "negative": -1,
    "mildly_negative": -1,
    "neutral": 0,
    "mildly_positive": 1,
    "positive": 1,
    "strongly_positive": 1,
}


@dataclass
class GeneratedSeries:
    ohlc: pd.DataFrame               # open/high/low/close at output frequency
    ticks: pd.DataFrame              # price/log_return/regime at tick frequency
    regime_path: np.ndarray          # per-tick regime label

    def regime_occupancy(self) -> dict[str, float]:
        vc = pd.Series(self.regime_path).value_counts(normalize=True)
        return {lab: float(vc.get(lab, 0.0)) for lab in LABELS}

    def attach_market_regimes(self, model: RegimeModel) -> None:
        """
        Add real-market columns to `ticks` for the generated timestamps:
        price (aclose), actual_log_return, t_stat, actual_regime,
        actual_regime_reason.
        """
        market = market_regime_frame(
            model.symbol,
            model.periodicity,
            self.ticks.index,
            model.window,
            model.cuts,
        )
        for col in market.columns:
            self.ticks[col] = market[col]

    def trend_score(self, n: int = 15) -> int:
        """
        Net directional count over the last `n` labelled ticks of
        `actual_regime` (real market; call attach_market_regimes first):
        (# positive-family) - (# negative-family).
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if "actual_regime" not in self.ticks.columns:
            raise RuntimeError("Call attach_market_regimes(model) before trend_score().")
        labels = self.ticks["actual_regime"].to_numpy()
        labelled = [lab for lab in labels if lab != ""]
        recent = labelled[-n:]
        return int(sum(REGIME_SCORE[lab] for lab in recent))


def _resample_ohlc(price: pd.Series, output_freq: str) -> pd.DataFrame:
    """
    Resample a tick price series to OHLC, labelling each bar with the FIRST day
    of the period when aggregating beyond a single day:
      - weekly  -> Monday (week start)
      - monthly/quarterly/yearly -> first calendar day
      - multi-day (e.g. "3D")    -> left edge
    Intraday and single-day frequencies keep pandas' default (left-labelled).
    """
    off = to_offset(output_freq)
    rule, kwargs = output_freq, {}
    if isinstance(off, Week):
        rule, kwargs = "W-MON", dict(label="left", closed="left")
    elif isinstance(off, (MonthEnd, MonthBegin)):
        rule = "MS"
    elif isinstance(off, (QuarterEnd, QuarterBegin)):
        rule = "QS"
    elif isinstance(off, (YearEnd, YearBegin)):
        rule = "YS"
    elif isinstance(off, Day) and off.n > 1:
        kwargs = dict(label="left", closed="left")
    return price.resample(rule, **kwargs).ohlc().dropna(how="all")


def _tick_index(periodicity: Periodicity, start: str, end: str,
                bars_per_session: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Build the tick timestamp index and a boolean array marking each session's
    first tick. Uses a business-day calendar (holidays are NOT removed unless a
    custom index is supplied by the caller).
    """
    days = pd.bdate_range(start=start, end=end)
    if len(days) == 0:
        raise ValueError(f"No business days in range {start}..{end}")

    if periodicity == Periodicity.DAILY:
        idx = pd.DatetimeIndex(days)
        session_first = np.ones(len(idx), dtype=bool)  # each day is its own session
        return idx, session_first

    if periodicity == Periodicity.PERMIN:
        stamps = []
        first_flags = []
        for d in days:
            base = pd.Timestamp(f"{d.date()} {SESSION_OPEN}")
            session = pd.date_range(base, periods=bars_per_session, freq="1min")
            stamps.append(session)
            flags = np.zeros(bars_per_session, dtype=bool)
            flags[0] = True
            first_flags.append(flags)
        idx = pd.DatetimeIndex(np.concatenate([s.values for s in stamps]))
        session_first = np.concatenate(first_flags)
        return idx, session_first

    raise ValueError(f"Unsupported periodicity for generation: {periodicity}")


def _regime_path(model: RegimeModel, n: int, rng: np.random.Generator) -> np.ndarray:
    """Semi-Markov regime label per tick."""
    idx = {lab: i for i, lab in enumerate(LABELS)}
    # Initial regime from occupancy.
    occ = np.array([model.occupancy.get(lab, 0.0) for lab in LABELS], dtype=float)
    if occ.sum() <= 0:
        raise ValueError("Model has zero occupancy across all regimes.")
    occ = occ / occ.sum()
    cur = int(rng.choice(7, p=occ))

    path = np.empty(n, dtype=object)
    filled = 0
    while filled < n:
        lab = LABELS[cur]
        durs = model.durations.get(lab, np.array([]))
        if durs.size == 0:
            dwell = 1
        else:
            dwell = int(rng.choice(durs))
        dwell = max(1, min(dwell, n - filled))
        path[filled:filled + dwell] = lab
        filled += dwell

        # Transition to the next regime (episode -> episode).
        row = model.transition[cur].copy()
        if row.sum() <= 0:
            cur = int(rng.choice(7, p=occ))     # dead-end: restart from occupancy
        else:
            cur = int(rng.choice(7, p=row / row.sum()))
    return path


def _build_samplers(model: RegimeModel, seed: int):
    """One persistent sampler per regime for fundamental, jitter innovation, gap."""
    fund, innov, gap = {}, {}, {}
    for i, lab in enumerate(LABELS):
        spec = model.specs[lab]
        f = spec["fundamental"]
        j = spec["jitter"]
        fund[lab] = MomentMatchedSampler(f, seed=seed + 1000 + i)
        # Jitter innovation: sd rescaled so stationary sd == j.dist.sd.
        from .distribution import DistributionSpec
        innov_sd = j.dist.sd * np.sqrt(1.0 - j.rho ** 2)
        innov_spec = DistributionSpec(mean=0.0, sd=innov_sd,
                                      skewness=j.dist.skewness, kurtosis=j.dist.kurtosis)
        innov[lab] = MomentMatchedSampler(innov_spec, seed=seed + 2000 + i)
        if lab in model.gaps:
            gap[lab] = MomentMatchedSampler(model.gaps[lab], seed=seed + 3000 + i)
    return fund, innov, gap


def _last_close_before(symbol: str, when: str) -> float:
    """Most recent adjusted daily close on/before `when` for `symbol`."""
    df = ohlcutils.data.load_symbol(
        symbol, end_time=when, days=30, src=Periodicity.DAILY
    )
    df = df[df.index <= pd.Timestamp(when, tz=df.index.tz)] if df.index.tz else \
        df[df.index <= pd.Timestamp(when)]
    if df.empty:
        raise ValueError(f"No close price available on/before {when} for {symbol}.")
    return float(df["aclose"].iloc[-1])


def simulate(
    model: RegimeModel,
    start: str,
    end: str,
    seed: int = 0,
    output_freq: str | None = None,
    start_price: float = 100.0,
    anchor_to_last_close: bool = False,
    bars_per_session: int | None = None,
) -> GeneratedSeries:
    """
    Generate synthetic OHLC from a calibrated RegimeModel.

    Parameters
    ----------
    model : RegimeModel       calibrated model (defines the tick periodicity)
    start, end : str          "YYYY-MM-DD" range for the synthetic series
    seed : int                master seed (fully reproducible)
    output_freq : str | None  pandas offset for OHLC bars (>= tick). Defaults to
                              the tick frequency (degenerate O=H=L=C bars).
    start_price : float       initial price level (used when anchor_to_last_close
                              is False)
    anchor_to_last_close : bool
                              if True, start from the symbol's last real adjusted
                              daily close on/before `start` (overrides start_price)
    bars_per_session : int    intraday bars per session (defaults to the
                              calibrated mean, rounded; daily ignores this)

    Returns
    -------
    GeneratedSeries
    """
    if anchor_to_last_close:
        start_price = _last_close_before(model.symbol, start)
    if start_price <= 0:
        raise ValueError(f"start_price must be > 0, got {start_price}")

    periodicity = model.periodicity
    if bars_per_session is None:
        bars_per_session = (SESSION_BARS_DEFAULT if periodicity != Periodicity.PERMIN
                            else max(1, round(model.bars_per_session)))
    idx, session_first = _tick_index(periodicity, start, end, bars_per_session)
    n = len(idx)

    rng = np.random.default_rng(seed)
    path = _regime_path(model, n, rng)
    fund_s, innov_s, gap_s = _build_samplers(model, seed)

    # --- per regime-run: fundamental block + AR(1) jitter (state carried) ---
    log_returns = np.empty(n, dtype=np.float64)
    jitter_prev = 0.0
    i = 0
    while i < n:
        lab = path[i]
        j = i + 1
        while j < n and path[j] == lab:
            j += 1
        L = j - i
        rho = model.specs[lab]["jitter"].rho
        f = fund_s[lab].sample(L)
        eps = innov_s[lab].sample(L)
        # AR(1): J_t = rho*J_{t-1} + eps_t, carrying state across runs.
        zi = np.array([rho * jitter_prev])
        J = lfilter([1.0], [1.0, -rho], eps, zi=zi)[0]
        jitter_prev = J[-1]
        log_returns[i:j] = f + J
        i = j

    # --- overnight gaps at intraday session opens (skip the very first tick) ---
    if periodicity != Periodicity.DAILY and model.gaps:
        for t in np.where(session_first)[0]:
            if t == 0:
                continue
            lab = path[t]
            sampler = gap_s.get(lab)
            if sampler is not None:
                log_returns[t] += float(sampler.sample(1)[0])

    # --- price path ---
    log_price = np.log(start_price) + np.cumsum(log_returns)
    price = np.exp(log_price)

    ticks = pd.DataFrame(index=idx)
    ticks.index.name = "timestamp"

    # --- resample to OHLC (synthetic path; ticks.price is real aclose after attach) ---
    if output_freq is None:
        output_freq = "1D" if periodicity == Periodicity.DAILY else "1min"
    ohlc = _resample_ohlc(pd.Series(price, index=idx), output_freq)

    return GeneratedSeries(ohlc=ohlc, ticks=ticks, regime_path=path)
