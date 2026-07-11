"""
regimes.py
----------
Calibrate a 7-regime synthetic-market model from real OHLC data (via ohlcutils).

A regime is a band of the *standardised* rolling mean return
    m(t) = mean(r[t-W+1 : t]) / sd(r[t-W+1 : t])
cut into seven labels (descriptive, in-window labelling by default):

    strongly_negative | negative | mildly_negative | neutral |
    mildly_positive | positive | strongly_positive

For each label we estimate:
  - fundamental DistributionSpec + jitter JitterSpec  (via fit.decompose_segments)
  - an overnight-gap DistributionSpec
plus regime dynamics: occupancy, episode durations, and a transition matrix.

Returns are computed from adjusted close (`aclose`). For intraday periodicities
overnight returns are excluded (autocorrelation is estimated within-session only).

No hidden fallbacks: thin labels / infeasible moments raise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import TypedDict

import ohlcutils
from ohlcutils.enums import Periodicity

from .distribution import DistributionSpec
from .jitter import JitterSpec
from .fit import decompose_segments, fit_distribution


class RegimeSpecPair(TypedDict):
    fundamental: DistributionSpec
    jitter: JitterSpec


LABELS = (
    "strongly_negative",
    "negative",
    "mildly_negative",
    "neutral",
    "mildly_positive",
    "positive",
    "strongly_positive",
)

DEFAULT_CUTS = (0.25, 0.75, 1.5)


@dataclass
class RegimeModel:
    """Calibrated regime model. Consumed by generate.simulate()."""
    symbol: str
    periodicity: Periodicity
    start: str
    end: str
    window: int
    cuts: tuple[float, float, float]
    specs: dict[str, RegimeSpecPair]             # label -> {"fundamental":..., "jitter":...}
    gaps: dict[str, DistributionSpec]            # label -> gap spec (empty for DAILY)
    transition: np.ndarray                       # 7x7, row-stochastic (episode->episode)
    occupancy: dict[str, float]                  # label -> fraction of bars
    durations: dict[str, np.ndarray]             # label -> array of episode lengths (bars)
    bars_per_session: float                      # mean labelled bars per session
    pooled_gap_labels: tuple[str, ...] = ()      # labels whose gap fell back to the global pool

    def labels(self) -> tuple[str, ...]:
        return LABELS

    def summary(self) -> pd.DataFrame:
        rows = []
        for lab in LABELS:
            fund = self.specs[lab]["fundamental"]
            jit = self.specs[lab]["jitter"]
            dur = self.durations.get(lab, np.array([]))
            gap = self.gaps.get(lab)
            rows.append({
                "label": lab,
                "occupancy_pct": round(100 * self.occupancy.get(lab, 0.0), 2),
                "n_episodes": int(dur.size),
                "mean_duration": round(float(dur.mean()), 2) if dur.size else 0.0,
                "fund_mean": fund.mean,
                "fund_sd": fund.sd,
                "fund_skew": round(fund.skewness, 3),
                "fund_exkurt": round(fund.kurtosis, 3),
                "jitter_sd": jit.dist.sd,
                "jitter_rho": round(jit.rho, 3),
                "gap_mean": gap.mean if gap else float("nan"),
                "gap_sd": gap.sd if gap else float("nan"),
            })
        return pd.DataFrame(rows).set_index("label")


def _band(m: float, cuts: tuple[float, float, float]) -> str:
    t1, t2, t3 = cuts
    if m < -t3:
        return "strongly_negative"
    if m < -t2:
        return "negative"
    if m < -t1:
        return "mildly_negative"
    if m <= t1:
        return "neutral"
    if m <= t2:
        return "mildly_positive"
    if m <= t3:
        return "positive"
    return "strongly_positive"


def explain_regime(t: float, cuts: tuple[float, float, float]) -> tuple[str, str]:
    """
    Map drift t-stat `t` to a regime label and a human-readable reason string.
    """
    t1, t2, t3 = cuts
    label = _band(t, cuts)
    band_desc = {
        "strongly_negative": f"t < -{t3}",
        "negative": f"-{t3} ≤ t < -{t2}",
        "mildly_negative": f"-{t2} ≤ t < -{t1}",
        "neutral": f"-{t1} ≤ t ≤ {t1}",
        "mildly_positive": f"{t1} < t ≤ {t2}",
        "positive": f"{t2} < t ≤ {t3}",
        "strongly_positive": f"t > {t3}",
    }
    reason = f"t={t:.3f} ({band_desc[label]}) → {label}"
    return label, reason


def daily_returns_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a _load_returns-style frame from daily ohlcutils bars (aclose, optional aopen)."""
    if df.empty or "aclose" not in df.columns:
        raise KeyError("Expected non-empty daily bars with column 'aclose'.")
    out = pd.DataFrame(index=df.index)
    out["aclose"] = df["aclose"].astype(float)
    if "aopen" in df.columns:
        out["aopen"] = df["aopen"].astype(float)
    else:
        out["aopen"] = out["aclose"]
    out["session"] = pd.DatetimeIndex(df.index).normalize()
    logc = pd.Series(np.log(out["aclose"].to_numpy(dtype=np.float64)), index=out.index)
    out["ret"] = logc.diff()
    return out


def _slice_returns_data(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    t0 = pd.Timestamp(start).normalize()
    t1 = pd.Timestamp(end).normalize()
    idx = pd.DatetimeIndex(data.index).tz_localize(None)
    mask = (idx >= t0) & (idx <= t1)
    sliced = data.loc[mask]
    if sliced.empty:
        raise ValueError(f"No return data in preload for {start}..{end}")
    return sliced


def market_regime_frame(
    symbol: str,
    periodicity: Periodicity,
    index: pd.DatetimeIndex,
    window: int,
    cuts: tuple[float, float, float] = DEFAULT_CUTS,
    load_kwargs: dict | None = None,
    preload: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Real-market log returns and regime labels for `index`, using the same
    drift t-stat labelling as calibration.

    Returns a DataFrame aligned to `index` with columns:
        price, actual_log_return, t_stat, actual_regime, actual_regime_reason
    """
    if len(index) == 0:
        raise ValueError("index must not be empty")
    if preload is not None:
        data = preload
    else:
        # Load extra history so the rolling window is full for the first gen tick.
        pad = window * 3 if periodicity == Periodicity.DAILY else window
        start = (index.min() - pd.tseries.offsets.BDay(pad)).strftime("%Y-%m-%d")
        end = index.max().strftime("%Y-%m-%d")
        data = _load_returns(symbol, periodicity, start, end, load_kwargs)
    ret = data["ret"]
    aclose = data["aclose"]
    s = pd.Series(ret.to_numpy(dtype=np.float64), index=data.index)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std(ddof=1)
    t_series = roll_mean / roll_std * np.sqrt(window)

    def _market_key(ts: pd.Timestamp) -> pd.Timestamp | None:
        ts = pd.Timestamp(ts)
        if ts in data.index:
            return ts
        if periodicity == Periodicity.DAILY:
            d = pd.Timestamp(ts).date()
            for ix in data.index:
                if pd.Timestamp(ix).date() == d:
                    return ix
            return None
        # intraday: align naive generated stamps to tz-aware market index
        didx = pd.DatetimeIndex(data.index)
        if didx.tz is not None and ts.tz is None:
            ts = ts.tz_localize(didx.tz)
            if ts in data.index:
                return ts
        return None

    rows = []
    for ts in index:
        key = _market_key(ts)
        if key is None:
            rows.append({
                "price": np.nan,
                "actual_log_return": np.nan,
                "t_stat": np.nan,
                "actual_regime": "",
                "actual_regime_reason": "no market bar for this timestamp",
            })
            continue
        px = float(aclose.loc[key])
        r = float(ret.loc[key]) if np.isfinite(ret.loc[key]) else np.nan
        tv = t_series.loc[key]
        if not np.isfinite(tv) or not np.isfinite(roll_std.loc[key]) or roll_std.loc[key] <= 0:
            rows.append({
                "price": px,
                "actual_log_return": r,
                "t_stat": np.nan,
                "actual_regime": "",
                "actual_regime_reason": f"need {window} prior labelled returns",
            })
        else:
            lab, reason = explain_regime(float(tv), cuts)
            rows.append({
                "price": px,
                "actual_log_return": r,
                "t_stat": float(tv),
                "actual_regime": lab,
                "actual_regime_reason": reason,
            })
    return pd.DataFrame(rows, index=index)


def _load_returns(symbol: str, periodicity: Periodicity, start: str, end: str,
                  load_kwargs: dict | None) -> pd.DataFrame:
    """
    Load OHLC and return a frame with columns:
        aclose, aopen, ret (within-session log return), session (date)
    The first bar of each session has ret = NaN (no within-session predecessor).
    """
    kwargs = dict(start_time=start, end_time=end, src=periodicity)
    if load_kwargs:
        kwargs.update(load_kwargs)
    df = ohlcutils.data.load_symbol(symbol, **kwargs)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} {start}..{end} @ {periodicity}")
    if periodicity == Periodicity.DAILY:
        return daily_returns_frame(df)
    for col in ("aclose", "aopen"):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' missing from loaded data.")
    out = pd.DataFrame(index=df.index)
    out["aclose"] = df["aclose"].astype(float)
    out["aopen"] = df["aopen"].astype(float)
    out["session"] = pd.DatetimeIndex(df.index).normalize()
    logc = pd.Series(np.log(out["aclose"].to_numpy(dtype=np.float64)), index=out.index)
    out["ret"] = logc.groupby(out["session"]).diff()
    return out


def _runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """Return contiguous runs as (start_idx, end_idx_exclusive, label)."""
    runs = []
    n = len(labels)
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append((i, j, labels[i]))
        i = j
    return runs


def calibrate(
    symbol: str,
    periodicity: Periodicity,
    start: str,
    end: str,
    window: int = 21,
    cuts: tuple[float, float, float] = DEFAULT_CUTS,
    min_obs_per_label: int = 200,
    min_gap_obs: int = 30,
    max_lag: int = 6,
    load_kwargs: dict | None = None,
    preload: pd.DataFrame | None = None,
) -> RegimeModel:
    """
    Calibrate a RegimeModel for `symbol` over [start, end] at `periodicity`.

    Parameters
    ----------
    symbol : str           e.g. "INFY_STK___"
    periodicity : Periodicity   DAILY or PERMIN
    start, end : str       "YYYY-MM-DD"
    window : int           rolling window (bars) for the standardised-mean label
    cuts : (t1,t2,t3)      symmetric band edges on standardised mean
    min_obs_per_label : int   raise if a label has fewer return obs
    min_gap_obs : int      raise if a label has fewer overnight gaps
    max_lag : int          lags used for AR(1) jitter estimation

    Returns
    -------
    RegimeModel
    """
    if window < 4:
        raise ValueError(f"window must be >= 4, got {window}")

    if preload is None:
        data = _load_returns(symbol, periodicity, start, end, load_kwargs)
    else:
        data = _slice_returns_data(preload, start, end)
    rets = data["ret"]
    valid = rets.notna()
    r = rets[valid]
    sessions = data["session"][valid].to_numpy()
    rv = r.to_numpy(dtype=np.float64)
    n = rv.size
    if n < window + min_obs_per_label:
        raise ValueError(
            f"Too few returns ({n}) for window={window} + min_obs_per_label={min_obs_per_label}."
        )

    # --- drift t-statistic -> label per bar ---
    # m = mean_W / (sd_W / sqrt(W)) = mean_W/sd_W * sqrt(W): scale-free trend
    # strength (~N(0,1) under a random walk), so the band cuts populate sensibly.
    s = pd.Series(rv)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std(ddof=1)
    m = (roll_mean / roll_std * np.sqrt(window)).to_numpy()
    labelled_mask = np.isfinite(m) & (roll_std.to_numpy() > 0)

    labels = np.array(
        [_band(m[i], cuts) if labelled_mask[i] else "" for i in range(n)],
        dtype=object,
    )

    # --- decomposition segments ---
    # Intraday: split at session boundaries (no cross-overnight lagged pairs).
    # Daily: returns are contiguous across days, so segment by label only.
    if periodicity == Periodicity.DAILY:
        key = labels
    else:
        key = np.array([f"{labels[i]}|{sessions[i]}" for i in range(n)], dtype=object)
    seg_by_label: dict[str, list[np.ndarray]] = {lab: [] for lab in LABELS}
    obs_by_label: dict[str, int] = {lab: 0 for lab in LABELS}
    for a, b, _k in _runs(key):
        lab = labels[a]
        if lab == "":
            continue
        seg = rv[a:b]
        seg_by_label[lab].append(seg)
        obs_by_label[lab] += seg.size

    # --- regime episodes (runs of equal label, across sessions) ---
    lab_only = labels.copy()
    episodes = [(a, b, lab) for (a, b, lab) in _runs(lab_only) if lab != ""]
    durations = {lab: [] for lab in LABELS}
    for a, b, lab in episodes:
        durations[lab].append(b - a)
    durations = {lab: np.array(v, dtype=int) for lab, v in durations.items()}

    # --- transition matrix (episode -> next episode) ---
    idx = {lab: i for i, lab in enumerate(LABELS)}
    T = np.zeros((7, 7), dtype=np.float64)
    ep_labels = [lab for _a, _b, lab in episodes]
    for prev, nxt in zip(ep_labels[:-1], ep_labels[1:]):
        T[idx[prev], idx[nxt]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    transition = np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums > 0)

    # --- occupancy ---
    total_labelled = sum(obs_by_label.values())
    occupancy = {lab: obs_by_label[lab] / total_labelled if total_labelled else 0.0
                 for lab in LABELS}

    # --- per-session open/close + opening label for gaps (intraday only) ---
    if periodicity == Periodicity.DAILY:
        # At daily scale the close-to-close return already embeds the overnight
        # move, so a separate gap process is redundant.
        gaps, pooled_gap_labels = {}, ()
    else:
        gaps, pooled_gap_labels = _calibrate_gaps(
            data, labels, valid.to_numpy(), min_gap_obs
        )

    # --- decompose each label ---
    specs: dict[str, RegimeSpecPair] = {}
    thin = []
    for lab in LABELS:
        if obs_by_label[lab] < min_obs_per_label:
            thin.append(f"{lab}({obs_by_label[lab]})")
            continue
        fundamental, jitter = decompose_segments(seg_by_label[lab], max_lag=max_lag)
        specs[lab] = {"fundamental": fundamental, "jitter": jitter}
    if thin:
        raise ValueError(
            "Insufficient observations for labels (min_obs_per_label="
            f"{min_obs_per_label}): {', '.join(thin)}. "
            "Lower min_obs_per_label, widen the date range, or adjust cuts/window."
        )

    bars_per_session = total_labelled / max(1, len(np.unique(sessions)))

    return RegimeModel(
        symbol=symbol, periodicity=periodicity, start=start, end=end,
        window=window, cuts=cuts, specs=specs, gaps=gaps,
        transition=transition, occupancy=occupancy, durations=durations,
        bars_per_session=float(bars_per_session),
        pooled_gap_labels=tuple(pooled_gap_labels),
    )


def _calibrate_gaps(data: pd.DataFrame, labels: np.ndarray, valid: np.ndarray,
                    min_gap_obs: int) -> tuple[dict[str, DistributionSpec], tuple[str, ...]]:
    """
    Overnight gap g = log(session_open / prev_session_close), bucketed by the
    regime label of the session's opening (first labelled) bar.

    `labels` is aligned to the *valid-return* rows; `valid` maps those back to
    the full frame index.

    Labels with fewer than `min_gap_obs` gaps fall back to a global gap spec
    (pooled over all sessions); the pooled labels are returned for transparency.
    """
    # Map full-frame positions of valid rows to their labels.
    full_labels = np.array([""] * len(data), dtype=object)
    full_labels[np.where(valid)[0]] = labels

    sess = data["session"].to_numpy()
    aopen = data["aopen"].to_numpy(dtype=float)
    aclose = data["aclose"].to_numpy(dtype=float)

    # Session boundaries in the full frame.
    runs = _runs(sess.astype(str))
    gap_by_label: dict[str, list[float]] = {lab: [] for lab in LABELS}
    prev_close = None
    for a, b, _k in runs:
        sess_open = aopen[a]
        # opening label: first labelled bar within the session
        open_label = ""
        for i in range(a, b):
            if full_labels[i] != "":
                open_label = full_labels[i]
                break
        if prev_close is not None and open_label != "" and sess_open > 0 and prev_close > 0:
            gap_by_label[open_label].append(float(np.log(sess_open / prev_close)))
        prev_close = aclose[b - 1]

    all_gaps = np.array(
        [g for lab in LABELS for g in gap_by_label[lab]], dtype=np.float64
    )
    if all_gaps.size < max(min_gap_obs, 4):
        raise ValueError(
            f"Too few overnight gaps overall ({all_gaps.size}) to fit a gap spec."
        )
    global_gap = fit_distribution(all_gaps)

    gaps: dict[str, DistributionSpec] = {}
    pooled = []
    for lab in LABELS:
        g = np.array(gap_by_label[lab], dtype=np.float64)
        if g.size < min_gap_obs:
            gaps[lab] = global_gap
            pooled.append(lab)
        else:
            gaps[lab] = fit_distribution(g)
    return gaps, tuple(pooled)
