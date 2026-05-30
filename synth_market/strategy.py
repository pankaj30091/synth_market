"""
strategy.py
-----------
Strategy engine for backtesting on synthetic tick data.

Design principles:
  - Strategies are stateless functions wrapped in a class; no hidden state
  - Signal generation and execution are separated cleanly
  - No try/except blocks; all errors surface immediately
  - Positions are +1 (long), -1 (short), 0 (flat)
  - Fills are at the NEXT tick's open (= current tick's close price),
    simulating one-tick execution lag
  - No transaction costs by default; can be passed as bps per trade

Included reference strategies:
  MeanReversionStrategy  — fades short-term moves (exploits rho < 0 jitter)
  MomentumStrategy       — follows short-term moves (exploits rho > 0 jitter)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Signal contract
# ---------------------------------------------------------------------------

Signal = Literal[1, -1, 0]   # long, short, flat


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """
    Abstract base. Subclasses implement `signals(prices)`.

    Parameters
    ----------
    cost_bps : float
        One-way transaction cost in basis points (applied per position change).
    """

    def __init__(self, cost_bps: float = 0.0):
        if cost_bps < 0:
            raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
        self.cost_bps = cost_bps

    @abstractmethod
    def signals(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute integer signal array for the full price series.

        Parameters
        ----------
        prices : np.ndarray of shape (n,)

        Returns
        -------
        np.ndarray of shape (n,) with values in {-1, 0, 1}
        Signal at tick t means: take this position at tick t+1 (lag 1 fill).
        """

    def run(self, df: pd.DataFrame) -> "BacktestResult":
        """
        Execute strategy on a price DataFrame produced by generate_price_series.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'price' column with a DatetimeIndex.

        Returns
        -------
        BacktestResult
        """
        prices = df["price"].to_numpy(dtype=np.float64)
        raw_signals = self.signals(prices)

        if raw_signals.shape != prices.shape:
            raise ValueError(
                f"signals() returned shape {raw_signals.shape}, "
                f"expected {prices.shape}"
            )
        if not np.all(np.isin(raw_signals, [-1, 0, 1])):
            raise ValueError("signals() must return values in {-1, 0, 1} only")

        return _execute(prices, raw_signals, self.cost_bps, df.index)


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

def _execute(
    prices: np.ndarray,
    signals: np.ndarray,
    cost_bps: float,
    index: pd.DatetimeIndex,
) -> "BacktestResult":
    n = len(prices)
    position  = np.zeros(n, dtype=np.float64)   # position held during tick t
    pnl       = np.zeros(n, dtype=np.float64)   # tick-level P&L
    trade_cost = cost_bps / 10_000               # fraction per trade

    for t in range(1, n):
        prev_pos  = position[t - 1]
        new_pos   = float(signals[t - 1])        # signal at t-1 fills at t
        position[t] = new_pos

        # P&L from holding prev_pos through this tick's log-return
        log_ret   = np.log(prices[t] / prices[t - 1])
        gross_pnl = prev_pos * log_ret

        # Transaction cost on position change (one-way)
        cost = abs(new_pos - prev_pos) * trade_cost

        pnl[t] = gross_pnl - cost

    cum_pnl    = np.cumsum(pnl)
    equity     = np.exp(cum_pnl)               # compound growth factor
    drawdown   = _drawdown_series(equity)

    return BacktestResult(
        index=index,
        prices=prices,
        position=position,
        pnl=pnl,
        cum_pnl=cum_pnl,
        equity=equity,
        drawdown=drawdown,
    )


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    index:     pd.DatetimeIndex
    prices:    np.ndarray
    position:  np.ndarray
    pnl:       np.ndarray
    cum_pnl:   np.ndarray
    equity:    np.ndarray
    drawdown:  np.ndarray

    def summary(self) -> dict[str, float]:
        total_return   = float(self.equity[-1] - 1.0)
        n              = len(self.pnl)
        annualised_ret = float((1 + total_return) ** (252 * 23400 / n) - 1)
        daily_pnl      = self.pnl[1:]             # drop t=0
        sharpe         = _sharpe(daily_pnl)
        max_dd         = float(self.drawdown.min())
        n_trades       = int(np.sum(np.diff(self.position) != 0))
        hit_rate       = float(np.mean(daily_pnl > 0)) if len(daily_pnl) > 0 else float("nan")

        return {
            "total_return_pct":    round(total_return * 100, 4),
            "annualised_return":   round(annualised_ret * 100, 4),
            "sharpe_ratio":        round(sharpe, 4),
            "max_drawdown_pct":    round(max_dd * 100, 4),
            "n_trades":            n_trades,
            "hit_rate_pct":        round(hit_rate * 100, 2),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "price":    self.prices,
                "position": self.position,
                "pnl":      self.pnl,
                "cum_pnl":  self.cum_pnl,
                "equity":   self.equity,
                "drawdown": self.drawdown,
            },
            index=self.index,
        )


def _sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return float("nan")
    mu  = np.mean(pnl)
    std = np.std(pnl, ddof=1)
    if std == 0:
        return float("nan")
    # Annualise: assume 1-second ticks, 23400 trading seconds/day, 252 days
    return float(mu / std * np.sqrt(252 * 23_400))


# ---------------------------------------------------------------------------
# Reference strategy: Mean Reversion
# ---------------------------------------------------------------------------

class MeanReversionStrategy(Strategy):
    """
    Fades short-term price moves.

    Signal: if price moved up over `lookback` ticks → short; down → long.
    Designed to exploit negative-rho (mean-reverting) jitter.

    Parameters
    ----------
    lookback : int
        Number of ticks to measure the short-term move. Must be >= 1.
    threshold : float
        Minimum absolute log-return to trigger a signal (filters noise).
    cost_bps : float
    """

    def __init__(self, lookback: int = 5, threshold: float = 0.0, cost_bps: float = 0.0):
        super().__init__(cost_bps)
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        self.lookback  = lookback
        self.threshold = threshold

    def signals(self, prices: np.ndarray) -> np.ndarray:
        n      = len(prices)
        sig    = np.zeros(n, dtype=np.int8)
        lb     = self.lookback
        thresh = self.threshold

        for t in range(lb, n):
            move = np.log(prices[t] / prices[t - lb])
            if move > thresh:
                sig[t] = -1       # fade the up-move → short
            elif move < -thresh:
                sig[t] = 1        # fade the down-move → long
        return sig


# ---------------------------------------------------------------------------
# Reference strategy: Momentum
# ---------------------------------------------------------------------------

class MomentumStrategy(Strategy):
    """
    Follows short-term price moves.

    Signal: if price moved up over `lookback` ticks → long; down → short.
    Designed to exploit positive-rho (momentum) jitter.

    Parameters
    ----------
    lookback : int
    threshold : float
    cost_bps : float
    """

    def __init__(self, lookback: int = 5, threshold: float = 0.0, cost_bps: float = 0.0):
        super().__init__(cost_bps)
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        self.lookback  = lookback
        self.threshold = threshold

    def signals(self, prices: np.ndarray) -> np.ndarray:
        n      = len(prices)
        sig    = np.zeros(n, dtype=np.int8)
        lb     = self.lookback
        thresh = self.threshold

        for t in range(lb, n):
            move = np.log(prices[t] / prices[t - lb])
            if move > thresh:
                sig[t] = 1        # follow the up-move → long
            elif move < -thresh:
                sig[t] = -1       # follow the down-move → short
        return sig