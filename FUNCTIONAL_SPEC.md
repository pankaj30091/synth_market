# synth_market — Functional Specification

**Audience:** Business and product users  
**Version:** 0.1 (as implemented)  
**Data source:** Historical OHLC via `ohlcutils` (NSE symbols, e.g. `INFY_STK___`)

---

## 1. Purpose

`synth_market` learns how a stock **behaves** in different market conditions from real history, then can:

1. **Summarise** those conditions as seven **trend regimes**, each with its own return distribution and microstructure noise profile.
2. **Generate** synthetic price paths that replay similar statistical behaviour for strategy testing and analysis.
3. **Overlay real market data** on generated timelines so users can compare actual prices, returns, and trend labels side by side.

It is designed for **strategy stress-testing and distributional analysis**, not for forecasting tomorrow’s closing price.

---

## 2. What the module does (two main capabilities)

### 2.1 Calibrate — “Learn the fingerprint of this stock”

**Input:** Symbol, bar frequency (daily or per-minute), calibration date range, labelling settings.

**Output:** A **Regime Model** containing, for each of seven trend regimes:

| Output | Meaning |
|--------|---------|
| **Fundamental distribution** | Typical return level, volatility, skew, fat-tail behaviour in that regime |
| **Jitter (microstructure)** | Short-term noise overlay: size, autocorrelation (bounce vs momentum) |
| **Overnight gap distribution** | Per-minute only: open vs prior close, by regime at session open |
| **Occupancy** | How often each regime occurred in history (%) |
| **Episode length** | How long a regime typically persists (bars) |
| **Transition matrix** | Which regime tends to follow which |

Calibration runs **fresh on every invocation** unless the user saves the model themselves (no built-in cache).

### 2.2 Simulate — “Generate synthetic prices from the model”

**Input:** Calibrated model, generation date range, random seed, optional start price anchor, output bar size (e.g. 15-min, weekly).

**Output:**

- **Synthetic OHLC** at the requested aggregation (built from an internal tick path).
- **Tick-level view** with **real market** close, return, regime label, and explanation for each timestamp in the generation window.

Synthetic generation uses a **semi-Markov regime walk**: stay in a regime for a sampled duration (from historical episode lengths), then transition according to the learned matrix, sampling returns from that regime’s specs.

---

## 3. Core concepts (plain language)

### 3.1 Price = Fundamental + Jitter

Each return is modelled as two independent parts:

- **Fundamental** — the underlying drift and shape (mean, volatility, skew, kurtosis) of returns in a regime.
- **Jitter** — short-term coloured noise (AR(1)): can mean-revert (bid-ask bounce) or show brief momentum depending on autocorrelation.

When **calibrating from history**, the module separates these from observed returns using autocorrelation structure.

### 3.2 Seven trend regimes

Every bar is classified into one of seven **bands** based on **trailing trend strength**, not the single-day return:

| Regime | Interpretation |
|--------|----------------|
| Strongly negative | Strong downtrend over the lookback window |
| Negative | Clear downtrend |
| Mildly negative | Slight downtrend |
| Neutral | No clear direction |
| Mildly positive | Slight uptrend |
| Positive | Clear uptrend |
| Strongly positive | Strong uptrend |

**Labelling rule:** Over the last *W* bars (default 21 daily, 30 per-minute), compute a scale-free trend score *t* (drift t-statistic). Map *t* into bands using cut points (default ±0.25, ±0.75, ±1.5).

**Important:** A large down day can still be labelled **positive** if the prior weeks were in an uptrend. The label describes **recent trend**, not “today’s sign.”

### 3.3 Descriptive vs predictive labelling

Regime labels are **descriptive** (what the trailing window looked like in hindsight). They are **not** a real-time trading signal and are **not** used to predict the next bar during generation.

---

## 4. Return distributions

Every regime is described by one or more **distribution specifications** — a compact statistical fingerprint of how returns behave in that trend bucket. These are not hand-picked; they are **estimated from historical data** during calibration and **replayed** during simulation.

### 4.1 The four-moment summary (`DistributionSpec`)

Each distribution is summarised by four numbers:

| Moment | Business meaning | Typical reading |
|--------|------------------|---------------|
| **Mean** | Average return per bar in this regime | Positive ⇒ upward drift; near zero ⇒ flat |
| **SD (standard deviation)** | Volatility — typical size of moves | Higher ⇒ more risk / larger swings |
| **Skewness** | Asymmetry of the return bell curve | Negative ⇒ more/larger down spikes; positive ⇒ more/larger up spikes |
| **Excess kurtosis** | Tail thickness vs a normal bell curve | 0 ≈ normal; **> 0 ⇒ fat tails** (extreme moves more common than Gaussian) |

Together, these four moments define the **shape and level** of returns. The module does not assume returns are normal; it fits flexible curves that can match skew and fat tails observed in real markets.

**Feasibility:** Not every combination of skew and kurtosis is mathematically possible. If estimated moments are infeasible, calibration **raises an error** rather than silently adjusting (except at generation time for edge cases — see §9).

### 4.2 Three distribution roles per regime

| Role | What it models | When used |
|------|----------------|-----------|
| **Fundamental** | Core return in a trend regime — drift + volatility + shape | Every bar while that regime is active |
| **Jitter** | Short-term microstructure noise layered on top | Every bar; memory carried via AR(1) |
| **Gap** | Overnight jump at session open (open vs prior close) | Per-minute only, at first bar of each day |

**Daily calibration:** Only **fundamental + jitter** are fitted. The daily close-to-close return already includes overnight movement, so a separate gap distribution is not used.

**Per-minute calibration:** Returns within each session exclude overnight; **gaps** are fitted separately and injected at the session open during simulation.

### 4.3 How returns are built (simulation)

For each bar in the synthetic path:

```
log_return = fundamental_sample + jitter_sample
price      = prior_price × exp(log_return)
```

At each new intraday session (per-minute):

```
first_bar_return += overnight_gap_sample   (from gap distribution for active regime)
```

**Fundamental** samples are independent (no memory bar-to-bar). **Jitter** has memory: today's jitter depends on yesterday's jitter (AR(1) process).

### 4.4 Jitter distribution (`JitterSpec`)

Jitter is specified separately from fundamental:

| Parameter | Meaning |
|-----------|---------|
| **Jitter SD** | Size of the microstructure noise layer (stationary volatility) |
| **Rho (ρ)** | Autocorrelation of jitter from one bar to the next |
| **Jitter mean** | Long-run average of jitter (default **0**; kept at zero when calibrating from data) |

**Interpreting rho:**

| ρ value | Market intuition |
|---------|------------------|
| **ρ ≈ 0** | White noise — no short-term pattern |
| **ρ < 0** | Mean reversion / bid-ask bounce — move one way, next bar tends to snap back |
| **ρ > 0** | Short-term momentum — move persists briefly before reverting |

Jitter also carries **shape** (skew, kurtosis) derived from history, not forced to be Gaussian.

### 4.5 Gap distribution (overnight jump)

**Gap** models the overnight price jump — the move from **prior session close** to **today’s open**:

```
gap = log(today_open / yesterday_close)
```

| Aspect | Detail |
|--------|--------|
| **When it applies** | Per-minute calibration and simulation only |
| **When it is skipped** | Daily calibration — the daily return already includes overnight |
| **When it is applied** | Added to the **first bar** of each new trading session (not every bar) |
| **Regime tagging** | Gap is labelled by the **trend regime at session open** (same t-stat rules) |

Each regime can have its own gap distribution (mean, SD, skew, kurtosis), summarised in the calibration table as **gap_mean** and **gap_sd** (full shape is stored internally).

**Per-regime vs pooled:** If a regime has too few overnight observations in history (below `--min-gap-obs`, default 10), its gap spec is **pooled with all sessions** and reported in `pooled_gap_labels`. This avoids fitting noise on thin buckets (e.g. strongly negative opens).

**Business intuition:**

| gap_mean | Typical reading |
|----------|-----------------|
| **> 0** | Opens tend to gap up overnight in this regime |
| **< 0** | Opens tend to gap down overnight in this regime |
| **≈ 0** | Overnight drift is neutral |

| gap_sd | Typical reading |
|--------|-----------------|
| **Higher** | Larger, less predictable overnight jumps (e.g. around volatile regimes) |
| **Lower** | Overnight opens close to prior close |

Gap is **independent** of intraday fundamental and jitter: within-session returns exclude overnight; the gap is re-injected once at the open so daily OHLC from per-minute paths reflects realistic open-vs-close behaviour.

### 4.6 How distributions are estimated from history (calibration)

For each regime bucket, the module pools all historical bars labelled with that regime and:

1. **Separates variance** into fundamental vs jitter using autocorrelation (lag-1 / lag-2 structure).
2. **Splits shape** (skew, kurtosis) between fundamental and jitter so that when the two independent layers are added back together, the **total return distribution matches** what was observed.
3. **Fits gap distributions** (per-minute) from overnight open-vs-close moves, tagged by the regime at session open.

The calibration summary table exposes the results:

| Summary column | Distribution field |
|----------------|-------------------|
| fund_mean | Fundamental mean |
| fund_sd | Fundamental SD |
| fund_skew | Fundamental skewness |
| fund_exkurt | Fundamental excess kurtosis |
| jitter_sd | Jitter stationary SD |
| jitter_rho | Jitter AR(1) coefficient |
| gap_mean, gap_sd | Gap distribution mean and SD |

### 4.7 Distribution families (technical note, simplified)

When **generating** random returns from a `DistributionSpec`, the module picks a mathematical curve that matches the four target moments as closely as possible:

| Family | Used when | Character |
|--------|-----------|-----------|
| **Normal** | Near-Gaussian (skew ≈ 0, kurtosis ≈ 0) | Symmetric, thin tails |
| **Johnson SU** | Fat-tailed / leptokurtic shapes | Unbounded, handles heavy tails |
| **Johnson SB** | Mildly platykurtic / bounded shapes | Covers cases SU cannot reach |

Selection is automatic. If a rare shape cannot be fitted exactly, the module may relax shape slightly (e.g. fall back to normal while preserving mean and SD) and **report failure** on infeasible inputs during calibration.

### 4.8 Worked example — reading one regime row

Example (illustrative daily row):

```
label: mildly_positive
fund_mean: 0.00   fund_sd: 0.01   fund_skew: 0.39   fund_exkurt: 15.7
jitter_sd: 0.00   jitter_rho: -0.11
```

**Reading:**

- In mildly-positive trend periods, average return was ~flat per day (`fund_mean ≈ 0`), with ~1% daily volatility.
- Returns were slightly right-skewed (`fund_skew > 0`) and **very fat-tailed** (`fund_exkurt >> 0`) — occasional large moves.
- Jitter was small but slightly mean-reverting (`ρ = -0.11`) — microstructure bounce at minute level; at daily scale jitter may be negligible (`jitter_sd ≈ 0`).

### 4.9 Manual distribution API (advanced)

Users can bypass calibration and specify distributions directly for controlled experiments:

```python
from synth_market import DistributionSpec, JitterSpec, PriceSeriesConfig, generate_price_series

fund = DistributionSpec(mean=0.0, sd=0.0003, skewness=-0.5, kurtosis=2.0)
jitter = JitterSpec(
    dist=DistributionSpec(mean=0.0, sd=0.0001, skewness=0.0, kurtosis=1.0),
    rho=-0.4,
)
cfg = PriceSeriesConfig(fundamental=fund, jitter=jitter, n_ticks=3600, seed=42)
df = generate_price_series(cfg)
```

This low-level path generates a single-regime synthetic series without the seven-regime model or ohlcutils history.

### 4.10 Reverse-engineering (fit from returns)

Given any contiguous return series, the module can estimate fundamental + jitter specs:

```python
from synth_market import decompose_returns

fundamental, jitter = decompose_returns(returns_array)
```

Used internally during calibration; also available for ad-hoc analysis of real or synthetic data.

---

## 5. User-facing outputs

### 5.1 Calibration summary table

One row per regime. Key columns:

| Column | Business meaning |
|--------|------------------|
| occupancy_pct | Share of history spent in this regime |
| n_episodes / mean_duration | How often it appears and typical run length |
| fund_mean, fund_sd | Average return and volatility in the regime |
| fund_skew, fund_exkurt | Asymmetry and tail thickness |
| jitter_sd, jitter_rho | Microstructure noise size; negative ρ ≈ bounce, positive ρ ≈ brief momentum |
| gap_mean, gap_sd | Overnight gap stats (per-minute only; daily skips gaps) |

### 5.2 Tick path (tail view in `example.py`)

After generation, the module attaches **real market** fields:

| Column | Source | Meaning |
|--------|--------|---------|
| **price** | Real market | Adjusted close (`aclose`) on that timestamp |
| **actual_log_return** | Real market | Log return for that bar |
| **actual_regime** | Real market | Trend regime from trailing *t*-stat |
| **actual_regime_reason** | Real market | e.g. `t=1.363 (0.75 < t ≤ 1.5) → positive` |

Synthetic OHLC is produced separately from an internal simulated path; the tick table is primarily for **comparing reality to the model’s regime definitions** on the same calendar.

### 5.3 Trend score

**Definition:** Over the last *n* bars (default 15, configurable):

```
score = (# positive-family labels) − (# negative-family labels)
```

- Mild/strong positive each count **+1**
- Mild/strong negative each count **−1**
- Neutral counts **0**

Range: **−n** to **+n**. Positive ⇒ net bullish trend labels recently; negative ⇒ net bearish.

### 5.4 Synthetic OHLC

Resampled bars (e.g. 15-min, weekly) from the simulated tick path. For aggregations longer than one day, bar timestamps use the **first day of the period** (e.g. Monday for weekly).

---

## 6. Parameters (business view)

### 6.1 Symbol and frequency

| Parameter | Meaning |
|-----------|---------|
| `--symbol` | Stock identifier in ohlcutils symbology (e.g. `INFY_STK___`, `MARUTI_STK___`) |
| `--periodicity daily` | Calibrate on daily bars; default output weekly OHLC |
| `--periodicity permin` | Calibrate on 1-minute bars; default output 15-min OHLC |

Daily and per-minute calibrations are **independent** — no cross-linking of regimes between frequencies.

### 6.2 Calibration window

| Parameter | Meaning |
|-----------|---------|
| `--cal-start`, `--cal-end` | History used to learn regime statistics |
| `--window` | Lookback bars for trend labelling (larger ⇒ smoother, longer episodes) |
| `--min-obs-per-label` | Minimum bars per regime required; fails if any regime is too thin |
| `--min-gap-obs` | Per-minute only: minimum overnight gaps to fit regime-specific gap; else pooled |

**Guidance:** Use a calibration window that reflects the market era you care about. Old windows may not match current volatility (e.g. a stock at ₹1160 vs model trained on ₹1600+ era).

### 6.3 Generation

| Parameter | Meaning |
|-----------|---------|
| `--gen-start`, `--gen-end` | Synthetic series calendar range |
| `--seed` | Reproducibility; same seed ⇒ same synthetic path |
| `--anchor-close` | Start synthetic path from last real daily close on/before gen-start (else starts at 100) |
| `--output-freq` | OHLC bar size (must be ≥ tick frequency): `15min`, `1D`, `1W`, etc. |
| `--score-n` | Lookback for trend score on real `actual_regime` labels |

---

## 7. How to run (example)

Install from repo root:

```bash
cd /path/to/synth_market
pip install .
```

Run the demo script:

```bash
python example.py -h

# Daily calibration + weekly synthetic OHLC, anchored to real start price
python example.py --periodicity daily --symbol MARUTI_STK___ \
  --cal-start 2021-01-01 --cal-end 2026-05-30 \
  --gen-start 2026-01-01 --gen-end 2026-05-30 \
  --anchor-close

# Per-minute calibration, 15-min OHLC
python example.py --periodicity permin --symbol INFY_STK___ \
  --cal-start 2025-01-01 --cal-end 2026-03-31 \
  --gen-start 2026-04-01 --gen-end 2026-04-30 \
  --anchor-close --score-n 15
```

Programmatic use:

```python
from ohlcutils.enums import Periodicity
from synth_market import calibrate, simulate

model = calibrate("INFY_STK___", Periodicity.DAILY, "2022-04-01", "2026-03-31")
print(model.summary())

gen = simulate(model, "2026-04-01", "2026-12-31", seed=42,
               output_freq="1W", anchor_to_last_close=True)
gen.attach_market_regimes(model)
print(gen.ticks.tail())
print(gen.trend_score(15))
```

---

## 8. Intended use cases

| Use case | Supported? |
|----------|------------|
| Summarise how a stock behaves in up/down/neutral trend buckets | Yes |
| Generate synthetic OHLC for strategy backtests | Yes |
| Test mean-reversion vs momentum under different microstructure (jitter ρ) | Yes (low-level API + reference strategies) |
| Compare real trend labels and trend score on a date range | Yes |
| Forecast exact future price or match spot at gen-end | **No** |
| Replace fundamental research or live trading signals | **No** |

---

## 9. Limitations and caveats

1. **Not a forecaster** — Synthetic paths wander from the start anchor; they do not target today’s market price at the end of the range.
2. **Calibration is historical** — Regime stats reflect the chosen `cal-start`/`cal-end` only.
3. **Regime labels lag** — Based on trailing windows; one bad day does not flip the label if the prior trend was strong.
4. **Thin regimes** — Rare buckets (e.g. strongly negative on illiquid names) may fail calibration or use pooled gap statistics.
5. **Moment decomposition** — Splitting fundamental vs jitter is approximate; some edge cases fall back to simpler noise models.
6. **Daily vs intraday** — Meaningful high/low on daily output requires either per-minute calibration or aggregating to weekly/monthly; one tick per day gives flat OHLC.
7. **No persistence of models** — Each run recalibrates unless the user saves/loads the model externally.
8. **Market calendar** — Missing holidays show as “no market bar”; generation uses business-day approximations unless extended.

---

## 10. Module structure (reference)

| Component | Role |
|-----------|------|
| `regimes.calibrate` | Build RegimeModel from ohlcutils history |
| `regimes.market_regime_frame` | Real prices, returns, labels, reasons for a date index |
| `generate.simulate` | Semi-Markov synthetic path + OHLC |
| `fit.decompose_*` | Split returns into fundamental + jitter |
| `distribution` / `jitter` | Moment-matched sampling and AR(1) noise |
| `strategy` | Optional reference backtest harness (mean-reversion, momentum) |
| `example.py` | CLI demo: calibrate → simulate → print summary |

---

## 11. Glossary

| Term | Definition |
|------|------------|
| **DistributionSpec** | Four-moment summary: mean, SD, skewness, excess kurtosis |
| **Regime** | One of seven trend buckets from trailing t-stat |
| **Episode** | Contiguous run of bars in the same regime |
| **Fundamental** | Regime’s core return distribution (drift + shape); i.i.d. per bar |
| **Jitter** | Short-term AR(1) noise on top of fundamental; carries memory via ρ |
| **Gap** | Overnight jump: log(open / prior close), per-minute sessions |
| **Rho (ρ)** | Jitter autocorrelation: negative = bounce, positive = brief momentum |
| **Excess kurtosis** | Tail fatness vs normal; > 0 = fat tails |
| **Johnson SU / SB** | Mathematical curve families used to match four moments when sampling |
| **t-stat (t)** | Standardised trend strength over the labelling window |
| **Trend score** | Count of positive minus negative regime labels over last *n* bars |

---

*For technical implementation details, see source modules under `synth_market/synth_market/` and `example.py`.*
