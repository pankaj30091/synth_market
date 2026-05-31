# synth_market — Functional Specification

**Audience:** Business and product users  
**Version:** 0.1 (as implemented)  
**Data source:** Historical OHLC via `ohlcutils` (NSE symbols, e.g. `INFY_STK___`)

---

## Executive Summary

**In one sentence:** `synth_market` learns how a stock behaved in different trend environments and generates synthetic price paths with similar statistical characteristics.

### What it is

- A **market-behaviour simulator**
- A **regime analysis tool**
- A **strategy stress-testing framework**

### What it is not

- A **price predictor**
- A **signal generator**
- A **forecasting model**

### System overview

```
Historical OHLC (ohlcutils)
            |
            v
     Trend labelling (7 bands, trailing t-stat)
            |
            v
       Regime pools (one per label)
            |
            +----> Fundamental fit (i.i.d. return component)
            |
            +----> Jitter fit (AR(1) microstructure)
            |
            +----> Gap fit (overnight jump; per-minute only)
            |
            v
        Regime Model
   (distributions + occupancy + dwell + transitions)
            |
            v
   Semi-Markov simulator (regime_path)
            |
            v
     Synthetic tick path
            |
            v
     Synthetic OHLC (resampled)
```

> **Important — synthetic vs actual regime labels**
>
> - **Synthetic regime** (`regime_path`) **drives the simulated price path** during generation.
> - **Actual regime** (`actual_regime`) is computed from **real historical prices** on the overlay timeline.
> - The two are **independent** and are **never expected to match** on a given date.
> - Comparing them side by side is for **analysis**, not validation that the simulator “got the day right.”

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
| **Fundamental distribution** | Typical return level, volatility, skew, fat-tail behaviour in that regime (see Appendix A) |
| **Jitter (microstructure)** | Short-term noise overlay: size, autocorrelation (bounce vs momentum); split from returns via ACF (Appendix A) |
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

Synthetic generation uses a **semi-Markov regime walk** (see §3.4): stay in a regime for a sampled duration (from historical episode lengths), then transition according to the learned matrix, sampling returns from that regime’s specs.

---

## 3. Core concepts (plain language)

### 3.1 Price = Fundamental + Jitter

> **Terminology — “fundamental” is not company fundamentals**
>
> In this module, **fundamental** means the **non-microstructure component of returns** — the lower-frequency, bar-to-bar i.i.d. part left after separating short-term coloured noise (jitter). It does **not** refer to earnings, valuation, balance-sheet data, or any corporate-fundamental research input.

Each return is modelled as two independent parts:

- **Fundamental** — the underlying drift and shape (mean, volatility, skew, kurtosis) of returns in a regime, with **no bar-to-bar memory**.
- **Jitter** — short-term coloured noise (AR(1)): can mean-revert (bid-ask bounce) or show brief momentum depending on autocorrelation.

When **calibrating from history**, the module separates these from observed returns using the methodology in **Appendix A** (ACF variance split + shared-shape cumulant matching).

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

#### Why seven regimes?

Readers often ask: why 7 and not 3 or 5?

| Design choice | Rationale |
|---------------|-----------|
| **Seven bands** | Enough granularity to distinguish **mild** vs **strong** trends on both sides of neutral, while keeping each bucket populated for typical liquid NSE names |
| **Three negative + neutral + three positive** | Symmetric structure around “no clear direction” |
| **Default cuts (±0.25, ±0.75, ±1.5)** | On a scale-free *t*-stat, these separate noise-like drift from clear and strong trends without requiring hand-tuned symbol-specific thresholds |
| **Configurable** | `window` and `cuts` can be changed; fewer/more bands are possible in principle but are not implemented — seven is the fixed label set |

Fewer buckets (e.g. 3) would merge mild and strong trends and lose behavioural detail. More buckets (e.g. 11) would thin per-regime samples and cause calibration failures on shorter histories or illiquid names. Seven is a practical compromise for **descriptive** trend taxonomy plus **stable moment estimation** per bucket.

### 3.3 Descriptive vs predictive labelling

Regime labels are **descriptive** (what the trailing window looked like in hindsight). They are **not** a real-time trading signal and are **not** used to predict the next bar during generation.

### 3.4 Regime dynamics — semi-Markov process

During **simulation**, the synthetic path does not pick a random regime independently on every bar. Instead it **walks through regimes** using a **semi-Markov** model learned from history:

1. **Start** in a regime drawn from historical **occupancy** (how often each regime appeared).
2. **Stay** in that regime for a **dwell time** — a random number of bars sampled from that regime’s historical **episode lengths**.
3. When the dwell expires, **transition** to the next regime using the **transition matrix**.
4. Repeat until the generation window is filled.

This produces **runs** of the same synthetic regime (episodes), with persistence and hand-offs that resemble the calibrated stock’s history.

#### Episode

An **episode** is a contiguous stretch of bars with the **same trend label** in the calibration history. Example: twelve daily bars labelled “positive” in a row = one positive episode of length 12.

Calibration records, per regime:

| Statistic | Meaning |
|-----------|---------|
| **n_episodes** | How many distinct runs of this regime occurred |
| **mean_duration** | Average episode length in bars |
| **durations** (internal) | Full list of episode lengths — used to bootstrap dwell times in simulation |

#### Transition matrix

A **7×7 matrix** where row *i* is “given we are leaving a regime-*i* episode, what is the probability the **next episode** is regime *j*?”

- Rows correspond to **current** regime (strongly negative … strongly positive).
- Columns correspond to **next** regime.
- Each row sums to 1 (or 0 if that regime never appeared as a hand-off in history).
- Counts are taken **episode → episode** (not bar → bar), so transitions happen only when a regime run ends.

**Business reading:** If “positive” often follows “mildly positive” in history, that pair gets high probability. If “strongly negative” rarely follows “strongly positive”, that cell stays small.

#### Semi-Markov vs plain Markov

| Model | Behaviour |
|-------|-----------|
| **Markov chain** | New regime (or same regime) chosen **every bar**; persistence comes only from self-transitions on the diagonal. |
| **Semi-Markov (this module)** | Regime is **held fixed** for a whole dwell period, then the next regime is chosen. Dwell length is **explicitly** sampled from historical episode lengths, not implied by a fixed transition rate. |

So the module captures both **“how long trends last”** (dwell) and **“what tends to follow what”** (transitions).

#### Simulation algorithm (summary)

```
cur ← sample initial regime from occupancy
while ticks remain:
    dwell ← sample from historical episode lengths for cur (min 1 bar)
    for each bar in dwell:
        return ← fundamental(cur) + jitter(cur)   (+ gap at session open if per-min)
    cur ← sample next regime from transition row for cur
        (if row is empty, restart from occupancy)
```

**Jitter carries memory across regime changes** (one AR(1) state for the whole path). **Fundamental** and **gap** samples are regime-specific at each bar.

#### What the Markov layer does *not* do

- It does **not** label the **real market** columns (`actual_regime`) on the tick overlay — those always come from trailing *t*-stats on actual prices.
- It does **not** forecast which regime the stock will be in on a given future date.
- It does **not** guarantee synthetic occupancy matches historical occupancy exactly (stochastic paths will vary by seed).

Access in code: `model.transition` (7×7 array), `model.durations`, `model.occupancy`; synthetic path labels in `GeneratedSeries.regime_path`.

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

**Statistical primer:** Definitions of log returns, moments, ACF, AR(1), cumulants, and related ideas are in **Appendix B** (optional reading for quant/review audiences).

**Feasibility:** Not every combination of skew and kurtosis is mathematically possible. If estimated moments are infeasible, calibration **raises an error** rather than silently adjusting (except at generation time for edge cases — see §9).

### 4.2 Three distribution roles per regime

| Role | What it models | When used |
|------|----------------|-----------|
| **Fundamental** | Core non-microstructure return in a trend regime — drift + volatility + shape (not company fundamentals) | Every bar while that regime is active |
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

> **Calibration methodology:** How returns are pooled, decomposed into fundamental + jitter, and fitted per regime is documented in **Appendix A** (for statisticians and implementers). Business readers can stop at §4.1–§4.5.

### 4.6 Distribution families (technical note, simplified)

When **generating** random returns from a `DistributionSpec`, the module picks a mathematical curve that matches the four target moments as closely as possible:

| Family | Used when | Character |
|--------|-----------|-----------|
| **Normal** | Near-Gaussian (skew ≈ 0, kurtosis ≈ 0) | Symmetric, thin tails |
| **Johnson SU** | Fat-tailed / leptokurtic shapes | Unbounded, handles heavy tails |
| **Johnson SB** | Mildly platykurtic / bounded shapes | Covers cases SU cannot reach |

Selection is automatic. If a rare shape cannot be fitted exactly, the module may relax shape slightly (e.g. fall back to normal while preserving mean and SD) and **report failure** on infeasible inputs during calibration.

### 4.7 Worked example — reading one regime row

```
label: mildly_positive
fund_mean: 0.00   fund_sd: 0.01   fund_skew: 0.39   fund_exkurt: 15.7
jitter_sd: 0.00   jitter_rho: -0.11
```

**Reading:**

- In mildly-positive trend periods, average return was ~flat per day (`fund_mean ≈ 0`), with ~1% daily volatility.
- Returns were slightly right-skewed (`fund_skew > 0`) and **very fat-tailed** (`fund_exkurt >> 0`) — occasional large moves.
- Jitter was small but slightly mean-reverting (`ρ = -0.11`) — microstructure bounce at minute level; at daily scale jitter may be negligible (`jitter_sd ≈ 0`).

### 4.8 Manual distribution API (advanced)

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

### 4.9 Reverse-engineering (fit from returns)

Given any contiguous return series, the module can estimate fundamental + jitter specs:

```python
from synth_market import decompose_returns

fundamental, jitter = decompose_returns(returns_array)
```

Used internally during calibration (see **Appendix A**); also available for ad-hoc analysis of real or synthetic data.

**Statistical decomposition detail** (pooling, ACF split, shape matching) is in **Appendix A** for readers who need the full methodology.

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

The **transition matrix** is stored on the model (`model.transition`, 7×7) but not printed in the summary table; see §3.4.

Synthetic run occupancy is available via `GeneratedSeries.regime_occupancy()` (from `regime_path`).

**Typical calibration output** (illustrative daily run on a liquid NSE name; values rounded):

| regime | occupancy | mean dur | fund mean | fund sd |
|--------|-----------|----------|-----------|---------|
| strongly_negative | 4.2% | 8.1 | −0.0032 | 0.028 |
| negative | 12.4% | 10.3 | −0.0014 | 0.021 |
| mildly_negative | 18.2% | 7.8 | −0.0006 | 0.017 |
| neutral | 31.1% | 6.5 | 0.0001 | 0.015 |
| mildly_positive | 17.6% | 8.9 | 0.0009 | 0.016 |
| positive | 28.7% | 11.2 | 0.0018 | 0.018 |
| strongly_positive | 5.8% | 9.7 | 0.0036 | 0.025 |

Occupancy sums to 100%. Strong buckets are rarer but more volatile; neutral dominates time but has the lowest `fund_sd`. Exact numbers vary by symbol, window, and calibration range — run `model.summary()` for your case.

### 5.2 Tick path (tail view in `example.py`)

> **Important — synthetic vs actual regime labels**
>
> | Label source | Column | Role |
> |--------------|--------|------|
> | **Synthetic** | `regime_path` (internal) | Drives simulated returns during generation |
> | **Actual (real market)** | `actual_regime` | Computed from real prices on the overlay timeline |
>
> These are **independent**. They are **not expected to match** on any given date. The tick overlay lets you compare real market behaviour to the model’s **definitions**, not to verify that simulation “predicted” the day’s regime.

After generation, the module attaches **real market** fields:

| Column | Source | Meaning |
|--------|--------|---------|
| **price** | Real market | Adjusted close (`aclose`) on that timestamp |
| **actual_log_return** | Real market | Log return for that bar |
| **actual_regime** | Real market | Trend regime from trailing *t*-stat |
| **actual_regime_reason** | Real market | e.g. `t=1.363 (0.75 < t ≤ 1.5) → positive` |

Synthetic OHLC is produced separately from an internal simulated path; the tick table is primarily for **comparing reality to the model’s regime definitions** on the same calendar.

### 5.3 Regime Balance Score (formerly “trend score”)

**Preferred name:** **Regime Balance Score** — a rolling count of historical **actual** regime labels, not a predictive trading signal.

**API name:** `GeneratedSeries.trend_score(n)` (unchanged in code).

**Definition:** Over the last *n* bars (default 15, configurable via `--score-n`):

```
score = (# positive-family labels) − (# negative-family labels)
```

- Mild/strong positive each count **+1**
- Mild/strong negative each count **−1**
- Neutral counts **0**

Range: **−n** to **+n**. Positive ⇒ more bullish labels recently in **real** history; negative ⇒ more bearish. Requires `attach_market_regimes(model)` first (uses `actual_regime`, not synthetic `regime_path`).

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
| `--score-n` | Lookback for Regime Balance Score on real `actual_regime` labels |

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
| Compare real regime labels and Regime Balance Score on a date range | Yes |
| Forecast exact future price or match spot at gen-end | **No** |
| Replace company-fundamental research or live trading signals | **No** |

---

## 9. Comparison with simpler models

| Model | Regimes | Persistence | Fat tails | Microstructure |
|-------|---------|-------------|-----------|----------------|
| Gaussian random walk | No | No | No | No |
| Return bootstrap | No | Limited (block bootstrap only) | Yes | No |
| Markov regime switching | Yes | Partial (per-bar transitions) | Depends | No |
| **synth_market** | Yes (7) | Yes (semi-Markov dwell) | Yes | Yes (jitter AR(1)) |

**Why not just bootstrap returns?** Bootstrap replays historical draws but does not separate regimes, episode persistence, or microstructure bounce/momentum. It also mixes unlike market environments unless manually segmented.

**Why not a Gaussian random walk?** Real returns show fat tails, skew, and short-term autocorrelation — especially at minute frequency. A Gaussian RW misses all three.

---

## 10. Validation

The module does not ship an automated validation report, but users should judge calibration and simulation quality with checks like these:

| Check | What to compare | Pass intuition |
|-------|-----------------|----------------|
| **Occupancy** | `model.summary()` occupancy vs `gen.regime_occupancy()` | Same order of magnitude; exact match not expected (stochastic paths) |
| **Episode durations** | Mean/median dwell in history vs synthetic `regime_path` runs | Similar persistence, not identical |
| **Volatility** | Pooled or per-regime SD in calibration vs synthetic returns | Within reasonable band |
| **ACF** | Lag-1 (and lag-2) autocorrelation, real vs synthetic | Jitter ρ should reproduce short-term structure when per-min |
| **Skew / kurtosis** | Pooled return moments by regime | Fat tails and asymmetry preserved approximately |
| **Tail quantiles** | 1%, 5%, 95%, 99% return quantiles, real vs synthetic | Tails should not collapse to Gaussian |

Programmatic hooks: `model.summary()`, `gen.regime_occupancy()`, `decompose_returns()` on held-out return slices, and comparing resampled synthetic OHLC to real OHLC over the same calendar (visual and moment checks).

---

## 11. Computational expectations

Rough runtimes on a typical dev machine with local ohlcutils data (order-of-magnitude; I/O and symbol history vary):

| Task | Typical range |
|------|----------------|
| Daily calibration (~5 years, liquid NSE name) | **< 2 s** |
| Per-minute calibration (~1 year) | **1–5 s** |
| Daily simulation (~1 year, weekly OHLC) | **< 1 s** |
| Per-minute simulation (~2 months, 15-min OHLC) | **< 1 s** |

Per-minute calibration over **multiple years** can take longer (large bar counts). Generation scales with tick count (bars × sessions). No GPU or distributed compute is used.

---

## 12. Known failure modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **Thin regimes** | Too few bars in a label vs `--min-obs-per-label` | Widen `cal-start`/`cal-end`, relax cuts, reduce `window`, or use a more liquid symbol |
| **Illiquid / sparse names** | Extreme moments, empty buckets | Prefer liquid NSE symbols; shorten history or coarsen periodicity |
| **Insufficient episode lengths** | Segments too short for lag-2 ACF | Longer history; daily instead of per-min; merge calibration window |
| **Infeasible skew/kurtosis** | Sample moments outside Pearson/Johnson feasible region | More data; different symbol; check for bad ticks in source |
| **Per-minute data gaps** | Missing minutes in ohlcutils feed | Clean source data; avoid broken sessions in calibration window |
| **Structural regime shifts** | Calibration era ≠ generation era (volatility, price level) | Recalibrate on recent history aligned with test period |
| **Corporate actions** | Bad adjustments in source `aclose` | Verify ohlcutils adjustment quality before trusting moments |
| **Pooled gap labels** | Too few overnight opens per extreme regime | Expected for thin buckets; reported in `pooled_gap_labels` |
| **Flat daily OHLC** | One synthetic tick per day | Use per-min calibration or aggregate to weekly/monthly for meaningful high/low |

---

## 13. Limitations and caveats

1. **Not a forecaster** — Synthetic paths wander from the start anchor; they do not target today’s market price at the end of the range.
2. **Calibration is historical** — Regime stats reflect the chosen `cal-start`/`cal-end` only.
3. **Regime labels lag** — Based on trailing windows; one bad day does not flip the label if the prior trend was strong.
4. **Moment decomposition is approximate** — See Appendix A; lag-2 inconsistency triggers a variance-share fallback.
5. **Daily vs intraday** — Meaningful high/low on daily output requires either per-minute calibration or aggregating to weekly/monthly.
6. **No built-in model persistence** — Each run recalibrates unless the user saves/loads the model externally.
7. **Market calendar** — Missing holidays show as “no market bar”; generation uses business-day approximations unless extended.

See **§12** for operational failure modes.

---

## 14. Module structure (reference)

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

## 15. Glossary

| Term | Definition |
|------|------------|
| **DistributionSpec** | Four-moment summary: mean, SD, skewness, excess kurtosis |
| **Regime** | One of seven trend buckets from trailing t-stat |
| **Episode** | Contiguous run of bars in the same regime |
| **Dwell time** | Number of bars the synthetic path stays in one regime before transitioning |
| **Transition matrix** | 7×7 probabilities of next regime given current regime (episode → episode) |
| **Semi-Markov** | Regime held for a random dwell, then transition; dwell lengths from history |
| **Occupancy** | Fraction of calibration bars spent in each regime |
| **Decomposition segment** | Contiguous return run used for ACF estimation (within session for per-min) |
| **Method of moments** | Estimate model parameters by matching sample autocovariances and cumulants |
| **Shared-shape split** | Fundamental and jitter use the same standardised skew/kurtosis so totals match |
| **Fundamental** | Non-microstructure return component (drift + shape); i.i.d. per bar — **not** company fundamentals |
| **Jitter** | Short-term AR(1) noise on top of fundamental; carries memory via ρ |
| **Gap** | Overnight jump: log(open / prior close), per-minute sessions |
| **Rho (ρ)** | Jitter autocorrelation: negative = bounce, positive = brief momentum |
| **Excess kurtosis** | Tail fatness vs normal; > 0 = fat tails |
| **Johnson SU / SB** | Mathematical curve families used to match four moments when sampling |
| **t-stat (t)** | Standardised trend strength over the labelling window |
| **Regime Balance Score** | Count of positive minus negative **actual** regime labels over last *n* bars; API: `trend_score()` |
| **actual_regime** | Real-market regime label from trailing *t*-stat on historical prices |
| **regime_path** | Synthetic regime labels driving simulation (independent of `actual_regime`) |

---

## Appendix A — Statistical decomposition methodology

This appendix documents how calibration turns pooled regime returns into **fundamental**, **jitter**, and **gap** specs. Core engine: `fit.decompose_segments`; orchestration: `regimes.calibrate`. For definitions of the underlying statistical concepts, see **Appendix B**.

### A.1 End-to-end pipeline (per regime)

```
Load aclose → log returns → label each bar (§3.2)
    → pool all bars with the same regime label
    → split into decomposition segments (A.2)
    → estimate four pooled moments + autocovariances
    → split variance: fundamental SD vs jitter SD + ρ (A.4)
    → split shape: shared skew/kurtosis (A.5)
    → back-transform jitter innovation moments (A.6)
    → (per-minute only) fit gap distribution separately (A.7)
```

Each of the seven regimes gets its **own** `(fundamental, jitter)` pair. Regimes with fewer than `--min-obs-per-label` bars **fail calibration** — no silent fallback to a global pool for return decomposition (gaps excepted; §4.5).

### A.2 What data is pooled, and how segments are built

**Pooling:** For regime label *L*, take every calibration bar labelled *L* and treat them as one empirical return sample.

**Segments:** Autocorrelation for jitter is estimated only from **contiguous runs** inside the pool — lagged pairs never cross segment boundaries.

| Periodicity | Segment rule | Why |
|-------------|--------------|-----|
| **Daily** | Contiguous run of the **same label** | Overnight already in close-to-close return |
| **Per-minute** | Same label **within one session** | No cross-overnight lag pairs |

**Gaps:** separate pool of `log(open / prior_close)` at session open; i.i.d. fit only.

### A.3 Statistical model

```
r(t) = F(t) + J(t)
```

| Component | Assumption |
|-----------|------------|
| **F(t)** | i.i.d. fundamental move |
| **J(t)** | AR(1): `J(t) = ρ·J(t−1) + ε(t)`, `E[J] = 0` |

Autocovariances: `γ(k) = σ_J² · ρ^k` for *k* ≥ 1; `γ(0) = σ_F² + σ_J²`.

### A.4 Variance split — method of moments (ACF)

Primary (lag-2): `ρ = γ(2)/γ(1)`, `σ_J² = γ(1)/ρ`, `σ_F² = γ(0) − σ_J²`.

- **No ACF:** |γ(1)/γ(0)| below ~1.96/√*n* → ρ = 0, jitter negligible.
- **Lag-2 inconsistent:** fallback with ~50% jitter variance share, ρ from γ(1), |ρ| clipped < 0.95.
- **Errors:** segments too short; σ_F² ≤ 0.

### A.5 Shape split — shared skew and kurtosis

```
s = κ₃ / (σ_F³ + σ_J³)
e = κ₄ / (σ_F⁴ + σ_J⁴)
```

Both layers share standardised *(s, e)*; `project_pearson_feasible` if needed. Infeasible → calibration error.

### A.6 Jitter innovation vs marginal

Innovation moments back-calculated: `κ_m(J) = κ_m(ε) / (1 − ρ^m)`. Negligible jitter → near-Gaussian innovation.

### A.7 Gap fitting

Direct four-moment `fit_distribution` on overnight gaps; pooled if < `--min-gap-obs`.

### A.8 Summary table mapping

| Summary column | What was estimated |
|----------------|-------------------|
| fund_mean | Pooled μ |
| fund_sd | σ_F |
| fund_skew, fund_exkurt | Shared *s*, *e* |
| jitter_sd | σ_J |
| jitter_rho | ρ |
| gap_mean, gap_sd | i.i.d. gap fit |

### A.9 Decomposition validation intuition

Synthetic single-regime returns should approximate: total variance, lag-1 ACF (when ρ ≠ 0), pooled skew/kurtosis. Approximate when lag-2 fallback used or *n* is small.

---

## Appendix B — Statistical background (optional reading)

**Audience:** Quant, risk, data-science, and review readers who want the statistical ideas behind the package — without reading source code. Business readers can skip this appendix.

Each topic below states the idea briefly and where `synth_market` uses it.

### B.1 Log returns

**Idea:** The log return over one bar is `r_t = log(P_t / P_{t-1}) ≈ (P_t − P_{t-1}) / P_{t-1}` for small moves. Log returns **add** over time: a two-bar move is `r_t + r_{t+1}`. They are symmetric-ish for modest percentages and standard in econometrics.

**In synth_market:** All calibration and simulation work on log returns from adjusted close (`aclose`). Prices are rebuilt as `P_t = P_{t-1} × exp(r_t)`.

### B.2 Mean, variance, and volatility (SD)

**Idea:** The **mean** is average return (drift). **Variance** is average squared deviation from the mean; **SD** (σ) is its square root — the usual scale for “typical” move size. Annualised volatility often uses √252 × daily SD, but the module stores **per-bar** SD in the same units as returns.

**In synth_market:** Each `DistributionSpec` carries mean and SD. Regime pools estimate these from history; simulation samples match them per active regime.

### B.3 Skewness and excess kurtosis

**Idea:** **Skewness** measures asymmetry: negative skew ⇒ longer/larger left tail (crash-like spikes); positive skew ⇒ right tail. **Excess kurtosis** measures tail thickness vs a normal bell curve: 0 ≈ Gaussian; **> 0** ⇒ fat tails (extreme moves more frequent than normal).

**In synth_market:** All four moments are stored per distribution. NSE equity returns often show positive excess kurtosis; ignoring it underestimates tail risk in stress tests.

### B.4 Cumulants and adding independent components

**Idea:** For **independent** random variables, variances add and **cumulants** add (third cumulant ≈ skew × σ³ scale; fourth cumulant relates to kurtosis). So a sum’s tail behaviour can be split across components if their shares of variance/cumulants are specified.

**In synth_market:** Returns are modelled as `F + J` (independent). The **shared-shape split** (Appendix A.5) assigns standardised skew/kurtosis to both layers so pooled cumulants match the sample.

### B.5 Autocorrelation and autocovariance (ACF)

**Idea:** **Autocovariance** γ(*k*) is the covariance between `r_t` and `r_{t−k}`. **Autocorrelation** is γ(*k*)/γ(0). γ(1) > 0 ⇒ short-term momentum; γ(1) < 0 ⇒ mean reversion (e.g. bid-ask bounce at tick/minute scale). **White noise** has γ(*k*) ≈ 0 for *k* ≥ 1.

**In synth_market:** Fundamental is i.i.d. (no memory). Jitter is AR(1), so γ(*k*) decays as ρ^*k*. ACF at lags 1–2 identifies σ_J and ρ (Appendix A.4). Segments exclude cross-session pairs on per-minute data.

### B.6 AR(1) process and innovations

**Idea:** An AR(1) series satisfies `J_t = ρ J_{t−1} + ε_t`, where **ε_t** is the **innovation** (shock), often i.i.d. with mean 0. |ρ| < 1 for stationarity. Negative ρ ⇒ oscillation (up then down); positive ρ ⇒ persistence. Stationary variance: Var(J) = Var(ε) / (1 − ρ²).

**In synth_market:** `JitterSpec` stores innovation moments and ρ. `JitterProcess` generates ε, then recurses to J, rescaling so stationary SD matches calibration.

### B.7 Method of moments (MoM)

**Idea:** Estimate model parameters by equating **sample moments** (mean, variance, autocovariances, etc.) to **theoretical moments** implied by the model — rather than maximising a likelihood (MLE). MoM is transparent and fast when the mapping is closed-form; it can be less efficient than MLE if the model is misspecified.

**In synth_market:** ρ, σ_F, σ_J come from matching γ(0), γ(1), γ(2) under the i.i.d. + AR(1) model. Shape uses cumulant matching. No MLE layer in the current pipeline.

### B.8 Semi-Markov and Markov chains

**Idea:** A **Markov chain** on regimes chooses the next state each step with fixed transition probabilities. A **semi-Markov** process also draws a **holding time** (dwell) in each state before transitioning — so persistence is explicit, not only from self-transitions.

**In synth_market:** Simulation samples dwell from empirical episode lengths, then next regime from the 7×7 transition matrix (§3.4). This is separate from return dynamics within a regime.

### B.9 Drift t-statistic (regime labelling)

**Idea:** Over window *W*, compute sample mean `m̄` and SD `s` of returns. The quantity `t = (m̄ / s) × √W` is a **standardised trend strength** (drift t-stat): scale-free, comparable across symbols and windows. Under a pure random walk with no drift, |t| is typically O(1); larger |t| ⇒ clearer directional drift over the window.

**In synth_market:** *t* maps to seven bands via cuts (default ±0.25, ±0.75, ±1.5). Labels are **descriptive** (trailing window), not forecasts.

### B.10 Johnson SU / SB and moment matching

**Idea:** Normal distributions fix only mean and variance. **Johnson** families (SU unbounded, SB bounded) can match **four moments** (mean, SD, skew, kurtosis) more flexibly. **Moment matching** picks parameters so theoretical moments ≈ target moments, then samples from that curve.

**In synth_market:** `MomentMatchedSampler` chooses Normal (near-Gaussian), Johnson SU (fat tails), or Johnson SB (mild/bounded cases). Calibration raises on infeasible targets; generation may relax in rare edge cases (§4.6).

### B.11 Pearson skew–kurtosis feasibility

**Idea:** Not every (skewness, kurtosis) pair is achievable by any distribution. The **Pearson feasible region** bounds excess kurtosis as a function of skew. Samples from short or noisy windows can land outside it.

**In synth_market:** `project_pearson_feasible` nudges estimates to the nearest valid pair before fitting. Persistent infeasibility → calibration error (§12).

### B.12 Further reading (external)

| Topic | Typical reference |
|-------|-------------------|
| Log returns & volatility | Tsay, *Analysis of Financial Time Series* |
| AR(1) / ACF | Hamilton, *Time Series Analysis*; Box–Jenkins ARMA intro |
| Semi-Markov processes | Pyke (1969); regime-switching survey papers |
| Johnson distributions | Johnson (1949); scipy `johnsonsu` / `johnsonsb` docs |
| Fat tails in equity returns | Cont (2001), *Empirical properties of asset returns* |

---

*For technical implementation details, see source modules under `synth_market/synth_market/` and `example.py`.*
