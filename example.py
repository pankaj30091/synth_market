"""
example.py
----------
End-to-end usage of synth_market:
  1. Calibrate a 7-regime model from real data (via ohlcutils).
  2. Generate synthetic price/OHLC data from that model.

Run from the repo root:
    ~/envs/base/bin/python example.py -h
    ~/envs/base/bin/python example.py --periodicity permin --anchor-close
    ~/envs/base/bin/python example.py --periodicity daily \\
        --cal-start 2022-04-01 --cal-end 2026-03-31 --output-freq 1W
"""

import argparse

import pandas as pd
from ohlcutils.enums import Periodicity

from synth_market import calibrate, simulate


# Defaults per periodicity (overridden by explicit CLI flags).
PROFILES = {
    "daily": {
        "periodicity": Periodicity.DAILY,
        "cal_start": "2022-04-01",
        "cal_end": "2026-03-31",
        "window": 21,
        "min_obs_per_label": 20,
        "min_gap_obs": 10,
        "output_freq": "1W",
    },
    "permin": {
        "periodicity": Periodicity.PERMIN,
        "cal_start": "2024-01-01",
        "cal_end": "2024-06-30",
        "window": 30,
        "min_obs_per_label": 200,
        "min_gap_obs": 10,
        "output_freq": "15min",
    },
}


def run(
    symbol: str,
    periodicity: str,
    cal_start: str,
    cal_end: str,
    window: int,
    min_obs_per_label: int,
    min_gap_obs: int,
    gen_start: str,
    gen_end: str,
    output_freq: str,
    seed: int,
    anchor_close: bool,
    score_n: int,
) -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)
    src = PROFILES[periodicity]["periodicity"]

    model = calibrate(
        symbol=symbol,
        periodicity=src,
        start=cal_start,
        end=cal_end,
        window=window,
        min_obs_per_label=min_obs_per_label,
        min_gap_obs=min_gap_obs,
    )

    print(f"=== {symbol} [{periodicity}]: calibrated {cal_start} .. {cal_end} ===")
    print(model.summary().to_string())
    print(f"\nmean labelled bars/session : {model.bars_per_session:.1f}")
    print(f"gap labels pooled to global : {model.pooled_gap_labels}")

    gen = simulate(
        model,
        start=gen_start,
        end=gen_end,
        seed=seed,
        output_freq=output_freq,
        anchor_to_last_close=anchor_close,
    )

    gen.attach_market_regimes(model)

    print("\n=== Generated price data ===")
    print(f"range: {gen_start} .. {gen_end}   seed: {seed}")
    print(f"tick rows: {len(gen.ticks)}   ohlc bars: {len(gen.ohlc)}")
    cols = ["price", "actual_log_return", "actual_regime", "actual_regime_reason"]
    print("\nTick price path (tail):")
    print(gen.ticks[cols].tail(15).round(4).to_string())
    print(f"\nOHLC @ {output_freq} (tail):")
    print(gen.ohlc.tail(6).round(2).to_string())
    print("\ngenerated regime occupancy:",
          {k: round(v, 2) for k, v in gen.regime_occupancy().items()})
    print(f"\ntrend score (last {score_n} ticks of actual_regime): "
          f"{gen.trend_score(score_n)}  [range {-score_n}..{score_n}]")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Calibrate a regime model and generate synthetic OHLC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="INFY_STK___", help="ohlcutils symbol")
    p.add_argument(
        "--periodicity", choices=list(PROFILES), default="permin",
        help="calibration source frequency; sets defaults for cal_* and output-freq",
    )

    cal = p.add_argument_group("calibration")
    cal.add_argument("--cal-start", default=None, metavar="YYYY-MM-DD",
                     help="history start for regime fitting "
                          "(default: permin 2024-01-01, daily 2022-04-01)")
    cal.add_argument("--cal-end", default=None, metavar="YYYY-MM-DD",
                     help="history end for regime fitting "
                          "(default: permin 2024-06-30, daily 2026-03-31)")
    cal.add_argument("--window", type=int, default=None,
                     help="rolling bars for drift t-stat labelling "
                          "(default: permin 30, daily 21)")
    cal.add_argument("--min-obs-per-label", type=int, default=None,
                     help="raise if a regime has fewer return observations "
                          "(default: permin 200, daily 20)")
    cal.add_argument("--min-gap-obs", type=int, default=None,
                     help="min overnight gaps per regime, per-min only "
                          "(default: 10)")

    gen = p.add_argument_group("generation")
    gen.add_argument("--gen-start", default="2026-04-01", metavar="YYYY-MM-DD",
                     help="synthetic series start")
    gen.add_argument("--gen-end", default="2026-04-30", metavar="YYYY-MM-DD",
                     help="synthetic series end")
    gen.add_argument("--output-freq", default=None,
                     help="pandas resample rule for OHLC (>= tick), e.g. 15min, 1D, 1W "
                          "(default: permin 15min, daily 1W)")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument(
        "--anchor-close", action="store_true",
        help="start price = last real daily aclose on/before --gen-start",
    )
    gen.add_argument("--score-n", type=int, default=15,
                     help="ticks of actual_regime to sum for the trend score")

    args = p.parse_args()
    prof = PROFILES[args.periodicity]

    run(
        symbol=args.symbol,
        periodicity=args.periodicity,
        cal_start=args.cal_start or prof["cal_start"],
        cal_end=args.cal_end or prof["cal_end"],
        window=args.window if args.window is not None else prof["window"],
        min_obs_per_label=(
            args.min_obs_per_label if args.min_obs_per_label is not None
            else prof["min_obs_per_label"]
        ),
        min_gap_obs=(
            args.min_gap_obs if args.min_gap_obs is not None else prof["min_gap_obs"]
        ),
        gen_start=args.gen_start,
        gen_end=args.gen_end,
        output_freq=args.output_freq or prof["output_freq"],
        seed=args.seed,
        anchor_close=args.anchor_close,
        score_n=args.score_n,
    )


if __name__ == "__main__":
    main()
