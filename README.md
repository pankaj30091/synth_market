# synth_market

Calibrate seven **trend regimes** from historical returns, then generate **synthetic OHLC** paths for strategy stress-testing. Not a forecaster or signal — details in **[FUNCTIONAL_SPEC.md](FUNCTIONAL_SPEC.md)**.

```bash
pip install .
python example.py --periodicity daily --symbol INFY_STK___ --anchor-close
```

`calibrate()` / `simulate()` need **ohlcutils** and an NSE-style symbol. See `example.py` and the spec for parameters.
