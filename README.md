# bettingedge

Betting research engine.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## End-to-end

```bash
# build canonical dataset
python -m engine.pipeline --rebuild-canonical --div I1 --seasons all
# build market model (using market implied probabilities)
python -m engine.pipeline --build-market --div I1 --train-ratio 0.8 --calibrate --model-source market
# generate picks for multiple markets
python -m engine.pipeline --picks --div I1 --ev-min 0.02 --markets 1x2 ou25
# optional quick backtest
python -m engine.pipeline --picks --div I1 --ev-min 0.02 --stake-mode fixed --stake-fraction 0.01
# launch UI
streamlit run ui/streamlit_app.py
```

## Structure

```
engine/      core library
ui/          streamlit UI
data/
  raw/       input CSVs
  processed/ generated datasets
scripts/     helper scripts
runs/        experiment outputs (gitignored)
```

## Supported markets

- `1x2` – home/draw/away
- `ou25` – over/under 2.5 goals
- `dnb` – draw no bet
- `dc` – double chance
- `ah` – Asian handicap (quarter lines supported)

## Commands

```bash
python -m engine.ingest.keys --check
```

## Metrics

- **ECE** – expected calibration error
- **Brier** – mean squared error between probabilities and outcomes
- **KS-p** – Kolmogorov–Smirnov test p-value
- **CAGR** – compound annual growth rate of equity
- **MaxDD** – maximum drawdown
- **Sharpe** – mean return divided by its standard deviation

## Reproducibility

Train/test splits and run artifacts are saved under `runs/<div>/run_*`.
Each run stores config, metrics, equity and trades to reproduce experiments.
