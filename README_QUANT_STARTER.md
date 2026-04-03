# Quant Research Starter (US Equities)

This starter provides a minimal but working modular setup for:
- data ingestion
- feature engineering
- label generation
- model training
- prediction
- backtesting

## Architecture

- `src/data/ingestion.py`: normalize OHLCV into typed `MarketBar` records.
- `src/features/engineering.py`: create simple, extensible alpha factors.
- `src/labels/generation.py`: forward-return direction labels.
- `src/models/training.py`: baseline linear classifier train/save/load.
- `src/models/prediction.py`: probability scoring + hard predictions.
- `src/backtest/engine.py`: long-only top-k backtest + strategy metrics.
- `src/pipeline.py`: orchestrates full training, inference, evaluation, and output persistence.

## Quickstart

```bash
python -m src.pipeline
```

Outputs:
- `data/processed/features.csv`
- `data/processed/labels.csv`
- `reports/tables/predictions.csv`
- `reports/tables/backtest_curve.csv`
- `reports/tables/metrics.json`
- `artifacts/model.json`

## Testing

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
