# Quant Research Starter (US Equities)

This starter pipeline provides a minimal but working modular setup for:
- data ingestion
- feature engineering
- label generation
- model training
- prediction
- backtesting

## Quickstart

```bash
python -m src.pipeline
```

Outputs:
- `data/processed/features.csv`
- `data/processed/labels.csv`
- `reports/tables/predictions.csv`
- `reports/tables/backtest_curve.csv`
- `artifacts/model.json`
