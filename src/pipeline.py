from __future__ import annotations

from datetime import datetime

from src.backtest.engine import backtest_metrics, run_long_only_backtest
from src.common.config import ExperimentConfig
from src.common.io_utils import write_csv, write_json
from src.data.ingestion import load_market_data
from src.features.engineering import generate_features
from src.labels.generation import generate_binary_labels
from src.models.evaluation import classification_metrics
from src.models.prediction import predict_rows
from src.models.training import load_model, save_model, train_linear_classifier


def _date_in_range(date_str: str, start: datetime.date, end: datetime.date) -> bool:
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    return start <= dt <= end


def _filter_by_date(
    rows: list[dict[str, object]],
    start: datetime.date,
    end: datetime.date,
) -> list[dict[str, object]]:
    return [row for row in rows if _date_in_range(str(row["date"]), start, end)]


def run_pipeline(config_path: str = "configs/experiment_v1.json") -> dict[str, dict[str, float]]:
    cfg = ExperimentConfig.from_json(config_path)

    bars = load_market_data(cfg.input_csv)
    features = generate_features(bars)
    labels = generate_binary_labels(features)

    write_csv(cfg.output_features_csv, features, list(features[0].keys()))
    write_csv(cfg.output_labels_csv, labels, list(labels[0].keys()))

    train_features = _filter_by_date(features, cfg.train_start, cfg.train_end)
    train_labels = _filter_by_date(labels, cfg.train_start, cfg.train_end)
    test_features = _filter_by_date(features, cfg.test_start, cfg.test_end)
    test_labels = _filter_by_date(labels, cfg.test_start, cfg.test_end)

    model = train_linear_classifier(train_features, train_labels)
    save_model(model, cfg.model_path)

    reloaded = load_model(cfg.model_path)
    predictions = predict_rows(reloaded, test_features)
    write_csv(cfg.prediction_csv, predictions, list(predictions[0].keys()))

    backtest_curve = run_long_only_backtest(predictions, test_features)
    write_csv(cfg.backtest_csv, backtest_curve, list(backtest_curve[0].keys()) if backtest_curve else ["date", "daily_return", "equity"])

    metrics = {
        "classification": classification_metrics(predictions, test_labels),
        "backtest": backtest_metrics(backtest_curve),
    }
    write_json(cfg.metrics_json, metrics)
    return metrics


if __name__ == "__main__":
    results = run_pipeline()
    print(results)
