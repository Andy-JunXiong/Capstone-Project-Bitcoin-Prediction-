from __future__ import annotations

from pathlib import Path

from src.backtest.engine import run_long_only_backtest
from src.common.config import ExperimentConfig
from src.common.io_utils import write_csv
from src.data.ingestion import load_market_data
from src.features.engineering import generate_features
from src.labels.generation import generate_binary_labels
from src.models.prediction import predict_rows
from src.models.training import load_model, save_model, train_linear_classifier


def run_pipeline(config_path: str = "configs/experiment_v1.json") -> None:
    cfg = ExperimentConfig.from_json(config_path)

    bars = load_market_data(cfg.input_csv)
    features = generate_features(bars)
    labels = generate_binary_labels(features)

    write_csv(cfg.output_features_csv, features, list(features[0].keys()))
    write_csv(cfg.output_labels_csv, labels, list(labels[0].keys()))

    model = train_linear_classifier(features, labels)
    model_path = Path("artifacts/model.json")
    save_model(model, model_path)

    reloaded = load_model(model_path)
    predictions = predict_rows(reloaded, features)
    write_csv(cfg.prediction_csv, predictions, list(predictions[0].keys()))

    backtest_curve = run_long_only_backtest(predictions, features)
    write_csv(cfg.backtest_csv, backtest_curve, list(backtest_curve[0].keys()))


if __name__ == "__main__":
    run_pipeline()
