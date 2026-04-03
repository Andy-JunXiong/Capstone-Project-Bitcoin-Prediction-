from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import json


@dataclass(frozen=True)
class ExperimentConfig:
    input_csv: Path
    output_features_csv: Path
    output_labels_csv: Path
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    prediction_csv: Path
    backtest_csv: Path
    metrics_json: Path
    model_path: Path

    @staticmethod
    def _parse_date(value: str, field: str) -> date:
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"Invalid {field}: {value}. Expected YYYY-MM-DD") from exc

    @staticmethod
    def from_json(path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        cfg = ExperimentConfig(
            input_csv=Path(payload["input_csv"]),
            output_features_csv=Path(payload["output_features_csv"]),
            output_labels_csv=Path(payload["output_labels_csv"]),
            train_start=ExperimentConfig._parse_date(payload["train_start"], "train_start"),
            train_end=ExperimentConfig._parse_date(payload["train_end"], "train_end"),
            test_start=ExperimentConfig._parse_date(payload["test_start"], "test_start"),
            test_end=ExperimentConfig._parse_date(payload["test_end"], "test_end"),
            prediction_csv=Path(payload["prediction_csv"]),
            backtest_csv=Path(payload["backtest_csv"]),
            metrics_json=Path(payload.get("metrics_json", "reports/tables/metrics.json")),
            model_path=Path(payload.get("model_path", "artifacts/model.json")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.train_start > self.train_end:
            raise ValueError("train_start must be <= train_end")
        if self.test_start > self.test_end:
            raise ValueError("test_start must be <= test_end")
        if self.train_end >= self.test_start:
            raise ValueError("Expected train_end < test_start for chronological split")
