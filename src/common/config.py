from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    input_csv: Path
    output_features_csv: Path
    output_labels_csv: Path
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    prediction_csv: Path
    backtest_csv: Path

    @staticmethod
    def from_json(path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return ExperimentConfig(
            input_csv=Path(payload["input_csv"]),
            output_features_csv=Path(payload["output_features_csv"]),
            output_labels_csv=Path(payload["output_labels_csv"]),
            train_start=payload["train_start"],
            train_end=payload["train_end"],
            test_start=payload["test_start"],
            test_end=payload["test_end"],
            prediction_csv=Path(payload["prediction_csv"]),
            backtest_csv=Path(payload["backtest_csv"]),
        )
