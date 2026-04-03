from __future__ import annotations

from pathlib import Path
import json
import unittest

from src.pipeline import run_pipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_runs_and_writes_outputs(self) -> None:
        metrics = run_pipeline("configs/experiment_v1.json")

        self.assertIn("classification", metrics)
        self.assertIn("backtest", metrics)

        expected_files = [
            Path("data/processed/features.csv"),
            Path("data/processed/labels.csv"),
            Path("reports/tables/predictions.csv"),
            Path("reports/tables/backtest_curve.csv"),
            Path("reports/tables/metrics.json"),
            Path("artifacts/model.json"),
        ]
        for file_path in expected_files:
            self.assertTrue(file_path.exists(), f"Expected output missing: {file_path}")

        parsed = json.loads(Path("reports/tables/metrics.json").read_text(encoding="utf-8"))
        self.assertIn("classification", parsed)
        self.assertIn("backtest", parsed)


if __name__ == "__main__":
    unittest.main()
