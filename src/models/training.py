from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

FEATURES = ["ret_1d", "ma5_gap", "range_20", "volume_ratio"]


@dataclass
class LinearClassifier:
    feature_names: list[str]
    weights: list[float]
    bias: float

    def score(self, row: dict[str, object]) -> float:
        total = self.bias
        for name, weight in zip(self.feature_names, self.weights):
            total += float(row[name]) * weight
        return total

    def predict_proba(self, row: dict[str, object]) -> float:
        score = self.score(row)
        return 1.0 / (1.0 + (2.718281828 ** (-score)))


def _fit_weight(feature_vals: list[float], targets: list[int]) -> float:
    if not feature_vals:
        return 0.0
    x_bar = sum(feature_vals) / len(feature_vals)
    y_bar = sum(targets) / len(targets)
    numer = sum((x - x_bar) * (y - y_bar) for x, y in zip(feature_vals, targets))
    denom = sum((x - x_bar) ** 2 for x in feature_vals)
    return (numer / denom) if denom else 0.0


def train_linear_classifier(rows: list[dict[str, object]], labels: list[dict[str, object]]) -> LinearClassifier:
    label_map = {(r["date"], r["symbol"]): int(r["target"]) for r in labels}
    aligned = [r for r in rows if (r["date"], r["symbol"]) in label_map]
    targets = [label_map[(r["date"], r["symbol"])] for r in aligned]

    weights: list[float] = []
    for feature in FEATURES:
        vals = [float(r[feature]) for r in aligned]
        weights.append(_fit_weight(vals, targets))

    base_rate = sum(targets) / len(targets) if targets else 0.5
    bias = base_rate - 0.5
    return LinearClassifier(feature_names=FEATURES, weights=weights, bias=bias)


def save_model(model: LinearClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"feature_names": model.feature_names, "weights": model.weights, "bias": model.bias}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(path: Path) -> LinearClassifier:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LinearClassifier(**payload)
