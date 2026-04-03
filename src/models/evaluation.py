from __future__ import annotations


def classification_metrics(
    predictions: list[dict[str, object]], labels: list[dict[str, object]]
) -> dict[str, float]:
    label_map = {(r["date"], r["symbol"]): int(r["target"]) for r in labels}
    y_true: list[int] = []
    y_pred: list[int] = []
    for row in predictions:
        key = (row["date"], row["symbol"])
        if key in label_map:
            y_true.append(label_map[key])
            y_pred.append(int(row["prediction"]))

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "n_samples": 0.0}

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "n_samples": float(len(y_true)),
    }
