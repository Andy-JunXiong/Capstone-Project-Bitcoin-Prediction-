from __future__ import annotations

from src.models.training import LinearClassifier


def predict_rows(model: LinearClassifier, rows: list[dict[str, object]], threshold: float = 0.5) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        proba = model.predict_proba(row)
        output.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "score": round(proba, 8),
                "prediction": int(proba >= threshold),
            }
        )
    return output
