from __future__ import annotations

from collections import defaultdict


def generate_binary_labels(feature_rows: list[dict[str, object]], horizon: int = 1) -> list[dict[str, object]]:
    by_symbol: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        by_symbol[str(row["symbol"])].append(row)

    labeled: list[dict[str, object]] = []
    for symbol, rows in by_symbol.items():
        rows = sorted(rows, key=lambda x: str(x["date"]))
        for idx, row in enumerate(rows):
            if idx + horizon >= len(rows):
                continue
            future_close = float(rows[idx + horizon]["close"])
            curr_close = float(row["close"])
            target = 1 if future_close > curr_close else 0
            labeled.append(
                {
                    "date": row["date"],
                    "symbol": symbol,
                    "target": target,
                }
            )
    return labeled
