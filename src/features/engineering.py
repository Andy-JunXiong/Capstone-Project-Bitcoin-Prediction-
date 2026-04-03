from __future__ import annotations

from collections import defaultdict, deque

from src.data.ingestion import MarketBar


def generate_features(bars: list[MarketBar]) -> list[dict[str, object]]:
    by_symbol: dict[str, list[MarketBar]] = defaultdict(list)
    for bar in bars:
        by_symbol[bar.symbol].append(bar)

    feature_rows: list[dict[str, object]] = []
    for symbol, series in by_symbol.items():
        series = sorted(series, key=lambda x: x.date)
        closes: deque[float] = deque(maxlen=20)
        vols: deque[float] = deque(maxlen=20)
        prev_close: float | None = None

        for bar in series:
            closes.append(bar.close)
            vols.append(bar.volume)

            ret_1d = (bar.close / prev_close - 1.0) if prev_close else 0.0
            trailing_closes = list(closes)
            ma_5 = sum(trailing_closes[-5:]) / min(len(trailing_closes), 5)
            range_20 = (max(trailing_closes) - min(trailing_closes)) / bar.close if len(trailing_closes) > 1 else 0.0
            avg_vol_20 = sum(vols) / len(vols)

            feature_rows.append(
                {
                    "date": bar.date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "close": bar.close,
                    "ret_1d": round(ret_1d, 8),
                    "ma5_gap": round((bar.close / ma_5) - 1.0, 8),
                    "range_20": round(range_20, 8),
                    "volume_ratio": round((bar.volume / avg_vol_20) if avg_vol_20 else 0.0, 8),
                }
            )
            prev_close = bar.close

    return sorted(feature_rows, key=lambda x: (x["date"], x["symbol"]))
