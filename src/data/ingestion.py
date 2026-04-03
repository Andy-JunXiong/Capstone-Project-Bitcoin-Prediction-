from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.common.io_utils import read_csv


@dataclass(frozen=True)
class MarketBar:
    date: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_market_data(path: Path, symbol_column: str = "symbol") -> list[MarketBar]:
    records = read_csv(path)
    bars: list[MarketBar] = []
    for row in records:
        bars.append(
            MarketBar(
                date=datetime.strptime(row["date"], "%Y-%m-%d"),
                symbol=row.get(symbol_column, "UNKNOWN"),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )
        )
    return sorted(bars, key=lambda x: (x.symbol, x.date))
