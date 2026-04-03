from __future__ import annotations

from collections import defaultdict


def run_long_only_backtest(
    predictions: list[dict[str, object]],
    features: list[dict[str, object]],
    top_k: int = 5,
    transaction_cost_bps: float = 5.0,
) -> list[dict[str, object]]:
    close_by_key = {(r["date"], r["symbol"]): float(r["close"]) for r in features}
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for p in predictions:
        by_date[str(p["date"])].append(p)

    dates = sorted(by_date.keys())
    equity = 1.0
    curve: list[dict[str, object]] = []

    for idx, day in enumerate(dates[:-1]):
        next_day = dates[idx + 1]
        ranked = sorted(by_date[day], key=lambda x: float(x["score"]), reverse=True)
        picks = ranked[:top_k]

        if not picks:
            curve.append({"date": day, "daily_return": 0.0, "equity": round(equity, 8)})
            continue

        daily_returns: list[float] = []
        for pick in picks:
            sym = str(pick["symbol"])
            c0 = close_by_key.get((day, sym))
            c1 = close_by_key.get((next_day, sym))
            if c0 and c1:
                daily_returns.append((c1 / c0) - 1.0)

        gross = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
        cost = (transaction_cost_bps / 10000.0) * 2
        net = gross - cost
        equity *= (1.0 + net)
        curve.append({"date": day, "daily_return": round(net, 8), "equity": round(equity, 8)})

    return curve


def backtest_metrics(curve: list[dict[str, object]]) -> dict[str, float]:
    if not curve:
        return {
            "total_return": 0.0,
            "avg_daily_return": 0.0,
            "max_drawdown": 0.0,
            "n_periods": 0.0,
        }

    daily_returns = [float(row["daily_return"]) for row in curve]
    equities = [float(row["equity"]) for row in curve]

    peak = equities[0]
    max_drawdown = 0.0
    for eq in equities:
        peak = max(peak, eq)
        drawdown = (eq / peak) - 1.0
        max_drawdown = min(max_drawdown, drawdown)

    return {
        "total_return": round(equities[-1] - 1.0, 6),
        "avg_daily_return": round(sum(daily_returns) / len(daily_returns), 6),
        "max_drawdown": round(max_drawdown, 6),
        "n_periods": float(len(curve)),
    }
