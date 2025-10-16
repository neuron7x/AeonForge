from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from markets.vpin import Trade, VPINCalculator, compute_vpin


@pytest.fixture
def trades() -> list[Trade]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    points = []
    for idx in range(100):
        side = "buy" if idx % 2 == 0 else "sell"
        points.append(
            Trade(
                timestamp=base + timedelta(seconds=idx),
                price=100.0 + idx * 0.1,
                volume=1.0,
                side=side,
            )
        )
    return points


def test_vpin_calculator_generates_values(trades: list[Trade]) -> None:
    calculator = VPINCalculator(bucket_volume=5, window=4)
    values = calculator.consume(trades)
    assert values  # ensure VPIN values were produced
    assert pytest.approx(values[-1], 0.001) == calculator.current_vpin


def test_compute_vpin_matches_streaming(trades: list[Trade]) -> None:
    calculator = VPINCalculator(bucket_volume=10, window=3)
    calculator.consume(trades)
    batch = compute_vpin(trades, bucket_volume=10, window=3)
    assert pytest.approx(batch, 0.0001) == calculator.current_vpin


def test_trade_validation() -> None:
    with pytest.raises(ValueError):
        Trade(
            timestamp=datetime.now(tz=timezone.utc),
            price=100,
            volume=0,
            side="buy",
        )
    with pytest.raises(ValueError):
        Trade(
            timestamp=datetime.now(tz=timezone.utc),
            price=-1,
            volume=1,
            side="buy",
        )
    with pytest.raises(ValueError):
        Trade(
            timestamp=datetime.now(tz=timezone.utc),
            price=1,
            volume=1,
            side="hold",
        )
