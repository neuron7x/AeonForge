from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from infrastructure.persistence import (
    ClickHouseAdapter,
    ParquetLakeAdapter,
    TimeSeriesPoint,
    TimescaleDBAdapter,
)


@pytest.fixture
def sample_points() -> list[TimeSeriesPoint]:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        TimeSeriesPoint(timestamp=ts, symbol="BTCUSDT", fields={"price": 48000.0, "volume": 1.5}),
        TimeSeriesPoint(timestamp=ts, symbol="ETHUSDT", fields={"price": 3000.0, "volume": 10.0}),
    ]


def test_timescaledb_write_batch(sample_points: list[TimeSeriesPoint]) -> None:
    adapter = TimescaleDBAdapter(dsn="postgresql://test")
    fake_conn = mock.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_cursor_ctx = mock.MagicMock()
    fake_cursor_ctx.__enter__.return_value = mock.MagicMock()
    fake_conn.cursor.return_value = fake_cursor_ctx
    with mock.patch("infrastructure.persistence.timeseries.psycopg2.connect", return_value=fake_conn):
        adapter.ensure_storage("ticks")
        adapter.write_batch("ticks", sample_points)
    assert fake_cursor_ctx.__enter__.return_value.execute.called


def test_clickhouse_adapter_roundtrip(sample_points: list[TimeSeriesPoint]) -> None:
    adapter = ClickHouseAdapter(endpoint="http://clickhouse")
    session_post = mock.MagicMock()
    fake_response = mock.MagicMock()
    fake_response.text = "{\"data\": [{\"timestamp\": \"2024-01-01T00:00:00\", \"symbol\": \"BTCUSDT\", \"payload\": {\"price\": 1}}]}"
    fake_response.raise_for_status.return_value = None
    session_post.side_effect = [fake_response, fake_response, fake_response]
    adapter._client.post = session_post
    adapter.ensure_storage("ticks")
    adapter.write_batch("ticks", sample_points)
    rows = adapter.read_range(
        "ticks",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        symbol="BTCUSDT",
    )
    assert rows[0]["symbol"] == "BTCUSDT"


def test_parquet_adapter(tmp_path: Path, sample_points: list[TimeSeriesPoint]) -> None:
    adapter = ParquetLakeAdapter(tmp_path)
    adapter.write_batch("ticks", sample_points)
    rows = adapter.read_range(
        "ticks",
        start=datetime(2023, 12, 31, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert {row["symbol"] for row in rows} == {"BTCUSDT", "ETHUSDT"}
