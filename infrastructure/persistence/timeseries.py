"""Time-series persistence adapters."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence

import httpx
import pandas as pd

try:  # pragma: no cover - pandas optional dependency probing
    from pandas.io import parquet as pd_parquet
except ImportError:  # pragma: no cover - defensive, pandas always available in tests
    pd_parquet = None
import psycopg2


@dataclass
class TimeSeriesPoint:
    """Canonical representation of a time-series measurement."""

    timestamp: datetime
    symbol: str
    fields: Mapping[str, float]

    def to_record(self) -> MutableMapping[str, object]:
        record: MutableMapping[str, object] = {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
        }
        record.update(self.fields)
        return record


class BaseTimeSeriesAdapter:
    """Common interface for time-series adapters."""

    def ensure_storage(self, *_, **__) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def write_batch(self, *_: TimeSeriesPoint) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def read_range(self, *_: object, **__: object) -> Sequence[Mapping[str, object]]:  # pragma: no cover
        raise NotImplementedError


class TimescaleDBAdapter(BaseTimeSeriesAdapter):
    """Adapter for TimescaleDB running on top of PostgreSQL."""

    def __init__(self, dsn: str, schema: str = "public", sslmode: str | None = "require") -> None:
        self.dsn = dsn
        self.schema = schema
        self.sslmode = sslmode

    def _connect(self):  # type: ignore[override]
        kwargs = {}
        if self.sslmode:
            kwargs["sslmode"] = self.sslmode
        return psycopg2.connect(self.dsn, **kwargs)

    def _quote_identifier(self, value: str) -> str:
        if not value.replace("_", "").isalnum():
            raise ValueError(f"Invalid identifier '{value}'")
        return f'"{value}"'

    def _qualified_table(self, table: str) -> str:
        return f"{self._quote_identifier(self.schema)}.{self._quote_identifier(table)}"

    def ensure_storage(self, table: str, time_column: str = "timestamp") -> None:
        qualified_table = self._qualified_table(table)
        quoted_time_column = self._quote_identifier(time_column)
        with self._connect() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {qualified_table} (
                        {quoted_time_column} TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        PRIMARY KEY ({quoted_time_column}, symbol)
                    );
                    """
                )
                cur.execute(
                    f"SELECT create_hypertable('{qualified_table}', '{time_column}', if_not_exists => TRUE);"
                )

    def write_batch(self, table: str, points: Sequence[TimeSeriesPoint]) -> None:
        if not points:
            return
        records = [point.to_record() for point in points]
        for record in records:
            record["payload"] = json.dumps({k: v for k, v in record.items() if k not in {"timestamp", "symbol"}})
        with self._connect() as conn:
            with conn.cursor() as cur:
                insert_sql = (
                    f"INSERT INTO {self._qualified_table(table)} (timestamp, symbol, payload) "
                    "VALUES (%s, %s, %s)"
                )
                for record in records:
                    cur.execute(
                        insert_sql,
                        (
                            record["timestamp"],
                            record["symbol"],
                            record["payload"],
                        ),
                    )
            conn.commit()

    def read_range(
        self,
        table: str,
        start: datetime,
        end: datetime,
        symbol: str | None = None,
    ) -> Sequence[Mapping[str, object]]:
        conditions = ["timestamp >= %s", "timestamp <= %s"]
        params: List[object] = [start, end]
        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
        qualified_table = self._qualified_table(table)
        where_clause = " AND ".join(conditions)
        query = (
            f"SELECT timestamp, symbol, payload FROM {qualified_table} "
            f"WHERE {where_clause} ORDER BY timestamp ASC"
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [
            {
                "timestamp": ts,
                "symbol": sym,
                **json.loads(payload),
            }
            for ts, sym, payload in rows
        ]


class ClickHouseAdapter(BaseTimeSeriesAdapter):
    """Adapter for ClickHouse's HTTP API."""

    def __init__(self, endpoint: str, user: str = "default", password: str = "", database: str = "default") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.user = user
        self.password = password
        self.database = database
        self._client = httpx.Client()

    def _request(self, query: str, *, data: str | None = None) -> requests.Response:
        params = {"database": self.database, "query": query}
        resp = self._client.post(
            f"{self.endpoint}/",
            params=params,
            data=data,
            auth=(self.user, self.password) if self.password else None,
            timeout=30,
        )
        resp.raise_for_status()
        return resp

    def ensure_storage(self, table: str) -> None:
        create_query = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            timestamp DateTime64(3) CODEC(Delta, ZSTD),
            symbol String,
            payload JSON
        )
        ENGINE = MergeTree
        ORDER BY (symbol, timestamp)
        SETTINGS index_granularity = 8192
        """
        self._request(create_query)

    def write_batch(self, table: str, points: Sequence[TimeSeriesPoint]) -> None:
        if not points:
            return
        payload = "\n".join(json.dumps(point.to_record(), default=str) for point in points)
        insert_query = f"INSERT INTO {table} FORMAT JSONEachRow"
        self._request(insert_query, data=payload)

    def read_range(
        self,
        table: str,
        start: datetime,
        end: datetime,
        symbol: str | None = None,
    ) -> Sequence[Mapping[str, object]]:
        conditions = [
            f"timestamp >= toDateTime64('{start.isoformat()}', 3)",
            f"timestamp <= toDateTime64('{end.isoformat()}', 3)",
        ]
        if symbol:
            conditions.append(f"symbol = '{symbol}'")
        query = f"SELECT timestamp, symbol, payload FROM {table} WHERE {' AND '.join(conditions)} ORDER BY timestamp"
        resp = self._request(query)
        rows = json.loads(resp.text).get("data", [])
        return [
            {
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "symbol": row["symbol"],
                **row.get("payload", {}),
            }
            for row in rows
        ]


class ParquetLakeAdapter(BaseTimeSeriesAdapter):
    """Adapter that persists time-series data in Parquet format."""

    def __init__(self, root_path: str | os.PathLike[str]) -> None:
        self.root = Path(root_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._supports_parquet = self._detect_parquet_support()

    def ensure_storage(self, *_, **__) -> None:
        # Parquet datasets require no upfront provisioning.
        return None

    def _dataset_path(self, table: str) -> Path:
        path = self.root / table
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_batch(self, table: str, points: Sequence[TimeSeriesPoint]) -> None:
        if not points:
            return
        dataset_path = self._dataset_path(table)
        now_ts = int(datetime.now(timezone.utc).timestamp())

        if self._supports_parquet:
            rows = []
            for point in points:
                record = point.to_record()
                timestamp = pd.Timestamp(record["timestamp"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
                else:
                    timestamp = timestamp.tz_convert("UTC")
                record["timestamp"] = timestamp
                rows.append(record)
            df = pd.DataFrame(rows)
            df.to_parquet(dataset_path / f"batch-{now_ts}.parquet", index=False)
        else:
            payload = []
            for point in points:
                record = point.to_record()
                ts = record["timestamp"]
                if isinstance(ts, datetime):
                    record["timestamp"] = ts.astimezone(timezone.utc).isoformat()
                payload.append(record)
            (dataset_path / f"batch-{now_ts}.jsonl").write_text(
                "\n".join(json.dumps(row, default=str) for row in payload),
                encoding="utf-8",
            )

    def read_range(
        self,
        table: str,
        start: datetime,
        end: datetime,
        symbol: str | None = None,
    ) -> Sequence[Mapping[str, object]]:
        dataset_path = self._dataset_path(table)
        if self._supports_parquet:
            frames: List[pd.DataFrame] = []
            for file in dataset_path.glob("*.parquet"):
                frames.append(pd.read_parquet(file))
            if not frames:
                return []
            df = pd.concat(frames, ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            if symbol:
                mask &= df["symbol"] == symbol
            filtered = df.loc[mask]
            return [
                {
                    **{k: v for k, v in row.items() if k not in {"timestamp"}},
                    "timestamp": row["timestamp"].to_pydatetime(),
                }
                for row in filtered.to_dict(orient="records")
            ]

        rows: List[Mapping[str, object]] = []
        for file in dataset_path.glob("*.jsonl"):
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                raw_ts = payload.get("timestamp")
                if isinstance(raw_ts, str):
                    ts = datetime.fromisoformat(raw_ts)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    payload["timestamp"] = ts.astimezone(timezone.utc)
                rows.append(payload)

        if not rows:
            return []

        filtered_rows = []
        for row in rows:
            ts = row["timestamp"]
            if not (start <= ts <= end):
                continue
            if symbol and row.get("symbol") != symbol:
                continue
            filtered_rows.append(row)
        return filtered_rows

    @staticmethod
    def _detect_parquet_support() -> bool:
        if pd_parquet is None:
            return False
        try:  # pragma: no cover - detection only
            pd_parquet.get_engine("auto")
            return True
        except (ImportError, ValueError):
            return False
