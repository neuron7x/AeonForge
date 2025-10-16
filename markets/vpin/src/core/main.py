"""Volume-Synchronized Probability of Informed Trading (VPIN) utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Trade:
    """Normalized trade representation used by :class:`VPINCalculator`."""

    timestamp: datetime
    price: float
    volume: float
    side: str

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        normalized_side = self.side.lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("Trade side must be 'buy' or 'sell'")
        object.__setattr__(self, "side", normalized_side)
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))
        else:
            object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        if self.volume <= 0:
            raise ValueError("Trade volume must be positive")
        if self.price <= 0:
            raise ValueError("Trade price must be positive")


@dataclass
class VPINBucket:
    """Internal accumulation bucket used to build VPIN windows."""

    volume_limit: float
    trades: List[Trade] = field(default_factory=list)
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    @property
    def imbalance(self) -> float:
        return abs(self.buy_volume - self.sell_volume)

    @property
    def is_complete(self) -> bool:
        return self.buy_volume + self.sell_volume >= self.volume_limit

    def add_trade(self, trade: Trade) -> None:
        remaining = self.volume_limit - (self.buy_volume + self.sell_volume)
        fill_volume = min(remaining, trade.volume)
        if fill_volume <= 0:
            return
        if trade.side == "buy":
            self.buy_volume += fill_volume
        else:
            self.sell_volume += fill_volume
        self.trades.append(trade)

    def spillover(self, trade: Trade) -> Trade | None:
        consumed = self.volume_limit - (self.buy_volume + self.sell_volume)
        if consumed >= trade.volume:
            return None
        residual_volume = trade.volume - consumed
        return Trade(
            timestamp=trade.timestamp,
            price=trade.price,
            volume=residual_volume,
            side=trade.side,
        )


class VPINCalculator:
    """Streaming VPIN calculator.

    The calculator consumes individual trades and produces VPIN values for a
    sliding window of buckets. Each bucket is defined by a target cumulative
    volume (``bucket_volume``) which is typically derived from historical
    average traded volume.
    """

    def __init__(self, bucket_volume: float, window: int = 50) -> None:
        if bucket_volume <= 0:
            raise ValueError("bucket_volume must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self.bucket_volume = bucket_volume
        self.window = window
        self._buckets: List[VPINBucket] = []
        self._active_bucket = VPINBucket(bucket_volume)

    def _finalize_active_bucket(self) -> None:
        if self._active_bucket.buy_volume + self._active_bucket.sell_volume == 0:
            return
        self._buckets.append(self._active_bucket)
        if len(self._buckets) > self.window:
            self._buckets.pop(0)
        self._active_bucket = VPINBucket(self.bucket_volume)

    def consume(self, trades: Iterable[Trade]) -> List[float]:
        """Consume trades and return newly generated VPIN values."""

        vpin_values: List[float] = []
        for trade in trades:
            residual: Trade | None = trade
            while residual is not None:
                self._active_bucket.add_trade(residual)
                if self._active_bucket.is_complete:
                    self._finalize_active_bucket()
                    if len(self._buckets) == self.window:
                        vpin_values.append(self.current_vpin)
                residual = self._active_bucket.spillover(residual)
        return vpin_values

    @property
    def current_vpin(self) -> float:
        if not self._buckets:
            return 0.0
        total_volume = self.bucket_volume * len(self._buckets)
        total_imbalance = sum(bucket.imbalance for bucket in self._buckets)
        return total_imbalance / total_volume

    def snapshot(self) -> Sequence[VPINBucket]:
        return list(self._buckets)


def compute_vpin(trades: Sequence[Trade], bucket_volume: float, window: int = 50) -> float:
    """Utility for batch VPIN computation."""

    calculator = VPINCalculator(bucket_volume=bucket_volume, window=window)
    calculator.consume(trades)
    return calculator.current_vpin
