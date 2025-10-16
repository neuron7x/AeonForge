"""Prometheus exporter for Phase 3 CBC-Ω² RCT monitoring.

The exporter exposes enrolment, adherence, and safety metrics so that Grafana
can display near real-time dashboards. It is designed to run alongside the data
collection microservice and ingest summarised events from Kafka or REST hooks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)


ENROLLMENT = Gauge(
    "cbc_omega2_enrollment_total",
    "Number of participants enrolled in the Phase 3 RCT",
    labelnames=("site", "arm"),
)

ADHERENCE = Gauge(
    "cbc_omega2_weekly_adherence",
    "Rolling 7-day adherence rate",
    labelnames=("cohort",),
)

ADVERSE_EVENTS = Counter(
    "cbc_omega2_adverse_event_total",
    "Cumulative adverse events logged during the trial",
    labelnames=("severity", "site"),
)

RECOVERY_INDEX = Histogram(
    "cbc_omega2_recovery_index",
    "Distribution of composite recovery index deltas",
    buckets=(-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5),
    labelnames=("cohort",),
)

EXPORT_LATENCY = Histogram(
    "cbc_omega2_export_latency_seconds",
    "Latency of processing aggregated event payloads",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)


@dataclass
class TrialEvent:
    """Canonical event structure used by the exporter."""

    kind: str
    payload: Dict[str, object] = field(default_factory=dict)


class TrialMetricsExporter:
    """Handles translation from trial events into Prometheus metrics."""

    def __init__(self) -> None:
        self._enrolled: Dict[tuple[str, str], int] = {}

    def process_events(self, events: Iterable[TrialEvent], latency: Optional[float] = None) -> None:
        """Ingest a batch of trial events and update metrics."""

        for event in events:
            if event.kind == "enrollment":
                self._handle_enrollment(event.payload)
            elif event.kind == "adherence":
                self._handle_adherence(event.payload)
            elif event.kind == "adverse_event":
                self._handle_adverse_event(event.payload)
            elif event.kind == "recovery_delta":
                self._handle_recovery_delta(event.payload)
            else:
                logger.warning("Unknown event type: %s", event.kind)

        if latency is not None:
            EXPORT_LATENCY.observe(latency)

    def _handle_enrollment(self, payload: Dict[str, object]) -> None:
        site = str(payload.get("site", "unknown"))
        arm = str(payload.get("arm", "unspecified"))
        delta = int(payload.get("delta", 1))
        key = (site, arm)
        self._enrolled[key] = self._enrolled.get(key, 0) + delta
        ENROLLMENT.labels(site=site, arm=arm).set(self._enrolled[key])

    def _handle_adherence(self, payload: Dict[str, object]) -> None:
        cohort = str(payload.get("cohort", "overall"))
        rate = float(payload.get("rate", 0.0))
        ADHERENCE.labels(cohort=cohort).set(rate)

    def _handle_adverse_event(self, payload: Dict[str, object]) -> None:
        severity = str(payload.get("severity", "unspecified"))
        site = str(payload.get("site", "unknown"))
        ADVERSE_EVENTS.labels(severity=severity, site=site).inc()

    def _handle_recovery_delta(self, payload: Dict[str, object]) -> None:
        cohort = str(payload.get("cohort", "overall"))
        delta = float(payload.get("delta", 0.0))
        RECOVERY_INDEX.labels(cohort=cohort).observe(delta)


def load_events(path: Path) -> Iterable[TrialEvent]:
    """Load trial events from a JSON lines file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            yield TrialEvent(kind=str(data["kind"]), payload=dict(data.get("payload", {})))


def run_exporter(event_source: Optional[Path] = None, port: int = 9400) -> None:
    """Start the Prometheus exporter and optionally replay historical events."""

    logging.basicConfig(level=logging.INFO)
    start_http_server(port)
    exporter = TrialMetricsExporter()

    if event_source is not None:
        exporter.process_events(load_events(event_source))
        logger.info("Replayed events from %s", event_source)

    logger.info("Exporter listening on port %d", port)

    # Placeholder loop for integration with the actual streaming source.
    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Exporter shutting down")


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Start the Phase 3 Prometheus exporter")
    parser.add_argument("--port", type=int, default=9400, help="Port for the metrics HTTP server")
    parser.add_argument("--replay", type=Path, help="Optional JSONL file to replay on startup")
    args = parser.parse_args()

    run_exporter(event_source=args.replay, port=args.port)


if __name__ == "__main__":
    _cli()
