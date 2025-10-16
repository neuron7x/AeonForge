from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import yaml


@dataclass
class SLO:
    name: str
    description: str
    sli: str
    target: float
    unit: str | None = None
    warning: float | None = None
    critical: float | None = None


@dataclass
class SLOEvaluation:
    slo: SLO
    value: float
    state: str


def load_slos(path: str | Path) -> List[SLO]:
    data = yaml.safe_load(Path(path).read_text())
    objectives = data.get("objectives", [])
    slos: List[SLO] = []
    for entry in objectives:
        alerting = entry.get("alerting", {})
        slos.append(
            SLO(
                name=entry["name"],
                description=entry.get("description", ""),
                sli=entry["sli"],
                target=float(entry["target"]),
                unit=entry.get("unit"),
                warning=float(alerting["warning"]) if "warning" in alerting else None,
                critical=float(alerting["critical"]) if "critical" in alerting else None,
            )
        )
    return slos


def evaluate_slos(slos: Iterable[SLO], measurements: Mapping[str, float]) -> List[SLOEvaluation]:
    evaluations: List[SLOEvaluation] = []
    for slo in slos:
        value = float(measurements.get(slo.sli, 0.0))
        state = "ok"
        higher_is_better = True
        if any(keyword in slo.sli for keyword in ("latency", "lag", "duration", "error", "burn")):
            higher_is_better = False
        elif any(keyword in slo.sli for keyword in ("success", "availability", "uptime", "throughput", "rate", "accuracy")):
            higher_is_better = True
        if slo.critical is not None:
            if higher_is_better:
                if value < slo.critical:
                    state = "critical"
            else:
                if value > slo.critical:
                    state = "critical"
        if state != "critical" and slo.warning is not None:
            if higher_is_better:
                if value < slo.warning:
                    state = "warning"
            else:
                if value > slo.warning:
                    state = "warning"
        if state == "ok":
            if higher_is_better:
                if value < slo.target:
                    state = "breaching"
            else:
                if value > slo.target:
                    state = "breaching"
        evaluations.append(SLOEvaluation(slo=slo, value=value, state=state))
    return evaluations
