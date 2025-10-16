from __future__ import annotations

from pathlib import Path

from infrastructure.observability.slo import evaluate_slos, load_slos


def test_load_and_evaluate(tmp_path: Path) -> None:
    config = tmp_path / "slos.yaml"
    config.write_text(
        """
service: test
objectives:
  - name: availability
    sli: success_rate
    target: 0.99
    alerting:
      warning: 0.98
      critical: 0.95
  - name: latency
    sli: latency_p95
    target: 0.3
    unit: seconds
    alerting:
      warning: 0.4
      critical: 0.5
"""
    )
    slos = load_slos(config)
    results = evaluate_slos(slos, {"success_rate": 0.97, "latency_p95": 0.45})
    states = {result.slo.name: result.state for result in results}
    assert states["availability"] == "warning"
    assert states["latency"] == "warning"
