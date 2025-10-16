# NeuroCrowd: Biometric-Calibrated Delegation

The lightweight NeuroCrowd control loop ships as
[`agi_core/integration/neurocrowd.py`](../agi_core/integration/neurocrowd.py) and is
covered by [`tests/test_neurocrowd.py`](../tests/test_neurocrowd.py).  This
section documents the current production snapshot after integrating biometric
calibration, injectable dependencies, and semantic memory logging.

## Core workflow

| Stage | Implementation | Notes |
| --- | --- | --- |
| Repository inspection | `_analyze_repo` builds a `RepoStructure` from the GitHub API surface, exposing file paths for prompt context.【F:agi_core/integration/neurocrowd.py†L143-L184】 | The helper is exercised indirectly via `generate_prompt`, which asserts that repository metadata is embedded in the prompt payload.【F:tests/test_neurocrowd.py†L69-L104】 |
| Strategy orchestration | `_load_default_strategies` and `_select_strategy` compose the strategy registry with overrideable defaults.【F:agi_core/integration/neurocrowd.py†L116-L166】 | Dummy factories in `tests/test_neurocrowd.py` verify that the correct project name and requirements are forwarded when eOI is high.【F:tests/test_neurocrowd.py†L106-L165】 |
| Biometric calibration | `execute_task` delegates to AI or CLI runners based on the biometric interface’s engagement overload index (eOI).【F:agi_core/integration/neurocrowd.py†L185-L239】 | The test suite covers both the AI-first path (`eoi >= threshold`) and the human-first path (`eoi < threshold`).【F:tests/test_neurocrowd.py†L106-L165】【F:tests/test_neurocrowd.py†L168-L198】 |
| Quality gate | `critic_review` enforces the canonical `timestamp/symbol/signal/confidence` schema and acceptable signal values before exiting the iteration loop.【F:agi_core/integration/neurocrowd.py†L241-L293】 | Negative-path tests assert the error messages for missing columns, wrong types, and invalid signals.【F:tests/test_neurocrowd.py†L200-L233】 |
| Iterative refinement | `iterate` records semantic episodes (when memory is provided) and appends remediation instructions until the critic passes or the retry budget is exhausted.【F:agi_core/integration/neurocrowd.py†L295-L343】 | Tests simulate both success-after-retry and failure-after-exhaustion scenarios.【F:tests/test_neurocrowd.py†L235-L360】 |

### Dependency injection surface

The constructor accepts overrides for the GitHub client, biometric interface,
strategy registry, CLI runner, data loader, and semantic memory adapter.  This
allows deterministic unit tests and production swaps without editing the class
body.【F:agi_core/integration/neurocrowd.py†L92-L141】  The default adapters use
`core.strategies`, `biometric_interface`, and an empty dataframe loader to keep
the module runnable in isolation.【F:agi_core/integration/neurocrowd.py†L107-L140】

### Semantic memory integration

Whenever a `memory` backend is supplied, NeuroCrowd records structured JSON
payloads for prompts, biometric readings, execution outcomes, and terminal
states.  Tests assert that the correct episode names and payloads are emitted
when operating in AI-driven mode.【F:agi_core/integration/neurocrowd.py†L185-L343】【F:tests/test_neurocrowd.py†L123-L165】  The JSON serializer flattens pandas
frames and timestamps for compatibility with vector stores.【F:agi_core/integration/neurocrowd.py†L345-L371】

## Quality assurance

The following automated checks are executed in CI (`make test`) and guarantee
that the NeuroCrowd surface remains stable:

- `tests/test_neurocrowd.py` covers prompt generation, AI routing, human
  routing, memory logging, critic validation, retry semantics, and helper
  utilities.【F:tests/test_neurocrowd.py†L1-L360】
- `tests/test_biometric.py` validates the biometric engine used by the API,
  ensuring eOI scores and ROC-calibrated thresholds behave as expected before
  NeuroCrowd consumes them via dependency injection.【F:tests/test_biometric.py†L1-L53】

To run the checks locally:

```bash
pip install -e .[qa]
pytest tests/test_neurocrowd.py tests/test_biometric.py
```

The command set mirrors CI and exercises the same code paths invoked in
production deployments.
