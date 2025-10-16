# API v0.3.3

## Fractal Coordination Layer

### `agi_core.fractal.EnergyAdaptiveScheduler`
Adaptive scheduler that distributes energy/compute budgets across a fractal
subtree.  It exposes:

- `update(utilisation: float) -> None`
- `allocate(weights: Iterable[float]) -> np.ndarray`
- `reset() -> None`

### `agi_core.fractal.FractalNode`
Base class for fractal modules.  Subclasses override `_process_local` to produce a
`ProcessResult`.  The node handles routing to children, energy budgeting and
aggregation.

### `agi_core.fractal.FractalComposite`
Composite node combining local heuristics with a dynamic set of children.
Supports custom routing functions for sparse activations.

### `agi_core.fractal.FractalLeaf`
Lightweight adapter around callable modules such as affordance maps or world
models.  Useful for embedding NumPy implementations without additional glue code.

### `agi_core.fractal.FractalCortex`
Top-level coordinator that injects `Signal` objects into the fractal tree and
records execution statistics for monitoring.

### `agi_core.integration.FractalMyceliumBridge`
Allows hybrid deployments where the fractal cortex delegates to the existing
`MycelialIntegration` fabric whenever additional suggestions are required.
