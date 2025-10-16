"""Fractal coordination primitives for aeonforge.

This package provides a scalable, energy-aware execution substrate that mirrors
the fractal architecture used by the project.  The implementation emphasises
composability so that modules can be reused across levels of abstraction while
remaining lightweight enough for high-frequency control loops.
"""

from .node import FractalNode, FractalLeaf, FractalComposite, FractalSignal, FractalContext
from .scheduler import EnergyAdaptiveScheduler
from .network import FractalCortex

__all__ = [
    "FractalNode",
    "FractalLeaf",
    "FractalComposite",
    "FractalSignal",
    "FractalContext",
    "EnergyAdaptiveScheduler",
    "FractalCortex",
]
