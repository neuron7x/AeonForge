from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(slots=True)
class Signal:
    type: str
    state: Optional[np.ndarray]=None
    goal: Optional[np.ndarray]=None
    payload: Optional[dict]=None

@dataclass(slots=True)
class ProcessResult:
    success: bool
    suggestion: Optional[np.ndarray]=None
    confidence: float=0.0
    error: Optional[str]=None
