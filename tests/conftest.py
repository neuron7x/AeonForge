"""Pytest configuration for the AGI Core test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure that the project root is available on ``sys.path`` so that the
# ``agi_core`` package can be imported when tests are executed directly from the
# ``tests`` directory (as enforced by ``testpaths`` in ``pytest.ini``).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
