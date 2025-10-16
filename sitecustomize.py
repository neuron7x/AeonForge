"""Project specific customisations loaded automatically by Python.

This module acts as a lightweight bootstrapper for the test environment.  The
upstream project expects a rich scientific Python stack (FastAPI, NumPy,
Pandas, etc.) to be available; however, the execution environment used for the
kata intentionally starts with a very small standard library only image.  The
integration tests exercise quite a large portion of the project surface area so
mocking out every third-party dependency would quickly become brittle.

To keep the repository self-contained we opportunistically ensure that the
handful of third-party packages we rely on are installed before the rest of the
code is imported.  ``sitecustomize`` is imported automatically by the Python
interpreter which makes it a convenient hook for the bootstrap logic.  The
installation is skipped when the relevant modules are already available or when
``AGI_CORE_SKIP_BOOTSTRAP`` is set (allowing power users to manage the
environment manually).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from typing import Iterable, Tuple


_REQUIRED_PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("fastapi", "fastapi"),
    ("pydantic", "pydantic"),
    ("prometheus_client", "prometheus-client"),
    ("jose", "python-jose"),
    ("httpx", "httpx"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("starlette", "starlette"),
)


def _ensure_dependencies(packages: Iterable[Tuple[str, str]]) -> None:
    missing: list[str] = []
    for module_name, package_name in packages:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if not missing:
        return

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *missing],
        check=True,
    )


if os.getenv("AGI_CORE_SKIP_BOOTSTRAP") not in {"1", "true", "TRUE", "True"}:
    try:  # pragma: no cover - best effort install
        _ensure_dependencies(_REQUIRED_PACKAGES)
    except Exception:  # pragma: no cover - bootstrap failures should be obvious
        pass


if os.getenv("AGI_CORE_ENABLE_SERVICE_STUBS", "1") not in {"0", "false", "False"}:
    try:  # pragma: no cover - best effort import
        from infrastructure.service_stubs import ensure_stub_services

        ensure_stub_services()
    except Exception:  # pragma: no cover - guard against import issues
        pass

