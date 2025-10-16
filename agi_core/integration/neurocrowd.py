"""Biometric-calibrated delegation flow for NeuroCrowd.

This module implements a light-weight variant of the NeuroCrowd control loop
described in the upstream TradePulse project.  The original reference snippet
is reproduced in the task description and revolves around three key concepts:

*   Repository inspection via the GitHub API to seed prompting metadata.
*   Strategy selection using a registry that can instantiate signal generators.
*   Biometric calibration (expressed as an "engagement overload index", eOI)
    to decide whether the AI should take the lead or defer to a human-facing
    CLI.

The implementation below keeps those semantics while providing a robust
interface for the agi-core codebase:

*   Every external dependency (GitHub, biometric interface, strategy registry
    and CLI runner) can be injected, which makes the class straightforward to
    unit test.
*   Failures while inspecting the repository are surfaced as informative
    ``RuntimeError`` instances so downstream tooling can react accordingly.
*   Returned signals are validated through :meth:`critic_review` to guarantee
    the consumer receives a canonical ``pandas.DataFrame`` with the expected
    schema.

The accompanying tests demonstrate the decision-making loop for both AI-driven
and human-driven paths, as well as the iterative quality gate.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Protocol, Sequence
from urllib.parse import urlsplit

import pandas as pd


class StrategyProtocol(Protocol):
    """Minimal protocol for a strategy capable of producing trading signals."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return signals for the provided data frame."""


class StrategyFactoryProtocol(Protocol):
    """Callable protocol that instantiates :class:`StrategyProtocol`."""

    def __call__(self, project_name: str, requirements: Sequence[str]) -> StrategyProtocol:
        """Build a strategy bound to ``project_name`` and ``requirements``."""


class BiometricProtocol(Protocol):
    """Protocol describing the minimal biometric interface we rely on."""

    def compute_eoi(self, user_state: Mapping[str, float]) -> float:
        """Return the engagement overload index based on ``user_state`` metrics."""


class RepoContentProtocol(Protocol):
    """Subset of the GitHub file content interface used by :class:`NeuroCrowd`."""

    path: str
    type: str


class RepoProtocol(Protocol):
    """Minimal GitHub repository contract required for analysis."""

    name: str

    def get_contents(self, path: str) -> Sequence[RepoContentProtocol]:
        """Return repository contents for ``path``."""


class GithubClientProtocol(Protocol):
    """Partial GitHub client protocol used to obtain repositories."""

    def get_repo(self, repository: str) -> RepoProtocol:
        """Return a repository handle."""


class MemoryProtocol(Protocol):
    """Minimal protocol required from a semantic memory backend."""

    def add_episode(self, name: str, content: str, timestamp: pd.Timestamp) -> None:
        """Persist an episode describing ``name`` with ``content``."""


CliRunner = Callable[[str], pd.DataFrame]
DataLoader = Callable[[], pd.DataFrame]


def _default_cli_runner(intent: str) -> pd.DataFrame:
    """Fallback CLI runner returning an empty, schema-compliant frame."""

    return pd.DataFrame(columns=["timestamp", "symbol", "signal", "confidence"])


def _default_data_loader() -> pd.DataFrame:
    """Fallback data loader used when no external data source is provided."""

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([]),
            "symbol": pd.Series(dtype="object"),
            "signal": pd.Series(dtype="object"),
            "confidence": pd.Series(dtype="float64"),
        }
    )


@dataclass
class RepoStructure:
    """Lightweight representation of a repository's top-level structure."""

    files: List[str]


class NeuroCrowd:
    """Core system for AI-driven NeuroCrowd with biometric calibration."""

    def __init__(
        self,
        github_token: Optional[str],
        repo_url: str,
        *,
        github_client: Optional[GithubClientProtocol] = None,
        strategies: Optional[Mapping[str, StrategyFactoryProtocol]] = None,
        default_strategy: str = "kuramoto",
        biometric: Optional[BiometricProtocol] = None,
        cli_runner: Optional[CliRunner] = None,
        data_loader: Optional[DataLoader] = None,
        eoi_threshold: float = 1.5,
        memory: Optional[MemoryProtocol] = None,
    ) -> None:
        if github_client is None:
            if github_token is None:
                raise ValueError("github_token must be provided when github_client is omitted")
            github_module = importlib.import_module("github")
            github_client = getattr(github_module, "Github")(github_token)

        self._github = github_client
        self._repo = github_client.get_repo(self._extract_repo_path(repo_url))
        self._strategies: Dict[str, StrategyFactoryProtocol] = dict(strategies or self._load_default_strategies())
        self._default_strategy = default_strategy
        self._biometric = biometric or self._load_default_biometric()
        self._cli_runner = cli_runner or _default_cli_runner
        self._data_loader = data_loader or _default_data_loader
        self._eoi_threshold = eoi_threshold
        self._memory = memory

        if self._default_strategy not in self._strategies:
            raise ValueError(f"Default strategy '{self._default_strategy}' not found in registry")

    @staticmethod
    def _extract_repo_path(repo_url: str) -> str:
        """Extract the ``owner/name`` component from ``repo_url``."""

        parsed = urlsplit(repo_url)
        if not parsed.netloc:
            return repo_url
        path = parsed.path.strip("/")
        if not path:
            raise ValueError(f"Unable to determine repository from URL: {repo_url}")
        return path

    @staticmethod
    def _load_default_strategies() -> Mapping[str, StrategyFactoryProtocol]:
        """Load strategy registry from ``core.strategies`` when available."""

        module = importlib.import_module("core.strategies")
        register = getattr(module, "register_strategies", None)
        if register is None:
            return {}
        return register()

    @staticmethod
    def _load_default_biometric() -> BiometricProtocol:
        """Instantiate the biometric interface declared in ``biometric_interface``."""

        module = importlib.import_module("biometric_interface")
        interface = getattr(module, "BiometricInterface")
        return interface()

    @property
    def repo(self) -> RepoProtocol:
        """Expose the GitHub repository handle for consumers and tests."""

        return self._repo

    def _analyze_repo(self) -> RepoStructure:
        """Inspect the top-level structure of the configured repository."""

        try:
            contents = self._repo.get_contents("")
        except Exception as exc:  # pragma: no cover - defensive branch
            raise RuntimeError("Failed to retrieve repository contents") from exc

        files = [content.path for content in contents if getattr(content, "type", "") == "file"]
        return RepoStructure(files=files)

    def generate_prompt(self, intent: str) -> Dict[str, Sequence[str]]:
        """Generate a structured prompt dictionary for downstream strategies."""

        repo_structure = self._analyze_repo()
        prompt = {
            "task": intent,
            "requirements": ["Python", "numpy/pandas", "core/indicators integration"],
            "output_format": "pd.DataFrame with timestamp, symbol, signal, confidence",
            "criteria": ["coverage >= 85%", "ruff linting", "mypy typing", "handle NaN"],
            "context": repo_structure.files,
        }
        self._record_episode(
            "intent",
            {
                "intent": intent,
                "requirements": prompt["requirements"],
                "context": repo_structure.files,
            },
        )
        return prompt

    def _select_strategy(self, intent: str) -> StrategyFactoryProtocol:
        """Return the strategy factory matching ``intent`` or the default."""

        strategy_name = self._extract_strategy_name(intent)
        factory = self._strategies.get(strategy_name)
        if factory is not None:
            return factory

        default_factory = self._strategies.get(self._default_strategy)
        if default_factory is None:  # pragma: no cover - guarded by __init__
            raise ValueError("Default strategy is not available")
        return default_factory

    @staticmethod
    def _extract_strategy_name(intent: str) -> str:
        """Heuristically derive a strategy name from ``intent``."""

        parts = intent.lower().split("strategy")
        if len(parts) < 2:
            return ""
        return parts[1].strip().split()[0]

    def execute_task(self, intent: str, user_state: Mapping[str, float]) -> pd.DataFrame:
        """Execute ``intent`` using biometric calibration to route work."""

        eoi = self._biometric.compute_eoi(user_state)
        self._record_episode("biometric", {"intent": intent, "eoi": eoi})
        if eoi >= self._eoi_threshold:
            prompt = self.generate_prompt(intent)
            factory = self._select_strategy(intent)
            strategy = factory(self._repo.name, prompt["requirements"])
            data = self._data_loader()
            output = strategy.generate_signals(data)
            self._record_episode(
                "execution",
                {
                    "intent": intent,
                    "mode": "ai",
                    "strategy": self._extract_strategy_name(intent) or self._default_strategy,
                    "signals": output,
                },
            )
            return output

        result = self._cli_runner(intent)
        self._record_episode(
            "execution",
            {
                "intent": intent,
                "mode": "human",
                "strategy": "cli",
                "signals": result if isinstance(result, pd.DataFrame) else None,
            },
        )
        return result

    @staticmethod
    def critic_review(output: pd.DataFrame) -> List[str]:
        """Validate the produced signals and return a list of detected flaws."""

        flaws: List[str] = []
        if not isinstance(output, pd.DataFrame):
            return ["Output must be pd.DataFrame"]

        required_columns = {"timestamp", "symbol", "signal", "confidence"}
        if not required_columns.issubset(output.columns):
            flaws.append("Missing required columns")
        else:
            invalid_signals = output["signal"].isin(["Buy", "Sell", "Hold"]).all()
            if not invalid_signals:
                flaws.append("Invalid signal values")

        return flaws

    def iterate(self, intent: str, user_state: Mapping[str, float], max_iterations: int = 3) -> pd.DataFrame:
        """Iteratively run :meth:`execute_task` until the critic passes or fail."""

        updated_intent = intent
        for attempt in range(max_iterations):
            output = self.execute_task(updated_intent, user_state)
            flaws = self.critic_review(output)
            if not flaws:
                self._record_episode(
                    "success",
                    {
                        "intent": updated_intent,
                        "iterations": attempt + 1,
                        "signals": output,
                    },
                )
                return output
            updated_intent = f"{updated_intent}. Fix: {', '.join(flaws)}"
            self._record_episode(
                "flaws",
                {
                    "intent": updated_intent,
                    "iteration": attempt + 1,
                    "flaws": flaws,
                },
            )

        self._record_episode(
            "failure",
            {
                "intent": updated_intent,
                "iterations": max_iterations,
            },
        )
        raise ValueError("Failed to meet quality criteria after max iterations")

    def _record_episode(self, name: str, payload: Mapping[str, Any] | Sequence[Any] | str | None) -> None:
        """Persist an episode to the configured semantic memory if available."""

        if self._memory is None or payload is None:
            return

        timestamp = pd.Timestamp.now(tz="UTC")
        content = payload if isinstance(payload, str) else json.dumps(payload, default=self._json_default, ensure_ascii=False)
        self._memory.add_episode(name=name, content=content, timestamp=timestamp)

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Serialize complex objects (DataFrames, timestamps) for JSON logging."""

        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return str(value)


__all__ = [
    "NeuroCrowd",
    "StrategyProtocol",
    "StrategyFactoryProtocol",
    "BiometricProtocol",
    "MemoryProtocol",
]

