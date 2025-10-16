from __future__ import annotations

import json
import importlib
import types
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence

import pandas as pd
import pytest

from agi_core.integration.neurocrowd import (
    BiometricProtocol,
    NeuroCrowd,
    StrategyFactoryProtocol,
    StrategyProtocol,
    _default_cli_runner,
    _default_data_loader,
)


@dataclass
class FakeContent:
    path: str
    type: str = "file"


class FakeRepo:
    def __init__(self, name: str, files: Sequence[FakeContent]):
        self.name = name
        self._files = list(files)

    def get_contents(self, path: str):  # pragma: no cover - exercised indirectly
        assert path == ""
        return list(self._files)


class FakeGithub:
    def __init__(self, repo: FakeRepo):
        self._repo = repo

    def get_repo(self, path: str) -> FakeRepo:
        assert path == self._repo.name
        return self._repo


class DummyStrategy(StrategyProtocol):
    def __init__(self, returned: pd.DataFrame):
        self.returned = returned
        self.calls = []

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(data)
        return self.returned


class DummyStrategyFactory(StrategyFactoryProtocol):
    def __init__(self, strategy: DummyStrategy):
        self.strategy = strategy
        self.args = []

    def __call__(self, project_name: str, requirements: Sequence[str]) -> DummyStrategy:
        self.args.append((project_name, list(requirements)))
        return self.strategy


class DummyBiometric(BiometricProtocol):
    def __init__(self, eoi: float):
        self.eoi = eoi
        self.calls = []

    def compute_eoi(self, user_state: Mapping[str, float]) -> float:
        self.calls.append(dict(user_state))
        return self.eoi


class DummyMemory:
    def __init__(self) -> None:
        self.calls = []

    def add_episode(self, name: str, content: str, timestamp: pd.Timestamp) -> None:
        self.calls.append((name, content, timestamp))


def _make_dataframe(signal: str = "Buy") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01T00:00:00Z"]),
            "symbol": ["TEST"],
            "signal": [signal],
            "confidence": [0.9],
        }
    )


def _make_neurocrowd(
    *,
    strategy_factory: StrategyFactoryProtocol,
    biometric: BiometricProtocol,
    cli_runner: Callable[[str], pd.DataFrame],
    data_loader: Callable[[], pd.DataFrame],
    memory: DummyMemory | None = None,
) -> NeuroCrowd:
    repo = FakeRepo("example/project", [FakeContent(path="README.md")])
    github = FakeGithub(repo)
    strategies: Dict[str, StrategyFactoryProtocol] = {"kuramoto": strategy_factory}
    return NeuroCrowd(
        github_token=None,
        repo_url="https://github.com/example/project",
        github_client=github,
        strategies=strategies,
        biometric=biometric,
        cli_runner=cli_runner,
        data_loader=data_loader,
        memory=memory,
    )


def test_generate_prompt_includes_repo_context():
    strategy = DummyStrategy(_make_dataframe())
    factory = DummyStrategyFactory(strategy)
    biometric = DummyBiometric(2.0)
    cli = lambda intent: _make_dataframe("Hold")
    loader = lambda: _make_dataframe()
    nc = _make_neurocrowd(
        strategy_factory=factory,
        biometric=biometric,
        cli_runner=cli,
        data_loader=loader,
    )

    prompt = nc.generate_prompt("add kuramoto strategy")

    assert prompt["task"] == "add kuramoto strategy"
    assert "README.md" in prompt["context"]
    assert "numpy/pandas" in prompt["requirements"]


def test_execute_task_prefers_ai_when_eoi_high():
    returned = _make_dataframe("Sell")
    strategy = DummyStrategy(returned)
    factory = DummyStrategyFactory(strategy)
    biometric = DummyBiometric(2.1)
    memory = DummyMemory()

    loader_called = []

    def loader():
        loader_called.append(True)
        return _make_dataframe()

    nc = _make_neurocrowd(
        strategy_factory=factory,
        biometric=biometric,
        cli_runner=lambda intent: _make_dataframe("Hold"),
        data_loader=loader,
        memory=memory,
    )

    output = nc.execute_task("build strategy kuramoto", {"hrv_sdnn": 50})

    assert output.equals(returned)
    assert loader_called
    assert factory.args[0][0] == "example/project"
    assert factory.strategy.calls  # data passed to strategy

    names = [name for name, _, _ in memory.calls]
    assert names == ["biometric", "intent", "execution"]

    payloads = {name: json.loads(content) for name, content, _ in memory.calls}
    assert payloads["biometric"] == {"intent": "build strategy kuramoto", "eoi": 2.1}
    assert payloads["intent"]["context"] == ["README.md"]
    assert payloads["execution"]["mode"] == "ai"
    assert payloads["execution"]["signals"][0]["signal"] == "Sell"


def test_execute_task_uses_cli_when_eoi_low():
    strategy = DummyStrategy(_make_dataframe("Sell"))
    factory = DummyStrategyFactory(strategy)
    biometric = DummyBiometric(0.5)

    cli_called = []

    def cli(intent: str) -> pd.DataFrame:
        cli_called.append(intent)
        return _make_dataframe("Hold")

    nc = _make_neurocrowd(
        strategy_factory=factory,
        biometric=biometric,
        cli_runner=cli,
        data_loader=lambda: _make_dataframe(),
    )

    output = nc.execute_task("evaluate strategy kuramoto", {"hrv_sdnn": 50})

    assert cli_called == ["evaluate strategy kuramoto"]
    assert output["signal"].iat[0] == "Hold"


def test_critic_review_flags_missing_columns():
    frame = pd.DataFrame({"symbol": ["TEST"]})
    assert "Missing required columns" in NeuroCrowd.critic_review(frame)


def test_critic_review_handles_non_dataframe_and_invalid_values():
    assert NeuroCrowd.critic_review("bad") == ["Output must be pd.DataFrame"]

    frame = _make_dataframe(signal="INVALID")
    assert "Invalid signal values" in NeuroCrowd.critic_review(frame)


def test_iterate_stops_after_successful_output():
    strategy = DummyStrategy(_make_dataframe("Sell"))
    factory = DummyStrategyFactory(strategy)
    biometric = DummyBiometric(2.0)
    memory = DummyMemory()
    nc = _make_neurocrowd(
        strategy_factory=factory,
        biometric=biometric,
        cli_runner=lambda intent: _make_dataframe("Hold"),
        data_loader=lambda: _make_dataframe("Sell"),
        memory=memory,
    )

    results = iter(
        [
            pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01"]), "symbol": ["T"], "signal": ["Buy"]}),
            _make_dataframe("Sell"),
        ]
    )

    def fake_execute(self, intent: str, user_state):
        return next(results)

    nc.execute_task = fake_execute.__get__(nc, NeuroCrowd)  # type: ignore[assignment]

    output = nc.iterate("improve strategy kuramoto", {"hrv_sdnn": 30})

    assert output["signal"].iat[0] == "Sell"
    names = [name for name, _, _ in memory.calls]
    assert "flaws" in names
    assert names[-1] == "success"

    flaw_payloads = [json.loads(content) for name, content, _ in memory.calls if name == "flaws"]
    assert flaw_payloads
    assert "Missing required columns" in flaw_payloads[0]["flaws"][0]
    assert "failure" not in names


def test_iterate_raises_when_quality_not_met():
    strategy = DummyStrategy(_make_dataframe("Sell"))
    factory = DummyStrategyFactory(strategy)
    biometric = DummyBiometric(2.0)
    memory = DummyMemory()
    nc = _make_neurocrowd(
        strategy_factory=factory,
        biometric=biometric,
        cli_runner=lambda intent: _make_dataframe("Hold"),
        data_loader=lambda: _make_dataframe("Sell"),
        memory=memory,
    )

    def always_bad(self, intent: str, user_state):
        return pd.DataFrame({"symbol": ["X"]})

    nc.execute_task = always_bad.__get__(nc, NeuroCrowd)  # type: ignore[assignment]

    with pytest.raises(ValueError):
        nc.iterate("bad strategy", {"hrv_sdnn": 10})

    names = [name for name, _, _ in memory.calls]
    assert names.count("flaws") == 3
    assert names[-1] == "failure"
    failure_payloads = [json.loads(content) for name, content, _ in memory.calls if name == "failure"]
    assert failure_payloads
    assert failure_payloads[0]["iterations"] == 3


def test_default_helpers_return_expected_schema():
    cli_frame = _default_cli_runner("intent")
    loader_frame = _default_data_loader()

    expected_columns = ["timestamp", "symbol", "signal", "confidence"]
    assert list(cli_frame.columns) == expected_columns
    assert list(loader_frame.columns) == expected_columns
    assert loader_frame.empty


def test_constructor_requires_default_strategy_present():
    strategy = DummyStrategy(_make_dataframe())
    factory = DummyStrategyFactory(strategy)
    repo = FakeRepo("example/project", [FakeContent(path="README.md")])
    github = FakeGithub(repo)

    with pytest.raises(ValueError):
        NeuroCrowd(
            github_token=None,
            repo_url="https://github.com/example/project",
            github_client=github,
            strategies={"other": factory},
            biometric=DummyBiometric(2.0),
            cli_runner=lambda intent: _make_dataframe(),
            data_loader=lambda: _make_dataframe(),
        )


def test_extract_repo_path_handles_variants():
    assert NeuroCrowd._extract_repo_path("example/project") == "example/project"

    with pytest.raises(ValueError):
        NeuroCrowd._extract_repo_path("https://github.com/")


def test_load_default_strategies_handles_missing_register(monkeypatch):
    class DummyModule:
        register_strategies = None

    monkeypatch.setattr(importlib, "import_module", lambda name: DummyModule())

    assert NeuroCrowd._load_default_strategies() == {}


def test_load_default_biometric_instantiates_interface(monkeypatch):
    init_calls = []

    class PatchedModule:
        class BiometricInterface:
            def __init__(self):
                init_calls.append("init")

            def compute_eoi(self, user_state: Mapping[str, float]) -> float:
                return sum(user_state.values())

    monkeypatch.setattr(importlib, "import_module", lambda name: PatchedModule())

    interface = NeuroCrowd._load_default_biometric()
    assert init_calls == ["init"]
    assert interface.compute_eoi({"a": 1.0, "b": 2.0}) == 3.0


def test_select_strategy_falls_back_to_default():
    strategy = DummyStrategy(_make_dataframe("Sell"))
    default_factory = DummyStrategyFactory(strategy)
    biometric = DummyBiometric(2.0)
    nc = _make_neurocrowd(
        strategy_factory=default_factory,
        biometric=biometric,
        cli_runner=lambda intent: _make_dataframe("Hold"),
        data_loader=lambda: _make_dataframe("Sell"),
    )

    factory = nc._select_strategy("unknown task")
    assert factory is default_factory


def test_extract_strategy_name_without_keyword():
    assert NeuroCrowd._extract_strategy_name("refactor module") == ""


def test_json_default_handles_unknown_types():
    value = NeuroCrowd._json_default(object())
    assert isinstance(value, str)


def test_constructor_import_path(monkeypatch):
    strategy_output = _make_dataframe("Sell")
    factory = DummyStrategyFactory(DummyStrategy(strategy_output))

    def fake_import(name: str):
        if name == "github":
            return types.SimpleNamespace(
                Github=lambda token: types.SimpleNamespace(
                    token=token,
                    get_repo=lambda path: FakeRepo(path, [FakeContent(path="README.md")]),
                )
            )
        if name == "core.strategies":
            return types.SimpleNamespace(register_strategies=lambda: {"kuramoto": factory})
        if name == "biometric_interface":
            class Interface:
                def __init__(self):
                    self.calls = []

                def compute_eoi(self, user_state: Mapping[str, float]) -> float:
                    self.calls.append(dict(user_state))
                    return 2.0

            return types.SimpleNamespace(BiometricInterface=Interface)
        raise AssertionError(f"Unexpected module: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    nc = NeuroCrowd(github_token="token123", repo_url="https://github.com/example/project")

    assert isinstance(nc.repo, FakeRepo)
    output = nc.execute_task("build strategy kuramoto", {"stress": 1.0})

    assert output.equals(strategy_output)
    assert factory.args[0][0] == "example/project"


def test_constructor_requires_token_when_client_missing():
    with pytest.raises(ValueError):
        NeuroCrowd(github_token=None, repo_url="https://github.com/example/project")

