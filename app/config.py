from __future__ import annotations

import os
from types import NoneType
try:  # Python 3.10+
    from types import UnionType
except ImportError:  # pragma: no cover - Python <3.10 compatibility
    UnionType = None  # type: ignore[assignment]
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

try:  # pragma: no cover - exercised implicitly during import
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - activated in minimal envs
    class SettingsConfigDict(dict):
        """Compatibility stub mirroring pydantic-settings configuration object."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    def _coerce_value(value: Any, annotation: Any) -> Any:
        """Best-effort coercion that covers the types we rely on in tests."""

        origin = get_origin(annotation)
        if origin is None:
            if annotation in (str, Any) or annotation is None:
                return value
            if annotation is int:
                return int(value)
            if annotation is float:
                return float(value)
            if annotation is bool:
                if isinstance(value, bool):
                    return value
                return str(value).lower() in {"true", "1", "yes"}
            return value

        if origin is list:
            (item_type,) = get_args(annotation) or (Any,)
            if isinstance(value, str):
                items = [item.strip() for item in value.split(",") if item.strip()]
            else:
                items = list(value)
            return [_coerce_value(item, item_type) for item in items]

        if origin in (Union, UnionType):
            for candidate in get_args(annotation):
                if candidate is NoneType:
                    if value in (None, "", "None"):
                        return None
                    continue
                try:
                    return _coerce_value(value, candidate)
                except (TypeError, ValueError):
                    continue
            return value

        return value

    class BaseSettings:  # pylint: disable=too-few-public-methods
        """Tiny subset of BaseSettings behaviour used in tests."""

        def __init__(self, **overrides: Any) -> None:
            hints = get_type_hints(self.__class__, include_extras=True)
            for name, annotation in hints.items():
                if name.startswith("_"):
                    continue

                if name in overrides:
                    value = overrides[name]
                elif (env_value := os.getenv(name)) is not None:
                    value = env_value
                elif hasattr(self.__class__, name):
                    attribute = getattr(self.__class__, name)
                    value = attribute() if callable(attribute) else attribute
                else:
                    raise RuntimeError(f"Environment variable {name} is required")

                coerced = _coerce_value(value, annotation)
                if isinstance(coerced, list):
                    coerced = list(coerced)
                setattr(self, name, coerced)


class Settings(BaseSettings):
    """Application configuration parsed from environment variables."""

    # Telegram
    BOT_TOKEN: str
    WEBHOOK_BASE_URL: str
    WEBHOOK_PATH: str = "/webhook"
    WEBHOOK_SECRET: str
    ADMIN_IDS: list[int] | str = []
    ADMIN_SECRET: str = "change-me"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://bot:botpass@db:5432/botdb"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    # GitHub
    GITHUB_TOKEN: str
    GITHUB_ORG: str = ""

    # Payout
    PAYOUT_WEBHOOK_URL: Optional[str] = None
    PAYOUT_SECRET: str = "change-me"
    PAYOUT_AMOUNT_CENTS: int = 10000

    # App
    TASK_DEADLINE_HOURS: int = 24
    MAX_ASSIGN_PER_USER: int = 100
    LOG_LEVEL: str = "info"
    ENVIRONMENT: str = "development"
    SENTRY_DSN: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()

if isinstance(settings.ADMIN_IDS, str):
    settings.ADMIN_IDS = [int(item.strip()) for item in settings.ADMIN_IDS.split(",") if item.strip()]

if not settings.CELERY_BROKER_URL:
    settings.CELERY_BROKER_URL = settings.REDIS_URL
if not settings.CELERY_RESULT_BACKEND:
    settings.CELERY_RESULT_BACKEND = settings.REDIS_URL
