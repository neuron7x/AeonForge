from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


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
