from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    # Telegram
    BOT_TOKEN: str
    WEBHOOK_BASE_URL: str
    WEBHOOK_PATH: str = "/webhook"
    WEBHOOK_SECRET: str
    ADMIN_IDS: List[int] | str = []
    ADMIN_SECRET: str = "change-me"

    # DB
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
    TAKE_RATE_LIMIT_PER_MINUTE: int = 5
    LOG_LEVEL: str = "info"
    ENVIRONMENT: str = "development"
    SENTRY_DSN: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Normalize admin ids if string
if isinstance(settings.ADMIN_IDS, str):
    settings.ADMIN_IDS = [int(x.strip()) for x in settings.ADMIN_IDS.split(",") if x.strip()]

# Defaults for Celery
if not settings.CELERY_BROKER_URL:
    settings.CELERY_BROKER_URL = settings.REDIS_URL
if not settings.CELERY_RESULT_BACKEND:
    settings.CELERY_RESULT_BACKEND = settings.REDIS_URL
