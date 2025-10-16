import time
from functools import wraps

import structlog
from prometheus_client import Counter, Gauge, Histogram


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

log = structlog.get_logger()

webhook_requests = Counter(
    "webhook_requests_total",
    "Total webhook requests",
    ["status"],
)
assignments_submitted = Counter(
    "assignments_submitted_total",
    "Total assignments submitted",
)
assignments_approved = Counter(
    "assignments_approved_total",
    "Total assignments approved",
)
verification_duration = Histogram(
    "verification_duration_seconds",
    "GitHub verification duration",
)
active_assignments = Gauge(
    "active_assignments",
    "Active assignments count",
)
queue_size = Gauge(
    "celery_queue_size",
    "Celery queue size",
)


def timed_task(func):
    """Decorator to measure task duration."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            log.info(f"{func.__name__} completed", duration=time.time() - start)
            return result
        except Exception as exc:  # pragma: no cover - logged for observability
            log.error(f"{func.__name__} failed", error=str(exc), duration=time.time() - start)
            raise

    return wrapper
