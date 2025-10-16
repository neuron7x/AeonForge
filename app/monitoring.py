import time
from functools import wraps

try:  # pragma: no cover - depends on optional dependency
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - activated in minimal envs
    class _NoOpMetric:
        def labels(self, **_):
            return self

        def inc(self, *_args, **_kwargs):
            return None

        def dec(self, *_args, **_kwargs):
            return None

        def set(self, *_args, **_kwargs):
            return None

        def observe(self, *_args, **_kwargs):
            return None

    class Counter(_NoOpMetric):  # type: ignore
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

    class Gauge(_NoOpMetric):  # type: ignore
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

    class Histogram(_NoOpMetric):  # type: ignore
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

try:  # pragma: no cover - depends on optional dependency
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - activated in slim envs
    structlog = None  # type: ignore


if structlog:  # pragma: no branch - configuration happens once at import time
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
else:  # Fallback to the stdlib logger with a similar API surface.
    import logging

    logging.basicConfig(level=logging.INFO)

    class _FallbackLogger:
        def __init__(self) -> None:
            self._logger = logging.getLogger("aeonforge")

        def info(self, event: str, **kwargs):
            message = event if not kwargs else f"{event} | {kwargs}"
            self._logger.info(message)

        def error(self, event: str, **kwargs):
            message = event if not kwargs else f"{event} | {kwargs}"
            self._logger.error(message)

    log = _FallbackLogger()

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
