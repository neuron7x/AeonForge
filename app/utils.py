from redis import asyncio as aioredis
from app.config import settings

_redis = None

async def get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis

async def rate_limit_take(user_id: int) -> bool:
    """Allow at most TAKE_RATE_LIMIT_PER_MINUTE per minute per user for /take."""
    r = await get_redis()
    key = f"take_rl:{user_id}"
    count = await r.incr(key)
    if count == 1:
        await r.expire(key, 60)
    return count <= settings.TAKE_RATE_LIMIT_PER_MINUTE
