#!/usr/bin/env bash
set -e

echo "🚀 PromptOps Bot Installer"

if [ ! -f .env ]; then
  echo "📝 Створюю .env із .env.template..."
  cp .env.template .env
  echo "⚠️ Заповни .env (BOT_TOKEN, WEBHOOK_BASE_URL, WEBHOOK_SECRET, GITHUB_TOKEN, ADMIN_IDS) та перезапусти install.sh"
  exit 0
fi

if ! command -v docker &>/dev/null; then
  echo "❌ Docker не встановлено"
  exit 1
fi

echo "🔨 Build..."
docker-compose build --no-cache

echo "🚀 Start..."
docker-compose up -d

echo "⏳ Чекаю DB..."
for i in {1..30}; do
  if docker-compose exec -T db pg_isready -U ${POSTGRES_USER:-bot} >/dev/null 2>&1; then
    echo "✅ DB ready"
    break
  fi
  sleep 2
done

echo "📦 Міграції..."
docker-compose exec web alembic upgrade head || {
  echo "❌ Alembic failure"; docker-compose logs web; exit 1;
}

echo "🌱 Додаю демо-завдання..."
docker-compose exec web python - << 'PY'
import asyncio
from app.db import AsyncSessionLocal
from app.models import Task

async def seed():
    async with AsyncSessionLocal() as s:
        res = await s.execute("select count(*) from tasks")
        if (res.scalar() or 0) == 0:
            tasks = [Task(title=f"Task {i}", text="Do X", requirement="PR URL", reward_cents=10000) for i in range(1, 51)]
            s.add_all(tasks)
            await s.commit()
            print("✅ Seeded 50 tasks")
        else:
            print("ℹ️ Tasks already present")
asyncio.run(seed())
PY

echo "📡 Встановлюю webhook з secret_token..."
docker-compose exec web python - << 'PY'
import os, requests
bot_token = os.getenv("BOT_TOKEN")
base = os.getenv("WEBHOOK_BASE_URL")
path = os.getenv("WEBHOOK_PATH", "/webhook")
secret = os.getenv("WEBHOOK_SECRET", "")
url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
resp = requests.post(url, json={"url": f"{base}{path}", "secret_token": secret, "allowed_updates": ["message"]})
print(resp.status_code, resp.text)
PY

echo "✅ Готово! Перевір /health та /metrics"
