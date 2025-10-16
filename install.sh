#!/usr/bin/env bash
set -e

echo "🚀 PromptOps Bot Installer"
echo "========================================"

if [ ! -f .env ]; then
    echo "📝 Створюю .env із .env.template..."
    cp .env.template .env
    echo ""
    echo "⚠️  ВАЖЛИВО: Відкрий .env та заповни:"
    echo "   - BOT_TOKEN (з @BotFather)"
    echo "   - WEBHOOK_BASE_URL (https://your-domain.com)"
    echo "   - GITHUB_TOKEN (ghp_...)"
    echo "   - ADMIN_IDS (твій Telegram ID)"
    echo ""
    echo "Потім перезапусти: bash install.sh"
    exit 0
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker не встановлено. Встанови Docker Desktop."
    exit 1
fi

echo "🔨 Будую Docker образи..."
docker-compose build --no-cache

echo "🚀 Запускаю сервіси..."
docker-compose up -d

echo "⏳ Чекаю на базу даних..."
for i in {1..30}; do
    if docker-compose exec -T db pg_isready -U bot > /dev/null 2>&1; then
        echo "✅ База готова"
        break
    fi
    echo "  Спроба $i/30..."
    sleep 2
done

echo "📦 Запускаю міграції..."
docker-compose exec web alembic upgrade head || {
    echo "❌ Міграції не вдалося запустити. Перевір .env та логи:"
    docker-compose logs web
    exit 1
}

echo "🌱 Додаю демо-завдання..."
docker-compose exec web python <<'PYEOF'
import asyncio

from app.db import AsyncSessionLocal
from app.models import Task


async def seed():
    async with AsyncSessionLocal() as session:
        tasks = [
            Task(
                title=f"Task #{i}",
                text=f"Зроби PR до репо та отримай {i*100}$",
                requirement="PR link",
                reward_cents=i * 100,
            )
            for i in range(1, 11)
        ]
        session.add_all(tasks)
        await session.commit()
    print(f"✅ Додано {len(tasks)} завдань")


asyncio.run(seed())
PYEOF

echo "📡 Встановлюю webhook..."
docker-compose exec web python <<'PYEOF'
import os

import requests

bot_token = os.getenv("BOT_TOKEN")
base_url = os.getenv("WEBHOOK_BASE_URL")
path = os.getenv("WEBHOOK_PATH", "/webhook")

if not bot_token:
    print("❌ BOT_TOKEN не встановлено")
    raise SystemExit(1)

webhook_url = f"{base_url}{path}"
url = f"https://api.telegram.org/bot{bot_token}/setWebhook"

resp = requests.post(url, json={"url": webhook_url, "allowed_updates": ["message"]})

if resp.status_code == 200:
    print(f"✅ Webhook встановлено: {webhook_url}")
else:
    print(f"❌ Помилка: {resp.text}")
PYEOF

echo ""
echo "✅ Установка завершена!"
echo ""
echo "📊 Команди:"
echo "  docker-compose logs -f web       # Логи вебсервера"
echo "  docker-compose logs -f worker    # Логи воркера"
echo "  docker-compose ps                # Статус сервісів"
echo "  docker-compose down              # Зупинити"
echo ""
echo "🧪 Тестування:"
echo "  Напиши боту: /start"
echo "  Потім: /tasks"
echo "  Потім: /take 1"
echo ""
