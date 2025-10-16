# PromptOps Bot 🤖 — Production-Ready

Telegram бот для роздачі завдань пакетами (batch), автоматичної верифікації GitHub PR і виплат після *повного* виконання пачки.

## 🚀 Швидкий старт
```bash
cp .env.template .env            # заповни BOT_TOKEN, WEBHOOK_BASE_URL, WEBHOOK_SECRET, GITHUB_TOKEN
chmod +x install.sh
./install.sh
```

## 📦 Що вміє
- Роздає N завдань за раз як *пачку* (batch). Оплата — тільки коли всі з цієї пачки approved.
- Дедлайн 24h (налаштовується).
- Автоверіфікація PR: merged + changed_files>0 (+ опціональна перевірка членства в org).
- Payout webhook після завершення пачки (або pending якщо не налаштовано).
- Atomic take (Postgres `FOR UPDATE SKIP LOCKED`), Redis rate-limit, Prometheus метрики, JSON-логи.
- Docker, docker-compose, Alembic, Celery, Redis, Postgres, CI/CD.

## 📚 Команди в Telegram
- `/tasks` — показати доступні завдання
- `/take N` — взяти N завдань (створює batch)
- `/my` — мої завдання
- `/submit ID URL` — надіслати PR-посилання
- `/help` — довідка

## 🧪 Тести
```
pytest -q
```

## 🔐 Безпека
- Вебхук перевіряє `X-Telegram-Bot-Api-Secret-Token` на точний збіг з `WEBHOOK_SECRET`.
- Секрети через `.env`/CI secrets. Не коміть токени.

## 🛠 Розгортання
- Локально: `docker-compose up -d`
- Reverse proxy (nginx) у `docker/nginx.conf` (TLS — через ваш termination, напр. Caddy/Traefik/Cloudflare Tunnel).
