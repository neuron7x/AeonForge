# Integration tests (docker-compose)

Run locally:

```bash
cd tests/integration
docker compose up -d --build
./wait_for_http.sh http://localhost:8000/health 60
curl -fsS http://localhost:8000/health
docker compose down -v
```
