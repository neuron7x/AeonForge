Цей PR додає початкову автоматизацію тестування та CI для репозиторію.

Зроблено:
- Додано workflow GitHub Actions (.github/workflows/python-tests.yml) для запуску pytest на Python 3.8–3.12 із збором покриття.
- Додано requirements-dev.txt з базовими dev-залежностями (pytest, pytest-cov, tox, coverage, pre-commit, codecov).
- Додано tox.ini для локального запуску тестів у кількох середовищах.
- Додано setup.cfg і .coveragerc для конфігурації pytest та coverage.
- Додано приклад smoke-тест: tests/test_smoke.py.

Як перевірити локально:
1. Оновіть pip та встановіть dev-залежності:
   python -m pip install --upgrade pip
   python -m pip install -r requirements-dev.txt

2. Запустіть тести локально:
   pytest

3. Або запустіть через tox (потрібні відповідні версії Python локально):
   tox

Додаткові примітки:
- Завантаження звіту coverage в Codecov відбувається лише якщо в репозиторії налаштовано secret CODECOV_TOKEN. Щоб увімкнути — додайте секрет у Settings → Secrets → Actions -> NEW SECRET з ім'ям CODECOV_TOKEN.
- Якщо потрібно — можу додати форматування/лінтінг (black, flake8, isort) та pre-commit конфіг.

Цей PR готовий до рев'ю.