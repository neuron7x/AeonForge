# AeonForge

**Коротко (TL;DR):** Мінімальне когнітивне ядро з людиною в контурі. 5 модулів (`Affordance`, `Relevance`, `DevLoop`, `CausalWM`, `MetaCtrl`) + конфіги. Без магії. Запустив — отримав артефакт/дію.

> Статус: **alpha**. Тестовано на Ubuntu 22.04 / Python 3.11 / CPU Torch.

---

## Що це?
AeonForge — це набір **практичних модулів** для побудови когнітивного циклу:
- `AffordanceMap` — повертає базові “можливі дії” для спостереження.
- `RelevanceFilter` — нормалізує ваги / відфільтровує шум.
- `DevLoop` — буфер досвіду, простий curiosity, офлайн replay.
- `CausalWorldModel` — примітивний `do()` для контрфактуального кроку.
- `MetaAttentionController` — приймає рішення “достатньо / стоп” по метриці.

**Навіщо:** скласти з цих блоків петлю “сприйняття → план → дія → пам’ять” з людиною у контурі (Human‑in‑the‑Loop).

---

## Вимоги
- Python **3.11**
- `pip`, `venv`
- CPU‑версія PyTorch (не потребує CUDA)

---

## Встановлення
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.* --only-binary=:all:
pip install -e .
```

> Якщо використовуєш `requirements.txt`: `pip install -r requirements.txt`

---

## Швидкий старт
Запустити демонстрацію:
```bash
python run.py +demo=true
```
Очікуваний вихід (спрощено):
```
Affordance: [0.125, 0.125, ...]
Relevance: [0.125, 0.125, ...]
DevLoop replay: [... останні записи ...]
WorldModel do(): {'action': 'adjust', 'effect': 'counterfactual-updated'}
Meta should_stop: True|False
```

---

## Мінімальний кодовий приклад
```python
from aeonforge import AffordanceMap, RelevanceFilter, DevLoop, CausalWorldModel, MetaAttentionController

aff = AffordanceMap()
rel = RelevanceFilter()
dev = DevLoop()
wm  = CausalWorldModel()
meta = MetaAttentionController(threshold=0.1)

obs = [0.2, 0.5, 0.3]
actions = aff.infer(obs)         # список ймовірностей дій
weights = rel.mask(actions)      # нормалізовані ваги
dev.log({"obs": obs, "a": actions, "w": weights})
cf = wm.do("action", "adjust")   # контрфакт
stop = meta.should_stop(0.92)    # зупинка за метрикою
print(actions, weights, cf, stop)
```

---

## Конфігурації (YAML)
`configs/default.yaml`
```yaml
seed: 42
task: demo
logging:
  level: INFO
cbc:
  enable: true
  eoi_threshold: 1.0
```

`configs/human_loop.yaml`
```yaml
agents:
  - role: architect
  - role: critic
  - role: integrator
publish:
  targets: [console]
```

---

## Структура репозиторію
```
.
├─ src/aeonforge/        # модулі ядра
│  ├─ affordance.py
│  ├─ relevance.py
│  ├─ devloop.py
│  ├─ world_model.py
│  ├─ meta.py
│  └─ demo.py
├─ configs/              # конфіги
│  ├─ default.yaml
│  └─ human_loop.yaml
├─ tests/                # мінімальні тести (smoke)
│  └─ test_smoke.py
├─ docs/                 # короткі технічні нотатки
├─ run.py                # CLI для демо
├─ pyproject.toml        # пакетна збірка (editable install)
└─ README.md
```

---

## Тести
```bash
pytest -q
```
> У CI використовується CPU‑Torch і `pip install -e .` (див. `.github/workflows/ci.yml`).

Мінімальний тест (щоб CI був зелений):
```python
def test_imports():
    import aeonforge as core
    assert hasattr(core, "AffordanceMap")
    assert hasattr(core, "MetaAttentionController")
```

---

## Обмеження / що **не** входить
- Немає складних середовищ RL та великих моделей — **тільки каркас**.
- Немає гарантованої стабільності API — **alpha**.
- Немає CUDA‑залежностей у CI за замовчуванням.

---

## Дорожня карта (коротко)
- [ ] Консистентні інтерфейси між модулями (вхід/вихід типізувати).
- [ ] Приклади інтеграції Human‑in‑the‑Loop (форми/CLI).
- [ ] Більше тестів: property‑based, інтеграційні.
- [ ] Логування подій та простий веб‑UI.
- [ ] Базові бенчмарки (час/памʼять).

---

## Ліцензія
MIT
