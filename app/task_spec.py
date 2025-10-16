from __future__ import annotations

import json
from typing import Any, Dict


def build_task_template(task: Any) -> Dict[str, Any]:
    payload = getattr(task, "payload", {}) or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}

    brief = payload.get("brief") or {}
    acceptance = payload.get("acceptance") or {}

    inputs = payload.get("inputs")
    if isinstance(inputs, str) and inputs:
        inputs = {"refs": [inputs]}
    elif inputs is None and getattr(task, "text", None):
        inputs = {"refs": [getattr(task, "text")]}  # type: ignore[arg-type]
    elif not isinstance(inputs, dict):
        inputs = {}

    created_at = getattr(task, "created_at")
    task_id = getattr(task, "id")
    reward_cents = getattr(task, "reward_cents", 0)
    bonus_quality_cents = getattr(task, "bonus_quality_cents", 0)
    bonus_speed_cents = getattr(task, "bonus_speed_cents", 0)

    data: Dict[str, Any] = {
        "task_id": f"T-{created_at.strftime('%Y')}-{task_id:06d}",
        "type": getattr(task, "type", None) or payload.get("type") or "generic",
        "brief": {
            "goal": brief.get("goal") or getattr(task, "title", ""),
            "tone": brief.get("tone") or payload.get("tone", "balanced"),
            "length": brief.get("length") or payload.get("length", "flex"),
            "must_include": brief.get("must_include") or payload.get("must_include", []),
            "must_avoid": brief.get("must_avoid") or payload.get("must_avoid", []),
            "negative_examples": brief.get("negative_examples") or payload.get("negative_examples", []),
        },
        "inputs": inputs,
        "acceptance": {
            "checks": acceptance.get("checks") or ["len_range", "no_pii", "no_toxic", "dedup<=0.85", "facts>=0.6"],
            "style": acceptance.get("style") or ["tone_match", "brand_terms_ok"],
        },
        "deadline_min": getattr(task, "deadline_minutes", 24 * 60),
        "payout": {
            "base_usd": round(reward_cents / 100, 2),
            "quality_bonus_max": round(bonus_quality_cents / 100, 2),
            "speed_bonus_max": round(bonus_speed_cents / 100, 2),
        },
        "level_required": getattr(task, "level_required", 0),
    }
    return data
