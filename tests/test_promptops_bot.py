from datetime import datetime, timezone
from types import SimpleNamespace

from app.task_spec import build_task_template
from app.quality import run_auto_qc, QCThresholds
from app.reputation import calculate_level


def _assert_template_structure(data: dict) -> None:
    required_top = {
        "task_id",
        "type",
        "brief",
        "inputs",
        "acceptance",
        "deadline_min",
        "payout",
        "level_required",
    }
    assert required_top.issubset(data.keys())
    assert isinstance(data["task_id"], str)
    assert isinstance(data["type"], str)
    assert isinstance(data["deadline_min"], int)
    assert isinstance(data["level_required"], int)

    brief = data["brief"]
    assert isinstance(brief, dict)
    for key in ("goal", "tone", "length"):
        assert isinstance(brief[key], str)
    for key in ("must_include", "must_avoid", "negative_examples"):
        assert isinstance(brief[key], list)

    acceptance = data["acceptance"]
    assert isinstance(acceptance, dict)
    assert isinstance(acceptance["checks"], list)
    assert isinstance(acceptance["style"], list)

    payout = data["payout"]
    assert isinstance(payout, dict)
    for field in ("base_usd", "quality_bonus_max", "speed_bonus_max"):
        assert isinstance(payout[field], float)


def test_task_template_matches_schema():
    task = SimpleNamespace(
        id=123,
        title="30s тизер про AeonForge",
        requirement="120-160 words",
        reward_cents=250,
        created_at=datetime.now(timezone.utc),
        level_required=2,
        deadline_minutes=45,
        type="content_outline",
        bonus_quality_cents=25,
        bonus_speed_cents=10,
        payload={
            "brief": {
                "goal": "30s тизер про AeonForge",
                "tone": "calm, factual",
                "length": "120-160 words",
                "must_include": ["AeonForge", "Human-in-the-Loop"],
                "must_avoid": ["обіцянки прибутку"],
                "negative_examples": ["клікбейт"],
            },
            "inputs": {"refs": ["https://example.com/readme"]},
            "acceptance": {"checks": ["len_range"], "style": ["tone_match"]},
        },
        text=None,
    )

    template = build_task_template(task)
    _assert_template_structure(template)
    assert template["brief"]["goal"] == "30s тизер про AeonForge"
    assert "refs" in template["inputs"]
    assert template["payout"]["base_usd"] == 2.5


def test_auto_qc_flags_pii_and_dedup():
    thresholds = QCThresholds()
    report_pii = run_auto_qc("Contact me at test@example.com", thresholds=thresholds)
    assert not report_pii["passed"]
    assert "pii_detected" in report_pii["reasons"]

    repetitive = "word " * 200
    report_dedup = run_auto_qc(repetitive, thresholds=thresholds)
    assert not report_dedup["passed"]
    assert any(reason.startswith("dedup") for reason in report_dedup["reasons"])


def test_calculate_level_progression():
    assert calculate_level(-5) == 0
    assert calculate_level(0) == 0
    assert calculate_level(15) >= 1
    assert calculate_level(40) >= 2
