from __future__ import annotations

from vaticore.copilot import explain_advisory
from vaticore.copilot.explain import _facts, _template_explanation
from vaticore.decisions import Advisory


def _advisory(genset: bool) -> Advisory:
    return Advisory(
        horizon_hours=24.0,
        expected_net_load_kwh=1200.0,
        conservative_net_load_kwh=1800.0,
        recommended_reserve_kwh=600.0,
        genset_recommended=genset,
        note="test note",
    )


def test_template_fallback_used_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = explain_advisory(_advisory(genset=True))
    assert result.generated_by == "template"
    assert "generator" in result.text.lower()


def test_template_mentions_no_generator_when_battery_covers() -> None:
    result = explain_advisory(_advisory(genset=False), api_key=None)
    assert result.generated_by == "template"
    assert "no generator" in result.text.lower() or "battery should cover" in result.text.lower()


def test_facts_are_grounded_in_the_numbers() -> None:
    facts = _facts(_advisory(genset=True))
    assert "1800 kWh" in facts
    assert "600 kWh" in facts
    assert "yes" in facts


def test_llm_path_falls_back_on_error(monkeypatch) -> None:
    # A key is present but the SDK import/call fails; must degrade, not raise.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-real")
    import vaticore.copilot.explain as mod

    def boom(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(mod, "_llm_explanation", boom)
    result = explain_advisory(_advisory(genset=False))
    assert result.generated_by == "template"
    assert result.text == _template_explanation(_advisory(genset=False))
