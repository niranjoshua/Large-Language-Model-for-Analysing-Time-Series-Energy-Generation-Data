"""Explain an advisory in plain language.

Grounding rule: the model is given the exact numbers from the advisory and asked
only to phrase them for an operator. It is instructed not to invent figures. If
no API key is configured, or the call fails, a deterministic template is used
instead, so the product never depends on the LLM being available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from vaticore.decisions import Advisory

# Latest capable Claude model. Override per deployment.
DEFAULT_MODEL = "claude-sonnet-5"

_SYSTEM_PROMPT = (
    "You are an operations assistant for a distributed energy operator (mini-grids "
    "and commercial solar). You explain a battery and generator advisory in plain, "
    "concise language a site operator can act on. Use ONLY the numbers provided. Do "
    "not invent or estimate any figure that is not given. Two or three sentences."
)


@dataclass(frozen=True)
class Explanation:
    """A plain-language explanation and how it was produced."""

    text: str
    generated_by: str  # "llm" or "template"


def _facts(advisory: Advisory) -> str:
    return (
        f"Horizon: {advisory.horizon_hours:.0f} hours. "
        f"Expected net load (P50): {advisory.expected_net_load_kwh:.0f} kWh. "
        f"Conservative net load (bad case): {advisory.conservative_net_load_kwh:.0f} kWh. "
        f"Recommended battery reserve: {advisory.recommended_reserve_kwh:.0f} kWh. "
        f"Generator recommended: {'yes' if advisory.genset_recommended else 'no'}. "
        f"Engine note: {advisory.note}"
    )


def _template_explanation(advisory: Advisory) -> str:
    horizon = f"over the next {advisory.horizon_hours:.0f} hours"
    if advisory.genset_recommended:
        action = (
            f"Plan to run the generator: even a full battery reserve of "
            f"{advisory.recommended_reserve_kwh:.0f} kWh is not enough for the "
            f"conservative net load of {advisory.conservative_net_load_kwh:.0f} kWh."
        )
    else:
        action = (
            f"The battery should cover demand; hold about "
            f"{advisory.recommended_reserve_kwh:.0f} kWh in reserve and no generator "
            "run is expected."
        )
    return (
        f"Expected net load {horizon} is about {advisory.expected_net_load_kwh:.0f} kWh. {action}"
    )


def _llm_explanation(advisory: Advisory, api_key: str, model: str) -> str:
    # Imported lazily so the package does not require the SDK unless the LLM path
    # is actually used. Install with: uv sync --extra copilot
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=300,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _facts(advisory)}],
    )
    # Content is a union of block types; pull text off text blocks defensively.
    parts = [getattr(block, "text", "") for block in message.content]
    text = " ".join(p for p in parts if p).strip()
    if not text:
        raise ValueError("empty LLM response")
    return text


def explain_advisory(
    advisory: Advisory,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Explanation:
    """Return a plain-language explanation of an advisory.

    Uses the Anthropic API when a key is available (argument or ANTHROPIC_API_KEY),
    and falls back to a deterministic template otherwise or on any error. The
    fallback guarantees the product works without the LLM.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            return Explanation(_llm_explanation(advisory, key, model), "llm")
        except Exception:
            # Never let the copilot break the product; degrade to the template.
            pass
    return Explanation(_template_explanation(advisory), "template")
