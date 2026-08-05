"""LLM copilot: plain-language explanations on top of the numeric forecast.

The copilot never computes forecasts or advice. It explains numbers the engine
already produced. That separation is deliberate: the forecast must stay
reproducible and defensible, while the copilot adds an accessible layer for
operators who act on litres of diesel, not quantiles.
"""

from vaticore.copilot.explain import Explanation, explain_advisory

__all__ = ["Explanation", "explain_advisory"]
