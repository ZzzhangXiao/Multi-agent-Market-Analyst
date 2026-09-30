"""
Structured trader output.

The trader LLM must return a TradeDecision via Groq tool-calling instead of
free text. Everything downstream (evaluation, IC, hit rate, ablations)
reads these fields, so they are validated here rather than parsed from prose.

Design choice consistent with the rest of the project: Python owns the
rules. The signal/score bands below are defined once, used in the prompt,
and enforced by a validator — the LLM cannot return BUY with a negative score.
"""
from typing import Literal
from pydantic import BaseModel, Field, model_validator

# ── Single source of truth for signal <-> score consistency ──
# Inclusive/exclusive edges are deliberate so every score maps to one signal.
SIGNAL_BANDS = {
    "BUY":    (0.2, 1.0),    #  0.2 <  score <=  1.0
    "HOLD":   (-0.2, 0.2),   # -0.2 <= score <=  0.2
    "REDUCE": (-0.6, -0.2),  # -0.6 <= score <  -0.2
    "AVOID":  (-1.0, -0.6),  # -1.0 <= score <  -0.6
}


def signal_for_score(score: float) -> str:
    """Deterministic score -> signal mapping (used by the validator and by
    future rule-only baselines, so both strategies share one mapping)."""
    if score > 0.2:
        return "BUY"
    if score >= -0.2:
        return "HOLD"
    if score >= -0.6:
        return "REDUCE"
    return "AVOID"


def bands_for_prompt() -> str:
    return (
        "BUY: score > 0.2 | HOLD: -0.2 to 0.2 | "
        "REDUCE: -0.6 to below -0.2 | AVOID: below -0.6"
    )


class TradeDecision(BaseModel):
    """Final trading decision for one ticker."""

    ticker: str = Field(description="The ticker being decided, exactly as given.")
    signal: Literal["BUY", "HOLD", "REDUCE", "AVOID"] = Field(
        description="Must be consistent with score: " + bands_for_prompt()
    )
    score: float = Field(
        ge=-1.0, le=1.0,
        description=(
            "Expected direction and strength of the ticker's return relative "
            "to its benchmark over the next ~21 trading days. -1 = strongly "
            "negative, 0 = no edge, +1 = strongly positive."
        ),
    )
    confidence: Literal["High", "Medium", "Low"]
    winning_side: Literal["bull", "bear", "none"] = Field(
        description="Which debate side won. 'none' if there was no debate or it was a draw."
    )
    key_evidence: list[str] = Field(
        min_length=1, max_length=4,
        description=(
            "1-4 short items, each citing a specific number copied exactly "
            "from the reports (e.g. 'RSI(14) 41.3, neutral')."
        ),
    )
    reasoning: str = Field(description="2-3 sentences.")
    key_risk: str = Field(description="The single biggest risk to this call.")

    @model_validator(mode="after")
    def _signal_matches_score(self):
        expected = signal_for_score(self.score)
        if self.signal != expected:
            raise ValueError(
                f"signal {self.signal} is inconsistent with score {self.score}; "
                f"score {self.score} maps to {expected} ({bands_for_prompt()})"
            )
        return self
