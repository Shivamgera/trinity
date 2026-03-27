"""Trade signal schema with chain-of-thought enforcement."""

from pydantic import BaseModel, field_validator
from typing import Literal


class TradeSignal(BaseModel):
    """Analyst agent output schema.

    IMPORTANT: 'reasoning' is the FIRST field to force chain-of-thought
    before the model commits to a decision. This is a deliberate prompt
    engineering choice --- LLMs that reason before deciding produce better-
    calibrated outputs.
    """

    reasoning: str
    decision: Literal["hold", "buy", "sell"]

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Reasoning must not be empty")
        return v
