"""Jev public schema for POST /v1/systemone (request and response)."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

MAX_QUESTIONS = 50
# Answer codes must be single tokens in both tokenizers: letters A..Z for choice (prompting.LETTERS),
# digits 0..9 for score (Qwen splits "10" into two tokens). Jev allows 255; exceeding these needs
# re-labeling and re-training with a new code alphabet.
MAX_CHOICE_OPTIONS = 26
MAX_SCORE_LEVELS = 10


def _as_text(v: Any) -> str:
    """Jev accepts strings or JSON values (objects/arrays) here; we render non-strings as JSON."""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    return json.dumps(v, ensure_ascii=False, indent=2)


Text = Annotated[str, BeforeValidator(_as_text)]


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Text
    criteria: dict[str, Text | None]

    @model_validator(mode="after")
    def _check(self) -> "ChoiceQuestion":
        if not 2 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 2..{MAX_CHOICE_OPTIONS} criteria")
        return self


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Text
    criteria: list[Text]

    @model_validator(mode="after")
    def _check(self) -> "ScoreQuestion":
        if not 2 <= len(self.criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score needs 2..{MAX_SCORE_LEVELS} levels")
        return self


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Text
    criteria: dict[Literal["true", "false"], Text] | None = None


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model: str = "fern"
    state: Annotated[Text, Field(min_length=1, pattern=r"\S")]
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _check(self) -> "SystemOneRequest":
        if not 1 <= len(self.questions) <= MAX_QUESTIONS:
            raise ValueError(f"1..{MAX_QUESTIONS} questions per request")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0
    truncated: bool = False  # prompt exceeded max_len; the start of `state` was dropped


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def option_keys(q: ChoiceQuestion | ScoreQuestion | NoulQuestion) -> list[str]:
    """Ordered option keys exactly as they appear in the prompt and in the label file."""
    if q.type == "choice":
        return list(q.criteria)
    if q.type == "score":
        return [str(i) for i in range(len(q.criteria))]
    return ["yes", "no"]


def build_answer(q: ChoiceQuestion | ScoreQuestion | NoulQuestion, probs: list[float]) -> Answer:
    """Turn a normalized distribution over option_keys(q) into a typed Jev answer.

    confidence follows the approximation in the Jev docs: (K*max - 1) / (K - 1), i.e. 0 for a
    uniform distribution and 1 for a one-hot one, so thresholds written for Jev carry over."""
    keys = option_keys(q)
    k = len(probs)
    confidence = max(0.0, (k * max(probs) - 1) / (k - 1))
    if q.type == "choice":
        best = max(range(len(keys)), key=probs.__getitem__)
        return ChoiceAnswer(choice=keys[best], probabilities=dict(zip(keys, probs)), confidence=confidence)
    if q.type == "score":
        return ScoreAnswer(
            score=sum(i * p for i, p in enumerate(probs)),
            legend={str(i): c for i, c in enumerate(q.criteria)},
            probabilities=dict(zip(keys, probs)),
            confidence=confidence,
        )
    return NoulAnswer(noul=probs[0])
