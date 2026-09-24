from __future__ import annotations

from dataclasses import dataclass

from ..schema import ChoiceQuestion, NoulQuestion, ScoreQuestion


@dataclass
class Example:
    id: str
    state: str
    question: ChoiceQuestion | ScoreQuestion | NoulQuestion
