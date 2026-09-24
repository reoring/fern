"""Render (state, question) into a chat prompt whose next token is the decision.

The same rendering is used for the teacher (llama-server) and the student (HF), so the
student learns exactly the distribution the teacher produced at the same position.
Options are surfaced as short labels (letters / digits / yes,no) so each answer is a
single token in both tokenizers; probabilities are read at that one position.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, option_keys

LETTERS = "ABCDEFGHIJ"

SYSTEM = (
    "You are a decision engine. You never explain. Read the state, then answer the "
    "question with exactly one option label."
)


@dataclass(frozen=True)
class Rendered:
    messages: list[dict[str, str]]
    labels: list[str]  # answer label per option, aligned with option_keys(q)
    prefix: str  # assistant text preceding the decision token


def render(state: str, q: ChoiceQuestion | ScoreQuestion | NoulQuestion) -> Rendered:
    keys = option_keys(q)
    if q.type == "choice":
        labels = list(LETTERS[: len(keys)])
        options = "\n".join(
            f"{lab}. {key}" + (f" — {desc}" if desc and desc != key else "")
            for lab, key, desc in zip(labels, keys, q.criteria.values())
        )
        task = f"{q.instructions}\n\nOptions:\n{options}\n\nAnswer with the letter only."
    elif q.type == "score":
        labels = keys
        options = "\n".join(f"{i}. {c}" for i, c in enumerate(q.criteria))
        task = f"{q.instructions}\n\nScale:\n{options}\n\nAnswer with the number only."
    else:
        labels = ["yes", "no"]
        crit = q.criteria or {}
        hint = ""
        if crit:
            hint = f"\nyes: {crit.get('true', '')}\nno: {crit.get('false', '')}"
        task = f"{q.instructions}{hint}\n\nAnswer yes or no only."
    user = f"State:\n{state}\n\nQuestion:\n{task}"
    return Rendered(
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        labels=labels,
        prefix="Answer:",
    )


def normalize_token(tok: str) -> str:
    """Collapse tokenizer variants (' A', 'A', 'Ġyes') onto the label alphabet."""
    return tok.replace("Ġ", " ").replace("▁", " ").strip().lower()
