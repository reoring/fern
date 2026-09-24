"""Have the teacher author Jev requests (state + questions) for agent-style decisions.

This is the only place we let the teacher decode. Output JSON is validated against the
Jev schema; invalid generations are dropped.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import ValidationError

from ..schema import SystemOneRequest
from ..teacher import Teacher
from . import Example

DOMAINS = [
    "customer support chat for an e-commerce store",
    "a coding agent deciding its next tool call",
    "a browser agent reading a web page screenshot description",
    "content moderation of a social media post",
    "triage of an incoming IT incident ticket",
    "a robot vacuum reading its sensor log",
    "a sales rep reviewing a CRM note",
    "a game AI reading the board/screen state",
    "a home assistant hearing a voice request",
    "medical intake form pre-screening (non-diagnostic)",
    "email inbox triage",
    "a trading bot reading a news headline and position",
    "a self-driving simulator reading a scene description",
    "a chess engine assistant reading a FEN position",
    "a code review bot reading a diff",
    "a recruiter screening a short resume summary",
    "a chatbot guardrail reading a user message",
    "a warehouse dispatcher reading an order queue",
    "a smart thermostat reading household sensors",
    "a legal assistant reading a contract clause",
]

STYLES = ["terse and realistic", "messy with typos and noise", "long and detailed", "ambiguous, requiring judgment", "adversarial or borderline"]

INSTRUCTION = """Write ONE realistic request for a fast decision model in this JSON schema:
{{
  "state": "<what the model looks at: logs, message, screen description, etc. 1-10 sentences>",
  "questions": {{
    "<id>": {{"type": "choice", "instructions": "<question>", "criteria": {{"<key>": "<meaning>", ...}}}},
    "<id>": {{"type": "score", "instructions": "<question>", "criteria": ["<lowest>", "...", "<highest>"]}},
    "<id>": {{"type": "noul", "instructions": "<yes/no question>"}}
  }}
}}
Rules: domain = {domain}. Style of the state = {style}. Include {n} questions mixing the three types (choice needs 2-6 keys, score needs 2-7 levels). Questions must be answerable from the state and genuinely require judgment.{lang_rule} Output only the JSON object."""

LANG_NAMES = {"ja": "Japanese", "zh": "Chinese", "de": "German", "es": "Spanish", "fr": "French", "ko": "Korean", "pt": "Portuguese"}


def _lang_rule(lang: str) -> str:
    if lang == "en":
        return ""
    return (
        f" Write the state, the instructions, the criteria keys and their meanings entirely in "
        f"{LANG_NAMES[lang]}, as a native speaker would in that domain; keep the JSON field names "
        f'("state", "questions", "type", "instructions", "criteria") and the type values in English.'
    )

JSON_RE = re.compile(r"\{.*\}", re.S)


def _examples(seed: int, i: int, req: SystemOneRequest, lang: str = "en") -> list[Example]:
    prefix = f"syn/{seed}" if lang == "en" else f"syn_{lang}/{seed}"  # English ids unchanged (data/syn.jsonl resumes)
    return [Example(id=f"{prefix}/{i}/{qid}", state=req.state, question=q) for qid, q in req.questions.items()]


async def generate(
    teacher: Teacher, n: int, cache: Path, *, seed: int = 0, langs: list[str] | None = None, max_tokens: int = 700, gen_concurrency: int = 32
) -> AsyncIterator[Example]:
    """Generate n requests; each validated request is appended to `cache` (jsonl: {i, request}) so
    a restart replays them without decoding again. Generation uses its own semaphore so the
    teacher's labeling slots are not starved by long decodes. `langs` cycles languages per request
    (default English only)."""
    rng = random.Random(seed)
    langs = langs or ["en"]
    params = [(rng.choice(DOMAINS), rng.choice(STYLES), rng.randint(1, 4), langs[i % len(langs)]) for i in range(n)]
    done: dict[int, SystemOneRequest] = {}
    if cache.exists():
        with cache.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["i"]] = SystemOneRequest.model_validate(row["request"])
    for i, req in done.items():
        for ex in _examples(seed, i, req, params[i][3]):
            yield ex

    sem = asyncio.Semaphore(gen_concurrency)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out = cache.open("a")

    async def one(i: int) -> list[Example]:
        domain, style, k, lang = params[i]
        messages = [{"role": "user", "content": INSTRUCTION.format(domain=domain, style=style, n=k, lang_rule=_lang_rule(lang))}]
        async with sem:
            prompt = await teacher.apply_chat_template(messages, thinking=False)
            body = await teacher.post(
                "/completion", {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.9, "top_p": 0.95}
            )
        m = JSON_RE.search(body["content"])
        if not m:
            return []
        try:
            req = SystemOneRequest.model_validate({"model": "gen", **json.loads(m.group())})
        except (ValidationError, json.JSONDecodeError):
            return []
        out.write(json.dumps({"i": i, "request": req.model_dump()}, ensure_ascii=False) + "\n")
        out.flush()
        return _examples(seed, i, req, lang)

    try:
        for fut in asyncio.as_completed([one(i) for i in range(n) if i not in done]):
            for ex in await fut:
                yield ex
    finally:
        out.close()
