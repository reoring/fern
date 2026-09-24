"""Teacher = llama-server. We never decode: one prefill, read next-token logprobs.

Contract verified against llama-server (/completion, n_predict=1, n_probs=N):
    {"completion_probabilities": [{"top_logprobs": [{"token": " A", "logprob": -0.1}, ...]}]}
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import httpx

from .prompting import Rendered, normalize_token

FLOOR_FACTOR = 0.1  # option absent from top-N gets exp(min_logprob) * FLOOR_FACTOR


@dataclass
class TeacherResult:
    probs: list[float]  # normalized over the option labels
    prompt_tokens: int
    covered: int  # how many option labels were found in top-N
    prompt_ms: float


class PromptTooLong(ValueError):
    """Prompt does not fit the teacher's per-slot context; the example is skipped."""


class Teacher:
    def __init__(self, base_url: str, *, n_probs: int = 32, concurrency: int = 8, think: int = 0, timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.n_probs = n_probs
        self.concurrency = concurrency
        self.think = think
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def post(self, path: str, payload: dict, *, retries: int = 6) -> dict:
        """POST with retry on transport errors / 5xx (llama-server drops connections occasionally)."""
        for attempt in range(retries):
            try:
                r = await self.client.post(path, json=payload)
                if r.status_code == 400 and "exceed_context" in r.text:
                    raise PromptTooLong(r.json()["error"]["message"])
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()
                err: Exception = httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
            except httpx.TransportError as e:
                err = e
            if attempt == retries - 1:
                raise err
            await asyncio.sleep(min(30, 2**attempt))
        raise AssertionError("unreachable")

    async def apply_chat_template(self, messages: list[dict[str, str]], *, thinking: bool) -> str:
        """Render the server's own chat template with thinking on/off (DeepSeek: `thinking`, Qwen: `enable_thinking`)."""
        body = await self.post(
            "/apply-template",
            {"messages": messages, "chat_template_kwargs": {"thinking": thinking, "enable_thinking": thinking}},
        )
        return body["prompt"]

    async def _think(self, prompt: str) -> str:
        """Let the model think up to `self.think` tokens, then close the block."""
        body = await self.post("/completion", {"prompt": prompt, "n_predict": self.think, "temperature": 0, "stop": ["</think>"]})
        return prompt + body["content"] + "</think>\n\n"

    async def decide(self, rendered: Rendered) -> TeacherResult:
        async with self.sem:
            prompt = await self.apply_chat_template(rendered.messages, thinking=self.think > 0)
            if self.think:
                prompt = await self._think(prompt)
            prompt += rendered.prefix
            body = await self.post(
                "/completion",
                {"prompt": prompt, "n_predict": 1, "n_probs": self.n_probs, "temperature": 0, "cache_prompt": True},
            )
        top = body["completion_probabilities"][0]["top_logprobs"]
        return TeacherResult(
            probs=option_probs(top, rendered.labels)[0],
            covered=option_probs(top, rendered.labels)[1],
            prompt_tokens=body["tokens_evaluated"] if "tokens_evaluated" in body else body["timings"]["prompt_n"],
            prompt_ms=body["timings"]["prompt_ms"],
        )


def option_probs(top_logprobs: list[dict], labels: list[str]) -> tuple[list[float], int]:
    """Sum token variants per label, floor missing labels, normalize over labels."""
    mass = {normalize_token(lab): 0.0 for lab in labels}
    min_lp = min(t["logprob"] for t in top_logprobs)
    for t in top_logprobs:
        key = normalize_token(t["token"])
        if key in mass:
            mass[key] += math.exp(t["logprob"])
    covered = sum(1 for v in mass.values() if v > 0)
    floor = math.exp(min_lp) * FLOOR_FACTOR
    raw = [mass[normalize_token(lab)] or floor for lab in labels]
    z = sum(raw)
    return [p / z for p in raw], covered
