"""Run the teacher over examples and write soft labels as jsonl.

Record: {id, state, question, option_keys, teacher_probs, teacher_model, think, covered}
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from .data import Example
from .prompting import render
from .schema import option_keys
from .teacher import PromptTooLong, Teacher, TeacherResult


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


async def _aiter(items: Iterable[Example]) -> AsyncIterator[Example]:
    for it in items:
        yield it


async def label(
    examples: Iterable[Example] | AsyncIterator[Example],
    out: Path,
    *,
    teacher: Teacher,
    teacher_name: str,
    limit: int | None = None,
) -> int:
    """Resumable: skips ids already in `out`. Returns number of newly written rows."""
    done = existing_ids(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stream = examples if hasattr(examples, "__anext__") else _aiter(examples)
    written, t0, tok = 0, time.time(), 0
    pending: set[asyncio.Task] = set()

    async def run(ex: Example) -> tuple[Example, TeacherResult | None]:
        try:
            return ex, await teacher.decide(render(ex.state, ex.question))
        except PromptTooLong as e:
            print(f"[label] skip {ex.id}: {e}", file=sys.stderr)
            return ex, None

    with out.open("a") as f:
        async def flush(ts: set[asyncio.Task]) -> None:
            nonlocal written, tok
            for t in ts:
                ex, res = await t
                if res is None:
                    continue
                tok += res.prompt_tokens
                f.write(json.dumps({
                    "id": ex.id,
                    "state": ex.state,
                    "question": ex.question.model_dump(),
                    "option_keys": option_keys(ex.question),
                    "teacher_probs": [round(p, 6) for p in res.probs],
                    "teacher_model": teacher_name,
                    "think": teacher.think,
                    "covered": res.covered,
                }, ensure_ascii=False) + "\n")
                written += 1
                if written % 200 == 0:
                    dt = time.time() - t0
                    print(f"[label] {written} rows  {written / dt:.1f} rows/s  {tok / dt:.0f} tok/s", file=sys.stderr)

        async for ex in stream:
            if ex.id in done:
                continue
            if limit is not None and written + len(pending) >= limit:
                break
            pending.add(asyncio.create_task(run(ex)))
            finished = {t for t in pending if t.done()}
            if len(pending) >= teacher.concurrency * 4 + 8:  # keep the server queue full but bounded
                finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            else:
                pending -= finished
            if finished:
                await flush(finished)
                f.flush()
        if pending:
            await flush(pending)
    return written


def read_labels(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with p.open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows
