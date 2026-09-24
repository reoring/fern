"""Teacher agreement on labeled rows, MMLU-Pro accuracy, and request latency."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

from .data.hf_tasks import mmlu_pro
from .prompting import render
from .schema import NoulQuestion, SystemOneRequest
from .student import Student
from .train import load_rows, rendered_of


@torch.inference_mode()
def agreement(student: Student, labels: list[Path], batch_size: int = 32, limit: int | None = None) -> dict:
    """Per label file (each is one gate in issue 009) plus the pooled numbers."""
    per: dict[str, dict] = {}
    tot = {"rows": 0, "agree": 0, "kl": 0.0, "brier": 0.0}
    for path in labels:
        rows = load_rows([path])[:limit]
        agree, kl, brier = 0, 0.0, 0.0
        for i in range(0, len(rows), batch_size):
            chunk = [rendered_of(r) for r in rows[i : i + batch_size]]
            logits, _, _ = student.forward_options([c[0] for c in chunk])
            for l, t in zip(logits, [c[1] for c in chunk]):
                p = torch.softmax(l, 0).cpu()
                agree += int(p.argmax() == t.argmax())
                kl += float((t * (t.clamp_min(1e-9).log() - p.clamp_min(1e-9).log())).sum())
                brier += float(((p - t) ** 2).sum())
        n = len(rows)
        per[str(path)] = {"rows": n, "agree": agree / n, "kl": kl / n, "brier": brier / n}
        for k, v in (("rows", n), ("agree", agree), ("kl", kl), ("brier", brier)):
            tot[k] += v
    n = tot["rows"]
    return {"rows": n, "agree": tot["agree"] / n, "kl": tot["kl"] / n, "brier": tot["brier"] / n, "per_file": per}


@torch.inference_mode()
def mmlu_pro_accuracy(student: Student, limit: int | None = 2000, batch_size: int = 32) -> dict:
    ds = list(mmlu_pro("test"))[:limit]
    gold = {r["question_id"]: r["answer_index"] for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test")}
    correct = 0
    for i in range(0, len(ds), batch_size):
        chunk = ds[i : i + batch_size]
        logits, _, _ = student.forward_options([render(e.state, e.question) for e in chunk])
        for e, l in zip(chunk, logits):
            qid = int(e.id.rsplit("/", 1)[1])
            correct += int(l.argmax().item() == gold[qid])
    return {"rows": len(ds), "accuracy": correct / len(ds)}


@torch.inference_mode()
def latency(student: Student, n_questions: int, reps: int = 50) -> dict:
    req = SystemOneRequest(
        state="Customer writes: my order arrived broken and I need it replaced today. " * 4,
        questions={f"q{i}": NoulQuestion(type="noul", instructions=f"Is statement {i} urgent?") for i in range(n_questions)},
    )
    qs = list(req.questions.values())
    for _ in range(5):
        student.decide(req.state, qs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        student.decide(req.state, qs)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"questions": n_questions, "p50_ms": statistics.median(ts), "p99_ms": ts[math.ceil(len(ts) * 0.99) - 1]}


def run(model_id: str, labels: list[Path], *, quant: str | None, mmlu_limit: int | None, agree_limit: int | None) -> dict:
    student = Student(model_id, quant=quant)
    report: dict = {"model": model_id, "quant": quant}
    if labels:
        report["agreement"] = agreement(student, labels, limit=agree_limit)
        print(json.dumps(report["agreement"]), file=sys.stderr)
    if mmlu_limit != 0:
        report["mmlu_pro"] = mmlu_pro_accuracy(student, limit=mmlu_limit)
        print(json.dumps(report["mmlu_pro"]), file=sys.stderr)
    report["latency"] = [latency(student, 1), latency(student, 10)]
    print(json.dumps(report["latency"]), file=sys.stderr)
    return report
