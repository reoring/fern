"""Progress view for scripts/label_all.sh output files."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

from .data.hf_tasks import KNOWLEDGE_SOURCES, SOURCES, mmlu_pro

V3_SOURCES = ["mmlu_aux", "wide", "xnli"]  # counted per file below, not in the v1 hf bar
V1_SOURCES = [s for s in SOURCES if s not in KNOWLEDGE_SOURCES and s not in V3_SOURCES]
# v3 label files (scripts/v3.sh) -> expected rows. synthetic files count requests × SYN_ROWS_PER_REQ.
V3_FILES = {
    "ab_think.jsonl": 4000,
    "syn_multi.jsonl": 25000,
    "eval_multi.jsonl": 1500,
    "xnli.jsonl": 14260,
    "eval_xnli.jsonl": 740,
    "wide.jsonl": 19000,
    "eval_wide.jsonl": 1000,
    "hf_aux_think.jsonl": 28500,
    "hf_choice_think.jsonl": 71000,
    "hf_rest.jsonl": 62700,
}

BAR = 24
SYN_ROWS_PER_REQ = 2.5  # observed: 8 requests → 22 rows


def _totals(data: Path) -> dict[str, int]:
    cache = data / ".totals.json"
    if cache.exists():
        return json.loads(cache.read_text())
    totals = {name: (min(cap, n) if cap else n) for name, (build, cap) in SOURCES.items() if name not in V3_SOURCES for n in [sum(1 for _ in build())]}
    totals["eval"] = sum(1 for _ in mmlu_pro("test"))
    cache.write_text(json.dumps(totals))
    return totals


def _counts(path: Path) -> Counter:
    c: Counter = Counter()
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    c[json.loads(line)["id"].split("/", 1)[0]] += 1
    return c


# id prefix -> source name
PREFIX = {"mmlu_pro": "mmlu_pro_val", "arc": "arc", "hellaswag": "hellaswag", "csqa": "csqa", "boolq": "boolq", "yelp": "yelp", "mnli": "mnli", "tweet_emotion": "tweet_emotion", "mmlu": "mmlu", "obqa": "openbookqa", "sciq": "sciq", "gsm8k": "gsm8k"}


def _bar(done: int, total: int) -> str:
    n = 0 if total == 0 else round(BAR * min(done, total) / total)
    pct = 0.0 if total == 0 else 100 * done / total
    return f"[{'#' * n}{'.' * (BAR - n)}] {pct:5.1f}%"


def _state(done: int, total: int, running: bool) -> str:
    # a few rows are skipped on purpose (over-long prompts, duplicate ids): >= 99% is complete
    return "done" if done >= total * 0.99 else ("running" if running else "pending")


def render(data: Path, syn_requests: int = 5000, prev: tuple[float, int] | None = None) -> tuple[str, tuple[float, int]]:
    """Returns (text, sample). Pass the previous sample as `prev` to get rate/ETA."""
    totals = _totals(data)
    hf = _counts(data / "hf.jsonl")
    hf_done = sum(hf.values())
    hf_total = sum(totals[s] for s in V1_SOURCES)
    syn_path, eval_path = data / "syn.jsonl", data / "eval_mmlu_pro.jsonl"
    syn_rows = sum(_counts(syn_path).values())
    syn_req = len({json.loads(l)["id"].rsplit("/", 2)[1] for l in syn_path.open() if l.strip()}) if syn_path.exists() else 0
    eval_done = sum(_counts(eval_path).values())
    syn_total_rows = int(syn_requests * SYN_ROWS_PER_REQ)

    hf_running = hf_done < hf_total * 0.99
    syn_running = not hf_running and syn_req < syn_requests * 0.99
    eval_running = not hf_running and not syn_running and eval_done < totals["eval"] * 0.99

    lines = [f"hf   {_bar(hf_done, hf_total)}  {_state(hf_done, hf_total, hf_running):8s} {hf_done:6d} / {hf_total} rows"]
    for name in V1_SOURCES:
        prefix = next(p for p, s in PREFIX.items() if s == name)
        lines.append(f"      {name:16s} {hf.get(prefix, 0):5d} / {totals[name]}")
    lines.append(
        f"syn  {_bar(syn_req, syn_requests)}  {_state(syn_req, syn_requests, syn_running):8s} {syn_rows:6d} rows  "
        f"{syn_req}/{syn_requests} requests  (~{syn_total_rows} rows expected)"
    )
    know_path = data / "hf_knowledge_think.jsonl"
    know_done = know_total = 0
    if know_path.exists():
        know = _counts(know_path)
        know_done, know_total = sum(know.values()), sum(totals[s] for s in KNOWLEDGE_SOURCES)
        lines.append(f"know {_bar(know_done, know_total)}  {_state(know_done, know_total, know_done < know_total * 0.99):8s} {know_done:6d} / {know_total} rows  (thinking teacher)")
        for name in KNOWLEDGE_SOURCES:
            prefix = next(p for p, s in PREFIX.items() if s == name)
            lines.append(f"      {name:16s} {know.get(prefix, 0):5d} / {totals[name]}")
    lines.append(f"eval {_bar(eval_done, totals['eval'])}  {_state(eval_done, totals['eval'], eval_running):8s} {eval_done:6d} / {totals['eval']} rows")
    think_path = data / "eval_mmlu_pro_think512.jsonl"
    think_done, think_total = (sum(_counts(think_path).values()), 2000) if think_path.exists() else (0, 0)
    if think_total:
        know_running = 0 < know_done < know_total * 0.99
        think_running = not know_running and think_done < think_total * 0.99
        lines.append(f"tevl {_bar(think_done, think_total)}  {_state(think_done, think_total, think_running):8s} {think_done:6d} / {think_total} rows  (eval, thinking teacher)")
    v3_done = v3_total = 0
    for fname, expect in V3_FILES.items():
        p = data / fname
        if not p.exists():
            continue
        n = sum(_counts(p).values())
        v3_done, v3_total = v3_done + n, v3_total + expect
        lines.append(f"v3   {_bar(n, expect)}  {_state(n, expect, n < expect * 0.99):8s} {n:6d} / {expect} rows  {fname}")
    lines.extend(_train_lines(data.parent / "runs"))
    total_done = hf_done + syn_rows + eval_done + know_done + think_done + v3_done
    stages = [(hf_done, hf_total), (syn_req, syn_requests), (eval_done, totals["eval"]), (know_done, know_total), (think_done, think_total), (v3_done, v3_total)]
    remaining = sum(max(0, t - d) for d, t in stages if d < t * 0.99)  # stages within 1% count as finished
    now = time.time()
    if prev is not None and remaining > 0:
        dt, drows = now - prev[0], total_done - prev[1]
        if dt > 0 and drows > 0:
            rate = drows / dt

            eta = remaining / rate
            done_at = time.strftime("%H:%M", time.localtime(now + eta))
            lines.append(f"\n{rate:.2f} rows/s   remaining ~{remaining} rows   ETA {eta / 3600:.1f}h (≈{done_at})   {time.strftime('%H:%M:%S')}")
        else:
            lines.append(f"\n(measuring rate...)   {time.strftime('%H:%M:%S')}")
    else:
        lines.append(f"\n{time.strftime('%H:%M:%S')}")
    return "\n".join(lines), (now, total_done)


TRAIN_RE = re.compile(r"^\[train\] step (?P<step>\d+)/(?P<total>\d+)\s+loss=(?P<loss>[\d.]+)\s+agree=(?P<agree>[\d.]+).*?(?P<tps>\d+) tok/s")
EVAL_RE = re.compile(r"^\[eval\] step (?P<step>\d+)\s+kl=(?P<kl>[\d.]+)\s+agree=(?P<agree>[\d.]+)")
STAGE_RE = re.compile(r"^(?P<ts>\S+) === (?P<stage>train|eval) (?P<run>\S+) ===")


def _train_lines(runs: Path) -> list[str]:
    """Parse runs/pipeline.log: latest train/eval stage, last step line, before/after eval."""
    log = runs / "pipeline.log"
    if not log.exists():
        return []
    stage = run = None
    started = 0.0
    last_train = None
    evals: list[re.Match] = []
    final: str | None = None
    with log.open() as f:
        for line in f:
            if m := STAGE_RE.match(line):
                stage, run = m["stage"], m["run"]
                started = time.mktime(time.strptime(m["ts"], "%Y-%m-%dT%H:%M:%S"))
                if stage == "train":
                    last_train, evals, final = None, [], None
            elif m := TRAIN_RE.match(line):
                last_train = m
            elif m := EVAL_RE.match(line):
                evals.append(m)
            elif "pipeline complete" in line:
                final = "complete"
            elif "train failed" in line or "eval failed" in line:
                final = line.split(" ", 1)[1].strip()
    if stage is None:
        return []
    out = []
    if last_train:
        step, total = int(last_train["step"]), int(last_train["total"])
        finished = step >= total or any(int(e["step"]) >= total for e in evals)
        if finished:
            step, state = total, "done"
        else:
            state = "running" if stage == "train" and not final else "stopped"
        eta = ""
        if state == "running" and step:
            eta = f"   ETA {(time.time() - started) / step * (total - step) / 3600:.1f}h"
        name = Path(run or "").name
        out.append(
            f"train {_bar(step, total)}  {state:8s} {step:5d} / {total} steps  loss={last_train['loss']}  "
            f"agree={last_train['agree']}  {last_train['tps']} tok/s{eta}  [{name}]"
        )
    elif stage == "train":
        out.append(f"train [........................]   0.0%  loading model / step 0 eval  [{Path(run or '').name}]")
    for e in evals:
        out.append(f"      held-out @step {e['step']:>5}: kl={e['kl']}  agree={e['agree']}")
    eval_json = runs / Path(run or "").name / "eval.json"
    if eval_json.exists():
        r = json.loads(eval_json.read_text())
        a, mp, lat = r.get("agreement", {}), r.get("mmlu_pro", {}), r.get("latency", [])
        out.append(
            f"eval  done   teacher-agree={a.get('agree', float('nan')):.3f}  kl={a.get('kl', float('nan')):.3f}  "
            f"mmlu_pro={mp.get('accuracy', float('nan')):.3f}  "
            + "  ".join(f"{l['questions']}q p50={l['p50_ms']:.0f}ms" for l in lat)
            + f"  [{Path(run or '').name}]"
        )
    elif stage == "eval" and not final:
        out.append(f"eval  running  (agreement → MMLU-Pro → latency)  [{Path(run or '').name}]")
    if final == "complete" and next_run_pending(runs):
        out.append("train pending  [v2] — starts after labeling stages")
    if final and final != "complete":
        out.append(f"!! {final}")
    return out


def next_run_pending(runs: Path) -> bool:
    """v2.sh logged '=== v2 start ===' but no '=== train runs/v2' yet."""
    log = (runs / "pipeline.log").read_text()
    return "=== v2 start ===" in log and "=== train runs/v2 ===" not in log
