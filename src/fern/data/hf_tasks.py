"""Public datasets → (state, question). Gold labels are ignored: the teacher labels."""

from __future__ import annotations

import random
import zlib
from collections.abc import Iterator

from datasets import load_dataset

from ..schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from . import Example

MAX_KEY = 160


def _choice(keys: list[str], instructions: str) -> ChoiceQuestion:
    seen: dict[str, int] = {}
    crit: dict[str, None] = {}
    for k in keys:
        k = " ".join(k.split())[:MAX_KEY] or "(empty)"
        n = seen.get(k, 0)
        seen[k] = n + 1
        crit[k if n == 0 else f"{k} ({n + 1})"] = None
    return ChoiceQuestion(type="choice", instructions=instructions, criteria=crit)


def mmlu_pro(split: str) -> Iterator[Example]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    for row in ds:
        opts = [o for o in row["options"] if o and o != "N/A"]
        if len(opts) < 2:
            continue
        yield Example(
            id=f"mmlu_pro/{split}/{row['question_id']}",
            state=f"Subject: {row['category']}\n\n{row['question']}",
            question=_choice(opts, "Which option is correct?"),
        )


def arc() -> Iterator[Example]:
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        ds = load_dataset("allenai/ai2_arc", cfg, split="train")
        for row in ds:
            yield Example(
                id=f"arc/{cfg}/{row['id']}",
                state=row["question"],
                question=_choice(row["choices"]["text"], "Which answer is correct?"),
            )


def hellaswag() -> Iterator[Example]:
    ds = load_dataset("Rowan/hellaswag", split="train")
    for row in ds:
        yield Example(
            id=f"hellaswag/{row['ind']}",
            state=f"{row['activity_label']}: {row['ctx']}",
            question=_choice(row["endings"], "Which continuation is most plausible?"),
        )


def commonsense_qa() -> Iterator[Example]:
    ds = load_dataset("tau/commonsense_qa", split="train")
    for row in ds:
        yield Example(
            id=f"csqa/{row['id']}",
            state=row["question"],
            question=_choice(row["choices"]["text"], "Which answer makes the most sense?"),
        )


def boolq() -> Iterator[Example]:
    ds = load_dataset("google/boolq", split="train")
    for i, row in enumerate(ds):
        yield Example(
            id=f"boolq/{i}",
            state=f"Passage:\n{row['passage']}",
            question=NoulQuestion(type="noul", instructions=f"Based on the passage: {row['question']}?"),
        )


def yelp() -> Iterator[Example]:
    ds = load_dataset("Yelp/yelp_review_full", split="train")
    levels = ["Terrible", "Poor", "Average", "Good", "Excellent"]
    for i, row in enumerate(ds):
        yield Example(
            id=f"yelp/{i}",
            state=f"Review:\n{row['text'][:2000]}",
            question=ScoreQuestion(type="score", instructions="How satisfied is the reviewer?", criteria=levels),
        )


def mnli() -> Iterator[Example]:
    ds = load_dataset("nyu-mll/glue", "mnli", split="train")
    crit = {"entailment": "The hypothesis follows from the premise", "neutral": "Undetermined", "contradiction": "The hypothesis conflicts with the premise"}
    for row in ds:
        yield Example(
            id=f"mnli/{row['idx']}",
            state=f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            question=ChoiceQuestion(type="choice", instructions="What is the relation between premise and hypothesis?", criteria=crit),
        )


def tweet_eval_emotion() -> Iterator[Example]:
    ds = load_dataset("cardiffnlp/tweet_eval", "emotion", split="train")
    crit = {"anger": None, "joy": None, "optimism": None, "sadness": None}
    for i, row in enumerate(ds):
        yield Example(id=f"tweet_emotion/{i}", state=f"Tweet: {row['text']}", question=ChoiceQuestion(type="choice", instructions="What emotion does the tweet express?", criteria=crit))


def mmlu() -> Iterator[Example]:
    """MMLU auxiliary_train (~100k) is mostly ARC/OBQA/RACE; use test+validation+dev of `all` for
    the real 57-subject questions (~16k) — disjoint from MMLU-Pro? No: MMLU-Pro reuses MMLU
    questions, so we exclude any question text that appears in MMLU-Pro test to keep eval clean."""
    banned = {r["question"].strip() for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test")}
    for split in ("test", "validation", "dev"):
        ds = load_dataset("cais/mmlu", "all", split=split)
        for i, row in enumerate(ds):
            if row["question"].strip() in banned:
                continue
            yield Example(
                id=f"mmlu/{split}/{i}",
                state=f"Subject: {row['subject'].replace('_', ' ')}\n\n{row['question']}",
                question=_choice(row["choices"], "Which option is correct?"),
            )


def openbookqa() -> Iterator[Example]:
    ds = load_dataset("allenai/openbookqa", "additional", split="train")
    for row in ds:
        yield Example(
            id=f"obqa/{row['id']}",
            state=f"Fact: {row['fact1']}\n\n{row['question_stem']}",
            question=_choice(row["choices"]["text"], "Which answer is correct?"),
        )


def sciq() -> Iterator[Example]:
    ds = load_dataset("allenai/sciq", split="train")
    rng = random.Random(1)
    for i, row in enumerate(ds):
        opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
        rng.shuffle(opts)
        yield Example(id=f"sciq/{i}", state=row["question"], question=_choice(opts, "Which answer is correct?"))


def gsm8k_mc() -> Iterator[Example]:
    """Numeric reasoning as a 4-way choice: correct answer plus three perturbed distractors."""
    import re

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(2)
    for i, row in enumerate(ds):
        m = re.search(r"####\s*(-?[\d,]+)", row["answer"])
        if not m:
            continue
        gold = int(m.group(1).replace(",", ""))
        pool = {gold}
        for f in (0.5, 2, 1.1, 0.9, 1.25):
            pool.add(int(round(gold * f)))
        pool.add(gold + rng.randint(1, 10))
        pool.add(gold - rng.randint(1, 10))
        pool.discard(gold)
        while len(pool) < 3:
            pool.add(gold + rng.randint(11, 100))
        opts = [str(gold)] + [str(x) for x in rng.sample(sorted(pool), 3)]
        rng.shuffle(opts)
        yield Example(id=f"gsm8k/{i}", state=row["question"], question=_choice(opts, "What is the correct numeric answer?"))


def mmlu_aux(cap: int = 30000) -> Iterator[Example]:
    """MMLU auxiliary_train (392k; RACE/ARC/OBQA-style 4-way). Rows are nested under 'train'."""
    ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    idx = list(range(len(ds)))
    random.Random(3).shuffle(idx)
    for i in idx[:cap]:
        row = ds[i]["train"]
        yield Example(id=f"mmlu_aux/{i}", state=row["question"], question=_choice(row["choices"], "Which option is correct?"))


WIDE_MIN, WIDE_MAX = 11, 26


def wide_choice(cap: int = 20000) -> Iterator[Example]:
    """11..26-way choice for the A..Z alphabet (issue 008-a): each ARC/CSQA/MMLU/OBQA question keeps
    its own options and is padded with options drawn from other questions of the *same* source,
    so distractors are plausible in genre. Exactly one original correct answer survives."""
    rng = random.Random(4)
    pools: dict[str, list[Example]] = {"arc": list(arc()), "csqa": list(commonsense_qa()), "mmlu": list(mmlu()), "obqa": list(openbookqa())}
    per = cap // len(pools) + 1
    for name, rows in pools.items():
        all_opts = [k for ex in rows for k in ex.question.criteria]
        rng.shuffle(rows)
        for ex in rows[:per]:
            own = list(ex.question.criteria)
            want = rng.randint(WIDE_MIN, WIDE_MAX)
            extra: list[str] = []
            while len(own) + len(extra) < want:
                cand = rng.choice(all_opts)
                if cand not in own and cand not in extra:
                    extra.append(cand)
            opts = own + extra
            rng.shuffle(opts)
            yield Example(id=f"wide/{ex.id}", state=ex.state, question=_choice(opts, ex.question.instructions))


XNLI_LANGS = ["ar", "de", "es", "fr", "hi", "ru", "th", "tr", "vi", "zh"]


def xnli(cap: int = 15000) -> Iterator[Example]:
    """NLI in 10 non-English languages (issue 008-g). Instructions stay English, the state is native;
    the same question shape as `mnli` so the student ties the task across languages."""
    ds = load_dataset("facebook/xnli", "all_languages", split="train")
    crit = {"entailment": "The hypothesis follows from the premise", "neutral": "Undetermined", "contradiction": "The hypothesis conflicts with the premise"}
    rng = random.Random(5)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for i in idx[:cap]:
        row = ds[i]
        lang = rng.choice(XNLI_LANGS)
        hyp = dict(zip(row["hypothesis"]["language"], row["hypothesis"]["translation"]))
        yield Example(
            id=f"xnli/{lang}/{i}",
            state=f"Premise: {row['premise'][lang]}\nHypothesis: {hyp[lang]}",
            question=ChoiceQuestion(type="choice", instructions="What is the relation between premise and hypothesis?", criteria=crit),
        )


# name -> (builder, per-source cap). Caps keep the mix balanced; MMLU-Pro test is held out for eval.
SOURCES: dict[str, tuple] = {
    "mmlu_pro_val": (lambda: mmlu_pro("validation"), None),
    "arc": (arc, None),
    "hellaswag": (hellaswag, 25000),
    "csqa": (commonsense_qa, None),
    "boolq": (boolq, None),
    "yelp": (yelp, 25000),
    "mnli": (mnli, 25000),
    "tweet_emotion": (tweet_eval_emotion, None),
    # knowledge-heavy additions (v2, issue 001/002)
    "mmlu": (mmlu, None),
    "openbookqa": (openbookqa, None),
    "sciq": (sciq, 5000),
    "gsm8k": (gsm8k_mc, 4000),
    # v3 (issue 009): more knowledge, 11..26-way choice, non-English
    "mmlu_aux": (mmlu_aux, None),
    "wide": (wide_choice, None),
    "xnli": (xnli, None),
}
KNOWLEDGE_SOURCES = ["mmlu", "openbookqa", "sciq", "gsm8k"]
EVAL_SOURCE = lambda: mmlu_pro("test")  # noqa: E731


def _is_eval(ex: Example, every: int = 20) -> bool:
    """Stable 1/`every` held-out slice keyed on the example id (independent of shuffle seed)."""
    return zlib.crc32(ex.id.encode()) % every == 0


def iter_sources(names: list[str] | None = None, seed: int = 0, split: str = "all") -> Iterator[Example]:
    """split: 'all' (v1/v2 files), 'train' (excludes the held-out slice), 'eval' (only the slice)."""
    rng = random.Random(seed)
    for name in names or list(SOURCES):
        build, cap = SOURCES[name]
        rows = list(build())
        rng.shuffle(rows)
        rows = rows[:cap] if cap else rows
        if split == "train":
            rows = [r for r in rows if not _is_eval(r)]
        elif split == "eval":
            rows = [r for r in rows if _is_eval(r)]
        yield from rows
