"""Gate for issue 009 Phase 0: does a thinking teacher beat the non-thinking one on commonsense?

Compares teacher argmax vs gold on ids present in BOTH files (hellaswag + csqa).
usage: uv run python scripts/ab_think.py data/hf.jsonl data/ab_think.jsonl
"""

import json
import sys
from collections import defaultdict

from datasets import load_dataset

gold: dict[str, int] = {}
for row in load_dataset("Rowan/hellaswag", split="train"):
    gold[f"hellaswag/{row['ind']}"] = int(row["label"])
for row in load_dataset("tau/commonsense_qa", split="train"):
    gold[f"csqa/{row['id']}"] = row["choices"]["label"].index(row["answerKey"])

files: dict[str, dict[str, tuple[int, float]]] = {}
for path in sys.argv[1:]:
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r["id"] in gold:
            p = r["teacher_probs"]
            rows[r["id"]] = (max(range(len(p)), key=p.__getitem__), max(p))
    files[path] = rows
common = set.intersection(*(set(r) for r in files.values()))
by_src: dict[str, list[str]] = defaultdict(list)
for i in common:
    by_src[i.split("/")[0]].append(i)
print(f"{len(common)} common questions: " + ", ".join(f"{k}={len(v)}" for k, v in by_src.items()))
for path, rows in files.items():
    parts = []
    for src, ids in by_src.items():
        acc = sum(rows[i][0] == gold[i] for i in ids) / len(ids)
        conf = sum(rows[i][1] for i in ids) / len(ids)
        parts.append(f"{src}: acc={acc:.3f} top-prob={conf:.3f}")
    print(f"{path}\n  " + "\n  ".join(parts))
