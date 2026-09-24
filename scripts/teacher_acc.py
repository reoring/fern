"""Teacher MMLU-Pro accuracy from label files, restricted to ids common to all files.
usage: uv run python scripts/teacher_acc.py data/eval_mmlu_pro.jsonl data/eval_mmlu_pro_iq2m.jsonl
"""

import json
import sys

from datasets import load_dataset

gold = {r["question_id"]: r["answer_index"] for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test")}
files = {}
for path in sys.argv[1:]:
    rows = {}
    for line in open(path):
        r = json.loads(line)
        p = r["teacher_probs"]
        rows[r["id"]] = (max(range(len(p)), key=p.__getitem__), max(p))
    files[path] = rows
common = set.intersection(*(set(r) for r in files.values()))
print(f"{len(common)} common questions")
for path, rows in files.items():
    acc = sum(rows[i][0] == gold[int(i.rsplit('/', 1)[1])] for i in common) / len(common)
    conf = sum(rows[i][1] for i in common) / len(common)
    print(f"{path}: acc={acc:.4f}  mean top-prob={conf:.3f}")
