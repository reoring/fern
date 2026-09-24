# fern

A 4B decision model with a Jev-compatible API. Send a `state` and up to 50 questions
(`choice` / `score` / `noul`); get back probability distributions in one forward pass —
no text generation, ~30 ms per request on one GPU.

Distilled from DeepSeek V4 Flash (teacher, 2-bit GGUF on a single 96 GB GPU) into
Qwen3.5-4B by matching the teacher's next-token distribution over the answer options.

## Quick start

```bash
uv sync
uv run hf download reoring/fern --local-dir runs/fern
uv run fern serve runs/fern                                  # http://127.0.0.1:8000
FERN_API_KEYS=k1,k2 uv run fern serve runs/fern --host 0.0.0.0   # require `Authorization: Bearer k1`
```

```bash
curl -sS http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades"}},
    "severity": {"type": "score", "instructions": "How severe is this?", "criteria": ["Low", "Medium", "High", "Critical"]},
    "urgent":   {"type": "noul", "instructions": "Does this need a reply today?"}
  }
}'
```

```json
{
  "model": "runs/fern",
  "answers": {
    "team":     {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.93, "technical": 0.07, "sales": 0.00}, "confidence": 0.90},
    "severity": {"type": "score", "score": 2.35, "legend": {"0": "Low", "1": "Medium", "2": "High", "3": "Critical"},
                 "probabilities": {"0": 0.01, "1": 0.07, "2": 0.49, "3": 0.43}, "confidence": 0.32},
    "urgent":   {"type": "noul", "noul": 0.97}
  },
  "usage": {"input_tokens": 278, "output_tokens": 0, "truncated": false}
}
```

`GET /health` returns the loaded model and last request latency; it is never authenticated.

## API semantics

Request/response follow the public Jev schema, so clients written for litjev or the hosted
API work unchanged within these limits:

- `choice.confidence` / `score.confidence` = `(K·max − 1)/(K − 1)` (0 = uniform, 1 = one-hot), the approximation given in the Jev docs, so Jev thresholds carry over.
- `score.score` = probability-weighted level index; `noul.noul` = P(yes).
- `usage.output_tokens` is always 0: nothing is generated, one forward pass reads the option logits.
- `usage.truncated: true` means the prompt exceeded `--max-len` (1,024 tokens) and the **start of `state`** was dropped (left truncation keeps the question). Treat such answers as unreliable; split or shorten `state`.
- Same input → same output (no sampling).
- `state` / `instructions` / `criteria` accept strings or JSON values. Non-strings are rendered as indented JSON text for the model, and `legend` echoes that text. `state` must be non-empty.
- **Max 10 options per choice/score** (Jev: 255). Teacher labels and the student use answer codes A–J / 0–9 only; more options would need re-labeling and re-training.
- 1–50 questions per request. Each question is its own prompt in one batched forward, so latency grows ~linearly: 1q ≈ 28 ms, 10q ≈ 48 ms, 50q ≈ 700 ms on an RTX PRO 6000.
- Lenient validation, unlike litjev (`extra="forbid"`): unknown top-level fields are ignored, `model` is optional and ignored, the response `model` is the loaded checkpoint path.
- Training data is English only; other languages work but with lower confidence.
- Auth: none by default (local use). With `FERN_API_KEYS` set (comma-separated), `/v1/systemone` requires `Authorization: Bearer <key>` (401 otherwise).

## Results

| | fern (v2) |
|---|---|
| Agreement with teacher, held-out training distribution | 0.95 (KL 0.07) |
| Agreement with thinking teacher, MMLU-Pro (2k) | 0.50 |
| MMLU-Pro accuracy (2k, 10-way) | 0.43 |
| Latency, 1 question / 10 questions (p50) | 28 ms / 48 ms |

It is a small model tuned for short states with clear criteria (routing, triage, labeling,
safety flags). Multi-step reasoning and long-document judgments are outside its range.

## Reproduce

One 96 GB GPU. Labeling ~7 h, training ~4 h per epoch.

```bash
scripts/teacher_up.sh UD-IQ2_XXS 8080     # downloads ~91 GB, serves the teacher on :8080
scripts/label_all.sh                      # → data/hf.jsonl data/syn.jsonl data/eval_mmlu_pro.jsonl
uv run fern progress --watch 10           # rows per stage, rate, ETA
# stop the teacher, then:
uv run fern train runs/v1 --data data/hf.jsonl --data data/syn.jsonl
uv run fern eval  runs/v1 --labels data/eval_mmlu_pro.jsonl --out runs/v1/eval.json
scripts/v2.sh                             # second epoch with thinking-teacher labels on knowledge sources
```

The teacher never generates text: for each example the server reads the next-token
distribution at the answer position over the option codes (`n_probs`), and the student is
trained with `KL(p_teacher ‖ p_student)` restricted to those codes. Sources: HellaSwag,
MNLI, Yelp reviews, CommonsenseQA, BoolQ, ARC, TweetEval, MMLU, OpenBookQA, SciQ, GSM8K
(multiple-choice form), plus ~15k synthetic Jev-style requests written by the teacher.

## License

Apache-2.0 (code). The model is derived from [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
(Apache-2.0) with soft labels from [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
(MIT). Training data includes Yelp reviews, whose dataset terms are non-commercial;
check them before commercial use of the weights.
