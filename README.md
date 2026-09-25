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
- **Max 26 options per `choice`, 10 levels per `score`** (Jev: 255). Answer codes must be single tokens (A–Z, 0–9); more would need a new code alphabet and re-training.
- 1–50 questions per request. Each question is its own prompt in one batched forward, so latency grows ~linearly: 1q ≈ 28 ms, 10q ≈ 48 ms, 50q ≈ 700 ms on an RTX PRO 6000.
- Lenient validation, unlike litjev (`extra="forbid"`): unknown top-level fields are ignored, `model` is optional and ignored, the response `model` is the loaded checkpoint path.
- Training data is mostly English, with NLI in 10 languages (XNLI) and ~25k synthetic requests in ja/zh/de/es/fr/ko. Teacher agreement on held-out non-English requests is ~0.77 vs ~0.89 on English (see Results).
- Auth: none by default (local use). With `FERN_API_KEYS` set (comma-separated), `/v1/systemone` requires `Authorization: Bearer <key>` (401 otherwise).

## Results

Agreement = argmax matches the teacher's argmax on held-out label files.

| | v2 | v3 |
|---|---|---|
| Held-out training distribution | 0.95 | 0.94 |
| Thinking teacher, MMLU-Pro (2k, 10-way) | 0.50 | 0.49 |
| MMLU-Pro accuracy (2k) | 0.43 | 0.42 |
| 11–26-way choice (920) | 0.83 | 0.84 |
| XNLI, 10 languages (740) | 0.82 | 0.84 |
| Synthetic requests, 6 non-English languages (1.5k) | 0.76 | 0.77 |
| Latency, 1 / 10 questions (p50, RTX PRO 6000) | 28 / 48 ms | 28 / 48 ms |

v3 adds 26-way choice, XNLI, MMLU auxiliary_train and multilingual synthetic data to the
mix and is trained from the base model. Most of the 26-way and multilingual capability is
already present in v2 (the base model generalises over the code alphabet); v3 is 1–2 pt
better there and 1 pt worse on MMLU-Pro.

It is a small model tuned for short states with clear criteria (routing, triage, labeling,
safety flags). Multi-step reasoning and long-document judgments are outside its range.

## Reproduce

One 96 GB GPU. v3: labeling ~15 h (multilingual generation and thinking labels dominate), training ~18 h.

```bash
scripts/teacher_up.sh UD-IQ2_XXS 8080     # downloads ~91 GB, serves the teacher on :8080
scripts/label_all.sh                      # → data/hf.jsonl data/syn.jsonl data/eval_mmlu_pro.jsonl
uv run fern progress --watch 10           # rows per stage, rate, ETA
# stop the teacher, then:
uv run fern train runs/v1 --data data/hf.jsonl --data data/syn.jsonl
uv run fern eval  runs/v1 --labels data/eval_mmlu_pro.jsonl --out runs/v1/eval.json
scripts/v2.sh                             # + thinking-teacher labels on knowledge sources, one more epoch
scripts/v3.sh                             # + 26-way choice, XNLI, MMLU aux, multilingual synthetic; train from base
```

The teacher never generates text for labels: for each example the server reads the
next-token distribution at the answer position over the option codes (`n_probs`), and the
student is trained with `KL(p_teacher ‖ p_student)` restricted to those codes. The only
decoding is for synthetic requests, which the teacher writes as JSON. Sources: HellaSwag,
MNLI, XNLI, Yelp reviews, CommonsenseQA, BoolQ, ARC, TweetEval, MMLU (+ auxiliary_train),
OpenBookQA, SciQ, GSM8K (multiple-choice form), 11–26-way variants of ARC/CSQA/MMLU/OBQA
padded with same-source distractors, plus ~40k synthetic Jev-style requests in 7 languages.

## License

Apache-2.0 (code). The model is derived from [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
(Apache-2.0) with soft labels from [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
(MIT). Training data includes Yelp reviews, whose dataset terms are non-commercial;
check them before commercial use of the weights.
