"""KL(teacher || student) on option-restricted logits. Full FT or LoRA."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import TypeAdapter

from .prompting import Rendered, render
from .schema import Question
from .student import Student, collate, option_logits

QuestionAdapter = TypeAdapter(Question)


def load_rows(paths: list[Path], weights: dict[str, float] | None = None) -> list[dict]:
    """Rows from all files. A weight w>1 for a file repeats its rows (floor) plus a random
    remainder so the file is seen ~w times per epoch; w<1 subsamples."""
    rows = []
    rng = random.Random(0)
    for p in paths:
        with p.open() as f:
            file_rows = [json.loads(l) for l in f if l.strip()]
        w = (weights or {}).get(str(p), 1.0)
        whole, frac = int(w), w - int(w)
        rows.extend(file_rows * whole)
        rows.extend(rng.sample(file_rows, int(len(file_rows) * frac)))
    return rows


def rendered_of(row: dict) -> tuple[Rendered, torch.Tensor]:
    q = QuestionAdapter.validate_python(row["question"])
    return render(row["state"], q), torch.tensor(row["teacher_probs"], dtype=torch.float32)


def distill_loss(logits: list[torch.Tensor], targets: list[torch.Tensor], ce_weight: float) -> tuple[torch.Tensor, float]:
    kl, ce, agree = 0.0, 0.0, 0
    for l, t in zip(logits, targets):
        t = t.to(l.device)
        logp = F.log_softmax(l, dim=0)
        kl = kl + F.kl_div(logp, t, reduction="sum")
        ce = ce + -logp[t.argmax()]
        agree += int(l.argmax() == t.argmax())
    n = len(logits)
    return (kl + ce_weight * ce) / n, agree / n


def train(
    data: list[Path],
    model_id: str,
    out: Path,
    *,
    lora: int = 0,
    epochs: float = 2,
    lr: float | None = None,
    batch_size: int = 16,
    grad_accum: int = 4,
    max_len: int = 1024,
    ce_weight: float = 0.0,
    warmup: float = 0.03,
    eval_frac: float = 0.02,
    max_steps: int | None = None,
    seed: int = 0,
    log_every: int = 10,
    grad_ckpt: bool = True,
    data_weight: dict[str, float] | None = None,
) -> Path:
    torch.manual_seed(seed)
    rows = load_rows(data, data_weight)
    random.Random(seed).shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_frac))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    print(f"[train] {len(train_rows)} train / {len(eval_rows)} eval rows from {len(data)} files", file=sys.stderr)

    student = Student(model_id, max_len=max_len)
    model, tok, device = student.model, student.tok, student.device
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if lora:
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=lora, lora_alpha=2 * lora, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    lr = lr or (2e-4 if lora else 1e-5)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)

    steps_per_epoch = math.ceil(len(train_rows) / (batch_size * grad_accum))
    total = max_steps or math.ceil(steps_per_epoch * epochs)
    warm = max(1, int(total * warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )

    def batches(rs: list[dict]):
        for i in range(0, len(rs), batch_size):
            chunk = [rendered_of(r) for r in rs[i : i + batch_size]]
            yield [c[0] for c in chunk], [c[1] for c in chunk]

    @torch.inference_mode()
    def evaluate() -> tuple[float, float]:
        model.eval()
        losses, agrees = [], []
        for rend, tg in batches(eval_rows):
            b = collate(tok, rend, max_len, device)
            logits = option_logits(model(input_ids=b.input_ids, attention_mask=b.attention_mask, logits_to_keep=1).logits[:, -1].float(), b.label_ids)
            loss, agree = distill_loss(logits, tg, 0.0)
            losses.append(loss.item()); agrees.append(agree)
        model.train()
        return sum(losses) / len(losses), sum(agrees) / len(agrees)

    kl0, agree0 = evaluate()
    print(f"[eval] step 0  kl={kl0:.4f}  agree={agree0:.3f}", file=sys.stderr)

    model.train()
    step, micro, t0, tokens = 0, 0, time.time(), 0
    run_loss, run_agree = 0.0, 0.0
    done = False
    while not done:
        for rend, tg in batches(train_rows):
            b = collate(tok, rend, max_len, device)
            tokens += int(b.attention_mask.sum())
            logits = option_logits(model(input_ids=b.input_ids, attention_mask=b.attention_mask, logits_to_keep=1).logits[:, -1].float(), b.label_ids)
            loss, agree = distill_loss(logits, tg, ce_weight)
            (loss / grad_accum).backward()
            run_loss += loss.item() / grad_accum; run_agree += agree / grad_accum
            micro += 1
            if micro % grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0:
                dt = time.time() - t0
                print(f"[train] step {step}/{total}  loss={run_loss / log_every:.4f}  agree={run_agree / log_every:.3f}  lr={sched.get_last_lr()[0]:.2e}  {tokens / dt:.0f} tok/s", file=sys.stderr)
                run_loss = run_agree = 0.0
            if step >= total:
                done = True
                break
        random.Random(seed + step).shuffle(train_rows)

    kl1, agree1 = evaluate()
    print(f"[eval] step {step}  kl={kl1:.4f}  agree={agree1:.3f}  (was kl={kl0:.4f} agree={agree0:.3f})", file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)
    if lora:
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "train_meta.json").write_text(json.dumps({
        "base": model_id, "lora": lora, "steps": step, "rows": len(train_rows), "lr": lr,
        "eval": {"kl_before": kl0, "agree_before": agree0, "kl_after": kl1, "agree_after": agree1},
        "data": [str(p) for p in data],
    }, indent=2))
    return out
