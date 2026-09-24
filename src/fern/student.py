"""Student = HF causal LM read like a decision head.

For each (state, question) we render the same chat prompt as the teacher, run one
forward, and take the logits at the final position restricted to the option labels.
Each label's logit is logsumexp over its single-token variants (" A" and "A").
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .prompting import Rendered, render
from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

THINK_CLOSE = "<think>\n\n</think>\n\n"


@functools.lru_cache(maxsize=None)
def label_token_ids(tok: PreTrainedTokenizerBase, label: str) -> list[int]:
    ids: list[int] = []
    for variant in (" " + label, label):
        enc = tok.encode(variant, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(enc[0])
    if not ids:
        raise ValueError(f"label {label!r} is not a single token in {tok.name_or_path}")
    return ids


_SYS_MARK, _USER_MARK = "\x00SYS\x00", "\x00USER\x00"


@functools.lru_cache(maxsize=None)
def template_skeleton(tok: PreTrainedTokenizerBase) -> tuple[str, str, str]:
    """Render the chat template once; later prompts are string substitutions.

    Chat templates for Qwen3.5-class models take ~100ms per jinja render, which
    dwarfs the model forward. Returns (head, mid, tail) such that the prompt for
    (system, user) is head + system + mid + user + tail.
    """
    text = tok.apply_chat_template(
        [{"role": "system", "content": _SYS_MARK}, {"role": "user", "content": _USER_MARK}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    if "<think>" in tok.get_vocab() and "<think>" not in text[-64:] and "</think>" not in text[-64:]:
        text += THINK_CLOSE
    head, rest = text.split(_SYS_MARK)
    mid, tail = rest.split(_USER_MARK)
    return head, mid, tail


def prompt_text(tok: PreTrainedTokenizerBase, r: Rendered) -> str:
    head, mid, tail = template_skeleton(tok)
    system, user = r.messages[0]["content"], r.messages[1]["content"]
    return head + system + mid + user + tail + r.prefix


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label_ids: list[list[list[int]]]  # per example, per option, token variants
    truncated: bool = False


def collate(tok: PreTrainedTokenizerBase, rendered: list[Rendered], max_len: int, device: torch.device) -> Batch:
    texts = [prompt_text(tok, r) for r in rendered]
    tok.padding_side = "left"
    tok.truncation_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return Batch(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        label_ids=[[label_token_ids(tok, lab) for lab in r.labels] for r in rendered],
        # a row that fills max_len exactly has (almost certainly) lost tokens on the left
        truncated=bool((enc["attention_mask"].sum(1) >= max_len).any()),
    )


def option_logits(last_logits: torch.Tensor, label_ids: list[list[list[int]]]) -> list[torch.Tensor]:
    """last_logits: [B, V] → list of [n_options] tensors (ragged across examples)."""
    out = []
    for b, opts in enumerate(label_ids):
        row = last_logits[b]
        out.append(torch.stack([torch.logsumexp(row[ids], dim=0) for ids in opts]))
    return out


class Student:
    def __init__(self, model_id: str, *, dtype: torch.dtype = torch.bfloat16, device: str = "cuda", quant: str | None = None, max_len: int = 1024):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs: dict = {"dtype": dtype}
        if quant == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **kwargs)
        self.device = torch.device(device)
        self.max_len = max_len

    def forward_options(self, rendered: list[Rendered]) -> tuple[list[torch.Tensor], int, bool]:
        batch = collate(self.tok, rendered, self.max_len, self.device)
        out = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, logits_to_keep=1)
        return option_logits(out.logits[:, -1].float(), batch.label_ids), int(batch.attention_mask.sum()), batch.truncated

    @torch.inference_mode()
    def decide(self, state: str, questions: list[ChoiceQuestion | ScoreQuestion | NoulQuestion]) -> tuple[list[list[float]], int, bool]:
        logits, n_tokens, truncated = self.forward_options([render(state, q) for q in questions])
        return [torch.softmax(l, dim=0).tolist() for l in logits], n_tokens, truncated
