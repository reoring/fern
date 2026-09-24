from __future__ import annotations

import asyncio
import glob
import json
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def _paths(patterns: list[str]) -> list[Path]:
    out = [Path(p) for pat in patterns for p in sorted(glob.glob(pat))]
    if not out:
        raise typer.BadParameter(f"no files match {patterns}")
    return out


@app.command()
def label(
    out: Path,
    source: str = typer.Option("hf", help="hf | synthetic | eval"),
    names: str = typer.Option("", help="comma-separated hf source names (default: all)"),
    n: int = typer.Option(20000, help="synthetic: number of generated requests"),
    limit: int | None = typer.Option(None, help="stop after this many new rows"),
    teacher_url: str = typer.Option("http://127.0.0.1:8080"),
    teacher_name: str = typer.Option("DeepSeek-V4-Flash-UD-IQ2_XXS"),
    concurrency: int = 8,
    think: int = typer.Option(0, help="teacher thinking budget in tokens (0 = read logits directly)"),
    seed: int = 0,
) -> None:
    """Label examples with the teacher's option distribution → jsonl."""
    from .data import hf_tasks, synthetic
    from .label import label as _label
    from .teacher import Teacher

    teacher = Teacher(teacher_url, concurrency=concurrency, think=think)

    async def main() -> int:
        if source == "hf":
            ex = hf_tasks.iter_sources(names.split(",") if names else None, seed=seed)
        elif source == "eval":
            ex = hf_tasks.EVAL_SOURCE()
        elif source == "synthetic":
            ex = synthetic.generate(teacher, n, out.with_name("syn_requests.jsonl"), seed=seed)
        else:
            raise typer.BadParameter(source)
        try:
            return await _label(ex, out, teacher=teacher, teacher_name=teacher_name, limit=limit)
        finally:
            await teacher.close()

    typer.echo(f"wrote {asyncio.run(main())} rows to {out}")


@app.command()
def progress(
    data_dir: Path = typer.Option(Path("data"), help="where the label stages write"),
    syn_n: int = typer.Option(5000, help="synthetic request count (SYN_N)"),
    watch: float = typer.Option(0, help="refresh every N seconds (0 = print once)"),
    window: float = typer.Option(10, help="one-shot: seconds to sample for the rate (0 = no rate/ETA)"),
) -> None:
    """Show labeling progress: rows per stage, rate, ETA."""
    import time

    from .progress import render

    text, prev = render(data_dir, syn_n)
    if not watch:
        if window > 0:
            time.sleep(window)
            text, prev = render(data_dir, syn_n, prev)
        typer.echo(text)
        return
    try:
        while True:
            typer.echo("\x1b[H\x1b[2J" + text)
            time.sleep(watch)
            text, prev = render(data_dir, syn_n, prev)
    except KeyboardInterrupt:
        pass


@app.command()
def train(
    out: Path,
    data: list[str] = typer.Option(..., help="jsonl globs"),
    model: str = "Qwen/Qwen3.5-4B",
    lora: int = typer.Option(0, help="LoRA rank; 0 = full fine-tune"),
    epochs: float = 2.0,
    lr: float | None = None,
    batch_size: int = 16,
    grad_accum: int = 4,
    max_len: int = 1024,
    ce_weight: float = 0.0,
    max_steps: int | None = None,
    seed: int = 0,
    eval_frac: float = 0.02,
    grad_ckpt: bool = typer.Option(True, help="gradient checkpointing; required at max_len 1024 on 96GB"),
    log_every: int = 10,
    data_weight: list[str] = typer.Option([], help="per-file sampling weight, e.g. data/hf_knowledge.jsonl=3"),
) -> None:
    """Distill teacher labels into the student."""
    from .train import train as _train

    weights = {}
    for spec in data_weight:
        path, _, w = spec.rpartition("=")
        weights[str(Path(path))] = float(w)
    _train(_paths(data), model, out, lora=lora, epochs=epochs, lr=lr, batch_size=batch_size, grad_accum=grad_accum,
           max_len=max_len, ce_weight=ce_weight, max_steps=max_steps, eval_frac=eval_frac, seed=seed,
           grad_ckpt=grad_ckpt, log_every=log_every, data_weight=weights)


@app.command()
def serve(model: str, host: str = "127.0.0.1", port: int = 8000, quant: str | None = None, max_len: int = 1024) -> None:
    """Serve POST /v1/systemone. Set FERN_API_KEYS=key1,key2 to require Bearer auth."""
    import uvicorn

    from .serve import api_keys_from_env, create_app

    uvicorn.run(create_app(model, quant=quant, max_len=max_len, api_keys=api_keys_from_env()), host=host, port=port)


@app.command()
def eval(
    model: str,
    labels: list[str] = typer.Option([], help="held-out teacher-labeled jsonl globs"),
    quant: str | None = None,
    mmlu_limit: int | None = typer.Option(2000, help="0 to skip"),
    agree_limit: int | None = None,
    out: Path | None = None,
) -> None:
    """Teacher agreement, MMLU-Pro accuracy, latency."""
    from .evaluate import run

    report = run(model, _paths(labels) if labels else [], quant=quant, mmlu_limit=mmlu_limit, agree_limit=agree_limit)
    text = json.dumps(report, indent=2)
    typer.echo(text)
    if out:
        out.write_text(text)


