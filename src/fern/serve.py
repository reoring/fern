"""POST /v1/systemone — all questions of a request in one forward."""

from __future__ import annotations

import os
import secrets
import time

from fastapi import Depends, FastAPI, HTTPException, Request

from .schema import SystemOneRequest, SystemOneResponse, Usage, build_answer
from .student import Student


def create_app(model_id: str, *, quant: str | None = None, max_len: int = 1024, api_keys: frozenset[str] = frozenset()) -> FastAPI:
    """`api_keys` empty → open (local use). Otherwise every /v1 call needs `Authorization: Bearer <key>`."""
    app = FastAPI(title="fern")
    student = Student(model_id, quant=quant, max_len=max_len)
    # warm up kernels so the first real request is not an outlier
    student.decide("warmup", [SystemOneRequest.model_validate({"state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}).questions["q"]])

    def require_key(request: Request) -> None:
        if not api_keys:
            return
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        if not any(secrets.compare_digest(token, k) for k in api_keys):
            raise HTTPException(401, "invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(require_key)])
    def systemone(req: SystemOneRequest) -> SystemOneResponse:
        t0 = time.perf_counter()
        ids = list(req.questions)
        probs, n_tokens, truncated = student.decide(req.state, [req.questions[i] for i in ids])
        resp = SystemOneResponse(
            model=model_id,
            answers={i: build_answer(req.questions[i], p) for i, p in zip(ids, probs)},
            usage=Usage(input_tokens=n_tokens, truncated=truncated),
        )
        app.state.last_ms = (time.perf_counter() - t0) * 1000
        return resp

    @app.get("/health")
    def health() -> dict:
        return {"model": model_id, "last_ms": getattr(app.state, "last_ms", None)}

    return app


def api_keys_from_env() -> frozenset[str]:
    """FERN_API_KEYS: comma-separated; unset/empty → no auth."""
    return frozenset(k.strip() for k in os.environ.get("FERN_API_KEYS", "").split(",") if k.strip())
