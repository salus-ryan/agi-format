"""Authenticated, session-isolated FastAPI server for the .agi runtime."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import sys
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

sys.path.insert(0, "/app")

import httpx
import structlog
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    Security,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from runtime.security import SlidingWindowRateLimiter

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "")
MEMORY_PATH = Path(os.environ.get("MEMORY_PATH", "/agent-memory"))
AGI_PACKAGE_PATH = os.environ.get("AGI_PACKAGE_PATH", "/app/skeleton")
GPU = os.environ.get("GPU", "L4")
API_PORT = int(os.environ.get("API_PORT", "8080"))
LORA_ADAPTER_PATH = Path(os.environ.get("LORA_ADAPTER_PATH", "/agent-memory/lora-adapter"))
API_KEY = os.environ.get("AGI_API_KEY", "")
AUTH_REQUIRED = os.environ.get("AGI_REQUIRE_AUTH", "true").lower() == "true"
MAX_SESSIONS = max(1, min(int(os.environ.get("AGI_MAX_SESSIONS", "32")), 256))
RATE_LIMIT_REQUESTS = max(1, int(os.environ.get("AGI_RATE_LIMIT_REQUESTS", "20")))
RATE_LIMIT_WINDOW = max(1, int(os.environ.get("AGI_RATE_LIMIT_WINDOW_SECONDS", "60")))
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16_000)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)


class ChatResponse(BaseModel):
    response: str
    step_log: dict
    state: dict
    session_id: str


class SessionAccess(BaseModel):
    session_id: str
    identity: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    for kernel in list(_kernels.values()):
        close = getattr(kernel, "close", None)
        if close:
            await close()
    _kernels.clear()
    _kernel_locks.clear()


app = FastAPI(
    title=".agi Reference Agent",
    version="0.2.0",
    docs_url=None if os.environ.get("AGI_DISABLE_DOCS", "true").lower() == "true" else "/docs",
    redoc_url=None,
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.environ.get("AGI_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Session-ID", "X-Request-ID"],
    )

bearer = HTTPBearer(auto_error=False)
rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
_kernels: OrderedDict[str, object] = OrderedDict()
_kernel_locks: dict[str, asyncio.Lock] = {}
_registry_lock = asyncio.Lock()
_training_lock = asyncio.Lock()
_started_at = time.monotonic()


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", request_id):
        request_id = uuid.uuid4().hex
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_failed", request_id=request_id, path=request.url.path)
        raise
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    log.info(
        "request_complete",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        status=response.status_code,
        elapsed_ms=round((time.monotonic() - started) * 1000, 1),
    )
    return response


def _authenticate(credentials: HTTPAuthorizationCredentials | None) -> str:
    if not AUTH_REQUIRED and not API_KEY:
        return "development"
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )
    supplied = credentials.credentials if credentials and credentials.scheme.lower() == "bearer" else ""
    if not supplied or not hmac.compare_digest(supplied, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return hashlib.sha256(supplied.encode("utf-8")).hexdigest()[:16]


async def require_access(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
    x_session_id: Annotated[str, Header(alias="X-Session-ID")] = "default",
) -> SessionAccess:
    identity = _authenticate(credentials)
    if not SESSION_PATTERN.fullmatch(x_session_id):
        raise HTTPException(status_code=400, detail="Invalid X-Session-ID")
    allowed, retry_after = rate_limiter.check(identity)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
    return SessionAccess(session_id=x_session_id, identity=identity)


async def get_kernel(session_id: str):
    """Get a session-isolated kernel, creating it once under a registry lock."""
    async with _registry_lock:
        if session_id in _kernels:
            kernel = _kernels.pop(session_id)
            _kernels[session_id] = kernel
            return kernel
        if len(_kernels) >= MAX_SESSIONS:
            raise HTTPException(
                status_code=503,
                detail="Session capacity reached; restart or increase AGI_MAX_SESSIONS",
            )

        from runtime.live_modules import LiveKernel
        from runtime.loader import load_agi

        package = load_agi(AGI_PACKAGE_PATH)
        session_memory = MEMORY_PATH / "sessions" / session_id
        session_memory.mkdir(parents=True, exist_ok=True)
        kernel = LiveKernel(
            package,
            vllm_base_url=VLLM_BASE_URL,
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            memory_path=str(session_memory),
            gpu=GPU,
            lora_adapter_path=str(LORA_ADAPTER_PATH),
            enable_online_training=session_id == "operator",
        )
        _kernels[session_id] = kernel
        _kernel_locks[session_id] = asyncio.Lock()
        log.info("session_kernel_initialized", session_id=session_id)
        return kernel


@app.get("/health")
async def health():
    """Liveness only; does not initialize a model-backed session."""
    return {
        "status": "alive",
        "version": app.version,
        "uptime_seconds": round(time.monotonic() - _started_at, 1),
    }


@app.get("/ready")
async def ready(response: Response):
    """Readiness checks both package conformance and the model server."""
    from runtime.loader import AGILoadError, load_agi

    checks: dict[str, str] = {}
    try:
        package = load_agi(AGI_PACKAGE_PATH)
        checks["package"] = f"ok:{package.name}:{package.format_version}"
        package.close()
    except AGILoadError as exc:
        checks["package"] = f"error:{exc}"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            model_response = await client.get(f"{VLLM_BASE_URL}/health")
            model_response.raise_for_status()
        checks["model"] = "ok"
    except (httpx.HTTPError, OSError) as exc:
        checks["model"] = f"error:{type(exc).__name__}"
    healthy = all(value.startswith("ok") for value in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if healthy else "not_ready", "checks": checks}


@app.get("/status")
async def agent_status(access: Annotated[SessionAccess, Depends(require_access)]):
    return (await get_kernel(access.session_id)).get_status()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, access: Annotated[SessionAccess, Depends(require_access)]):
    kernel = await get_kernel(access.session_id)
    lock = _kernel_locks[access.session_id]
    async with lock:
        result = await kernel.step_async(
            req.message, max_tokens=req.max_tokens, temperature=req.temperature
        )
    return ChatResponse(
        response=result["response"],
        step_log=result["step_log"],
        state=result["state"],
        session_id=access.session_id,
    )


@app.get("/memories")
async def memories(access: Annotated[SessionAccess, Depends(require_access)]):
    return {"memories": (await get_kernel(access.session_id)).get_memories(limit=50)}


@app.get("/action_log")
async def action_log(access: Annotated[SessionAccess, Depends(require_access)]):
    return {"actions": (await get_kernel(access.session_id)).get_action_log(limit=50)}


@app.get("/world_model")
async def world_model(access: Annotated[SessionAccess, Depends(require_access)]):
    kernel = await get_kernel(access.session_id)
    return {"summary": kernel.world_model.summary(), "geolocation": kernel.geolocation.snapshot()}


@app.get("/geolocation")
async def geolocation(access: Annotated[SessionAccess, Depends(require_access)]):
    return (await get_kernel(access.session_id)).geolocation.snapshot()


@app.get("/learning")
async def learning(access: Annotated[SessionAccess, Depends(require_access)]):
    kernel = await get_kernel(access.session_id)
    result = {
        "learner": kernel.learner.summary(),
        "consolidator": kernel.consolidator.summary(),
        "verification": kernel.verifier.summary(),
        "online_training_enabled": kernel.lora_trainer is not None,
    }
    if kernel.lora_trainer:
        result["lora"] = kernel.lora_trainer.summary()
    return result


@app.get("/knowledge")
async def knowledge(access: Annotated[SessionAccess, Depends(require_access)]):
    return (await get_kernel(access.session_id)).knowledge_graph.summary()


@app.get("/knowledge/query")
async def knowledge_query(
    q: Annotated[str, Query(min_length=1, max_length=1_000)],
    access: Annotated[SessionAccess, Depends(require_access)],
):
    kernel = await get_kernel(access.session_id)
    facts = kernel.knowledge_graph.query_text(q)
    return {"query": q, "facts": [fact.to_dict() for fact in facts[:20]]}


@app.get("/dynamics")
async def dynamics(access: Annotated[SessionAccess, Depends(require_access)]):
    return (await get_kernel(access.session_id)).world_model.dynamics.summary()


@app.post("/train")
async def trigger_training(access: Annotated[SessionAccess, Depends(require_access)]):
    kernel = await get_kernel(access.session_id)
    if not kernel.lora_trainer:
        raise HTTPException(status_code=403, detail="Online training is disabled")
    if kernel.lora_trainer.is_training:
        return {"status": "already_training"}
    async with _training_lock, _kernel_locks[access.session_id]:
        result = await kernel.lora_trainer.train_async()
        if result.get("promoted"):
            kernel.llm.served_model_name = "agent-lora"
        return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")
