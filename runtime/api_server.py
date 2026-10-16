"""FastAPI server for the .agi agent.

Standalone module so it can be launched via `uvicorn runtime.api_server:create_app`.
The app factory reads config from environment variables set by the Modal entrypoint.
"""

import json
import os
import sys

sys.path.insert(0, "/app")

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

# ── Config from env ──────────────────────────────────────────────────────────

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
MEMORY_PATH = os.environ.get("MEMORY_PATH", "/agent-memory")
AGI_PACKAGE_PATH = os.environ.get("AGI_PACKAGE_PATH", "/app/skeleton")
GPU = os.environ.get("GPU", "L4")
API_PORT = int(os.environ.get("API_PORT", "8080"))

# ── Request/Response models ──────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 2048
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    step_log: dict
    state: dict


# ── App factory ──────────────────────────────────────────────────────────────

_kernel = None


def get_kernel():
    """Lazy-init the kernel on first request."""
    global _kernel
    if _kernel is None:
        from runtime.loader import load_agi
        from runtime.live_modules import LiveKernel

        package = load_agi(AGI_PACKAGE_PATH)
        _kernel = LiveKernel(
            package,
            vllm_base_url=VLLM_BASE_URL,
            model_name=MODEL_NAME,
            memory_path=MEMORY_PATH,
            gpu=GPU,
        )
        log.info("kernel_initialized_on_first_request")
    return _kernel


app = FastAPI(title="AGI Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    kernel = get_kernel()
    return {
        "status": "healthy",
        "agent": kernel.package.name,
        "model": MODEL_NAME,
        "gpu": GPU,
        "steps_completed": kernel.total_steps,
        "episodic_memories": len(kernel.memory_store),
    }


@app.get("/status")
async def status():
    return get_kernel().get_status()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    kernel = get_kernel()
    result = await kernel.step_async(
        req.message,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    return ChatResponse(
        response=result["response"],
        step_log=result["step_log"],
        state=result["state"],
    )


@app.get("/memories")
async def memories():
    return {"memories": get_kernel().get_memories(limit=50)}


@app.get("/action_log")
async def action_log():
    return {"actions": get_kernel().get_action_log(limit=50)}


@app.get("/world_model")
async def world_model():
    kernel = get_kernel()
    return {
        "summary": kernel.world_model.summary(),
        "geolocation": kernel.geolocation.snapshot(),
    }


@app.get("/geolocation")
async def geolocation():
    return get_kernel().geolocation.snapshot()


@app.get("/learning")
async def learning():
    kernel = get_kernel()
    return {
        "learner": kernel.learner.summary(),
        "consolidator": kernel.consolidator.summary(),
        "verification": kernel.verifier.summary(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")
