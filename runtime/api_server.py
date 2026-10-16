"""FastAPI server for the .agi agent.

Standalone module launched via uvicorn. Reads config from environment variables.
"""

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
MODEL_REVISION = os.environ.get("MODEL_REVISION", "")
MEMORY_PATH = os.environ.get("MEMORY_PATH", "/agent-memory")
AGI_PACKAGE_PATH = os.environ.get("AGI_PACKAGE_PATH", "/app/skeleton")
GPU = os.environ.get("GPU", "L4")
API_PORT = int(os.environ.get("API_PORT", "8080"))
LORA_ADAPTER_PATH = os.environ.get("LORA_ADAPTER_PATH", "/agent-memory/lora-adapter")

# ── Request/Response models ──────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 2048
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    step_log: dict
    state: dict


# ── Kernel ───────────────────────────────────────────────────────────────────

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
            model_revision=MODEL_REVISION,
            memory_path=MEMORY_PATH,
            gpu=GPU,
            lora_adapter_path=LORA_ADAPTER_PATH,
        )
        log.info("kernel_initialized_on_first_request")
    return _kernel


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="AGI Agent", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
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
        "has_lora_adapter": kernel.lora_trainer.has_adapter if kernel.lora_trainer else False,
    }


@app.get("/status")
async def status():
    return get_kernel().get_status()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    kernel = get_kernel()
    result = await kernel.step_async(
        req.message, max_tokens=req.max_tokens, temperature=req.temperature,
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
    result = {
        "learner": kernel.learner.summary(),
        "consolidator": kernel.consolidator.summary(),
        "verification": kernel.verifier.summary(),
    }
    if kernel.lora_trainer:
        result["lora"] = kernel.lora_trainer.summary()
    return result


@app.post("/train")
async def trigger_training():
    """Manually trigger a LoRA training run."""
    kernel = get_kernel()
    if not kernel.lora_trainer:
        return {"error": "LoRA trainer not configured"}
    if kernel.lora_trainer.is_training:
        return {"status": "already_training"}
    result = await kernel.lora_trainer.train_async()
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="warning")
