"""
agi-format: Real .agi agent on Modal with Qwen3-8B.

Serves an autonomous agent backed by Qwen3-8B via vLLM on an L4 GPU.
The agent follows the .agi spec: perceive -> reason -> plan -> act -> learn.

Architecture:
  - vLLM subprocess on :8000 (internal, GPU inference)
  - FastAPI subprocess on :8080 (external, agent API)
  - Modal routes external traffic to :8080

Deploy:   modal deploy modal_app.py
Dev:      modal serve modal_app.py
"""

import subprocess
import time

import modal

MINUTES = 60

# ── Model ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# ── Infrastructure ───────────────────────────────────────────────────────────

GPU = "L4"
VLLM_PORT = 8000
API_PORT = 8080

# ── Image ────────────────────────────────────────────────────────────────────

HF_CACHE_PATH = "/root/.cache/huggingface/hub"
MEMORY_PATH = "/agent-memory"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "fastapi[standard]==0.115.12",
        "httpx==0.28.1",
        "structlog==24.4.0",
        "numpy<2",
        "uvicorn==0.34.3",
    )
    .env({
        "TORCH_CPP_LOG_LEVEL": "FATAL",
        "HF_HUB_CACHE": HF_CACHE_PATH,
    })
    .add_local_dir("skeleton", remote_path="/app/skeleton")
    .add_local_dir("runtime", remote_path="/app/runtime")
)

# ── Volumes ──────────────────────────────────────────────────────────────────

hf_cache_vol = modal.Volume.from_name("sal-model-cache")
agent_memory_vol = modal.Volume.from_name("agi-agent-memory", create_if_missing=True)

# ── App ──────────────────────────────────────────────────────────────────────

APP_NAME = "agi-agent"
app = modal.App(name=APP_NAME)


# ── Agent function ───────────────────────────────────────────────────────────


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        MEMORY_PATH: agent_memory_vol,
    },
    timeout=3600,
    scaledown_window=5 * MINUTES,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=API_PORT, startup_timeout=10 * MINUTES)
def agent():
    """Start vLLM (internal) + FastAPI agent server (external).

    Both are subprocess.Popen -- this function returns immediately
    and Modal routes traffic to API_PORT.
    """
    import httpx

    # 1. Start vLLM on internal port
    vllm_cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level", "error",
        "--disable-uvicorn-access-log",
        "--disable-log-requests",
        "--enforce-eager",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.92",
    ]
    vllm_proc = subprocess.Popen(vllm_cmd)

    # Wait for vLLM health before starting the API
    deadline = time.time() + 5 * MINUTES
    while time.time() < deadline:
        if (rc := vllm_proc.poll()) is not None:
            raise RuntimeError(f"vLLM exited with code {rc}")
        try:
            httpx.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5).raise_for_status()
            break
        except (httpx.ConnectError, httpx.HTTPStatusError):
            time.sleep(3)
    else:
        raise TimeoutError("vLLM not ready within 5 minutes")

    print(f"vLLM ready on :{VLLM_PORT}")

    # 2. Start FastAPI agent server on external port
    api_env = {
        **dict(__import__("os").environ),
        "VLLM_BASE_URL": f"http://127.0.0.1:{VLLM_PORT}",
        "MODEL_NAME": MODEL_NAME,
        "MEMORY_PATH": MEMORY_PATH,
        "AGI_PACKAGE_PATH": "/app/skeleton",
        "GPU": GPU,
        "API_PORT": str(API_PORT),
    }
    api_cmd = [
        "python", "-m", "uvicorn",
        "runtime.api_server:app",
        "--host", "0.0.0.0",
        "--port", str(API_PORT),
        "--log-level", "warning",
    ]
    subprocess.Popen(api_cmd, env=api_env, cwd="/app")

    print(f"FastAPI agent server starting on :{API_PORT}")
