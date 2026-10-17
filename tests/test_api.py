"""API authentication, validation, and session-isolation tests."""

import asyncio
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("AGI_API_KEY", "test-key")
    monkeypatch.setenv("AGI_REQUIRE_AUTH", "true")
    monkeypatch.setenv("AGI_DISABLE_DOCS", "true")
    import runtime.api_server as api_server

    api_server = importlib.reload(api_server)
    with TestClient(api_server.app) as client:
        yield api_server, client


def auth_headers(session: str = "default") -> dict[str, str]:
    return {"Authorization": "Bearer test-key", "X-Session-ID": session}


def test_health_is_public_and_minimal(api):
    _, client = api
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert "memories" not in response.json()
    assert response.headers["cache-control"] == "no-store"


def test_protected_endpoint_requires_bearer_token(api):
    _, client = api
    response = client.get("/status")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_invalid_session_id_is_rejected(api):
    _, client = api
    response = client.get("/status", headers=auth_headers("../../escape"))
    assert response.status_code == 400


def test_status_uses_requested_session(api, monkeypatch):
    api_server, client = api
    seen: list[str] = []

    class FakeKernel:
        def __init__(self, session_id):
            self.session_id = session_id

        def get_status(self):
            return {"session": self.session_id}

    async def fake_get_kernel(session_id):
        seen.append(session_id)
        return FakeKernel(session_id)

    monkeypatch.setattr(api_server, "get_kernel", fake_get_kernel)
    first = client.get("/status", headers=auth_headers("alpha"))
    second = client.get("/status", headers=auth_headers("beta"))
    assert first.json() == {"session": "alpha"}
    assert second.json() == {"session": "beta"}
    assert seen == ["alpha", "beta"]


def test_chat_serializes_per_session_and_validates_input(api, monkeypatch):
    api_server, client = api

    class FakeKernel:
        async def step_async(self, message, max_tokens, temperature):
            return {
                "response": f"echo:{message}",
                "step_log": {"ok": True},
                "state": {"tokens": max_tokens, "temperature": temperature},
            }

    async def fake_get_kernel(session_id):
        return FakeKernel()

    monkeypatch.setattr(api_server, "get_kernel", fake_get_kernel)
    api_server._kernel_locks["alpha"] = asyncio.Lock()
    response = client.post(
        "/chat",
        headers=auth_headers("alpha"),
        json={"message": "hello", "max_tokens": 32, "temperature": 0},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "alpha"
    assert response.json()["response"] == "echo:hello"

    invalid = client.post(
        "/chat", headers=auth_headers("alpha"), json={"message": "", "max_tokens": 100_000}
    )
    assert invalid.status_code == 422
