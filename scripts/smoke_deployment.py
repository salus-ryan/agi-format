#!/usr/bin/env python3
"""Black-box smoke test for a deployed .agi API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def request(url: str, *, method: str = "GET", key: str = "", session: str = "smoke", body=None, timeout=30):
    headers = {"Accept": "application/json", "X-Session-ID": session}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"detail": str(exc)}
        return exc.code, payload


def request_with_retry(*args, attempts: int = 3, **kwargs):
    last_error = None
    for attempt in range(attempts):
        try:
            return request(*args, **kwargs)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after {attempts} attempts: {last_error}") from last_error


def wait_for(base_url: str, path: str, expected_status: int, deadline_seconds: int):
    deadline = time.monotonic() + deadline_seconds
    last = None
    while time.monotonic() < deadline:
        try:
            last = request(f"{base_url}{path}", timeout=15)
            if last[0] == expected_status:
                return last
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last = (0, {"error": type(exc).__name__})
        time.sleep(3)
    raise RuntimeError(f"{path} did not reach HTTP {expected_status}; last={last}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("AGI_API_KEY", ""))
    parser.add_argument("--session", default=f"smoke-{int(time.time())}")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    if not args.api_key:
        parser.error("--api-key or AGI_API_KEY is required")

    health = wait_for(base_url, "/health", 200, args.startup_timeout)
    ready = wait_for(base_url, "/ready", 200, args.startup_timeout)

    unauthenticated = request_with_retry(f"{base_url}/status", session=args.session)
    if unauthenticated[0] != 401:
        raise RuntimeError(f"Protected endpoint returned {unauthenticated[0]} without auth")

    status_response = request_with_retry(
        f"{base_url}/status", key=args.api_key, session=args.session, timeout=60
    )
    if status_response[0] != 200:
        raise RuntimeError(f"Authenticated status failed: {status_response}")

    result = {
        "health": health[1],
        "ready": ready[1],
        "auth_boundary": "ok",
        "status": "ok",
    }
    if not args.skip_chat:
        chat = request_with_retry(
            f"{base_url}/chat",
            method="POST",
            key=args.api_key,
            session=args.session,
            body={"message": "Reply with exactly: smoke-ok", "max_tokens": 32, "temperature": 0},
            timeout=240,
        )
        if chat[0] != 200 or "smoke-ok" not in chat[1].get("response", "").lower():
            raise RuntimeError(f"Chat smoke failed: HTTP {chat[0]} {chat[1]}")
        result["chat"] = "ok"
        result["chat_response"] = chat[1]["response"][:100]

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
