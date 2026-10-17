"""Security primitives shared by the API and action runtime."""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from urllib.parse import urlparse

SECRET_KEY_PATTERN = re.compile(r"(?i)(authorization|api[_-]?key|password|secret|token)")
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|password|secret|token)\s*[:=]\s*([^\s,;]+)"
)


def redact_sensitive(value):
    """Recursively redact common credential fields without logging their values."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SECRET_KEY_PATTERN.search(str(key)) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    if isinstance(value, str):
        return SECRET_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    return value


class SecurityError(ValueError):
    """Raised when an operation violates the runtime security policy."""


@dataclass(frozen=True)
class NetworkPolicy:
    """Deny-by-default outbound HTTP policy."""

    allowed_domains: tuple[str, ...] = ()
    allow_http: bool = False

    def validate_url(self, url: str) -> str:
        parsed = urlparse(url)
        allowed_schemes = {"https", "http"} if self.allow_http else {"https"}
        if parsed.scheme.lower() not in allowed_schemes:
            raise SecurityError("URL scheme is not allowed")
        if parsed.username or parsed.password:
            raise SecurityError("URL credentials are not allowed")
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if not hostname:
            raise SecurityError("URL must include a hostname")
        if parsed.port not in (None, 80, 443):
            raise SecurityError("URL port is not allowed")
        if not self.allowed_domains:
            raise SecurityError("Outbound network access is disabled")
        if not any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in self.allowed_domains
        ):
            raise SecurityError(f"Domain is not allowlisted: {hostname}")

        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
            }
        except socket.gaierror as exc:
            raise SecurityError(f"Cannot resolve hostname: {hostname}") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise SecurityError("Private, local, reserved, and metadata addresses are blocked")
        return url


class SlidingWindowRateLimiter:
    """Small in-process limiter suitable for a single API container."""

    def __init__(self, requests: int, window_seconds: int):
        if requests <= 0 or window_seconds <= 0:
            raise ValueError("Rate-limit values must be positive")
        self.requests = requests
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.requests:
                retry_after = max(1, int(self.window_seconds - (current - events[0])))
                return False, retry_after
            events.append(current)
            if not events:
                self._events.pop(key, None)
            return True, 0
