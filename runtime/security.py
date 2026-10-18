"""Network policy shared by the reference tool runner."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


class SecurityError(ValueError):
    """Raised when an operation violates the runtime security policy."""


@dataclass(frozen=True)
class ResolvedTarget:
    """A validated URL plus the public address the request must be sent to."""

    url: str
    hostname: str
    address: str


@dataclass(frozen=True)
class NetworkPolicy:
    """Deny-by-default outbound HTTP policy."""

    allowed_domains: tuple[str, ...] = ()
    allow_http: bool = False

    def validate_url(self, url: str) -> ResolvedTarget:
        """Validate a URL and pin it to a resolved public address.

        Callers MUST connect to `address` rather than resolving the hostname again;
        a second lookup could be answered with a private address (DNS rebinding).
        """
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

        default_port = 443 if parsed.scheme.lower() == "https" else 80
        try:
            records = socket.getaddrinfo(
                hostname, parsed.port or default_port, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise SecurityError(f"Cannot resolve hostname: {hostname}") from exc
        addresses = {ipaddress.ip_address(record[4][0]) for record in records}
        if not addresses:
            raise SecurityError(f"Cannot resolve hostname: {hostname}")
        if any(not address.is_global for address in addresses):
            raise SecurityError("Private, local, reserved, and metadata addresses are blocked")
        chosen = sorted(addresses, key=lambda address: (address.version, int(address)))[0]
        return ResolvedTarget(url=url, hostname=hostname, address=str(chosen))
