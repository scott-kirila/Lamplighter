"""The localhost trust boundary, enforced.

The server binds ``127.0.0.1`` and carries no authentication: the security model
is "only this machine can reach it". A *browser* breaks that assumption in two
ways the network layer cannot see, because the user's own browser is reachable
by any page they visit and can reach this port on their behalf.

* **WebSockets ignore the same-origin policy.** Any page open in another tab can
  ``new WebSocket("ws://127.0.0.1:<port>/ws")``, receive the connect-time
  ``sync`` (the entire project, plus generated source on request), and send a
  ``validate`` message that *replaces* the project — a write that persists to
  ``.lamplighter/graph.json`` and outlives a kernel restart. The browser sends
  ``Origin`` on that handshake but will not act on it: only the server can refuse.
* **DNS rebinding** points an attacker-controlled name at ``127.0.0.1``, making
  their page same-origin with this server and reopening the whole HTTP API
  (start runs, download weights, list registered data). The page's ``Host``
  header still names the attacker's domain, which is what closes it.

Both checks are host allowlists, not authentication — they assert the boundary
the README already claims rather than adding a new one.

Two deliberate calls:

* **A missing ``Origin`` is allowed.** Non-browser clients (the notebook client,
  ``curl``, ``websockets``) send none, and they are not the threat — a browser
  always sends one on a cross-origin request. ``Origin: null`` (sandboxed iframe,
  ``file://``) is *not* missing and is rejected.
* **The port is not checked, only the host.** An SSH tunnel legitimately maps a
  different local port (``ssh -L 9000:127.0.0.1:8000`` makes the browser's origin
  ``localhost:9000``), and every attack this blocks is already blocked by the
  host alone.
"""
from __future__ import annotations

from urllib.parse import urlsplit

# Loopback under every spelling a browser or tunnel may produce. ``urlsplit``
# strips the brackets from an IPv6 authority, so ``::1`` is the form we compare.
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

_allowed: set[str] = set(_LOOPBACK)


def allow_host(host: str) -> None:
    """Also accept ``host``. Called by the session for a non-loopback bind — the
    ``Lamplighter(host=...)`` case, which already warns that it exposes the
    kernel. Loopback is always allowed and never removed."""
    name = _hostname(host)
    if name:
        _allowed.add(name)


def allowed_hosts() -> set[str]:
    """The current allowlist (a copy) — for diagnostics and tests."""
    return set(_allowed)


def _hostname(value: str | None) -> str:
    """The bare host from an ``Origin`` or ``Host`` header — scheme, port and
    IPv6 brackets removed, lowercased. ``""`` for anything unparseable, so a
    malformed header fails closed rather than matching by accident."""
    if not value:
        return ""
    text = value.strip()
    if not text:
        return ""
    # A Host header has no scheme; `//` makes urlsplit read it as an authority.
    if "//" not in text:
        text = "//" + text
    try:
        return urlsplit(text).hostname or ""
    except ValueError:
        return ""


def origin_ok(origin: str | None) -> bool:
    """Is this ``Origin`` allowed to open a WebSocket? ``None`` (header absent)
    is a non-browser client and passes; every present value must be loopback."""
    if origin is None:
        return True
    return _hostname(origin) in _allowed


def host_ok(host: str | None) -> bool:
    """Is this ``Host`` one we answer to? Rejecting anything else is what makes
    DNS rebinding fail — the rebound page still sends the attacker's domain."""
    if host is None:
        # HTTP/1.1 requires Host; its absence means a client we didn't build for.
        return True
    return _hostname(host) in _allowed
