"""Keenable web search + extract provider — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider` (the
plugin-facing ABC). Lives in hermes-agent at ``plugins/web/keenable/``.

Keenable is a web search API built for AI agents. The provider works
out-of-the-box against the keyless public endpoints, and switches to the
authenticated endpoints when ``KEENABLE_API_KEY`` is set. Both ``search``
(``/v1/search``) and ``extract`` (``/v1/fetch``) are implemented.

Config keys this provider responds to::

    web:
      search_backend: "keenable"     # explicit per-capability
      extract_backend: "keenable"    # explicit per-capability
      backend: "keenable"            # shared fallback

Env vars::

    KEENABLE_API_KEY=keen_...        # optional; https://keenable.ai/console
    KEENABLE_API_URL=https://...     # optional; defaults to https://api.keenable.ai
"""

from __future__ import annotations

import ipaddress
import logging
import os
from importlib import metadata
from typing import Any, Dict, List
from urllib.parse import urlparse

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.keenable.ai"
_AUTH_SEARCH = "/v1/search"
_PUBLIC_SEARCH = "/v1/search/public"
_AUTH_FETCH = "/v1/fetch"
_PUBLIC_FETCH = "/v1/fetch/public"

# Per CLAUDE.md §5: base URL is env-only and must be HTTPS (loopback allowed
# for local dev only).
_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _user_agent() -> str:
    try:
        version = metadata.version("hermes-agent")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return f"keenable-hermes-agent/{version}"


def _base_url() -> str:
    """Read the Keenable base URL from ``KEENABLE_API_URL``; default to prod.

    Enforces HTTPS unless the host is a loopback (for local development).
    """
    raw = (os.getenv("KEENABLE_API_URL") or "").strip() or _DEFAULT_BASE_URL
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"KEENABLE_API_URL must be http(s); got scheme {parsed.scheme!r}"
        )
    if parsed.scheme == "http" and parsed.hostname not in _LOCALHOST_HOSTS:
        raise ValueError(
            f"KEENABLE_API_URL must be HTTPS (host {parsed.hostname!r} is not loopback)"
        )
    return raw.rstrip("/")


def _api_key() -> str:
    """Return the configured API key (stripped) or empty string."""
    return os.getenv("KEENABLE_API_KEY", "").strip()


def _format_error(status: int, body: Any) -> str:
    """Surface the backend's helpful message + status code.

    Keenable returns ``{"error": "...", "message": "..."}`` on most failures.
    Examples:
      - 401 → auth failure (key invalid)
      - 402 → out of credits
      - 429 → rate limited (public: 2 RPS + hourly cap; org: configurable)
      - 5xx → server error
    The backend's body often includes upgrade/auth instructions — keep it.
    """
    code_label = {
        401: "auth failed",
        402: "out of credits",
        403: "forbidden",
        429: "rate limited",
    }.get(status, f"HTTP {status}")
    detail = ""
    if isinstance(body, dict):
        msg = body.get("message") or body.get("error")
        if msg:
            detail = f" — {msg}"
    elif isinstance(body, str) and body:
        detail = f" — {body[:200]}"
    return f"Keenable {code_label}{detail}"


# ---------------------------------------------------------------------------
# URL guards for extract() — CLAUDE.md §8.
# ---------------------------------------------------------------------------

# Hosts we refuse to fetch even before sending to Keenable. The backend has
# server-side SSRF protection too, but client-side guard saves a round trip
# and avoids any chance of metadata-service URLs leaking via DNS.
_FORBIDDEN_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::", "::1",
    "169.254.169.254",  # AWS/GCP metadata service
    "metadata.google.internal",
}


def _is_safe_fetch_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Reject non-http(s) and obvious private targets."""
    try:
        parsed = urlparse(url)
    except Exception as exc:  # noqa: BLE001
        return False, f"unparseable URL: {exc}"
    if parsed.scheme not in ("http", "https"):
        return False, f"only http(s) URLs allowed (got {parsed.scheme!r})"
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL has no host"
    if host in _FORBIDDEN_HOSTS:
        return False, f"host {host!r} is not allowed"
    # Block obviously private/link-local/loopback IP literals.
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, f"IP {host!r} is private/loopback/link-local"
    except ValueError:
        # Not a literal IP — that's fine, the backend will resolve.
        pass
    return True, ""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class KeenableWebSearchProvider(WebSearchProvider):
    """Keenable search + extract provider.

    Always available — uses the keyless public endpoints when no key is set,
    and the authenticated endpoints when ``KEENABLE_API_KEY`` is configured.
    """

    @property
    def name(self) -> str:
        return "keenable"

    @property
    def display_name(self) -> str:
        return "Keenable"

    def is_available(self) -> bool:
        """Always available — Keenable has a keyless public endpoint."""
        return True

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # ---- search ---------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the Keenable API.

        Returns ``{"success": True, "data": {"web": [{"title", "url",
        "description", "position"}]}}`` on success, or ``{"success": False,
        "error": str}`` on failure (parsed from the backend body when possible).
        """
        import httpx

        try:
            base = _base_url()
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        key = _api_key()
        headers: Dict[str, str] = {
            "User-Agent": _user_agent(),
            "Content-Type": "application/json",
        }
        if key:
            headers["X-API-Key"] = key
            url = f"{base}{_AUTH_SEARCH}"
        else:
            url = f"{base}{_PUBLIC_SEARCH}"

        payload: Dict[str, Any] = {"query": query, "mode": "pro"}

        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=30)
        except httpx.TimeoutException as exc:
            logger.warning("Keenable search timeout: %s", exc)
            return {"success": False, "error": "Keenable request timed out"}
        except httpx.RequestError as exc:
            logger.warning("Keenable search request error: %s", exc)
            return {"success": False, "error": f"Could not reach Keenable: {exc}"}

        if resp.status_code >= 400:
            body: Any
            try:
                body = resp.json()
            except Exception:  # noqa: BLE001
                body = resp.text
            logger.warning("Keenable search %d: %r", resp.status_code, body)
            return {"success": False, "error": _format_error(resp.status_code, body)}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Keenable search response parse error: %s", exc)
            return {"success": False, "error": "Keenable returned a non-JSON body"}

        raw_results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(raw_results, list):
            return {"success": False, "error": "Unexpected Keenable response shape"}

        truncated = raw_results[: max(1, int(limit))]
        web_results = [
            {
                "title": str(r.get("title", "") if isinstance(r, dict) else ""),
                "url": str(r.get("url", "") if isinstance(r, dict) else ""),
                "description": str(
                    (r.get("description") or r.get("snippet") or "")
                    if isinstance(r, dict) else ""
                ),
                "position": i + 1,
            }
            for i, r in enumerate(truncated)
        ]

        logger.info(
            "Keenable search '%s': %d results (from %d raw, limit %d)",
            query, len(web_results), len(raw_results), limit,
        )
        return {"success": True, "data": {"web": web_results}}

    # ---- extract --------------------------------------------------------

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Fetch markdown content for each URL via Keenable ``/v1/fetch``.

        Returns the legacy list-of-results shape per the ABC contract:
            [{"url", "title", "content", "raw_content", "metadata", "error"?}]

        Per-URL failures (bad scheme, private host, HTTP error) become items
        with an ``error`` field rather than raising — the agent can react.
        """
        import httpx

        try:
            base = _base_url()
        except ValueError as exc:
            return [
                {"url": u, "title": "", "content": "", "raw_content": "",
                 "error": str(exc), "metadata": {"sourceURL": u}}
                for u in urls
            ]

        key = _api_key()
        headers = {"User-Agent": _user_agent()}
        if key:
            headers["X-API-Key"] = key
            fetch_url = f"{base}{_AUTH_FETCH}"
        else:
            fetch_url = f"{base}{_PUBLIC_FETCH}"

        out: List[Dict[str, Any]] = []
        for u in urls:
            ok, reason = _is_safe_fetch_url(u)
            if not ok:
                out.append({
                    "url": u, "title": "", "content": "", "raw_content": "",
                    "error": f"refused: {reason}",
                    "metadata": {"sourceURL": u},
                })
                continue

            try:
                resp = httpx.get(
                    fetch_url, params={"url": u}, headers=headers, timeout=30,
                )
            except httpx.TimeoutException:
                out.append({
                    "url": u, "title": "", "content": "", "raw_content": "",
                    "error": "Keenable fetch timed out",
                    "metadata": {"sourceURL": u},
                })
                continue
            except httpx.RequestError as exc:
                out.append({
                    "url": u, "title": "", "content": "", "raw_content": "",
                    "error": f"Could not reach Keenable: {exc}",
                    "metadata": {"sourceURL": u},
                })
                continue

            if resp.status_code >= 400:
                body: Any
                try:
                    body = resp.json()
                except Exception:  # noqa: BLE001
                    body = resp.text
                out.append({
                    "url": u, "title": "", "content": "", "raw_content": "",
                    "error": _format_error(resp.status_code, body),
                    "metadata": {"sourceURL": u},
                })
                continue

            try:
                data = resp.json()
            except Exception:  # noqa: BLE001
                out.append({
                    "url": u, "title": "", "content": "", "raw_content": "",
                    "error": "Keenable returned a non-JSON body",
                    "metadata": {"sourceURL": u},
                })
                continue

            if not isinstance(data, dict):
                out.append({
                    "url": u, "title": "", "content": "", "raw_content": "",
                    "error": "Unexpected Keenable response shape",
                    "metadata": {"sourceURL": u},
                })
                continue

            title = str(data.get("title") or "")
            content = str(data.get("content") or "")
            md: Dict[str, Any] = {"sourceURL": data.get("url") or u, "title": title}
            for k in ("description", "author", "published_at"):
                if data.get(k):
                    md[k] = data[k]

            out.append({
                "url": data.get("url") or u,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": md,
            })

        return out

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Keenable",
            "badge": "free",
            "tag": "Works without an API key; set KEENABLE_API_KEY for the paid tier.",
            "env_vars": [
                {
                    "key": "KEENABLE_API_KEY",
                    "prompt": "Keenable API key (optional)",
                    "url": "https://keenable.ai/console",
                },
            ],
        }
