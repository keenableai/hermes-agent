"""Tests for the Keenable web search + extract provider.

Covers:
- KeenableWebSearchProvider.is_available() — always True (keyless public endpoint)
- KeenableWebSearchProvider.search() — happy path, endpoint selection,
  HTTP error, request error, bad JSON, malformed body, 401/402/429
- KeenableWebSearchProvider.extract() — happy path, endpoint selection,
  URL guards (non-http(s), private/loopback/link-local/metadata hosts),
  per-URL error entries
- Base-URL handling via KEENABLE_API_URL env var (HTTPS enforced unless loopback)
- API key never leaks into error strings or logs
- _is_backend_available("keenable") integration (always True)
- _get_backend() recognizes "keenable" as a valid configured backend
- _get_backend() picks "keenable" as the universal fallback when nothing
  else is configured
- _get_backend() does NOT override a configured paid backend with keenable
- check_web_api_key() includes keenable in availability check
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Identity / availability
# ---------------------------------------------------------------------------


class TestKeenableProviderIdentity:
    def test_always_available_without_key(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider
        assert KeenableWebSearchProvider().is_available() is True

    def test_always_available_with_key(self, monkeypatch):
        monkeypatch.setenv("KEENABLE_API_KEY", "keen_xxx")
        from plugins.web.keenable.provider import KeenableWebSearchProvider
        assert KeenableWebSearchProvider().is_available() is True

    def test_provider_name(self):
        from plugins.web.keenable.provider import KeenableWebSearchProvider
        assert KeenableWebSearchProvider().name == "keenable"

    def test_display_name(self):
        from plugins.web.keenable.provider import KeenableWebSearchProvider
        assert KeenableWebSearchProvider().display_name == "Keenable"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.keenable.provider import KeenableWebSearchProvider
        assert issubclass(KeenableWebSearchProvider, WebSearchProvider)

    def test_advertises_both_capabilities(self):
        from plugins.web.keenable.provider import KeenableWebSearchProvider
        p = KeenableWebSearchProvider()
        assert p.supports_search() is True
        assert p.supports_extract() is True

    def test_setup_schema_marks_free_tier(self):
        from plugins.web.keenable.provider import KeenableWebSearchProvider
        schema = KeenableWebSearchProvider().get_setup_schema()
        assert schema["badge"] == "free"
        assert {v["key"] for v in schema["env_vars"]} == {"KEENABLE_API_KEY"}


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestKeenableProviderSearch:
    _SAMPLE_RESPONSE = {
        "results": [
            {"title": "A", "url": "https://a.example.com", "description": "desc A"},
            {"title": "B", "url": "https://b.example.com", "description": "desc B"},
            {"title": "C", "url": "https://c.example.com", "description": "desc C"},
        ]
    }

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.text = "" if json_data is None else str(json_data)
        return m

    @staticmethod
    def _mock_text_resp(text, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.side_effect = ValueError("not json")
        m.text = text
        return m

    def test_happy_path_normalizes_results(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = KeenableWebSearchProvider().search("test query", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {
            "title": "A", "url": "https://a.example.com",
            "description": "desc A", "position": 1,
        }
        assert web[2]["position"] == 3

    def test_uses_public_endpoint_without_key(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            KeenableWebSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://api.keenable.ai/v1/search/public"
        assert "X-API-Key" not in captured["headers"]
        assert captured["headers"]["User-Agent"].startswith("keenable-hermes-agent/")
        assert captured["json"]["query"] == "q"
        assert captured["json"]["mode"] == "pro"

    def test_uses_authenticated_endpoint_with_key(self, monkeypatch):
        monkeypatch.setenv("KEENABLE_API_KEY", "keen_test_123")
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return self._mock_resp({"results": []})

        with patch("httpx.post", side_effect=fake_post):
            KeenableWebSearchProvider().search("q", limit=5)

        assert captured["url"] == "https://api.keenable.ai/v1/search"
        assert captured["headers"].get("X-API-Key") == "keen_test_123"

    def test_limit_truncates_client_side(self, monkeypatch):
        """Keenable API has no `max_results`; the provider slices client-side
        to honor the ABC contract."""
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp(self._SAMPLE_RESPONSE)):
            result = KeenableWebSearchProvider().search("q", limit=2)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2

    def test_empty_results(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp({"results": []})):
            result = KeenableWebSearchProvider().search("nothing", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_falls_back_to_snippet_for_description(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        payload = {"results": [{"title": "x", "url": "https://x.example.com", "snippet": "snip"}]}
        with patch("httpx.post", return_value=self._mock_resp(payload)):
            result = KeenableWebSearchProvider().search("q", limit=5)
        assert result["data"]["web"][0]["description"] == "snip"

    @pytest.mark.parametrize(
        "status, body, label, must_include",
        [
            (401, {"error": "Unauthorized", "message": "Invalid API key"},
             "auth failed", "Invalid API key"),
            (402, {"error": "Payment required", "message": "Out of credits"},
             "out of credits", "credits"),
            (429, {"error": "Rate limit", "message": "Too many requests"},
             "rate limited", "Too many"),
            (500, {"error": "Internal", "message": "boom"},
             "HTTP 500", "boom"),
        ],
    )
    def test_distinguishes_status_codes(self, monkeypatch, status, body, label, must_include):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp(body, status_code=status)):
            result = KeenableWebSearchProvider().search("q", limit=5)

        assert result["success"] is False
        assert label in result["error"]
        assert must_include in result["error"]

    def test_non_json_error_body(self, monkeypatch):
        """A 502 with an HTML body must not leak JSONDecodeError."""
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch(
            "httpx.post",
            return_value=self._mock_text_resp("<html>502</html>", status_code=502),
        ):
            result = KeenableWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "HTTP 502" in result["error"]

    def test_timeout_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.post", side_effect=httpx.TimeoutException("slow")):
            result = KeenableWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_request_error_returns_failure(self, monkeypatch):
        import httpx
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.post", side_effect=httpx.RequestError("boom")):
            result = KeenableWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "boom" in result["error"] or "Keenable" in result["error"]

    def test_results_not_a_list(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.post", return_value=self._mock_resp({"results": "oops"})):
            result = KeenableWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "Unexpected" in result["error"]

    def test_api_key_not_in_error(self, monkeypatch):
        import httpx
        monkeypatch.setenv("KEENABLE_API_KEY", "keen_SUPERSECRET_DO_NOT_LEAK")
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.post", side_effect=httpx.RequestError("boom")):
            result = KeenableWebSearchProvider().search("q", limit=5)
        assert "SUPERSECRET" not in result["error"]
        assert "keen_SUPERSECRET_DO_NOT_LEAK" not in result["error"]


# ---------------------------------------------------------------------------
# Base URL handling via KEENABLE_API_URL
# ---------------------------------------------------------------------------


class TestKeenableBaseUrl:
    @staticmethod
    def _ok_resp():
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {"results": []}
        m.text = ""
        return m

    def test_custom_base_url_via_env(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.setenv("KEENABLE_API_URL", "https://staging.keenable.ai")
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return self._ok_resp()

        with patch("httpx.post", side_effect=fake_post):
            KeenableWebSearchProvider().search("q", limit=5)
        assert captured["url"] == "https://staging.keenable.ai/v1/search/public"

    def test_http_localhost_allowed(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.setenv("KEENABLE_API_URL", "http://localhost:8080")
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return self._ok_resp()

        with patch("httpx.post", side_effect=fake_post):
            KeenableWebSearchProvider().search("q", limit=5)
        assert captured["url"].startswith("http://localhost:8080")

    def test_http_non_localhost_rejected(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.setenv("KEENABLE_API_URL", "http://attacker.example.com")
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        result = KeenableWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "HTTPS" in result["error"]


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


class TestKeenableProviderExtract:
    _SAMPLE_FETCH = {
        "url": "https://example.com/",
        "title": "Example Domain",
        "content": "# Example Domain\n\nMarkdown body.",
        "description": "Example",
        "author": "RFC 2606",
        "published_at": "1999-01-01",
    }

    @staticmethod
    def _mock_resp(json_data, status_code=200):
        m = MagicMock()
        m.status_code = status_code
        m.json.return_value = json_data
        m.text = str(json_data)
        return m

    def test_returns_documents(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.get", return_value=self._mock_resp(self._SAMPLE_FETCH)):
            docs = KeenableWebSearchProvider().extract(["https://example.com"])

        assert len(docs) == 1
        doc = docs[0]
        assert doc["url"].startswith("http")
        assert doc["title"] == "Example Domain"
        assert doc["content"].startswith("# Example Domain")
        assert doc["raw_content"] == doc["content"]
        assert doc["metadata"]["description"] == "Example"
        assert doc["metadata"]["author"] == "RFC 2606"
        assert doc["metadata"]["published_at"] == "1999-01-01"

    def test_uses_public_fetch_without_key(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["params"] = kwargs.get("params", {})
            captured["headers"] = kwargs.get("headers", {})
            return self._mock_resp(self._SAMPLE_FETCH)

        with patch("httpx.get", side_effect=fake_get):
            KeenableWebSearchProvider().extract(["https://example.com"])

        assert captured["url"] == "https://api.keenable.ai/v1/fetch/public"
        assert captured["params"]["url"] == "https://example.com"
        assert "X-API-Key" not in captured["headers"]

    def test_uses_authenticated_fetch_with_key(self, monkeypatch):
        monkeypatch.setenv("KEENABLE_API_KEY", "keen_test_456")
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            return self._mock_resp(self._SAMPLE_FETCH)

        with patch("httpx.get", side_effect=fake_get):
            KeenableWebSearchProvider().extract(["https://example.com"])

        assert captured["url"] == "https://api.keenable.ai/v1/fetch"
        assert captured["headers"].get("X-API-Key") == "keen_test_456"

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "data:text/html,<script>",
        "javascript:alert(1)",
        "ftp://example.com/x",
    ])
    def test_rejects_non_http_schemes_without_network(self, monkeypatch, url):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.get") as mocked:
            docs = KeenableWebSearchProvider().extract([url])

        assert mocked.call_count == 0, "guard must run BEFORE the network call"
        assert docs[0]["error"].startswith("refused")

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1/",
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://0.0.0.0/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
    ])
    def test_rejects_private_or_metadata_hosts_without_network(self, monkeypatch, url):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.get") as mocked:
            docs = KeenableWebSearchProvider().extract([url])

        assert mocked.call_count == 0
        assert docs[0]["error"].startswith("refused")

    def test_mixed_batch_filters_unsafe_only(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        with patch("httpx.get", return_value=self._mock_resp(self._SAMPLE_FETCH)) as mocked:
            docs = KeenableWebSearchProvider().extract(
                ["https://example.com", "file:///etc/passwd"]
            )
        assert mocked.call_count == 1
        assert "error" not in docs[0] or docs[0].get("error") is None
        assert docs[1]["error"].startswith("refused")

    def test_per_url_http_error_becomes_error_entry(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
        from plugins.web.keenable.provider import KeenableWebSearchProvider

        body = {"error": "Rate limit", "message": "Slow down"}
        with patch("httpx.get", return_value=self._mock_resp(body, status_code=429)):
            docs = KeenableWebSearchProvider().extract(["https://example.com"])
        assert len(docs) == 1
        assert docs[0]["error"].startswith("Keenable rate limited")
        assert "Slow down" in docs[0]["error"]
        assert docs[0]["content"] == ""


# ---------------------------------------------------------------------------
# Integration with web_tools.py: priority chain + availability
# ---------------------------------------------------------------------------


class TestKeenableBackendWiring:
    def test_is_backend_available_always_true(self, monkeypatch):
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        from tools.web_tools import _is_backend_available
        assert _is_backend_available("keenable") is True

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "keenable"})
        assert web_tools._get_backend() == "keenable"

    def test_keenable_is_universal_fallback(self, monkeypatch):
        """When no other backend is configured or available, keenable wins."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in (
            "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
            "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "keenable"

    def test_keenable_does_not_override_paid_provider(self, monkeypatch):
        """Tavily (higher priority) should win over the keenable fallback."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in (
            "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
            "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TAVILY_API_KEY", "tvly")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "tavily"

    def test_keenable_does_not_override_brave_free(self, monkeypatch):
        """Even brave-free (free tier) outranks keenable when configured."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in (
            "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
            "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._get_backend() == "brave-free"

    def test_check_web_api_key_true_when_keenable_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "keenable"})
        monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
        assert web_tools.check_web_api_key() is True
