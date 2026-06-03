"""Keenable web search plugin — bundled, auto-loaded."""


from plugins.web.keenable.provider import KeenableWebSearchProvider


def register(ctx) -> None:
    """Register the Keenable provider with the plugin context."""
    ctx.register_web_search_provider(KeenableWebSearchProvider())
