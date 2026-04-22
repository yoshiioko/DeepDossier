"""Research tools — web, Wikipedia, ArXiv. Never raise; always return str."""

from __future__ import annotations

from tavily import TavilyClient
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

from src.agent.config import Settings


def web_search(query: str, settings: Settings, max_results: int = 5) -> str:
    """Search the web via Tavily. Returns formatted results or an error string."""
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        results = client.search(query, max_results=max_results)
        formatted = []
        for r in results.get("results", []):
            formatted.append(
                f"## {r.get('title', 'No title')}\n"
                f"URL: {r.get('url', '')}\n"
                f"{r.get('content', '')}"
            )
        return "\n\n".join(formatted) or "[web_search: no results]"
    except Exception as e:
        return f"[web_search error: {e}]"


def wikipedia_search(query: str, max_results: int = 3) -> str:
    """Search Wikipedia. Returns article text or an error string."""
    try:
        wrapper = WikipediaAPIWrapper(top_k_results=max_results)
        return wrapper.run(query)
    except Exception as e:
        return f"[wikipedia_search error: {e}]"


def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search ArXiv for papers. Returns paper summaries or an error string."""
    try:
        wrapper = ArxivAPIWrapper(top_k_results=max_results)
        return wrapper.run(query)
    except Exception as e:
        return f"[arxiv_search error: {e}]"
