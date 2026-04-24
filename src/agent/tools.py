"""Research tools — web, Wikipedia, ArXiv. Never raise; always return str."""

from __future__ import annotations

import arxiv
import wikipedia as wiki_lib
from tavily import TavilyClient

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
    """Search Wikipedia. Returns article summaries with source URLs."""
    try:
        titles = wiki_lib.search(query, results=max_results)
        if not titles:
            return "[wikipedia_search: no results]"
        formatted = []
        for title in titles[:max_results]:
            try:
                page = wiki_lib.page(title, auto_suggest=False)
                formatted.append(
                    f"## {page.title}\n"
                    f"URL: {page.url}\n"
                    f"{page.summary}"
                )
            except Exception:
                continue
        return "\n\n".join(formatted) or "[wikipedia_search: no results]"
    except Exception as e:
        return f"[wikipedia_search error: {e}]"


def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search ArXiv for papers. Returns paper summaries with source URLs."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results)
        formatted = []
        for paper in client.results(search):
            formatted.append(
                f"## {paper.title}\n"
                f"URL: {paper.entry_id}\n"
                f"Authors: {', '.join(str(a) for a in paper.authors[:3])}\n"
                f"{paper.summary}"
            )
        return "\n\n".join(formatted) or "[arxiv_search: no results]"
    except Exception as e:
        return f"[arxiv_search error: {e}]"
