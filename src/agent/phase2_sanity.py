"""Phase 2 smoke test — run with: uv run python -m src.agent.phase2_sanity"""

from __future__ import annotations

import sys

from pydantic_ai.models.test import TestModel

from src.agent.agents import synthesize_agent
from src.agent.config import Settings
from src.agent.prompts import build_synthesize_prompt, build_compiler_prompt
from src.agent.schemas import SubQuery, SubResult
from src.agent.subgraph import build_researcher_subgraph
from src.agent.tools import web_search, wikipedia_search, arxiv_search


def ok(msg: str) -> None:
    print(f"\033[32m[ok]\033[0m  {msg}")


def fail(msg: str) -> None:
    print(f"\033[31m[FAIL]\033[0m {msg}")
    sys.exit(1)


# ── Check 1: Tools return strings and never raise ─────────────────────────────

def check_tools() -> None:
    settings = Settings(google_api_key="x", tavily_api_key="x")

    result = web_search("test query", settings=settings)
    assert isinstance(result, str)
    ok(f"web_search returned str (len={len(result)})")

    result = wikipedia_search("Python programming language")
    assert isinstance(result, str)
    ok(f"wikipedia_search returned str (len={len(result)})")

    result = arxiv_search("large language models")
    assert isinstance(result, str)
    ok(f"arxiv_search returned str (len={len(result)})")

# ── Check 2: Prompts build correctly ─────────────────────────────────────────

def check_prompts() -> None:
    sq = SubQuery(topic="quantum computing", tool_hint="paper", rationale="academic research")
    prompt = build_synthesize_prompt(sq, raw_content="Some raw content here.")
    assert "quantum computing" in prompt
    assert "raw content" in prompt.lower()
    ok("build_synthesize_prompt contains topic and raw content")

    sr = SubResult(topic="quantum computing", summary="Qubits are great.", confidence=0.9)
    compiler_prompt = build_compiler_prompt("What is quantum computing?", [sr])
    assert "quantum computing" in compiler_prompt.lower()
    assert "0.90" in compiler_prompt
    ok("build_compiler_prompt contains query and confidence score")

# ── Check 3: synthesize_agent runs with TestModel ────────────────────────────

def check_synthesize_agent() -> None:
    settings = Settings(google_api_key="x", tavily_api_key="x")
    sq = SubQuery(topic="AI safety", tool_hint="web", rationale="important topic")

    try:
        with synthesize_agent.override(model=TestModel()):
            result = synthesize_agent.run_sync(
                build_synthesize_prompt(sq, "Some raw content about AI safety."),
                deps=settings,
            )
        assert isinstance(result.output, SubResult)
        ok(f"synthesize_agent (TestModel) returned SubResult: topic='{result.output.topic}'")
    except Exception as e:
        fail(f"synthesize_agent failed: {e}")

# ── Check 4: Full subgraph run with TestModel ─────────────────────────────────

def check_subgraph() -> None:
    settings = Settings(google_api_key="x", tavily_api_key="x")
    sq = SubQuery(topic="climate change", tool_hint="wiki", rationale="broad coverage")

    try:
        with synthesize_agent.override(model=TestModel()):
            subgraph = build_researcher_subgraph(settings)
            initial_state = {
                "sub_query": sq,
                "raw_content": "",
                "sub_results": [],
            }
            final_state = subgraph.invoke(initial_state)

        assert len(final_state["sub_results"]) > 0
        assert isinstance(final_state["sub_results"][0], SubResult)
        ok(f"Full subgraph returned SubResult with confidence={final_state['sub_results'][0].confidence}")
    except Exception as e:
        fail(f"Subgraph run failed: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== DeepDossier Phase 2 Sanity Check ===\n")
    check_tools()
    check_prompts()
    check_synthesize_agent()
    check_subgraph()
    print("\n✅ All Phase 2 checks passed.\n")


if __name__ == "__main__":
    main()
