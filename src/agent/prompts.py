"""Prompt builder functions for synthesize_node and compiler_node."""

from __future__ import annotations

from src.agent.schemas import SubQuery, SubResult


def build_synthesize_prompt(sub_query: SubQuery, raw_content: str) -> str:
    """Build the prompt for synthesize_agent.

    The agent will distil raw_content into a structured SubResult.
    """
    return f"""You are a research synthesizer. Your task is to distil the following raw content
into a concise, structured research finding.

## Research Topic
{sub_query.topic}

## Rationale
{sub_query.rationale}

## Raw Content
{raw_content}

## Instructions
- Write a clear, factual summary of the key findings relevant to the topic.
- Extract all URLs from the raw content into the sources list.
- Assign a confidence score (0.0–1.0) based on source quality and content relevance.
- If the raw content is empty, an error string, or clearly irrelevant, set confidence=0.0
  and note the failure in the summary.
- Be concise — the summary should be 2–4 sentences.
"""


def build_compiler_prompt(query: str, sub_results: list[SubResult]) -> str:
    """Build the prompt for compiler_agent.

    The agent will synthesise all sub_results into a full DossierOutput.
    """
    results_text = "\n\n".join(
        f"### Topic: {r.topic}\n"
        f"Confidence: {r.confidence:.2f}\n"
        f"Summary: {r.summary}\n"
        f"Sources: {', '.join(r.sources) if r.sources else 'none'}"
        for r in sub_results
    )

    return f"""You are a professional research compiler. Your task is to synthesise the following
research findings into a comprehensive, well-structured dossier.

## Original Research Query
{query}

## Research Findings
{results_text}

## Instructions
- Write a compelling title for the dossier.
- Write a 3–5 sentence executive summary covering the key conclusions.
- Organise the findings into logical sections (DossierSection objects), each with a
  heading, body, and sources list.
- Compile all_sources as a deduplicated list of every URL across all findings.
- List any limitations where confidence was low (< 0.5) or sources were absent.
- Write final_markdown as a complete Markdown document rendering all sections.
- Be honest about gaps — do not fabricate information.
"""


def build_planner_prompt(query: str, known_topics: list[str], max_sub_queries: int = 5) -> str:
    """Build the prompt for planner_node (Instructor + Gemini).

    The LLM must return a PlannerOutput with at least one SubQuery.
    """
    known_section = (
        "The following topics are already in memory — do NOT create sub-queries for them:\n"
        + "\n".join(f"- {t}" for t in known_topics)
        if known_topics
        else "No topics are in memory yet — research everything relevant."
    )

    return (
        "You are a research planning agent. Decompose the following query into "
        "specific, focused sub-research tasks.\n\n"
        f"## User Query\n{query}\n\n"
        f"## Memory Status\n{known_section}\n\n"
        "## Instructions\n"
        f"- Create between 1 and {max_sub_queries} sub-queries.\n"
        "- Each sub-query must have:\n"
        "    - topic: a specific, focused research topic (not the full query)\n"
        '    - tool_hint: exactly one of "web", "wiki", or "paper"\n'
        '        - "web"   → breaking news, company info, current events (Tavily)\n'
        '        - "wiki"  → background concepts, history, definitions (Wikipedia)\n'
        '        - "paper" → academic research, technical depth (ArXiv)\n'
        "    - rationale: why this sub-topic matters to answering the overall query\n"
        "- Do NOT include sub-queries for known topics listed above.\n"
        "- planning_notes: briefly explain your decomposition strategy.\n"
    )
