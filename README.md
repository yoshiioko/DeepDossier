# DeepDossier

A multi-agent research pipeline that decomposes complex queries into parallel sub-tasks,
stores findings in a persistent ChromaDB vector store, and produces a structured dossier
via a CLI runner. Built with **LangGraph**, **Pydantic AI**, and **Google Gemini (`gemini-2.5-flash`)**.

```
START → planner_node → dispatcher_node → [ResearcherSubgraph × N] (parallel, Send API)
                                                  ↓
                        aggregator_node → [INTERRUPT: human approval]
                                                  ↓
                        memory_writer_node ──► ChromaDB
                                                  ↓
                        compiler_node ──► DossierOutput → END
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.12+ |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | latest |
| Google API key | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| Tavily API key | [Tavily](https://app.tavily.com/) |

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/DeepDossier.git
cd DeepDossier
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Configure environment
```bash
cp .env.example .env
```

Open `.env` and fill in your keys:
```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Run via CLI (with human-in-the-loop approval prompt)
```bash
uv run python main.py
```

You will be prompted to enter a research question. The planner decomposes it into
sub-queries, researchers run in parallel, and you are asked to approve before findings
are written to ChromaDB. A final structured dossier is printed to the terminal.

---

## Running Tests

Unit tests use Pydantic AI's `TestModel` — zero API calls, runs in under 1 second:
```bash
uv run pytest -q -m "not integration"
```

With coverage report:
```bash
uv run pytest -q -m "not integration" --cov=src --cov-report=term-missing
```

Integration tests require real API keys and cost credits:
```bash
uv run pytest -q -m integration
```

---

## Project Structure

```
src/agent/
├── config.py           # Settings — API keys, model names, thresholds
├── state.py            # SupervisorState + SubgraphState
├── schemas.py          # Pydantic v2 models (PlannerOutput, SubResult, DossierOutput, …)
├── tools.py            # web_search (Tavily), wikipedia_search, arxiv_search
├── prompts.py          # Prompt builders for each node
├── memory.py           # BaseMemory protocol + ChromaDBMemory implementation
├── agents.py           # Pydantic AI singletons: synthesize_agent, compiler_agent
├── nodes.py            # All supervisor node functions
├── subgraph.py         # ResearcherSubgraph: fetch_node + synthesize_node
├── graph.py            # build_supervisor_graph() — wiring, interrupt, Send dispatch
├── runner.py           # run_once(), run_cli_async(), run_cli()
└── sanity.py           # ChromaDB round-trip smoke test (no LLM)
tests/
├── test_schemas.py     # Schema validation and model_validator assertions
├── test_memory.py      # ChromaDB protocol with mock; min_score filtering
├── test_subgraph.py    # Researcher subgraph unit tests (TestModel, zero API calls)
├── test_nodes.py       # Supervisor node unit tests (TestModel, confidence gating)
├── test_graph.py       # Graph routing + HITL interrupt/resume tests
└── test_integration.py # Full graph run (@pytest.mark.integration)
main.py                 # CLI entrypoint
```

---

## Architecture Notes

- **Two LLM paths**: Instructor + `google.generativeai` (`GenerativeModel`) in `planner_node` only;
  Pydantic AI (`google-gla` model string) for `synthesize_agent` and `compiler_agent` in `agents.py`. Never mixed in the same node.
- **Parallel dispatch** uses LangGraph's `Send` API in `dispatcher_node` — not
  `asyncio.gather`. Each `ResearcherSubgraph` instance is a fully isolated child graph.
- **Structured output**: `planner_node` uses Instructor (`instructor.from_gemini`) for
  `PlannerOutput` with automatic LLM self-correction on `ValidationError`. Pydantic AI
  handles `SubResult` and `DossierOutput`.
- **Human-in-the-loop**: `interrupt()` is a node, not an edge condition. Resume via
  `graph.update_state(config, {"human_approved": True})` then `await graph.ainvoke(None, config)`.
- **Persistence**: `AsyncSqliteSaver` keeps graph state across process restarts.
  Each independent dossier run is scoped by `thread_id`.
- **Vector memory**: `BaseMemory` protocol in `memory.py` — swap `ChromaDBMemory` for
  `Mem0Memory` in one line without changing any callers.
- **Tool routing**: `fetch_node` reads a `tool_hint` on each sub-query set by `planner_node`:
  `"web"` → Tavily, `"wiki"` → Wikipedia, `"paper"` → ArXiv.
- **Confidence-gated memory**: only `SubResult` entries with `confidence ≥ 0.7` are written
  to ChromaDB; retrieved chunks with cosine similarity ≥ 0.7 are injected into the compiler
  prompt as RAG context.
