# AGENTS.md — DeepDossier

**DeepDossier** is a multi-agent research pipeline (LangGraph + Pydantic AI + FastAPI) that decomposes user queries into parallel sub-research tasks, stores findings in a persistent ChromaDB vector store, and streams structured dossier output over SSE.

---

## Architecture

```
User Query → planner_node → dispatcher_node → [ResearcherSubgraph × N] (parallel)
                                                        ↓
                          aggregator_node → [INTERRUPT: human approval] → memory_writer_node → compiler_node → DossierOutput
```

- **Supervisor graph** (`src/agent/graph.py`): orchestrates all nodes; uses `SqliteSaver` for persistence across restarts.
- **ResearcherSubgraph** (`src/agent/subgraph.py`): child graph with `fetch_node` + `synthesize_node`, spawned in parallel via LangGraph's `Send` API — not `asyncio.gather`.
- **Two LLM paths**: LangChain (`ChatGoogleGenerativeAI`) for graph nodes; Pydantic AI (`GeminiModel`) for `agents.py` agents. Never mix both in the same node.
- **Instructor** wraps the Gemini client only in `planner_node` for `PlannerOutput` structured output with auto-retry on `ValidationError`.
- **ChromaDB** (`src/agent/memory.py`): all calls wrapped in `asyncio.to_thread()`. Backed by `BaseMemory` protocol — swap to `Mem0Memory` without changing callers.

## Key Files

| File | Role |
|------|------|
| `src/agent/graph.py` | `build_supervisor_graph()` — node wiring, interrupt checkpoint, Send dispatch |
| `src/agent/nodes.py` | All supervisor node functions; calls Pydantic AI agents internally |
| `src/agent/subgraph.py` | `ResearcherSubgraph`: `fetch_node` + `synthesize_node` |
| `src/agent/agents.py` | Pydantic AI module-level singletons: `synthesize_agent`, `compiler_agent` |
| `src/agent/memory.py` | `BaseMemory` protocol + `ChromaDBMemory` |
| `src/agent/schemas.py` | All Pydantic v2 models (`PlannerOutput`, `SubResult`, `DossierOutput`, etc.) |
| `src/agent/api.py` | FastAPI app: `/health`, `/research`, `/stream/{thread_id}`, `/approve/{thread_id}`, `/dossier/{thread_id}` |
| `src/agent/state.py` | `SupervisorState` + `SubgraphState` |
| `src/agent/config.py` | `Settings` — all env vars with defaults |

## Developer Workflows

```bash
uv sync                                          # install all deps (runtime + dev)
uv run pre-commit install                        # install hooks (once after clone)

uv run python -m src.agent.sanity                # Phase 1: ChromaDB round-trip smoke test
uv run python -m src.agent.phase2_sanity         # Phase 2: subgraph + tools smoke test

uv run python main.py                            # CLI full graph run (with HITL approval prompt)
uv run uvicorn src.agent.api:app --reload --port 8000  # API server (dev)

uv run pytest -q -m "not integration"           # unit tests — zero API calls (uses TestModel)
uv run pytest -q -m "not integration" --cov=src --cov-report=term-missing
uv run pytest -q -m integration                 # integration tests (costs credits, needs .env)

uv run ruff check src tests && uv run ruff format src tests
uv run mypy src                                  # strict mode
docker compose up --build                        # containerised run (Phase 7)
```

## Critical Conventions

- **Package manager**: `uv` exclusively — never `pip install`.
- **Parallel dispatch**: always use LangGraph `Send` API in `dispatcher_node`; never `asyncio.gather`.
- **Interrupt/resume**: `interrupt()` is a node, not an edge condition. Resume with `graph.update_state(config, {"human_approved": True})` then `graph.invoke(None, config)`.
- **State serialisation**: store `DossierOutput` as `.model_dump()` dict in `SupervisorState` (not as Pydantic object) for `SqliteSaver` JSON compatibility.
- **Async safety**: FastAPI routes call `agent.run()` (async); never call `agent.run_sync()` in async context.
- **Tool routing**: `fetch_node` selects tool via `tool_hint` on sub-query: `"web"` → Tavily, `"wiki"` → Wikipedia, `"paper"` → ArXiv.
- **Pydantic AI agents**: defined as module-level singletons in `agents.py`; called via `agent.run_sync(prompt, deps=settings)` from within LangGraph nodes. In tests, `TestModel` replaces the real model — no monkey-patching.
- **SSE events**: always use typed `EventPayload` Pydantic model — never raw dicts.
- **Logging**: `structlog` throughout; bind `run_id` + `node_name` to every log event; JSON in prod, coloured in dev.
- **Test markers**: `@pytest.mark.integration` for real API/DB tests; unit tests using `TestModel` need no markers.

## Environment Setup

Copy `.env.example` → `.env`. Required keys:

```env
GOOGLE_API_KEY=...          # Gemini (both LangChain and Pydantic AI paths)
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=deep-dossier
LANGSMITH_TRACING=true
```

Notable optional overrides: `PLANNER_MODEL_NAME=gemini-2.5-pro` (stronger model for planning), `CHROMA_PATH=./chroma_db`, `MAX_PARALLEL_RESEARCHERS=5`.

