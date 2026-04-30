# AGENTS.md — DeepDossier

**DeepDossier** is a multi-agent research pipeline (LangGraph + Pydantic AI) that decomposes user queries into parallel sub-research tasks, stores findings in a persistent ChromaDB vector store, and produces a structured dossier via a CLI runner.

---

## Architecture

```
User Query → planner_node → dispatcher_node → [ResearcherSubgraph × N] (parallel)
                                                        ↓
                          aggregator_node → [INTERRUPT: human approval] → memory_writer_node → compiler_node → DossierOutput
```

- **Supervisor graph** (`src/agent/graph.py`): orchestrates all nodes; uses `AsyncSqliteSaver` for persistence across restarts. Exposed as an `@asynccontextmanager` — always enter it with `async with build_supervisor_graph(settings, memory) as graph`.
- **ResearcherSubgraph** (`src/agent/subgraph.py`): child graph with `fetch_node` + `synthesize_node`, spawned in parallel via LangGraph's `Send` API — not `asyncio.gather`.
- **Two LLM paths**: Instructor + `google.generativeai` (`GenerativeModel`) in `planner_node` only; Pydantic AI (`google-gla` model string) for `synthesize_agent` and `compiler_agent` in `agents.py`. Never mix both in the same node.
- **Instructor** wraps `google.generativeai.GenerativeModel` in `planner_node` for `PlannerOutput` structured output with auto-retry on `ValidationError` (`max_retries=3`, `Mode.GEMINI_JSON`).
- **ChromaDB** (`src/agent/memory.py`): all calls are `async` and awaited directly inside async node functions. Backed by `BaseMemory` protocol — swap to `Mem0Memory` without changing callers.

## Key Files

| File | Role |
|------|------|
| `src/agent/graph.py` | `build_supervisor_graph()` — async context manager; `_build_graph()` wires nodes, edges, Send dispatch |
| `src/agent/nodes.py` | All supervisor node functions (`async def`); calls Pydantic AI agents with `await agent.run()` |
| `src/agent/subgraph.py` | `ResearcherSubgraph`: `fetch_node` (sync) + `synthesize_node` (async) |
| `src/agent/agents.py` | Pydantic AI module-level singletons: `synthesize_agent`, `compiler_agent` |
| `src/agent/memory.py` | `BaseMemory` protocol + `ChromaDBMemory` |
| `src/agent/schemas.py` | All Pydantic v2 models (`PlannerOutput`, `SubResult`, `DossierOutput`, etc.) |
| `src/agent/runner.py` | `run_once()` (async) + `run_cli_async()` (async) + `run_cli()` (sync shim via `asyncio.run()`) |
| `src/agent/state.py` | `SupervisorState` + `SubgraphState` |
| `src/agent/config.py` | `Settings` — all env vars with defaults |

## Developer Workflows

```bash
uv sync                                          # install all deps (runtime + dev)

uv run python -m src.agent.sanity                # Phase 1: ChromaDB round-trip smoke test
uv run python -m src.agent.phase2_sanity         # Phase 2: subgraph + tools smoke test

uv run python main.py                            # CLI full graph run (with HITL approval prompt)

uv run pytest -q -m "not integration"           # unit tests — zero API calls (uses TestModel)
uv run pytest -q -m "not integration" --cov=src --cov-report=term-missing
uv run pytest -q -m integration                 # integration tests (costs credits, needs .env)

uv run ruff check src tests && uv run ruff format src tests
uv run mypy src                                  # strict mode
```

## Critical Conventions

- **Package manager**: `uv` exclusively — never `pip install`.
- **Fully async pipeline**: all supervisor nodes and `synthesize_node` are `async def`. Use `await agent.run()` — never `agent.run_sync()`. Never call `asyncio.run()` inside a node or anywhere inside the running event loop.
- **Single event loop**: `asyncio.run()` is called exactly once, at the top of `run_cli()` in `runner.py`. Everything below it uses `await`.
- **Graph invocation**: always use `await graph.ainvoke()` or `await graph.astream()` — never `graph.invoke()` (sync runner rejects async nodes).
- **Checkpointer**: `AsyncSqliteSaver` (requires `aiosqlite`). `build_supervisor_graph` is an `@asynccontextmanager` — enter it with `async with` to manage the DB connection lifetime.
- **Node wiring adapters**: LangGraph calls nodes with a single `state` argument. Bridge the `(state, settings)` / `(state, settings, memory)` signatures using named adapter functions inside `_build_graph()` and `build_researcher_subgraph()` — not lambdas, not `functools.partial`.
- **Parallel dispatch**: always use LangGraph `Send` API in `dispatcher_node`; never `asyncio.gather`.
- **Interrupt/resume**: `interrupt()` is a node, not an edge condition. Resume with `graph.update_state(config, {"human_approved": True})` then `await graph.ainvoke(None, config)`.
- **State serialisation**: store `DossierOutput` as `.model_dump()` dict in `SupervisorState` (not as Pydantic object) for `AsyncSqliteSaver` JSON compatibility.
- **Tool routing**: `fetch_node` selects tool via `tool_hint` on sub-query: `"web"` → Tavily, `"wiki"` → Wikipedia, `"paper"` → ArXiv.
- **Pydantic AI agents**: defined as module-level singletons in `agents.py`; called via `await agent.run(prompt, deps=settings)` from within async node functions. In tests, `TestModel` replaces the real model — no monkey-patching. Full subgraph tests use `await subgraph.ainvoke()`.
- **Memory hit logging**: `planner_node` emits `planner_node.memory_hit` (with `known_topics`) or `planner_node.memory_miss` on every run so ChromaDB cache usage is visible in structured logs.
- **Logging**: `structlog` throughout; bind `run_id` + `node_name` to every log event; JSON in prod, coloured in dev.
- **Test markers**: `@pytest.mark.integration` for real API/DB tests; unit tests using `TestModel` need no markers.

## Environment Setup

Copy `.env.example` → `.env`. Required keys:

```env
GOOGLE_API_KEY=...          # Gemini (both Instructor/google.generativeai and Pydantic AI paths)
TAVILY_API_KEY=...
```

Notable optional overrides: `PLANNER_MODEL_NAME=gemini-2.5-pro` (stronger model for planning), `CHROMA_PATH=./chroma_db`, `MAX_PARALLEL_RESEARCHERS=5`.

> **Dependency note**: use `pydantic-ai-slim[google,openai,logfire]` — **not** `pydantic-ai[gemini]` (that extra doesn't exist and installs all provider extras including `mistralai 2.x`, which conflicts with `instructor`). The slim package gives you only the Google Gemini path with no Mistral dependency.

> **Warning suppression**: `google.generativeai` emits a `FutureWarning` on import (the package is deprecated in favour of `google.genai`). This is suppressed via `warnings.filterwarnings()` in `main.py` and `nodes.py` before the import executes. This is intentional — the migration to `google.genai` / `instructor.from_genai` is tracked as a future task.
