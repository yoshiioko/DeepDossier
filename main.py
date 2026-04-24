"""DeepDossier CLI entrypoint."""

from src.agent.config import Settings
from src.agent.graph import build_supervisor_graph
from src.agent.memory import ChromaDBMemory
from src.agent.runner import run_cli


if __name__ == "__main__":
    settings = Settings()
    memory = ChromaDBMemory(settings)
    graph = build_supervisor_graph(settings, memory)
    run_cli(settings, memory, graph)
