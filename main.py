"""DeepDossier CLI entrypoint."""

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.generativeai|instructor",
)

from dotenv import load_dotenv

load_dotenv()  # ensures GOOGLE_API_KEY is in os.environ for Pydantic AI

from src.agent.config import Settings
from src.agent.graph import build_supervisor_graph
from src.agent.memory import ChromaDBMemory
from src.agent.runner import run_cli


if __name__ == "__main__":
    settings = Settings()
    memory = ChromaDBMemory(settings)
    run_cli(settings, memory, build_supervisor_graph(settings, memory))
