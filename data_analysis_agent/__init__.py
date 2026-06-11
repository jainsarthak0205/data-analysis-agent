"""Agentic data analysis over the Palmer Penguins dataset."""

from data_analysis_agent.dataset import (
    Dataset,
    bundled_penguins_path,
    load_penguins,
)
from data_analysis_agent.agent import (
    AgentConfig,
    AgentResult,
    DataAnalysisAgent,
)
from data_analysis_agent.llm import (
    ClaudeClient,
    FakeLLMClient,
    LLMClient,
)

__all__ = [
    "Dataset",
    "bundled_penguins_path",
    "load_penguins",
    "AgentConfig",
    "AgentResult",
    "DataAnalysisAgent",
    "ClaudeClient",
    "FakeLLMClient",
    "LLMClient",
]
