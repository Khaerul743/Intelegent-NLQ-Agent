from langgraph.checkpoint.memory import MemorySaver

from src.base import BaseAgent
from src.infrastructure import DuckDbManager
from src.schema import (
    DatasetDetailInformation,
)
from src.tools import (
    RetrieveDatasetTool,
)

from .nodes import AgentNLQNode
from .prompts import AgentNLQPrompt
from .workflow import AgentNLQWorkflow


class AgentNLQ(BaseAgent):
    def __init__(
        self,
        dataset_detail_information: DatasetDetailInformation,
        directory_datasets_path: str,
        llm_provider: str,
        llm_model: str,
    ):
        self.checkpointer = MemorySaver()
        # Setup duckdb manager datasets
        self.duckdb_manager = DuckDbManager(directory_datasets_path)

        # setup tool needed
        self.retrieve_dataset_tool = RetrieveDatasetTool(
            dataset_detail_information, llm_provider, llm_model, self.duckdb_manager
        )

        # Setup agent prompt
        self.prompts = AgentNLQPrompt(self.retrieve_dataset_tool.tool_prompt)
        # Setup agent nodes
        self.nodes = AgentNLQNode(
            self.prompts, self.retrieve_dataset_tool, llm_model, llm_provider
        )

        # Setup agent workflow
        self.workflow = AgentNLQWorkflow(self.checkpointer, self.nodes)

        super().__init__(self.nodes, self.workflow)
