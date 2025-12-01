from src.infrastructure import DuckDbManager
from src.schema import DatasetDetailInformation

from .models import RetreiveDatasetModel
from .prompts import RetrieveDatasetPrompt
from .tool_nodes import RetrieveDatasetNodes
from .tool_workflow import RetrieveDatasetWorkflow


class RetrieveDatasetTool:
    def __init__(
        self,
        detail_dataset_information: DatasetDetailInformation,
        llm_provider: str,
        llm_model: str,
        duckdb_manager: DuckDbManager,
    ):
        self.duckdb_manager = duckdb_manager
        self.tool_prompt = RetrieveDatasetPrompt(
            detail_dataset_information, self.duckdb_manager
        )
        self.tool_nodes = RetrieveDatasetNodes(
            self.tool_prompt, self.duckdb_manager, llm_provider, llm_model
        )
        self.tool_workflow = RetrieveDatasetWorkflow(self.tool_nodes)

    def read_dataset(self, data_description_needed) -> list[str] | None:
        """
        Tool ini digunakan untuk mengambil data dari database.
        Params:
            - data_description_needed: Deskripsikan secara detail data apa yang harus diambil/query.
        """
        try:
            result = self.tool_workflow.run(
                RetreiveDatasetModel(data_description_needed=data_description_needed)
            )
        except RuntimeError as e:
            return [str(e)]
        return result.get("result", None)
