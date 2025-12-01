from duckdb import CatalogException

from src.base import BaseNode
from src.infrastructure import DuckDbManager

from .models import (
    RetreiveDatasetModel,
    StructuredOutputGenerateQuery,
    StructuredOutputQueryNeeded,
    StructuredOutputValidateTableExist,
    StructuredOutputValidationResult,
)
from .prompts import RetrieveDatasetPrompt


class RetrieveDatasetNodes(BaseNode):
    def __init__(
        self,
        prompt: RetrieveDatasetPrompt,
        duckdb_manager: DuckDbManager,
        llm_provicer: str,
        llm_model: str,
    ):
        self.prompts = prompt
        self.duckdb_manager = duckdb_manager

        self._retry = 0
        super().__init__(llm_model, llm_provicer)

    def analyst_table_exits(self, state: RetreiveDatasetModel):
        prompt = self.prompts.analyst_table_exist(state.data_description_needed)
        response = self.call_llm_with_structured_output(
            prompt, StructuredOutputValidateTableExist, "dict"
        )
        if not response["is_table_exist"]:
            raise RuntimeError(response["description_analyst_result"])

        return {
            "is_table_exist": response["is_table_exist"],
            "tables_description": response["description_analyst_result"],
        }

    def analyst_table_router(self, state: RetreiveDatasetModel):
        if state.is_table_exist:
            return "next"
        return "end"

    def analyst_query_needed(self, state: RetreiveDatasetModel):
        prompt = self.prompts.analyst_query_needed(
            state.data_description_needed, state.tables_description
        )
        response = self.call_llm_with_structured_output(
            prompt, StructuredOutputQueryNeeded
        )

        return {"analyst_query_needed_result": response}

    def generate_query(self, state: RetreiveDatasetModel):
        prompt = self.prompts.generate_query(state.analyst_query_needed_result)
        response = self.call_llm_with_structured_output(
            prompt, StructuredOutputGenerateQuery, "dict"
        )
        return {"list_queries": response}

    def query_to_db(self, state: RetreiveDatasetModel):
        if state.list_queries is None:
            return {"result": []}

        results: list[str] = []
        for query_item in state.list_queries.list_queries:
            try:
                query_result = self.duckdb_manager.get_data(
                    query_item.query, query_item.table_name
                )
                results.append(str(query_result))
            except CatalogException as e:
                results.append(str(e))
            except ValueError as e:
                results.append(str(e))
            except Exception as e:
                results.append(str(e))

        combined_result = state.result + ["\n".join(results)]
        return {"result": combined_result}

    def query_result_validation(self, state: RetreiveDatasetModel):
        if state.analyst_query_needed_result is not None:
            problem_solve = state.analyst_query_needed_result.problem_solving
        else:
            problem_solve = None
        prompt = self.prompts.validation_result(
            state.data_description_needed,
            state.result,
            state.tables_description,
            problem_solve,
        )
        response = self.call_llm_with_structured_output(
            prompt, StructuredOutputValidationResult, "dict"
        )
        response_data = StructuredOutputValidationResult(
            is_valid=response["is_valid"], next_step_query=response["next_step_query"]
        )
        return {
            "is_valid": response_data.is_valid,
            "analyst_query_needed_result": response_data.next_step_query,
        }

    def validation_router(self, state: RetreiveDatasetModel):
        if state.is_valid or self._retry >= 3:
            return "next"
        self._retry += 1
        return "query_again"
