from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .models import RetreiveDatasetModel
from .tool_nodes import RetrieveDatasetNodes


class RetrieveDatasetWorkflow:
    def __init__(self, tool_nodes: RetrieveDatasetNodes):
        self.tool_nodes = tool_nodes
        self.build = self._build_workflow()

    def _build_workflow(self) -> CompiledStateGraph[RetreiveDatasetModel]:
        graph: StateGraph = StateGraph(RetreiveDatasetModel)
        graph.add_node("analyst_table", self.tool_nodes.analyst_table_exits)
        graph.add_node("analyst_query_needed", self.tool_nodes.analyst_query_needed)
        graph.add_node("generate_query", self.tool_nodes.generate_query)
        graph.add_node("query_to_db", self.tool_nodes.query_to_db)
        graph.add_node("validation_result", self.tool_nodes.query_result_validation)

        graph.add_edge(START, "analyst_table")
        graph.add_conditional_edges(
            "analyst_table",
            self.tool_nodes.analyst_table_router,
            {"next": "analyst_query_needed", "end": END},
        )
        graph.add_edge("analyst_query_needed", "generate_query")
        graph.add_edge("generate_query", "query_to_db")
        graph.add_edge("query_to_db", "validation_result")
        graph.add_conditional_edges(
            "validation_result",
            self.tool_nodes.validation_router,
            {"query_again": "generate_query", "next": END},
        )

        return graph.compile()

    def run(self, state: RetreiveDatasetModel):
        return self.build.invoke(state)
