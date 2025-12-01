from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.base import BaseAgentStateModel, BaseWorkflow

from .nodes import AgentNLQNode


class AgentNLQWorkflow(BaseWorkflow):
    def __init__(
        self,
        state_saver: BaseCheckpointSaver,
        agent_node: AgentNLQNode,
    ):
        self.checkpointer = state_saver
        self.nodes = agent_node
        self.build = self._build_workflow()

    def _build_workflow(
        self,
    ) -> CompiledStateGraph[BaseAgentStateModel]:
        graph = StateGraph(BaseAgentStateModel)

        # Main nodes
        graph.add_node("main_agent", self.nodes.main_agent)
        graph.add_node("anwser_tool_message", self.nodes.answer_tool_message)

        # Tool node
        graph.add_node(
            "read_file",
            ToolNode(
                tools=[
                    self.nodes.retrieve_dataset_tool.read_dataset,
                ]
            ),
        )

        # Workflow
        graph.add_edge(START, "main_agent")
        graph.add_conditional_edges(
            "main_agent",
            self.nodes.conditional_tool_call,
            {"tool_call": "read_file", "end": END},
        )
        graph.add_edge("read_file", "anwser_tool_message")
        graph.add_edge("anwser_tool_message", END)

        return graph.compile(checkpointer=self.checkpointer)

    def run(self, state: BaseAgentStateModel, thread_id: str):
        return self.build.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

    def show(self):
        pass
