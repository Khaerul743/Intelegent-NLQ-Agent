from langchain_core.messages import HumanMessage

from src.base import BaseAgentStateModel, BaseNode
from src.tools import RetrieveDatasetTool

from .prompts import AgentNLQPrompt


class AgentNLQNode(BaseNode):
    def __init__(
        self,
        prompt: AgentNLQPrompt,
        retrieve_dataset_tool: RetrieveDatasetTool,
        llm_model: str,
        llm_provider: str,
    ):
        super().__init__(llm_model, llm_provider)
        self.prompts = prompt
        self.retrieve_dataset_tool = retrieve_dataset_tool

    def main_agent(self, state: BaseAgentStateModel):
        prompt = self.prompts.main_agent(state.user_message)
        messages = self.get_prompt_setup(prompt, state.messages)

        response = self.call_llm_with_tool(
            messages,
            [
                self.retrieve_dataset_tool.read_dataset,
            ],
        )

        self.estimate_total_tokens(prompt, state.user_message, response.content)

        return {
            "messages": list(state.messages)
            + [HumanMessage(content=state.user_message)]
            + [response],
            "response": response.content,
        }

    def answer_tool_message(self, state: BaseAgentStateModel):
        prompt = self.prompts.main_agent(state.user_message)
        messages = self.get_prompt_setup(prompt, state.messages)
        response = self.call_llm(messages)
        self.estimate_total_tokens(prompt, state.user_message, response.content)

        return {
            "messages": list(state.messages) + [response],
            "response": response.content,
        }
