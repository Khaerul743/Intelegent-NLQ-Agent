import asyncio
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import tiktoken
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from src.core.utils.logger import get_logger

from .base_model import BaseAgentStateModel
from .components.memory import LongTermMemory

R = TypeVar("R")
load_dotenv()


class BaseNode:
    def __init__(
        self,
        llm_model: str,
        provider: str,
        use_long_memory: bool = False,
        user_memory_id: Optional[str] = None,
    ):
        self.llm_model = llm_model
        self.provider = provider.lower()
        self._llm = None  # lazy init
        self.logger = get_logger(__name__)
        self._total_token: int = 0
        self.use_long_memory = use_long_memory
        self.memory_id = user_memory_id
        self._memory = None

        try:
            self.tokenizer = tiktoken.encoding_for_model(llm_model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def memory(self) -> LongTermMemory:
        if not self.memory_id:
            raise ValueError("User memory id is required")

        if not self._memory and self.use_long_memory:
            self._memory = LongTermMemory(self.memory_id)
            self.logger.info("Initialized memory is success")
            return self._memory

        return self._memory

    def _is_include_long_memory(self) -> bool:
        if self.use_long_memory:
            return True
        return False

    def save_context(self, user_message: str, response: str):
        if self._is_include_long_memory():
            message = [
                HumanMessage(content=user_message),
                AIMessage(content=response),
            ]
            self.memory.add_context(message)

    @property
    def llm(self):
        """Lazy initialization of LLM instance"""
        if self._llm is None:
            self._llm = self._get_llm_provider(self.provider, self.llm_model)
            self.logger.info(
                f"Initialized LLM provider: {self.provider} ({self.llm_model})"
            )
        return self._llm

    def _get_llm_provider(self, provider: str, model: str):
        """Return the appropriate LLM instance based on provider."""
        if provider == "openai":
            return ChatOpenAI(model=model)
        elif provider == "anthropic":
            return ChatAnthropic(
                model_name=model,
                temperature=0.7,
                timeout=60,  # default timeout 60 detik
                stop=None,
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def call_llm(self, messages: Any) -> Any:
        """Generalized method to call LLM (async or sync safe)."""
        try:
            llm = self.llm

            # if hasattr(llm, "ainvoke") and asyncio.iscoroutinefunction(llm.ainvoke):
            #     response = await llm.ainvoke(messages)
            if hasattr(llm, "invoke"):
                response = llm.invoke(messages)
            else:
                raise TypeError("Provided LLM does not support invoke/ainvoke.")

            return response

        except Exception as e:
            self.logger.error(f"Error while invoking LLM: {e}")
            raise

    def call_llm_with_tool(self, messages: Any, tools: Sequence[Any]) -> Any:
        """
        Call LLM with tools bound to it.

        Args:
            messages: Messages to send to LLM (can be list of BaseMessage or string)
            tools: Sequence of tools to bind to LLM (e.g., [tool1, tool2, ...])

        Returns:
            LLM response with tool bindings

        Raises:
            TypeError: If LLM does not support bind_tools or invoke/ainvoke
            Exception: If error occurs during LLM invocation
        """
        try:
            llm = self.llm

            # Bind tools to LLM
            if not hasattr(llm, "bind_tools"):
                raise TypeError("Provided LLM does not support bind_tools method.")

            llm_with_tools = llm.bind_tools(tools)
            self.logger.debug(f"Bound {len(tools)} tools to LLM")

            # Call LLM with tools (async or sync safe)
            # if hasattr(llm_with_tools, "ainvoke") and asyncio.iscoroutinefunction(
            #     llm_with_tools.ainvoke
            # ):
            # response = await llm_with_tools.ainvoke(messages)
            if hasattr(llm_with_tools, "invoke"):
                response = llm_with_tools.invoke(messages)
            else:
                raise TypeError("LLM with tools does not support invoke/ainvoke.")

            return response

        except Exception as e:
            self.logger.error(f"Error while invoking LLM with tools: {e}")
            raise

    def call_llm_with_structured_output(
        self,
        messages: Any,
        output_model: type[BaseModel],
        output_type: Literal["base", "dict"] = "base",
    ):
        """Call LLM and return a parsed pydantic model instance as a dictionary (structured output)."""
        try:
            llm = self.llm.with_structured_output(output_model)

            if hasattr(llm, "invoke"):
                response = llm.invoke(messages)
            else:
                raise TypeError("Provided LLM does not support invoke/ainvoke.")

            if output_type == "base":
                return response

            # If the LLM already returned a dict, return it directly
            if isinstance(response, dict):
                return response

            # If the LLM returned a BaseModel instance, convert to dict (supports pydantic v1 & v2)
            if isinstance(response, BaseModel):
                try:
                    if hasattr(response, "model_dump"):
                        return response.model_dump()  # pydantic v2
                    return response.dict()  # pydantic v1
                except Exception as e:
                    self.logger.error(
                        f"Failed to convert BaseModel response to dict: {e}"
                    )
                    raise

            # Otherwise, attempt to parse the raw response into the provided pydantic model then convert to dict
            try:
                parsed = output_model.parse_obj(response)  # type: ignore[arg-type]
                if hasattr(parsed, "model_dump"):
                    return parsed.model_dump()
                return parsed.dict()
            except Exception as e:
                self.logger.error(
                    f"Failed to parse LLM response into {output_model} and convert to dict: {e}"
                )
                raise

        except Exception as e:
            self.logger.error(f"Error while invoking LLM: {e}")
            raise

    def retry_with_backoff(
        self,
        func: Callable[[], Union[R, Awaitable[R]]],
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Union[R, Awaitable[R]]:
        """Retry function with exponential backoff (sync or async)."""

        async def _async_wrapper() -> R:
            for attempt in range(max_retries):
                try:
                    result = func()
                    # Bisa return coroutine, maka perlu di-await
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        self.logger.error(
                            f"Max retries ({max_retries}) exceeded. Last error: {e}"
                        )
                        raise
                    delay = base_delay * (2**attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)

        try:
            # Kalau event loop sudah jalan (berarti dalam async context)
            asyncio.get_running_loop()
            return _async_wrapper()
        except RuntimeError:
            # Kalau tidak ada loop, berarti dipanggil secara sync
            for attempt in range(max_retries):
                try:
                    result = func()
                    if asyncio.iscoroutine(result):
                        # Jalankan coroutine di event loop baru
                        return asyncio.run(result)
                    return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        self.logger.error(
                            f"Max retries ({max_retries}) exceeded. Last error: {e}"
                        )
                        raise
                    delay = base_delay * (2**attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)

    def get_total_token(self):
        return self._total_token

    def _sum_token(self, token: int):
        self._total_token += token

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using tiktoken"""
        try:
            token = len(self.tokenizer.encode(text))
            return token
        except Exception as e:
            self.logger.warning(f"Error estimating tokens: {str(e)}")
            return len(text) // 4  # fallback

    def _handle_prompt_token(self, prompts: List[BaseMessage]) -> str:
        list_prompt = []
        for prompt in prompts:
            list_prompt.append(prompt.content)

        return " ".join(list_prompt)

    def estimate_total_tokens(
        self, prompts: List[BaseMessage], user_message: str, response_llm: str
    ):
        tokens = (
            self._estimate_tokens(self._handle_prompt_token(prompts))
            + self._estimate_tokens(user_message)
            + self._estimate_tokens(response_llm)
        )
        self._sum_token(tokens)
        return tokens

    def estimate_structured_output_tokens(
        self, prompt: str, response_content: str = ""
    ) -> int:
        """Estimate tokens for structured output calls"""
        try:
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = (
                self._estimate_tokens(response_content) if response_content else 50
            )
            return input_tokens + output_tokens + 20
        except Exception as e:
            self.logger.warning(f"Error estimating structured output tokens: {str(e)}")
            return 100

    def get_all_previous_messages(self, messages: Sequence[BaseMessage]):
        all_previous_messages = messages
        print(f"LEN: {len(all_previous_messages)}")
        return all_previous_messages

    def get_prompt_setup(
        self, agent_prompt: List[BaseMessage], state_messages: Sequence[BaseMessage]
    ) -> List[Any]:
        all_previous_messages = self.get_all_previous_messages(state_messages)
        setup_prompt: list[Any] = (
            [agent_prompt[0]] + list(all_previous_messages) + [agent_prompt[1]]
        )
        return setup_prompt

    def get_content_state_last_message(self, state_messages: Sequence[BaseMessage]):
        return state_messages[-1].content

    def get_state_last_message(
        self, state_messages: Sequence[BaseMessage]
    ) -> BaseMessage:
        return state_messages[-1]

    def conditional_tool_call(self, state: BaseAgentStateModel):
        last_message = self.get_state_last_message(state.messages)
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_call"
        return "end"
