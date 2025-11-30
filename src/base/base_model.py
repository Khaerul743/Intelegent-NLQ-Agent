from typing import Annotated, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


class BaseAgentStateModel(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_message: str = ""
    response: Optional[str] = "none"
