import time
from typing import Any, Dict

from .base_model import BaseAgentStateModel
from .base_node import BaseNode
from .base_workflow import BaseWorkflow


class BaseAgent:
    def __init__(self, agent_node: BaseNode, workflow: BaseWorkflow):
        self.workflow = workflow
        self.agent_node = agent_node
        self._response_time = 0
        self._result = None

    def get_response_time(self):
        return self._response_time

    def get_token_usage(self):
        return self.agent_node.get_total_token()

    def get_llm_model(self):
        return self.agent_node.llm_model

    def execute(
        self, state: BaseAgentStateModel, thread_id: str
    ) -> Dict[str, Any] | Any:
        start_time = time.perf_counter()
        result = self.workflow.run(state, thread_id)
        self._result = result
        end_time = time.perf_counter()
        self._response_time = round(end_time - start_time, 2)
        return result

    def show_execute_detail(self):
        """
        Return a neat, human-readable summary of messages from the last execution result,
        with emoji decorations to make the output more readable.

        If an AI message has an empty string (""), it's treated as a tool call / processing
        indicator and replaced with a suitable emoji/text so it doesn't appear blank.
        """
        if self._result is None:
            return "No execution result available."

        # locate messages in possible shapes
        state = self._result
        messages = None
        if isinstance(state, dict):
            messages = (
                state.get("messages")
                or state.get("state", {}).get("messages")
                or state.get("conversation", {}).get("messages")
            )
        else:
            messages = getattr(state, "messages", None)

        if not messages:
            return "No messages found in state."

        # role -> emoji+label map
        role_label_map = {
            "system": "⚙️ System:",
            "user": "👤 User:",
            "assistant": "🤖 AI:",
            "ai": "🤖 AI:",
            "tool": "🛠️ Tool:",
        }

        lines = []
        for idx, msg in enumerate(messages, start=1):
            # extract common fields from dict-like or object-like message
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("type") or msg.get("sender")
                content = msg.get("content") or msg.get("text") or msg.get("message")
                tool_name = msg.get("tool_name") or msg.get("name") or msg.get("tool")
                # sometimes tool output is nested
                if content is None:
                    content = msg.get("tool_output") or msg.get("output")
            else:
                role = (
                    getattr(msg, "role", None)
                    or getattr(msg, "type", None)
                    or getattr(msg, "sender", None)
                )
                content = (
                    getattr(msg, "content", None)
                    or getattr(msg, "text", None)
                    or getattr(msg, "message", None)
                )
                tool_name = (
                    getattr(msg, "tool_name", None)
                    or getattr(msg, "name", None)
                    or getattr(msg, "tool", None)
                )

            # normalize content to string
            if content is None:
                content = ""
            else:
                # keep multi-line content readable
                content = str(content)

            # If content is empty string, treat as tool call / processing indicator
            if not content.strip():
                if tool_name:
                    content = f"⏳ Calling tool: {tool_name}..."
                else:
                    low_role = (role or "").lower() if isinstance(role, str) else ""
                    if low_role in ("assistant", "ai", "bot"):
                        content = "⏳ (processing...)"
                    else:
                        content = "—"

            # choose label
            label = None
            if isinstance(role, str):
                label = role_label_map.get(role.lower())
            if not label:
                # if tool_name exists, prefer tool label
                if tool_name:
                    label = f"🛠️ Tool:"
                else:
                    label = "💬 Message:"

            # format line
            if tool_name:
                lines.append(f"{idx}. 🛠️ [Tool: {tool_name}] {content}")
            else:
                lines.append(f"{idx}. {label} {content}")

        return "\n".join(lines)

    def show_workflow(self):
        return self.workflow.show()

    def get_response(self):
        if self._result is None:
            return None

        return self._result.get("response")
