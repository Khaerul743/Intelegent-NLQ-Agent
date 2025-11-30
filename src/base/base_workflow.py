from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
)


class BaseWorkflow(ABC):
    @abstractmethod
    def run(self, state, thread_id: str) -> Dict[str, Any] | Any:
        pass

    @abstractmethod
    def show(self):
        pass
