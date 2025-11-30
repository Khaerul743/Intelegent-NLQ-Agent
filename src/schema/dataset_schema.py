from typing import Any, Optional

from pydantic import BaseModel


class DatasetDetailInformation(BaseModel):
    available_datasets: list[str]
    dataset_descriptions: dict[str, Any]
