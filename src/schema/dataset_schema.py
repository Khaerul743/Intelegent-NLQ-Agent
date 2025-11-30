from typing import Any

from pydantic import BaseModel


class DatasetDetailInformation(BaseModel):
    available_datasets: list[str]
    dataset_descriptions: dict[str, Any]


# EXAMPLE
# dataset = DatasetDetailInformation(
#     available_datasets=["customers"],
#     dataset_descriptions={"customers": "Data customer"},
# )
