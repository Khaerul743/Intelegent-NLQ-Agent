from src.infrastructure import DuckDbManager
from src.schema import DatasetDetailInformation
from src.tools import RetrieveDatasetTool

dataset = DatasetDetailInformation(
    available_datasets=["customers"],
    dataset_descriptions={"customers": "Data customer"},
)

duckdb = DuckDbManager("dataset")
tool = RetrieveDatasetTool(dataset, "openai", "gpt-3.5-turbo", duckdb)

result = tool.read_dataset("Customer dengan total pengeluaran terbanyak")

print(result)
