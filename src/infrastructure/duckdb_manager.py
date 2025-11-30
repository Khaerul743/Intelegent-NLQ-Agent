import os

import duckdb
import pandas as pd
from duckdb import CatalogException


class DuckDbManager:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def _get_dataframe(self, table_name: str) -> pd.DataFrame:
        if os.path.exists(f"{self.directory_path}/{table_name}.csv"):
            df = pd.read_csv(f"{self.directory_path}/{table_name}.csv")
        elif os.path.exists(f"{self.directory_path}/{table_name}.xls"):
            df = pd.read_excel(f"{self.directory_path}/{table_name}.xls")
        elif os.path.exists(f"{self.directory_path}/{table_name}.xlsx"):
            df = pd.read_excel(f"{self.directory_path}/{table_name}.xlsx")
        else:
            raise ValueError(
                "Error while getting dataframe: Dataset file not found. Please enter the correct table name or directory folder path"
            )
        return df

    def get_data(self, query: str, table_name: str):
        try:
            df = self._get_dataframe(table_name)

            db_path = os.path.join(self.directory_path, f"{table_name}.db")

            # Create directory if it doesn't exist
            if not os.path.exists(db_path):
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                print("Database created")
            else:
                print("Database already exists")

            # Always connect and register the table with dynamic name
            connection = duckdb.connect(database=db_path)
            connection.register(table_name, df)

            try:
                result = connection.execute(query).df()
            except CatalogException as e:
                raise e
            finally:
                connection.close()

            return result.to_string()
        except CatalogException as e:
            raise e
        except ValueError as e:
            raise e
        except Exception as e:
            raise e

    def get_dataset_info(self, table_name: str) -> str:
        try:
            df = self._get_dataframe(table_name)

            # Basic information
            num_rows = len(df)
            num_cols = len(df.columns)

            # Column information
            columns_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null_count = df[col].count()
                null_count = num_rows - non_null_count

                # Get sample values for better understanding
                sample_values = df[col].dropna().head(3).tolist()
                sample_str = ", ".join([str(val) for val in sample_values])

                columns_info.append(
                    f"- **{col}** ({dtype}): {non_null_count} non-null values, {null_count} null values. Sample values: {sample_str}"
                )

            # Create comprehensive description
            description = f"""Table name: **{table_name}**
        Total rows: **{num_rows}**
        Total columns: **{num_cols}**

        **Column Details:**
        {chr(10).join(columns_info)}

        **Data Summary:**
        - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB
        - Duplicate rows: {df.duplicated().sum()}"""

            return description
        except ValueError as e:
            raise e
        except Exception as e:
            raise e
