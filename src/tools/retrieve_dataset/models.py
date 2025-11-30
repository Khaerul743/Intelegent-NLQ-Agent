from typing import Optional

from pydantic import BaseModel, Field


class StructuredOutputValidateTableExist(BaseModel):
    is_table_exist: bool = Field(
        description="Apakah minimal ada satu table dari data yang diminta tersedia di TABLE YANG TERSEDIA."
    )
    description_analyst_result: str = Field(
        description="Deskripsi LENGKAP dari hasil analisis. Harus memuat kategori 'Table to query (exists)' dan 'Table not available (skip or need action)'."
    )


class TableRequeired(BaseModel):
    table_name: str = Field(description="Nama table yang akan digunakan.")
    required_colums: list[str] = Field(
        description="Kolom apa saja yang dibutuhkan dari data", default_factory=list
    )
    filters: str = Field(
        description="Filter atau kondisi yang diperlukan dalam pengambilan data",
        default="",
    )


class StructuredOutputQueryNeeded(BaseModel):
    problem: str = Field(
        description="Tujuan query, apa yang ingin diambil dari data yang tersedia"
    )
    problem_solving: str = Field(
        description="Reasoning lengkap langkah-langkah bagaimana cara mengambil data tersebut"
    )
    table_required: list[TableRequeired] = Field(
        description="Table yang akan digunakan."
    )


class Query(BaseModel):
    table_name: str = Field(description="Nama table yang akan digunakan.")
    query: str = Field(
        description="Query SQL yang akan digunakan untuk menjawab pertanyaan"
    )


class StructuredOutputGenerateQuery(BaseModel):
    list_queries: list[Query] = Field(
        description="Kumpulan dari query yang akan digunakan. Pastikan nama table dan query sesuai."
    )


class StructuredOutputValidationResult(BaseModel):
    is_valid: bool = Field(
        description="Apakah data result sudah sesuai dengan yang diminta."
    )
    next_step_query: Optional[StructuredOutputQueryNeeded] = Field(
        description="Langkah-langkah query selanjutnya yang akan dilakukan, jika data kurang/tidak sesuai yang diminta.",
        default=None,
    )


class RetreiveDatasetModel(BaseModel):
    data_description_needed: str
    is_table_exist: bool = False
    tables_description: str = ""
    analyst_query_needed_result: Optional[StructuredOutputQueryNeeded] = None
    list_queries: Optional[StructuredOutputGenerateQuery] = None
    result: list[str] = []
    is_valid: bool = False
