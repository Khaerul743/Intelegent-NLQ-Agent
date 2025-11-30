from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.infrastructure import DuckDbManager
from src.schema import DatasetDetailInformation

from .models import StructuredOutputQueryNeeded


class RetrieveDatasetPrompt:
    def __init__(
        self,
        dataset_information: DatasetDetailInformation,
        duckdb_manager: DuckDbManager,
    ):
        self.dataset_information = dataset_information
        self.duckdb_manager = duckdb_manager

    def _get_detail_data_parts(self) -> str:
        try:
            detail_dataparts = []
            for i in self.dataset_information.available_datasets:
                detail_datapart = self.duckdb_manager.get_dataset_info(i)
                detail_dataparts.append(detail_datapart)

            detail_data_parts_str = "\n\n".join(detail_dataparts)
            return detail_data_parts_str
        except ValueError as e:
            return "Tidak ada table yang tersedia"

    def _get_data_descriptions(self) -> str:
        list_data_desc = []
        for k, v in self.dataset_information.dataset_descriptions:
            list_data_desc.append(f"{k}: {v}")

        return "\n".join(list_data_desc)

    def analyst_table_exist(self, main_instruction: str) -> list[BaseMessage]:
        detail_data = self._get_detail_data_parts()

        return [
            SystemMessage(
                content=f"""
Kamu adalah **Table Existence Analyzer** yang cerdas dan teliti.
Tugas utama: Menganalisis ketersediaan tabel yang diperlukan untuk memenuhi permintaan pengguna SEBELUM membuat query.

INSTRUKSI UTAMA:
1.  **Identifikasi Kebutuhan:** Baca permintaan pengguna (`main_instruction`) dan tentukan daftar **SEMUA** tabel yang idealnya diperlukan untuk memenuhi kebutuhan data.
2.  **Verifikasi Ketersediaan:** Cocokkan daftar tabel yang diperlukan dengan `TABLE YANG TERSEDIA` di bawah ini.
3.  **Kategorisasi Wajib:** Kategorikan **SEMUA** tabel yang teridentifikasi menjadi dua daftar, dan **kedua daftar ini wajib dimasukkan ke dalam description_analyst_result**:
    * **"Table to query (exists)"**: Tabel yang **DITEMUKAN** di `TABLE YANG TERSEDIA`. Untuk setiap tabel, berikan ringkasan singkat mengapa tabel itu relevan (1-2 kalimat).
    * **"Table not available (skip or need action)"**: Tabel yang **TIDAK DITEMUKAN** di `TABLE YANG TERSEDIA`. Jelaskan secara singkat mengapa data terkait tidak bisa di-query karena tabelnya tidak ada (misalnya: 'Tabel customer tidak ada, sehingga data customer tidak dapat diambil. Bagian ini akan dilewati atau memerlukan tindakan perbaikan.').
4.  **Akurasi Nama:** JANGAN mengarang nama tabel atau kolom. Gunakan nama **EXACT** persis seperti tertulis di `TABLE YANG TERSEDIA`.
5.  **Batasan:** JANGAN MENULIS QUERY SQL. Hanya deskripsi dan rekomendasi operasional.

BERIKUT ADALAH DESKRIPSI DATA YANG TERSEDIA:
{self._get_data_descriptions()}

TABLE YANG TERSEDIA:
{detail_data}
"""
            ),
            HumanMessage(
                content=f"""
Silakan lakukan analisis berikut berdasarkan instruksi di atas.

MAIN INSTRUCTION:
{main_instruction}
"""
            ),
        ]

    def analyst_query_needed(
        self, main_instruction: str, tables_description: str
    ) -> list[BaseMessage]:
        detail_data = self._get_detail_data_parts()
        return [
            SystemMessage(
                content=f"""
Kamu adalah *Query Reasoning Agent* dalam arsitektur LangGraph.
Tugas utama kamu adalah **menganalisis kebutuhan data** dan **menentukan langkah-langkah** yang diperlukan untuk membangun query ke database.

⚠️ **PERINGATAN SANGAT PENTING**  
Output dari agent ini *langsung digunakan* oleh node berikutnya untuk membangun query SQL.  
Kesalahan dalam penamaan kolom = query akan error.  
Jangan mengarang kolom yang tidak ada.  
JANGAN MEMBUAT QUERY SQL. Kamu hanya memberikan reasoning.

---

## 🎯 TUGAS:
1. Analisis apa yang ingin diambil oleh user (problem).
2. Evaluasi struktur data: nama kolom, tipe data, contoh data, jumlah baris.
3. Tentukan langkah-langkah logis untuk mengambil data tersebut (problem_solving).
4. Identifikasi kolom yang dibutuhkan **dengan nama EXACT**.
5. Identifikasi filter/kondisi (jika ada) dalam bentuk plain-text, bukan SQL.

---

## 🧠 CARA KERJA:
1. **Pahami permintaan pengguna**: apa informasi utama yang ingin diambil?
2. **Analisis data tersedia**: cek kolom-kolom yang relevan.
3. **Pecah langkah-langkah**:
   - apakah butuh filter?
   - apakah butuh sorting?
   - apakah butuh aggregasi?
   - apakah butuh group-by?
4. **Validasi kolom**:
   - semua kolom yang kamu sebutkan harus ada dalam struktur data.
   - jangan menebak nama kolom.

---
##DESKRIPSI DATA YANG TERSEDIA:
{self._get_data_descriptions()}

## 🗂️ TABLE YANG TERSEDIA:
{detail_data}
"""
            ),
            HumanMessage(
                content=f"""
Berikut adalah deskripsi data yang perlu diambil dari database, untuk itu pastikan kamu analisis serta berikan langkah-langkan dengan detail:
{main_instruction}

Berikut adalah tambahan informasi tentang table:
{tables_description}
"""
            ),
        ]

    def generate_query(
        self,
        query_needed: Optional[StructuredOutputQueryNeeded] = None,
    ) -> list[BaseMessage]:
        detail_data = self._get_detail_data_parts()

        detail_query = []
        problem = ""
        problem_solving = ""
        if query_needed is not None:
            problem = query_needed.problem
            problem_solving = query_needed.problem_solving
            for i in query_needed.table_required:
                detail_query.append(
                    f"*Table name: {i.table_name}\n Required column: {', '.join(i.required_colums)}\n Filters: {i.filters}"
                )

        return [
            SystemMessage(
                content=f"""
Kamu adalah agent yang bertugas untuk men-generate query database berdasarkan analisis problem dan solusi yang telah dibuat oleh agent reasoning sebelumnya.

## TUGAS:
1. Analisis problem dan problem_solving yang diberikan
2. Generate query SQL yang sesuai untuk menjawab pertanyaan pengguna
3. Pastikan query menggunakan nama kolom yang EXACT sesuai dengan struktur database

## CARA KERJA:
1. **Baca Problem**: Pahami apa yang ingin diselesaikan
2. **Analisis Solusi**: Lihat langkah-langkah yang telah direncanakan
3. **Generate Query**: Buat query SQL berdasarkan solusi yang diberikan

## BERIKUT ADALAH INFORMASI DATA YANG TERSEDIA:
{detail_data}

Pastikan nama table yang digunakan sesuai dengan nama table yang tersedia.
**OUTPUT**: Berikan HANYA query SQL tanpa penjelasan atau tambahan apapun.
"""
            ),
            HumanMessage(
                content=f"""                
Generate query SQL berdasarkan problem dan solusi yang telah dianalisis.
## PROBLEM:
{problem}

## SOLUSI:
{problem_solving} 

Berikut adalah deskripsi query yang dapat membantu kamu untuk menghasilkan query yang sesuai:
{"\n\n".join(detail_query)}
"""
            ),
        ]

    def validation_result(
        self,
        main_instruction: str,
        data_result: list[str],
        tables_description: str,
        previous_problem_solve: Optional[str] = None,
    ) -> list[BaseMessage]:
        detail_data = self._get_detail_data_parts()
        data_result_str = "\n\n".join(data_result)

        return [
            SystemMessage(
                content=f"""
Kamu adalah Validation Agent untuk memverifikasi hasil query sebelumnya dalam konteks arsitektur data kami.
Tugas:
1. Verifikasi apakah data yang dikembalikan (data_result) sudah memenuhi permintaan pengguna (main_instruction) dan sesuai dengan langkah solusi sebelumnya (previous_problem_solve).
2. Identifikasi masalah secara spesifik: kolom hilang, nama kolom salah, tipe data tidak sesuai, jumlah baris tidak memadai, filter tidak sesuai, atau data tidak relevan.
3. Jika masih kurang, berikan langkah-langkah terperinci untuk mengambil data tambahan atau memperbaiki query (langkah-langkah operasional, bukan SQL).
4. Jangan membuat nama kolom atau tabel yang tidak ada. Seluruh nama kolom yang disebut harus ada di {detail_data}.

- Gunakan nama kolom EXACT seperti yang ada di detail tabel. Jangan menebak.

BERIKUT DESKRIPSI DATA YANG TERSEDIA:
{self._get_data_descriptions()}
"""
            ),
            HumanMessage(
                content=f"""
Validasi hasil berikut ini berdasarkan instruksi dan solusi sebelumnya.

MAIN INSTRUCTION:
{main_instruction}

PREVIOUS SOLUTION (problem_solving):
{previous_problem_solve}

DATA_RESULT (baris contoh atau ringkasan hasil query):
{data_result_str}

INFORMASI TABLE:
{tables_description}

informasi table cukup penting karena informasi tersebut memberikan deskripsi table yang tersedia sehingga perlu di query, dan table yang tidak tersedia sehingga tidak perlu diquery.
"""
            ),
        ]
