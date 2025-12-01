## NLQ Agent

NLQ Agent adalah _AI agent_ berbasis **LangGraph** dan **LangChain** yang mampu:

- **Menerima pertanyaan natural language** dari user
- **Menganalisis kebutuhan data**, memilih tabel yang relevan
- **Membangun dan mengeksekusi query SQL** ke dataset lokal (CSV/Excel via DuckDB)
- **Mengembalikan hasil dan menjelaskannya** ke user dengan bahasa natural

Project ini dirancang agar **mudah diperluas**: Anda cukup menambahkan file dataset dan konfigurasi deskripsinya, lalu agent akan dapat memahami dan melakukan query ke dataset tersebut.

---

## Teknologi & Library yang Digunakan

- **LangGraph (`langgraph`)**  
  - Mengatur _workflow_ agent sebagai _state machine_ (graph)  
  - Dipakai di:
    - `AgentNLQWorkflow` (`src/agent/workflow.py`)
    - `RetrieveDatasetWorkflow` (`src/tools/retrieve_dataset/tool_workflow.py`)

- **LangChain (`langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`)**  
  - Abstraksi LLM dan integrasi provider:
    - OpenAI (`ChatOpenAI`)
    - Anthropic (`ChatAnthropic`)
    - Google Generative AI (`ChatGoogleGenerativeAI`)
  - Digunakan di `BaseNode` (`src/base/base_node.py`) dan seluruh node agent/tool.

- **DuckDB (`duckdb`)**  
  - _In-process database_ untuk menjalankan query SQL di atas **dataframe** (CSV / Excel)
  - Manajemen oleh `DuckDbManager` (`src/infrastructure/duckdb_manager.py`).

- **Pandas (`pandas`)**  
  - Membaca file **CSV / XLS / XLSX** menjadi `DataFrame` sebelum diregister ke DuckDB.

- **Pydantic (`pydantic`)**  
  - Model state dan konfigurasi:
    - `BaseAgentStateModel` (`src/base/base_model.py`)
    - `DatasetDetailInformation` (`src/schema/dataset_schema.py`)
    - Berbagai _structured output model_ untuk tool retrieve dataset (`src/tools/retrieve_dataset/models.py`).

- **tiktoken**  
  - Estimasi jumlah token untuk prompt dan respon di `BaseNode`.

- **python-dotenv (`python-dotenv`)**  
  - Memuat environment variables (misal API key LLM) dari file `.env`.

Dependency utama didefinisikan di `pyproject.toml`.

---

## Arsitektur Project

Struktur direktori utama:

- **`main.py`**  
  Entry point aplikasi (loop CLI sederhana).

- **`src/base/`** – _Base abstractions_:
  - `BaseAgent` (`src/base/base_agent.py`)  
    - Wrapper umum untuk menjalankan workflow dan menghitung waktu respon.
  - `BaseAgentStateModel` (`src/base/base_model.py`)  
    - State standar agent (menyimpan `messages`, `user_message`, dan `response`).
  - `BaseNode` (`src/base/base_node.py`)  
    - Abstraksi node yang:
      - Menginisialisasi LLM berdasarkan provider
      - Memanggil LLM (biasa, dengan tools, dan structured output)
      - Menghitung estimasi token.
  - `BaseWorkflow` (`src/base/base_workflow.py`)  
    - Abstraksi workflow berbasis LangGraph.

- **`src/infrastructure/`**:
  - `DuckDbManager` (`src/infrastructure/duckdb_manager.py`)  
    - Membaca file dataset (`.csv`, `.xls`, `.xlsx`) dari folder tertentu
    - Meregister `DataFrame` sebagai tabel di DuckDB
    - Mengeksekusi query SQL dan mengembalikan hasil sebagai string
    - Menyediakan fungsi `get_dataset_info` untuk mendeskripsikan struktur dataset (jumlah kolom, tipe kolom, contoh nilai, dll).

- **`src/schema/`**:
  - `DatasetDetailInformation` (`src/schema/dataset_schema.py`)  
    - Model konfigurasi yang berisi:
      - `available_datasets: list[str]` – daftar nama dataset / tabel
      - `dataset_descriptions: dict[str, Any]` – deskripsi singkat tiap dataset

- **`src/tools/`**:
  - `RetrieveDatasetTool` (`src/tools/retrieve_dataset/retrieve_datasets.py`)  
    - Tool LangChain yang digunakan agent utama untuk:
      - Menganalisis kebutuhan query
      - Menentukan tabel dan kolom yang relevan
      - Men-generate SQL
      - Menjalankan SQL ke DuckDB
      - Memvalidasi apakah hasil sudah menjawab pertanyaan user.
  - `RetrieveDatasetPrompt` (`src/tools/retrieve_dataset/prompts.py`)  
    - Kumpulan prompt sistem untuk beberapa tahap:
      - Analisis ketersediaan tabel (`analyst_table_exist`)
      - Analisis kebutuhan query (`analyst_query_needed`)
      - Generate SQL (`generate_query`)
      - Validasi hasil (`validation_result`)
    - Memanfaatkan `DuckDbManager.get_dataset_info` untuk menyusun deskripsi tabel.
  - `RetrieveDatasetNodes` (`src/tools/retrieve_dataset/tool_nodes.py`)  
    - Node-node LangGraph untuk tiap langkah:
      - Cek ketersediaan tabel (`analyst_table_exits`)
      - Analisis kebutuhan query
      - Generate query
      - Eksekusi query ke DuckDB
      - Validasi hasil dan _retry_ jika perlu.
  - `RetrieveDatasetWorkflow` (`src/tools/retrieve_dataset/tool_workflow.py`)  
    - Graph LangGraph khusus untuk alur retrieve dataset.

- **`src/agent/`**:
  - `AgentNLQ` (`src/agent/agent.py`)  
    - Implementasi agent utama berbasis `BaseAgent`.
    - Menginisialisasi:
      - `MemorySaver` (checkpoint LangGraph)
      - `DuckDbManager` (berdasarkan path directory dataset)
      - `RetrieveDatasetTool`
      - `AgentNLQPrompt`
      - `AgentNLQNode`
      - `AgentNLQWorkflow`
  - `AgentNLQPrompt` (`src/agent/prompts.py`)  
    - Prompt sistem utama untuk agent:
      - Menjelaskan peran agent
      - Menjelaskan kapan dan bagaimana menggunakan tool `Retrieve_dataset`
      - Menyertakan deskripsi dataset dan detail dataset dari `RetrieveDatasetPrompt`.
  - `AgentNLQNode` (`src/agent/nodes.py`)  
    - Node utama yang:
      - Menyusun prompt (system + riwayat pesan + pesan user)
      - Memanggil LLM dengan tools (`call_llm_with_tool`)
      - Menghitung total token
      - Menyusun state baru (`messages`, `response`).
  - `AgentNLQWorkflow` (`src/agent/workflow.py`)  
    - Graph LangGraph untuk agent utama:
      - Node `main_agent` (memanggil LLM dengan tool)
      - Node `read_file` (ToolNode yang menjalankan `read_dataset`)
      - Node `anwser_tool_message` (LLM menjawab berdasarkan hasil tool)
      - Edge kondisional dengan `conditional_tool_call`.

---

## Alur Kerja Agent Secara Singkat

1. **User** memasukkan pertanyaan di CLI (`main.py`).
2. **AgentNLQ** membuat state awal (`BaseAgentStateModel`) dan menjalankan workflow.
3. **Node `main_agent`**:
   - Menggabungkan prompt sistem + history + pesan user.
   - Memanggil LLM dengan tool `read_dataset`.
   - Jika LLM memutuskan memanggil tool:
     - State dialirkan ke node `read_file` (ToolNode).
4. **Tool `ReadDatasetTool.read_dataset`**:
   - Menjalankan `RetrieveDatasetWorkflow`:
     - Analisis tabel tersedia
     - Analisis kebutuhan query
     - Generate SQL
     - Eksekusi SQL via DuckDB
     - Validasi hasil (bisa retry sampai 3x).
5. **Node `anwser_tool_message`**:
   - LLM diminta menyusun jawaban final ke user berdasarkan hasil tool.
6. **CLI** menampilkan jawaban dan menyimpan waktu respon + total token.

---

## Cara Menjalankan Project

### 1. Persiapan Environment

1. Pastikan Python versi **>= 3.13** (sesuai `pyproject.toml`).
2. Instal dependency (disarankan menggunakan `uv` atau `pip`):

```bash
pip install -e .
```

Atau, jika menggunakan `uv`:

```bash
uv sync
```

3. Siapkan file `.env` di root project untuk API key LLM, contoh:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_genai_api_key
```

> Gunakan hanya provider & model yang ingin Anda pakai di `main.py`.

### 2. Menjalankan CLI

Jalankan:

```bash
python main.py
```

Contoh alur:

- Masukkan pertanyaan, misalnya:  
  `Tampilkan 10 customer dengan total transaksi terbesar.`
- Agent akan:
  - Menganalisis dataset yang ada
  - Jika perlu, memanggil tool untuk query ke DuckDB
  - Menjawab dengan hasil dan penjelasan.
- Untuk keluar, ketik: `exit`, `e`, atau `ex`.

---

## Cara Menambahkan Dataset Baru

Bagian ini penting agar **agent dapat “memahami” dataset baru** yang Anda tambahkan.

### 1. Letakkan File Dataset di Folder Dataset

Secara default, project ini mengasumsikan folder dataset di:

- `dataset/` (lihat parameter kedua di `AgentNLQ` pada `main.py`)

Langkah:

- Simpan file Anda di folder `dataset/` dengan format:
  - `nama_dataset.csv`, atau
  - `nama_dataset.xls`, atau
  - `nama_dataset.xlsx`
- **Nama file (tanpa ekstensi)** akan menjadi **nama tabel** yang digunakan di konfigurasi dan query.

Contoh:

- File: `dataset/customers.csv` → nama tabel: `"customers"`
- File: `dataset/orders.xlsx` → nama tabel: `"orders"`

### 2. Konfigurasi `DatasetDetailInformation`

Konfigurasi ini memberi tahu agent:

- Dataset apa saja yang tersedia (`available_datasets`)
- Deskripsi singkat tiap dataset (`dataset_descriptions`)

Model ada di `src/schema/dataset_schema.py`:

```python
from src.schema import DatasetDetailInformation

dataset = DatasetDetailInformation(
    available_datasets=["customers"],
    dataset_descriptions={"customers": "Data customer"},
)
```

Untuk menambahkan dataset baru, misalnya `orders.csv`:

```python
from src.schema import DatasetDetailInformation

dataset = DatasetDetailInformation(
    available_datasets=["customers", "orders"],
    dataset_descriptions={
        "customers": "Data customer (profil dan informasi dasar pelanggan)",
        "orders": "Data transaksi pesanan yang berisi detail order per customer",
    },
)
```

**Catatan penting:**

- Nilai di `available_datasets` **harus sama persis** dengan nama file (tanpa ekstensi) di folder `dataset/`.
- `dataset_descriptions` sangat membantu agent untuk:
  - Menentukan dataset mana yang relevan dengan pertanyaan
  - Menjelaskan data ke user.

### 3. Update Inisialisasi Agent di `main.py`

Contoh `main.py` yang ada saat ini:

```python
from src.agent import AgentNLQ
from src.base import BaseAgentStateModel
from src.schema import DatasetDetailInformation

dataset = DatasetDetailInformation(
    available_datasets=["customers"],
    dataset_descriptions={"customers": "Data customer"},
)

agent = AgentNLQ(dataset, "dataset", "openai", "gpt-4o-mini")
```

Jika Anda sudah menambahkan dataset baru (`orders`), cukup sesuaikan object `dataset` seperti contoh sebelumnya.  
Parameter `AgentNLQ`:

- `dataset_detail_information`: instance `DatasetDetailInformation`
- `directory_datasets_path`: path folder dataset (misal `"dataset"`)
- `llm_provider`: `"openai"`, `"anthropic"`, atau `"google"`
- `llm_model`: nama model sesuai provider (misal `"gpt-4o-mini"`, dsb.)

### 4. Bagaimana Agent Memakai Konfigurasi Dataset?

Setelah `DatasetDetailInformation` Anda isi dengan benar:

1. `AgentNLQ` meneruskan object ini ke `RetrieveDatasetTool`.
2. `RetrieveDatasetPrompt` menggunakan:
   - `available_datasets` untuk meminta **detail struktur tabel** ke `DuckDbManager`.
   - `dataset_descriptions` untuk membuat deskripsi naratif tentang data yang tersedia.
3. Informasi ini dimasukkan ke berbagai prompt:
   - Agent utama (`AgentNLQPrompt.main_agent`)
   - Tool reasoning (analisis tabel, analisis kebutuhan query, validasi).
4. Dengan demikian, LLM dapat:
   - Mengetahui konteks tiap dataset
   - Memilih tabel yang paling relevan
   - Menghindari penyebutan tabel/kolom yang tidak ada.

---

## Tips & Best Practice Saat Menambah Dataset

- **Gunakan nama file & tabel yang deskriptif**  
  Contoh: `customers`, `orders`, `products`, `transactions_monthly`.

- **Isi `dataset_descriptions` dengan cukup detail**, misalnya:
  - Tujuan dataset (contoh: “daftar seluruh transaksi penjualan”)
  - Hubungan dengan dataset lain (contoh: “relasi ke `customers` via `customer_id`”)

- **Pastikan header kolom konsisten**  
  Karena LLM akan menggunakan nama kolom persis seperti yang dibaca DuckDB, hindari:
  - Spasi yang tidak perlu
  - Nama kolom yang terlalu ambigu (misal: `val1`, `val2`).

- **Uji dengan pertanyaan sederhana dulu**  
  Misalnya:
  - “Tampilkan 5 baris pertama dari data orders”
  - “Berapa jumlah baris di tabel customers?”

---

## Ringkasan

- Project **NLQ Agent** menyediakan kerangka kerja lengkap untuk membangun **NLQ (Natural Language Query) Agent** berbasis LangGraph & LangChain.
- Agent dapat:
  - Memahami pertanyaan user
  - Memilih dataset yang tepat
  - Menyusun dan menjalankan query SQL ke DuckDB
  - Menjawab dengan penjelasan yang ramah.
- Untuk menambahkan dataset baru, fokus pada:
  - Menaruh file di folder dataset
  - Mengisi `DatasetDetailInformation` dengan nama tabel dan deskripsi yang tepat
  - Menjalankan ulang agent dan menguji dengan beberapa pertanyaan.


