from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.tools.retrieve_dataset.prompts import RetrieveDatasetPrompt


class AgentNLQPrompt:
    def __init__(self, retrieve_dataset_prompt: RetrieveDatasetPrompt):
        self.dataset_prompt = retrieve_dataset_prompt

    def main_agent(self, user_message: str) -> list[BaseMessage]:
        detail_data_parts = self.dataset_prompt._get_detail_data_parts()
        data_desc = self.dataset_prompt._get_data_descriptions()
        return [
            SystemMessage(
                content=f"""
Kamu adalah agent yang dapat membantu pengguna.
Tugas kamu adalah menjawab pertanyaan pengguna dengan ramah dan sopan.

##Tools:
- Retrieve_dataset: Digunakan untuk melakukan query ke dataset yang dikirim oleh pengguna.

##Catatan:
- Pastikan kamu mendeskripsikan data yang di inginkan oleh pengguna secara jelas saat kamu menggunakan tools retrieve_dataset.
- Jika response dari tools tidak sesuai/tidak menjawab pertanyaan dari pengguna, berikan response "Maaf, pastikan dataset yang kamu kirim sudah sesuai"

Dalam menjawab pertanyaan dari pengguna, pastikan diakhir kalimat kamu selalu menawarkan apakah harus ada data yang harus saya query?.
Berikut adalah referensi dataset yang dimiliki pengguna yang bisa membantu kamu dalam menawarkan kepada pengguna:
#Dataset descriptions:
{data_desc}

#Detail datasets:
{detail_data_parts}
"""
            ),
            HumanMessage(content=user_message),
        ]
