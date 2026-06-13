# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage, SystemMessage

class AnswerGenerator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    def generate_answer(self, query: str, retrieved_contexts: list):
        # Combine context
        context_texts = []
        for ctx in retrieved_contexts:
            meta = ctx["metadata"]
            source = meta.get("source", "Unknown")
            page = meta.get("page_number", "?")
            context_texts.append(f"[Sumber: {source}, Halaman: {page}]\n{ctx['content']}")
            
        full_context = "\n\n---\n\n".join(context_texts)
        
        system_prompt = """Kamu adalah asisten analis keuangan senior yang ahli dalam mengekstrak dan menganalisis laporan perusahaan. 
TUGAS UTAMA:
Jawab pertanyaan pengguna secara terstruktur, jelas, dan profesional HANYA berdasarkan konteks yang diberikan. Konteks dapat berupa teks naratif, tabel (Markdown), atau deskripsi visual/grafik.
ATURAN KETAT:
1. TANPA HALUSINASI: Jangan pernah mengarang informasi, berasumsi, atau menggunakan pengetahuan di luar konteks yang diberikan.
2. JAWABAN TIDAK DITEMUKAN: Jika informasi untuk menjawab pertanyaan tidak ada di dalam konteks, Anda WAJIB menjawab dengan: "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang diberikan."
3. KUTIPAN WAJIB: Setiap klaim atau data yang Anda sebutkan harus disertai sumber yang spesifik.
FORMAT OUTPUT & KUTIPAN:
- Gunakan poin-poin (bullet points) atau paragraf pendek agar mudah dibaca.
- Jika dari teks, tuliskan halamannya di akhir jawaban anda. Contoh: (Halaman 12).
- Jika dari data visual/ilustrasi/tabel, tuliskan judul visualnya di akhir jawaban anda. Contoh: (Sumber Jawaban : Chart “Komposisi DPK Bank Mandiri” (Halaman 6)).

Konteks Dokumen:
{context}
"""
        
        prompt = system_prompt.format(context=full_context)
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]
        
        print("Menghasilkan jawaban dari LLM Gemini...")
        response = self.llm.invoke(messages)
        return response.content
