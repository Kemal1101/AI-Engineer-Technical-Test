# Multimodal Retrieval-Augmented Generation (RAG) System

Sistem Multimodal RAG ini dirancang untuk membaca, mengekstrak, dan menganalisis laporan kompleks, seperti Laporan Keuangan Bank Mandiri 2025. Sistem ini menggunakan teknologi terkini untuk memahami berbagai jenis data dari dokumen PDF, termasuk teks naratif, tabel, dan grafik/visual, kemudian menjawab pertanyaan pengguna berdasarkan konteks tersebut dengan tingkat akurasi tinggi dan minim halusinasi.

Sistem ini memiliki dua komponen utama:
1. **Pipeline Ingesti (Ingestion)**: Mengubah dokumen PDF menjadi data terstruktur (teks, markdown tabel, dan narasi deskripsi gambar) yang di-chunk, di-embed, dan disimpan ke dalam database vektor.
2. **REST API**: Endpoint FastAPI yang menerima query dari pengguna, mencari potongan dokumen (chunks) paling relevan (retrieval), dan menggunakan model *Large Language Model (LLM)* untuk menyusun jawaban akhir beserta referensi halamannya (generation).

## Fitur Utama
* **Multimodal Parsing**: Mampu mengekstrak teks biasa, mengonversi tabel menjadi format Markdown, serta mendeskripsikan gambar/grafik menggunakan model LLM Vision.
* **Semantic Chunking**: Membagi dokumen menjadi beberapa potongan bermakna berdasarkan kesamaan semantik (bukan sekadar batasan jumlah karakter/token).
* **Vector Search Database**: Penyimpanan vektor dengan skalabilitas tinggi menggunakan PostgreSQL dengan ekstensi `pgvector`, dilengkapi dengan pencarian efisien menggunakan index HNSW (Hierarchical Navigable Small World).
* **High-Accuracy LLM Generation**: Menggunakan model `gemini-2.5-flash` dari Google Generative AI yang berfokus memberikan jawaban faktual dari konteks yang diberikan, lengkap dengan referensi (keterangan sumber dan halaman).

## Tech Stack
* **Bahasa Pemrograman**: Python 3
* **Framework API**: FastAPI, Uvicorn
* **Orkestrasi LLM**: LangChain, LangChain Experimental
* **LLM & Vision**: Google Gemini (`gemini-2.5-flash`)
* **Embedding Model**: HuggingFace (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`)
* **Database Vector**: PostgreSQL dengan ekstensi `pgvector`
* **Ekstraksi PDF**: PyMuPDF (`fitz`), `pdfplumber`

## Struktur Direktori
```
.
├── ingestion/
│   ├── chunker.py    # Modul untuk semantic chunking & embedding
│   ├── indexer.py    # Modul untuk menyimpan vector embeddings ke PostgreSQL
│   └── parser.py     # Modul untuk ekstraksi teks, gambar, dan tabel dari PDF
├── query/
│   ├── generator.py  # Modul untuk menghasilkan jawaban dengan Google Gemini LLM
│   └── retriever.py  # Modul untuk mencari teks relevan dari database vektor
├── main.py           # Entry point untuk FastAPI
├── ingest.py         # Script CLI untuk menjalankan proses Ingesti dokumen PDF
├── database.sql      # Schema query untuk setup tabel database PostgreSQL & pgvector
├── requirements.txt  # Daftar dependensi library Python
└── .env              # (Dibuat manual) Environment variable (Kunci API & DB URL)
```

## Prasyarat
Sebelum menjalankan sistem ini, pastikan Anda telah menginstal dan menyiapkan hal-hal berikut:
1. **Python 3.9+** terinstal di sistem Anda.
2. **PostgreSQL** dengan ekstensi **`pgvector`** yang sudah terinstal dan aktif. (Bisa menggunakan layanan seperti Supabase).
3. **Google API Key** untuk menggunakan layanan LLM Gemini dari Google AI Studio.

## Instalasi
1. Clone atau masuk ke direktori proyek.
2. Buat Virtual Environment (opsional namun disarankan):
   ```bash
   python -m venv venv
   # Di Windows
   venv\Scripts\activate
   # Di Linux/Mac
   source venv/bin/activate
   ```
3. Instal semua dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

## Konfigurasi Environment (`.env`)
Buat file bernama `.env` di *root* direktori proyek, dan isi dengan variabel berikut:

```env
DATABASE_URL=postgresql://[user]:[password]@[host]:[port]/[db_name]
GOOGLE_API_KEY=your_google_gemini_api_key_here
```
> Ganti nilai `DATABASE_URL` dengan *connection string* database PostgreSQL Anda dan `GOOGLE_API_KEY` dengan kunci API Gemini Anda.

## Setup Database
Anda perlu menyiapkan tabel dan ekstensi `pgvector` di database PostgreSQL Anda. Jalankan query SQL yang terdapat pada file `database.sql` pada tool database administration Anda (misal: DBeaver, pgAdmin, atau CLI psql).

```sql
-- 1. Aktifkan ekstensi pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Buat tabel
CREATE TABLE laporan_mandiri_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    metadata JSONB NOT NULL
);

-- 3. Buat index HNSW untuk pencarian cosine similarity yang cepat
CREATE INDEX ON laporan_mandiri_chunks USING hnsw (embedding vector_cosine_ops);
```

## Penggunaan

### 1. Ingesti Dokumen (Memasukkan PDF ke Database)
Sebelum API dapat menjawab pertanyaan, Anda harus melakukan *ingest* pada dokumen PDF agar datanya tersimpan di database vektor.

Jalankan script `ingest.py` dan berikan path file PDF yang akan diproses:
```bash
python ingest.py "Laporan_Keuangan_Bank_Mandiri_2025.pdf"
```
**Proses ini akan melakukan:**
* Membaca teks pada tiap halaman PDF.
* Mengekstrak tabel dan mengonversinya ke bentuk Markdown.
* Mengekstrak gambar/grafik dan mendeskripsikan isinya dengan Gemini Vision.
* Membagi data tersebut menggunakan `SemanticChunker`.
* Menghasilkan *vector embeddings* lokal dengan model `sentence-transformers`.
* Menyimpan konten, metadata, dan vektor ke database PostgreSQL.

### 2. Menjalankan REST API
Setelah proses ingesti selesai, jalankan server API dengan perintah berikut:
```bash
python main.py
```
*Atau dengan uvicorn langsung:*
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API akan berjalan di `http://localhost:8000`.

### 3. Menggunakan API
Anda bisa menguji endpoint `/query` menggunakan `curl`, Postman, atau langsung dari dokumentasi interaktif Swagger UI di `http://localhost:8000/docs`.

**Endpoint `/query` (POST)**

**Request Body:**
```json
{
  "query": "Berapa total aset Bank Mandiri pada tahun 2025?",
  "top_k": 5
}
```
*   `query`: Pertanyaan yang ingin diajukan.
*   `top_k` (opsional): Jumlah potongan dokumen relevan yang akan dicari di database (default: 5).

**Response (Contoh):**
```json
{
  "answer": "Total aset Bank Mandiri pada tahun 2025 tercatat sebesar Rp 1.500 Triliun. (Halaman 24).",
  "sources": [
    {
      "source": "Laporan_Keuangan_Bank_Mandiri_2025.pdf",
      "page": 24
    },
    {
      "source": "Laporan_Keuangan_Bank_Mandiri_2025.pdf",
      "page": 45
    }
  ]
}
```
