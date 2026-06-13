-- 1. Aktifkan ekstensi pgvector di PostgreSQL
create extension if not exists vector;

-- 2. Buat tabel untuk menyimpan potongan dokumen (chunks)
create table laporan_mandiri_chunks (
    id uuid primary key default gen_random_uuid(),
    content text not null,                -- Menyimpan teks asli, Markdown tabel, atau deskripsi gambar
    embedding vector(768),               -- Menyimpan hasil embedding (768 dimensi)
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Metadata kolom untuk mempermudah filtering dan debugging
    metadata jsonb not null              -- Menyimpan {"page_number": 12, "source": "laporan_2025.pdf", "type": "table/text/chart"}
);

-- 3. Buat indeks HNSW untuk pencarian kemiripan yang cepat (Cosine Similarity)
create index on laporan_mandiri_chunks using hnsw (embedding vector_cosine_ops);