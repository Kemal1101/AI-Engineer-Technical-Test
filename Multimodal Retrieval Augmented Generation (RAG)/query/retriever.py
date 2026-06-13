# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import psycopg
# pyrefly: ignore [missing-import]
from pgvector.psycopg import register_vector
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

class DocumentRetriever:
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is missing.")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def retrieve_context(self, query: str, top_k: int = 5):
        # 1. Embed query
        print(f"Mencari konteks untuk query: '{query}'")
        query_embedding = self.embeddings.embed_query(query)
        
        # 2. Vector search in database
        results = []
        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # Use cosine distance (<=>) for HNSW index
                cur.execute(
                    """
                    SELECT content, metadata, embedding <=> %s::vector AS distance
                    FROM laporan_mandiri_chunks
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (query_embedding, top_k)
                )
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "content": row[0],
                        "metadata": row[1],
                        "distance": row[2]
                    })
        return results
