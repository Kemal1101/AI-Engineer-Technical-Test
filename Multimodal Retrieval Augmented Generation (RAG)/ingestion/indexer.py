import os
import json
# pyrefly: ignore [missing-import]
import psycopg
# pyrefly: ignore [missing-import]
from pgvector.psycopg import register_vector
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

class DatabaseIndexer:
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is missing.")

    def store_chunks(self, chunks_with_embeddings):
        """
        chunks_with_embeddings: list of dicts with 'content', 'embedding', and 'metadata'
        """
        print(f"Menyimpan {len(chunks_with_embeddings)} chunk ke database...")
        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)
            
            with conn.cursor() as cur:
                # Prepare data
                data_to_insert = [
                    (item["content"], item["embedding"], json.dumps(item["metadata"]))
                    for item in chunks_with_embeddings
                ]
                
                cur.executemany(
                    """
                    INSERT INTO laporan_mandiri_chunks (content, embedding, metadata)
                    VALUES (%s, %s::vector, %s)
                    """,
                    data_to_insert
                )
            conn.commit()
        print("Penyimpanan berhasil.")
