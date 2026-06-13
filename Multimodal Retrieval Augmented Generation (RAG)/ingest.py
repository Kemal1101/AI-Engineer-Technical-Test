import sys
import argparse
import os
from ingestion.parser import DocumentParser
from ingestion.chunker import DocumentChunker
from ingestion.indexer import DatabaseIndexer

def run_ingestion(pdf_path):
    print(f"Mulai proses ingestion untuk: {pdf_path}")
    
    # 1. Parsing
    parser = DocumentParser()
    parsed_data = parser.parse_document(pdf_path)
    print(f"Berhasil mengekstrak {len(parsed_data)} elemen dari dokumen.")

    # 2. Chunking
    chunker = DocumentChunker()
    chunks = chunker.create_chunks(parsed_data)
    print(f"Berhasil membuat {len(chunks)} semantic chunks.")
    
    # Generate embeddings
    print("Mendapatkan vector embeddings dari HuggingFace...")
    page_contents = [chunk.page_content for chunk in chunks]
    embeddings = chunker.embeddings.embed_documents(page_contents)
    
    # Prepare data for Database
    chunks_with_embeddings = []
    source_name = os.path.basename(pdf_path)
    for i, chunk in enumerate(chunks):
        chunks_with_embeddings.append({
            "content": chunk.page_content,
            "embedding": embeddings[i],
            "metadata": {
                "page_number": chunk.metadata.get("page", 0),
                "source": source_name,
                "type": "semantic_chunk"
            }
        })
        
    # 3. Indexing (Store to DB)
    indexer = DatabaseIndexer()
    indexer.store_chunks(chunks_with_embeddings)
    print("Ingestion selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF into Multimodal RAG Database")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()
    
    run_ingestion(args.pdf_path)
