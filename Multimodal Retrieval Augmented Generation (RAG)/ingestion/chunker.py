# pyrefly: ignore [missing-import]
from langchain_experimental.text_splitter import SemanticChunker
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

class DocumentChunker:
    def __init__(self):
        # We use HuggingFace local embeddings (768 dimensions)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.text_splitter = SemanticChunker(self.embeddings)

    def create_chunks(self, parsed_data):
        """
        Takes the structured parsed data (text, table, image_description)
        Groups them by page, then applies Semantic Chunker.
        """
        # Group by page
        pages_dict = {}
        for item in parsed_data:
            page_num = item["page"]
            content = item["content"]
            
            if page_num not in pages_dict:
                pages_dict[page_num] = []
            
            pages_dict[page_num].append(content)
            
        docs = []
        for page_num, contents in pages_dict.items():
            page_text = "\n\n".join(contents)
            docs.append(Document(page_content=page_text, metadata={"page": page_num}))
            
        # Perform semantic chunking
        print("Membagi dokumen menggunakan SemanticChunker...")
        chunks = self.text_splitter.split_documents(docs)
        
        return chunks
