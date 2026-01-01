import os
import logging
from typing import List, Dict, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import config

logging.basicConfig(level=logging.INFO)

class NCERTRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        """Loads the FAISS index from disk."""
        if not os.path.exists(config.VECTOR_DB_DIR) or not os.listdir(config.VECTOR_DB_DIR):
            logging.warning(f"Vector DB not found at {config.VECTOR_DB_DIR}. Please run ingestion first.")
            return None
        
        try:
            return FAISS.load_local(config.VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error(f"Error loading Vector DB: {e}")
            return None

    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL, filters: Dict = None) -> List[Document]:
        """
        Retrieves relevant documents for a query.
        
        Args:
            query: The user's question.
            top_k: Number of documents to retrieve.
            filters: Optional metadata filters (e.g., {"subject": "Science"}). 
                     Note: standard FAISS doesn't support complex filtering easily without metadata wrappers,
                     so strictly speaking this might require a different vector store or post-filtering. 
                     For this implementation, we will use basic vector search and optional post-filtering if needed.
        """
        if not self.vector_store:
            logging.error("Vector Store is not initialized.")
            return []

        # basic search
        docs = self.vector_store.similarity_search(query, k=top_k)
        
        # If we had metadata filters, we would apply them here if the vector store supports it
        # or post-filter the results. For simplicity with basic FAISS, we return the top results.
        # Improvement: Fetch 2*top_k and filter manually if strictly needed.
        
        return docs

if __name__ == "__main__":
    retriever = NCERTRetriever()
    if retriever.vector_store:
        results = retriever.retrieve("What is photosynthesis?")
        for doc in results:
            print(f"Content: {doc.page_content[:100]}...")
            print(f"Source: {doc.metadata.get('source')}")
            print("-" * 20)
