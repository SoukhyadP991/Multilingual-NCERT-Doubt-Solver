import logging
import time
from typing import Dict, Any
from src.retrieval import NCERTRetriever
from src.generation import LocalLLMGenerator
from src.utils import detect_language

logging.basicConfig(level=logging.INFO)

class RAGPipeline:
    def __init__(self):
        self.retriever = NCERTRetriever()
        self.generator = LocalLLMGenerator()

    def process_query(self, query: str, filters: Dict = None) -> Dict[str, Any]:
        """
        End-to-end processing of a user query.
        Returns a dictionary with the answer, context, and metadata.
        """
        start_time = time.time()
        
        # 1. Detect Language
        lang = detect_language(query)
        logging.info(f"Detected language: {lang}")

        # 2. Retrieve
        # We can append language instruction to query if needed, but for now raw query is better for embeddings
        retrieved_docs = self.retriever.retrieve(query, filters=filters)
        
        if not retrieved_docs:
            return {
                "answer": "I don't know based on NCERT textbooks. (No relevant content found)",
                "source_documents": [],
                "language": lang,
                "latency": time.time() - start_time
            }

        # 3. Generate Answer
        # Add language instruction to the prompt context implicitly via system prompt or here
        # We might want to wrap the generator call to enforce output language
        answer = self.generator.generate_answer(query, retrieved_docs)

        # 4. Post-processing (optional language check)
        
        latency = time.time() - start_time
        
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "language": lang,
            "latency": latency
        }

if __name__ == "__main__":
    pipeline = RAGPipeline()
    # Mock run only if model exists
    if pipeline.generator.llm and pipeline.retriever.vector_store:
        result = pipeline.process_query("What is the capital of India?") # Expect "I don't know" or similar if no PDF
        print(result)
