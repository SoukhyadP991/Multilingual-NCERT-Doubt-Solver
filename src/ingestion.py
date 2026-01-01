import os
import glob
import logging
from typing import List, Dict, Optional
import pytesseract
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IngestionPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdfs(self, directory: str) -> List[Document]:
        """Loads all PDFs from a directory."""
        pdf_files = glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
        documents = []
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {directory}")
            return []

        for pdf_path in pdf_files:
            logging.info(f"Processing: {pdf_path}")
            try:
                # Try standard Pypdf loading first
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # Check if text extraction was successful (not empty)
                # If extraction is poor (scanned PDF), use OCR
                if not docs or len(docs[0].page_content.strip()) < 10:
                    logging.info(f"Standard extraction failed for {pdf_path}. Switching to OCR.")
                    docs = self.ocr_pdf(pdf_path)
                
                # Enrich metadata
                filename = os.path.basename(pdf_path)
                for doc in docs:
                    doc.metadata["source"] = filename
                    # Simplistic extraction of metadata from filename if possible, 
                    # e.g., "Grade10_Science_Ch1.pdf"
                    # For now, we keep it generic or rely on filename
                
                documents.extend(docs)
                
            except Exception as e:
                logging.error(f"Error processing {pdf_path}: {e}")
        
        return documents

    def ocr_pdf(self, pdf_path: str) -> List[Document]:
        """Performs OCR on a PDF file."""
        images = convert_from_path(pdf_path)
        docs = []
        
        try:
            for i, image in enumerate(images):
                # Tesseract OCR
                text = pytesseract.image_to_string(image, lang='eng+hin') # English + Hindi
                
                # Create Document object
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "page": i + 1,
                    "is_ocr": True
                }
                docs.append(Document(page_content=text, metadata=metadata))
        except pytesseract.TesseractNotFoundError:
            logging.error(f"Tesseract OCR not found. Please install Tesseract and add to PATH to process scanned PDF: {pdf_path}")
        except Exception as e:
            logging.error(f"OCR failed for {pdf_path}: {e}")
            
        return docs

    def clean_text(self, text: str) -> str:
        """Basic text normalization."""
        # Add specific cleaning rules here if needed
        return text.strip()

    def create_vector_db(self, documents: List[Document]):
        """Chunks documents and creates a FAISS index."""
        if not documents:
            logging.info("No documents to index.")
            return

        logging.info(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logging.info(f"Created {len(chunks)} chunks.")

        if not chunks:
            logging.warning("No chunks created.")
            return

        logging.info("Creating FAISS index...")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save locally
        save_path = config.VECTOR_DB_DIR
        vector_store.save_local(save_path)
        logging.info(f"Vector DB saved to {save_path}")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)

    pipeline = IngestionPipeline()
    logging.info("Starting ingestion process...")
    
    docs = pipeline.load_pdfs(config.RAW_DATA_DIR)
    if docs:
        pipeline.create_vector_db(docs)
    else:
        logging.info(f"Please place PDF files in {config.RAW_DATA_DIR}")
