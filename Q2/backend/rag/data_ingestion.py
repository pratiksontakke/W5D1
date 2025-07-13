import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader,
    CSVLoader, JSONLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from loguru import logger
from tqdm import tqdm
import pandas as pd
import json

from config import settings, DATA_PATHS
from pydantic_models import IntentType, IngestionRequest, IngestionResponse, DocumentChunk
from chroma_utils import ChromaManager

class DocumentProcessor:
    """Handles processing of different document types"""
    
    def __init__(self):
        self.supported_extensions = {
            '.txt': self._load_text,
            '.md': self._load_text,
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.doc': self._load_docx,
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.html': self._load_html,
            '.htm': self._load_html
        }
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load text files"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return []
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF files"""
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return []
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load Word documents"""
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading Word file {file_path}: {e}")
            return []
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV files"""
        try:
            loader = CSVLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return []
    
    def _load_json(self, file_path: str) -> List[Document]:
        """Load JSON files"""
        try:
            loader = JSONLoader(file_path, jq_schema='.', text_content=False)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return []
    
    def _load_html(self, file_path: str) -> List[Document]:
        """Load HTML files"""
        try:
            loader = UnstructuredHTMLLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading HTML file {file_path}: {e}")
            return []
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its extension"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {file_extension}")
            return []
        
        documents = self.supported_extensions[file_extension](file_path)
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                'source': file_path,
                'file_type': file_extension,
                'file_name': Path(file_path).name,
                'processed_at': time.time()
            })
        
        return documents
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory"""
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return documents
        
        # Get all files recursively
        all_files = []
        for ext in self.supported_extensions.keys():
            all_files.extend(directory.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(all_files)} files to process")
        
        # Process files with progress bar
        for file_path in tqdm(all_files, desc="Loading documents"):
            file_docs = self.load_document(str(file_path))
            documents.extend(file_docs)
        
        return documents

class TextChunker:
    """Handles text chunking with different strategies"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces"""
        logger.info(f"Chunking {len(documents)} documents")
        
        chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            try:
                doc_chunks = self.splitter.split_documents([doc])
                
                # Add chunk metadata
                for i, chunk in enumerate(doc_chunks):
                    chunk.metadata.update({
                        'chunk_id': f"{doc.metadata.get('source', 'unknown')}_{i}",
                        'chunk_index': i,
                        'total_chunks': len(doc_chunks),
                        'chunk_size': len(chunk.page_content)
                    })
                
                chunks.extend(doc_chunks)
                
            except Exception as e:
                logger.error(f"Error chunking document {doc.metadata.get('source', 'unknown')}: {e}")
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

class EmbeddingGenerator:
    """Generates embeddings for documents"""
    
    def __init__(self, embedding_type: str = "openai"):
        self.embedding_type = embedding_type
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if self.embedding_type == "openai":
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key
            )
        elif self.embedding_type == "sentence_transformer":
            return SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            if self.embedding_type == "openai":
                # OpenAI embeddings support batch processing
                embeddings = await self.embeddings.aembed_documents(texts)
                return embeddings
            else:
                # For other embeddings, use synchronous method
                embeddings = self.embeddings.embed_documents(texts)
                return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

class DataIngestionPipeline:
    """Main data ingestion pipeline"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.chroma_manager = ChromaManager()
    
    async def ingest_data(self, request: IngestionRequest) -> IngestionResponse:
        """Main ingestion pipeline"""
        start_time = time.time()
        errors = []
        
        try:
            logger.info(f"Starting ingestion for {request.intent_type} from {request.data_path}")
            
            # Step 1: Load documents
            documents = self.document_processor.load_directory(request.data_path)
            if not documents:
                return IngestionResponse(
                    success=False,
                    documents_processed=0,
                    chunks_created=0,
                    processing_time=time.time() - start_time,
                    errors=["No documents found or loaded"]
                )
            
            # Step 2: Chunk documents
            chunks = self.text_chunker.chunk_documents(documents)
            
            # Step 3: Add intent metadata
            for chunk in chunks:
                chunk.metadata['intent'] = request.intent_type.value
            
            # Step 4: Store in vector database
            collection_name = f"{request.intent_type.value}_documents"
            
            if request.overwrite:
                # Clear existing collection
                self.chroma_manager.delete_collection(collection_name)
            
            # Store chunks in Chroma
            await self._store_chunks_in_chroma(chunks, collection_name)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Ingestion completed: {len(documents)} documents, {len(chunks)} chunks in {processing_time:.2f}s")
            
            return IngestionResponse(
                success=True,
                documents_processed=len(documents),
                chunks_created=len(chunks),
                processing_time=processing_time,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Ingestion pipeline error: {e}")
            return IngestionResponse(
                success=False,
                documents_processed=0,
                chunks_created=0,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _store_chunks_in_chroma(self, chunks: List[Document], collection_name: str):
        """Store chunks in Chroma vector database"""
        try:
            # Prepare texts and metadata
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Create or get collection
            collection = self.chroma_manager.get_or_create_collection(collection_name)
            
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in tqdm(range(0, len(texts), batch_size), desc="Storing chunks"):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = [f"chunk_{i}_{j}" for j in range(len(batch_texts))]
                
                # Generate embeddings
                embeddings = await self.embedding_generator.generate_embeddings(batch_texts)
                
                # Add to collection
                collection.add(
                    embeddings=embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"Stored {len(chunks)} chunks in collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Error storing chunks in Chroma: {e}")
            raise
    
    async def ingest_all_intents(self) -> Dict[str, IngestionResponse]:
        """Ingest data for all intent types"""
        results = {}
        
        for intent_type in IntentType:
            data_path = DATA_PATHS.get(intent_type.value)
            if data_path and os.path.exists(data_path):
                request = IngestionRequest(
                    data_path=data_path,
                    intent_type=intent_type,
                    overwrite=True
                )
                
                result = await self.ingest_data(request)
                results[intent_type.value] = result
            else:
                logger.warning(f"Data path not found for {intent_type.value}: {data_path}")
                results[intent_type.value] = IngestionResponse(
                    success=False,
                    documents_processed=0,
                    chunks_created=0,
                    processing_time=0,
                    errors=[f"Data path not found: {data_path}"]
                )
        
        return results
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        stats = {}
        
        for intent_type in IntentType:
            collection_name = f"{intent_type.value}_documents"
            try:
                collection = self.chroma_manager.get_collection(collection_name)
                if collection:
                    count = collection.count()
                    stats[intent_type.value] = {
                        'document_count': count,
                        'collection_exists': True
                    }
                else:
                    stats[intent_type.value] = {
                        'document_count': 0,
                        'collection_exists': False
                    }
            except Exception as e:
                stats[intent_type.value] = {
                    'document_count': 0,
                    'collection_exists': False,
                    'error': str(e)
                }
        
        return stats

# Global ingestion pipeline instance
ingestion_pipeline = DataIngestionPipeline() 