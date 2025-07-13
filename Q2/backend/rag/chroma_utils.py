import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
from loguru import logger

from config import settings

class ChromaManager:
    """Manager for Chroma vector database operations"""
    
    def __init__(self):
        self.client = None
        self.persistent_path = settings.chroma_db_path
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Chroma client"""
        try:
            # Ensure the directory exists
            os.makedirs(self.persistent_path, exist_ok=True)
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persistent_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            logger.info(f"Chroma client initialized with path: {self.persistent_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise
    
    def get_or_create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Get or create a collection"""
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {}
            )
            logger.info(f"Collection '{name}' ready")
            return collection
        except Exception as e:
            logger.error(f"Error getting/creating collection '{name}': {e}")
            raise
    
    def get_collection(self, name: str):
        """Get an existing collection"""
        try:
            return self.client.get_collection(name)
        except Exception as e:
            logger.error(f"Error getting collection '{name}': {e}")
            return None
    
    def delete_collection(self, name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(name)
            logger.info(f"Collection '{name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting collection '{name}': {e}")
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            collection = self.get_collection(name)
            if collection:
                return {
                    'name': collection.name,
                    'count': collection.count(),
                    'metadata': collection.metadata
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting collection info for '{name}': {e}")
            return {}
    
    def query_collection(self, collection_name: str, query_texts: List[str], 
                        n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        """Query a collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return {"error": f"Collection '{collection_name}' not found"}
            
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )
            
            return results
        except Exception as e:
            logger.error(f"Error querying collection '{collection_name}': {e}")
            return {"error": str(e)}
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     embeddings: List[List[float]], metadatas: List[Dict],
                     ids: List[str]):
        """Add documents to a collection"""
        try:
            collection = self.get_or_create_collection(collection_name)
            
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            raise
    
    def update_documents(self, collection_name: str, documents: List[str],
                        embeddings: List[List[float]], metadatas: List[Dict],
                        ids: List[str]):
        """Update documents in a collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            collection.update(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Updated {len(documents)} documents in collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error updating documents in collection '{collection_name}': {e}")
            raise
    
    def delete_documents(self, collection_name: str, ids: List[str]):
        """Delete documents from a collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            collection.delete(ids=ids)
            
            logger.info(f"Deleted {len(ids)} documents from collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error deleting documents from collection '{collection_name}': {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            collections = self.list_collections()
            stats = {
                'total_collections': len(collections),
                'collections': {}
            }
            
            total_documents = 0
            for collection_name in collections:
                info = self.get_collection_info(collection_name)
                stats['collections'][collection_name] = info
                total_documents += info.get('count', 0)
            
            stats['total_documents'] = total_documents
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if Chroma database is healthy"""
        try:
            # Try to list collections
            collections = self.client.list_collections()
            return True
        except Exception as e:
            logger.error(f"Chroma health check failed: {e}")
            return False
    
    def reset_database(self):
        """Reset the entire database (use with caution)"""
        try:
            collections = self.list_collections()
            for collection_name in collections:
                self.delete_collection(collection_name)
            
            logger.info("Database reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            raise

# Global Chroma manager instance
chroma_manager = ChromaManager()
