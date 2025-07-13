import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Configuration settings for the RAG system"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    
    # Ollama Configuration (for local LLM fallback)
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama2", env="OLLAMA_MODEL")
    
    # Vector Database Configuration
    chroma_db_path: str = Field("./chroma_db", env="CHROMA_DB_PATH")
    vector_db_type: str = Field("chroma", env="VECTOR_DB_TYPE")
    
    # RAG Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    retrieval_k: int = Field(5, env="RETRIEVAL_K")
    temperature: float = Field(0.7, env="TEMPERATURE")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(True, env="DEBUG")
    
    # Evaluation Configuration
    evaluation_dataset_path: str = Field("./data/evaluation/", env="EVALUATION_DATASET_PATH")
    metrics_log_path: str = Field("./logs/metrics.log", env="METRICS_LOG_PATH")
    
    # Intent Classification Configuration
    intent_confidence_threshold: float = Field(0.7, env="INTENT_CONFIDENCE_THRESHOLD")
    fallback_response_enabled: bool = Field(True, env="FALLBACK_RESPONSE_ENABLED")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./rag_database.db", env="DATABASE_URL")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/rag.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Intent categories
INTENT_CATEGORIES = {
    "technical": {
        "description": "Technical support queries about code, APIs, bugs, and implementation",
        "keywords": ["error", "bug", "code", "API", "implementation", "technical", "debug", "fix"],
        "prompt_template": """
        You are a technical support specialist. Use the following technical documentation and code examples to help solve the user's technical issue.
        
        Context: {context}
        
        Question: {question}
        
        Provide a detailed technical solution with code examples if applicable. If you cannot find the answer in the documentation, say "I don't have enough technical information to solve this issue."
        
        Answer:
        """
    },
    "billing": {
        "description": "Billing, account, and pricing related queries",
        "keywords": ["billing", "payment", "price", "cost", "account", "subscription", "invoice", "plan"],
        "prompt_template": """
        You are a billing support specialist. Use the following pricing information and account policies to help with the user's billing question.
        
        Context: {context}
        
        Question: {question}
        
        Provide clear information about billing, pricing, or account policies. If you cannot find the answer in the policies, say "I don't have enough billing information to answer this question."
        
        Answer:
        """
    },
    "features": {
        "description": "Feature requests and product roadmap queries",
        "keywords": ["feature", "request", "roadmap", "enhancement", "new", "future", "upcoming", "product"],
        "prompt_template": """
        You are a product specialist. Use the following roadmap and feature information to help with the user's feature-related question.
        
        Context: {context}
        
        Question: {question}
        
        Provide information about current features, planned features, or product roadmap. If you cannot find the answer in the roadmap, say "I don't have enough product information to answer this question."
        
        Answer:
        """
    }
}

# Data paths for different intents
DATA_PATHS = {
    "technical": "./data/technical/",
    "billing": "./data/billing/",
    "features": "./data/features/"
} 