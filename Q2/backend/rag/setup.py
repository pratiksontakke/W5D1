#!/usr/bin/env python3
"""
Setup script for RAG Customer Support System
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
from loguru import logger

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        logger.error("Python 3.9 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/technical",
        "data/billing", 
        "data/features",
        "logs",
        "models",
        "evaluation_results",
        "chroma_db",
        "dashboard_outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    
    if not env_file.exists():
        env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Ollama Configuration (for local LLM fallback)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Vector Database Configuration
CHROMA_DB_PATH=./chroma_db
VECTOR_DB_TYPE=chroma

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5
TEMPERATURE=0.7

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Evaluation Configuration
EVALUATION_DATASET_PATH=./data/evaluation/
METRICS_LOG_PATH=./logs/metrics.log

# Intent Classification Configuration
INTENT_CONFIDENCE_THRESHOLD=0.7
FALLBACK_RESPONSE_ENABLED=True

# Database Configuration
DATABASE_URL=sqlite:///./rag_database.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/rag.log
"""
        
        with open(env_file, "w") as f:
            f.write(env_content)
        
        logger.info("Created .env file - please update with your API keys")
    else:
        logger.info(".env file already exists")

def check_openai_key():
    """Check if OpenAI API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        logger.warning("OpenAI API key not configured. Please update .env file")
        return False
    
    logger.info("OpenAI API key configured")
    return True

async def test_system():
    """Test system components"""
    logger.info("Testing system components...")
    
    try:
        # Test imports
        from config import settings
        from llm_wrapper import llm_wrapper
        from intent_classifier import intent_classifier
        from chroma_utils import chroma_manager
        from rag_engine import rag_engine
        
        logger.info("All imports successful")
        
        # Test database connection
        if chroma_manager.health_check():
            logger.info("ChromaDB connection successful")
        else:
            logger.warning("ChromaDB connection failed")
        
        # Test LLM connection (if API key is configured)
        if check_openai_key():
            try:
                health = await llm_wrapper.health_check()
                if health.get("openai", False):
                    logger.info("OpenAI connection successful")
                else:
                    logger.warning("OpenAI connection failed")
            except Exception as e:
                logger.warning(f"OpenAI connection test failed: {e}")
        
        logger.info("System test completed")
        
    except Exception as e:
        logger.error(f"System test failed: {e}")
        return False
    
    return True

async def ingest_sample_data():
    """Ingest sample data"""
    logger.info("Ingesting sample data...")
    
    try:
        from data_ingestion import ingestion_pipeline
        from pydantic_models import IngestionRequest, IntentType
        
        # Ingest data for each intent
        for intent_type in IntentType:
            data_path = f"./data/{intent_type.value}"
            
            if Path(data_path).exists() and any(Path(data_path).iterdir()):
                request = IngestionRequest(
                    data_path=data_path,
                    intent_type=intent_type,
                    overwrite=True
                )
                
                result = await ingestion_pipeline.ingest_data(request)
                
                if result.success:
                    logger.info(f"Ingested {result.documents_processed} documents for {intent_type.value}")
                else:
                    logger.error(f"Failed to ingest data for {intent_type.value}: {result.errors}")
            else:
                logger.warning(f"No data found for {intent_type.value}")
        
        logger.info("Sample data ingestion completed")
        
    except Exception as e:
        logger.error(f"Sample data ingestion failed: {e}")
        return False
    
    return True

async def run_quick_evaluation():
    """Run a quick evaluation"""
    logger.info("Running quick evaluation...")
    
    try:
        from evaluation import evaluator
        
        # Generate test cases
        test_cases = evaluator.test_generator.generate_test_cases()[:6]  # Quick test
        
        # Run evaluation
        results = await evaluator.run_batch_evaluation(test_cases)
        
        logger.info(f"Quick evaluation completed:")
        logger.info(f"  - Total cases: {results.total_cases}")
        logger.info(f"  - Passed cases: {results.passed_cases}")
        logger.info(f"  - Overall accuracy: {results.overall_accuracy:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick evaluation failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 RAG Customer Support System Setup Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Update your .env file with your OpenAI API key")
    print("2. Start the API server:")
    print("   python main.py")
    print("\n3. Test the system:")
    print("   curl -X POST 'http://localhost:8000/query' \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"query\": \"How do I fix API errors?\"}'")
    print("\n4. Access the API documentation:")
    print("   http://localhost:8000/docs")
    print("\n5. Run evaluation:")
    print("   curl -X POST 'http://localhost:8000/evaluate/batch'")
    print("\n6. Generate dashboard:")
    print("   python dashboard.py run")
    print("\nFor more information, see the README.md file")
    print("="*60)

async def main():
    """Main setup function"""
    logger.info("Starting RAG Customer Support System setup...")
    
    # Basic checks
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Create environment file
    create_env_file()
    
    # Test system
    system_ok = await test_system()
    
    if system_ok:
        # Ingest sample data
        if check_openai_key():
            await ingest_sample_data()
            await run_quick_evaluation()
        else:
            logger.warning("Skipping data ingestion - OpenAI API key not configured")
    
    # Print next steps
    print_next_steps()
    
    logger.info("Setup completed!")

if __name__ == "__main__":
    asyncio.run(main()) 