import asyncio
import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import json

from pydantic_models import (
    QueryRequest, QueryResponse, StreamingResponse as StreamingResp,
    IngestionRequest, IngestionResponse, HealthCheck, MetricsResponse,
    LLMSwitchRequest, LLMSwitchResponse, TestCase, BatchEvaluationResult,
    IntentType, LLMProvider
)
from config import settings
from rag_engine import rag_engine
from data_ingestion import ingestion_pipeline
from evaluation import evaluator
from llm_wrapper import llm_wrapper
from chroma_utils import chroma_manager

# Global state for metrics tracking
metrics_store = {
    "total_queries": 0,
    "queries_by_intent": {"technical": 0, "billing": 0, "features": 0},
    "total_response_time": 0.0,
    "total_confidence": 0.0,
    "success_count": 0,
    "recent_metrics": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting RAG API server...")
    
    # Initialize components
    try:
        # Check system health
        health = await rag_engine.health_check()
        if not health.get("overall", False):
            logger.warning("Some system components are not healthy")
        
        logger.info("RAG API server started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down RAG API server...")

# Create FastAPI app
app = FastAPI(
    title="RAG Customer Support API",
    description="A comprehensive RAG system for customer support with intent detection and evaluation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for updating metrics
async def update_metrics(response: QueryResponse):
    """Update global metrics"""
    try:
        metrics_store["total_queries"] += 1
        metrics_store["queries_by_intent"][response.intent.intent.value] += 1
        metrics_store["total_response_time"] += response.response_time
        metrics_store["total_confidence"] += response.confidence
        
        if response.confidence > settings.intent_confidence_threshold:
            metrics_store["success_count"] += 1
        
        # Keep recent metrics (last 100)
        metrics_store["recent_metrics"].append({
            "timestamp": time.time(),
            "response_time": response.response_time,
            "confidence": response.confidence,
            "intent": response.intent.intent.value
        })
        
        if len(metrics_store["recent_metrics"]) > 100:
            metrics_store["recent_metrics"].pop(0)
            
    except Exception as e:
        logger.error(f"Metrics update error: {e}")

# Main RAG endpoints
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a RAG query"""
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        response = await rag_engine.process_query(request)
        
        # Update metrics in background
        background_tasks.add_task(update_metrics, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """Process a RAG query with streaming response"""
    try:
        logger.info(f"Processing streaming query: {request.query[:100]}...")
        
        async def generate_stream():
            async for chunk in rag_engine.process_streaming_query(request):
                yield f"data: {json.dumps(chunk.dict())}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data ingestion endpoints
@app.post("/ingest", response_model=IngestionResponse)
async def ingest_data(request: IngestionRequest):
    """Ingest data for a specific intent"""
    try:
        logger.info(f"Starting data ingestion for {request.intent_type.value}")
        
        response = await ingestion_pipeline.ingest_data(request)
        
        return response
        
    except Exception as e:
        logger.error(f"Data ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/all")
async def ingest_all_data():
    """Ingest data for all intents"""
    try:
        logger.info("Starting data ingestion for all intents")
        
        results = await ingestion_pipeline.ingest_all_intents()
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingest/stats")
async def get_ingestion_stats():
    """Get data ingestion statistics"""
    try:
        stats = ingestion_pipeline.get_ingestion_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Ingestion stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# LLM management endpoints
@app.post("/llm/switch", response_model=LLMSwitchResponse)
async def switch_llm_provider(request: LLMSwitchRequest):
    """Switch LLM provider"""
    try:
        success = await llm_wrapper.switch_provider(request.provider, request.model)
        
        return LLMSwitchResponse(
            success=success,
            current_provider=llm_wrapper.get_current_provider(),
            current_model=llm_wrapper.get_current_model(),
            message=f"Successfully switched to {request.provider}" if success else "Failed to switch provider"
        )
        
    except Exception as e:
        logger.error(f"LLM switch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm/status")
async def get_llm_status():
    """Get current LLM status"""
    try:
        health = await llm_wrapper.health_check()
        queue_status = llm_wrapper.get_queue_status()
        
        return {
            "current_provider": llm_wrapper.get_current_provider().value,
            "current_model": llm_wrapper.get_current_model(),
            "health": health,
            "queue_status": queue_status
        }
        
    except Exception as e:
        logger.error(f"LLM status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Evaluation endpoints
@app.post("/evaluate/single")
async def evaluate_single_query(test_case: TestCase):
    """Evaluate a single test case"""
    try:
        result = await evaluator.evaluate_single_query(test_case)
        return result
        
    except Exception as e:
        logger.error(f"Single evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/batch", response_model=BatchEvaluationResult)
async def evaluate_batch(test_cases: List[TestCase] = None):
    """Run batch evaluation"""
    try:
        logger.info("Starting batch evaluation")
        
        result = await evaluator.run_batch_evaluation(test_cases)
        
        # Save results
        timestamp = int(time.time())
        filepath = f"./evaluation_results/batch_eval_{timestamp}.json"
        evaluator.save_evaluation_results(result, filepath)
        
        return result
        
    except Exception as e:
        logger.error(f"Batch evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate/generate-tests")
async def generate_test_cases():
    """Generate test cases for evaluation"""
    try:
        test_cases = evaluator.test_generator.generate_test_cases()
        return {"test_cases": test_cases}
        
    except Exception as e:
        logger.error(f"Test generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate/report")
async def get_evaluation_report():
    """Get latest evaluation report"""
    try:
        # Run a quick evaluation
        result = await evaluator.run_batch_evaluation()
        report = evaluator.generate_evaluation_report(result)
        
        return {"report": report}
        
    except Exception as e:
        logger.error(f"Evaluation report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System monitoring endpoints
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """System health check"""
    try:
        health = await rag_engine.health_check()
        
        return HealthCheck(
            status="healthy" if health.get("overall", False) else "unhealthy",
            components=health
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheck(
            status="unhealthy",
            components={"error": str(e)}
        )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics"""
    try:
        total_queries = metrics_store["total_queries"]
        
        # Calculate averages
        avg_response_time = (
            metrics_store["total_response_time"] / total_queries 
            if total_queries > 0 else 0.0
        )
        
        avg_confidence = (
            metrics_store["total_confidence"] / total_queries 
            if total_queries > 0 else 0.0
        )
        
        success_rate = (
            metrics_store["success_count"] / total_queries 
            if total_queries > 0 else 0.0
        )
        
        return MetricsResponse(
            total_queries=total_queries,
            queries_by_intent=metrics_store["queries_by_intent"],
            average_response_time=avg_response_time,
            average_confidence=avg_confidence,
            success_rate=success_rate,
            recent_metrics=metrics_store["recent_metrics"][-10:]  # Last 10 metrics
        )
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        stats = rag_engine.get_system_stats()
        return stats
        
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Database management endpoints
@app.get("/database/stats")
async def get_database_stats():
    """Get vector database statistics"""
    try:
        stats = chroma_manager.get_database_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Database stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/reset")
async def reset_database():
    """Reset the vector database (use with caution)"""
    try:
        chroma_manager.reset_database()
        return {"message": "Database reset successfully"}
        
    except Exception as e:
        logger.error(f"Database reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Intent classification endpoints
@app.post("/intent/classify")
async def classify_intent(query: str):
    """Classify intent of a query"""
    try:
        from intent_classifier import intent_classifier
        
        result = intent_classifier.classify_intent(query)
        return result
        
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/intent/model-info")
async def get_intent_model_info():
    """Get intent classification model information"""
    try:
        from intent_classifier import intent_classifier
        
        info = intent_classifier.get_model_info()
        return info
        
    except Exception as e:
        logger.error(f"Intent model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Customer Support API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "stream": "/query/stream",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/version")
async def get_version():
    """Get API version"""
    return {"version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting RAG API server...")
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
