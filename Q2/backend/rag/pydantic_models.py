from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class IntentType(str, Enum):
    """Intent categories for query classification"""
    TECHNICAL = "technical"
    BILLING = "billing"
    FEATURES = "features"

class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str = Field(..., description="User query text", min_length=1)
    max_sources: int = Field(5, description="Maximum number of sources to return", ge=1, le=10)
    intent: Optional[IntentType] = Field(None, description="Optional intent override")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")

class SourceDocument(BaseModel):
    """Model for source document metadata"""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: Optional[float] = Field(None, description="Relevance score")
    
class IntentClassification(BaseModel):
    """Model for intent classification results"""
    intent: IntentType = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Classification confidence", ge=0.0, le=1.0)
    scores: Dict[str, float] = Field(..., description="Scores for all intents")

class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    intent: IntentClassification = Field(..., description="Intent classification")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(..., description="Overall response confidence", ge=0.0, le=1.0)
    response_time: float = Field(..., description="Response time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
class StreamingResponse(BaseModel):
    """Model for streaming response chunks"""
    chunk: str = Field(..., description="Response chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class EvaluationMetrics(BaseModel):
    """Model for evaluation metrics"""
    faithfulness: float = Field(..., description="Faithfulness score", ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., description="Answer relevancy score", ge=0.0, le=1.0)
    context_precision: float = Field(..., description="Context precision score", ge=0.0, le=1.0)
    context_recall: float = Field(..., description="Context recall score", ge=0.0, le=1.0)
    response_time: float = Field(..., description="Response time in seconds")
    token_usage: int = Field(..., description="Token usage count")

class TestCase(BaseModel):
    """Model for test cases"""
    question: str = Field(..., description="Test question")
    expected_intent: IntentType = Field(..., description="Expected intent")
    ground_truth: Optional[str] = Field(None, description="Ground truth answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Test case metadata")

class EvaluationResult(BaseModel):
    """Model for evaluation results"""
    test_case: TestCase = Field(..., description="Test case")
    response: QueryResponse = Field(..., description="System response")
    metrics: EvaluationMetrics = Field(..., description="Evaluation metrics")
    passed: bool = Field(..., description="Whether test passed")

class BatchEvaluationResult(BaseModel):
    """Model for batch evaluation results"""
    total_cases: int = Field(..., description="Total number of test cases")
    passed_cases: int = Field(..., description="Number of passed cases")
    failed_cases: int = Field(..., description="Number of failed cases")
    overall_accuracy: float = Field(..., description="Overall accuracy", ge=0.0, le=1.0)
    intent_accuracy: Dict[str, float] = Field(..., description="Accuracy per intent")
    average_metrics: EvaluationMetrics = Field(..., description="Average metrics")
    results: List[EvaluationResult] = Field(..., description="Individual results")

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(None, description="Chunk embedding")
    
class IngestionRequest(BaseModel):
    """Request model for data ingestion"""
    data_path: str = Field(..., description="Path to data directory")
    intent_type: IntentType = Field(..., description="Intent type for the data")
    overwrite: bool = Field(False, description="Whether to overwrite existing data")

class IngestionResponse(BaseModel):
    """Response model for data ingestion"""
    success: bool = Field(..., description="Whether ingestion was successful")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")

class HealthCheck(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field("1.0.0", description="API version")
    components: Dict[str, bool] = Field(default_factory=dict, description="Component health status")

class MetricsResponse(BaseModel):
    """Model for metrics dashboard response"""
    total_queries: int = Field(..., description="Total queries processed")
    queries_by_intent: Dict[str, int] = Field(..., description="Queries by intent")
    average_response_time: float = Field(..., description="Average response time")
    average_confidence: float = Field(..., description="Average confidence score")
    success_rate: float = Field(..., description="Success rate")
    recent_metrics: List[EvaluationMetrics] = Field(..., description="Recent metrics")

class LLMProvider(str, Enum):
    """LLM provider options"""
    OPENAI = "openai"
    OLLAMA = "ollama"

class LLMSwitchRequest(BaseModel):
    """Request to switch LLM provider"""
    provider: LLMProvider = Field(..., description="LLM provider to switch to")
    model: Optional[str] = Field(None, description="Optional model name")

class LLMSwitchResponse(BaseModel):
    """Response for LLM provider switch"""
    success: bool = Field(..., description="Whether switch was successful")
    current_provider: LLMProvider = Field(..., description="Current active provider")
    current_model: str = Field(..., description="Current active model")
    message: str = Field(..., description="Status message")
