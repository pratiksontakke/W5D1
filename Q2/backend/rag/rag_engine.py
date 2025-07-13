import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from loguru import logger
import numpy as np

from pydantic_models import (
    QueryRequest, QueryResponse, IntentType, IntentClassification,
    SourceDocument, StreamingResponse
)
from config import settings, INTENT_CATEGORIES
from llm_wrapper import llm_wrapper
from intent_classifier import intent_classifier
from chroma_utils import chroma_manager
from data_ingestion import EmbeddingGenerator

class RAGRetriever:
    """Handles document retrieval from vector database"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.chroma_manager = chroma_manager
    
    async def retrieve_documents(self, query: str, intent: IntentType, 
                               max_results: int = 5) -> List[SourceDocument]:
        """Retrieve relevant documents based on query and intent"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embeddings([query])
            
            # Get collection name based on intent
            collection_name = f"{intent.value}_documents"
            
            # Query the collection
            results = self.chroma_manager.query_collection(
                collection_name=collection_name,
                query_texts=[query],
                n_results=max_results
            )
            
            if "error" in results:
                logger.error(f"Retrieval error: {results['error']}")
                return []
            
            # Convert results to SourceDocument objects
            source_docs = []
            if results.get('documents') and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 0.0
                    
                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity_score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                    
                    source_docs.append(SourceDocument(
                        content=doc,
                        metadata=metadata,
                        score=similarity_score
                    ))
            
            logger.info(f"Retrieved {len(source_docs)} documents for intent {intent.value}")
            return source_docs
            
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return []
    
    def calculate_context_utilization(self, query: str, context: str) -> float:
        """Calculate how well the context matches the query"""
        try:
            # Simple word overlap calculation
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words.intersection(context_words))
            utilization = overlap / len(query_words)
            
            return min(utilization, 1.0)
            
        except Exception as e:
            logger.error(f"Context utilization calculation error: {e}")
            return 0.0

class RAGGenerator:
    """Handles response generation using LLM"""
    
    def __init__(self):
        self.llm = llm_wrapper
    
    async def generate_response(self, query: str, context: str, intent: IntentType) -> str:
        """Generate response using context and intent-specific prompt"""
        try:
            # Get intent-specific prompt template
            prompt_template = INTENT_CATEGORIES[intent.value]["prompt_template"]
            
            # Format the prompt
            formatted_prompt = prompt_template.format(
                context=context,
                question=query
            )
            
            # Generate response
            response = await self.llm.generate(formatted_prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    async def generate_streaming_response(self, query: str, context: str, 
                                        intent: IntentType) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        try:
            # Get intent-specific prompt template
            prompt_template = INTENT_CATEGORIES[intent.value]["prompt_template"]
            
            # Format the prompt
            formatted_prompt = prompt_template.format(
                context=context,
                question=query
            )
            
            # Generate streaming response
            async for chunk in self.llm.generate_stream(formatted_prompt):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming response generation error: {e}")
            yield "I apologize, but I'm having trouble generating a response right now."

class ConfidenceCalculator:
    """Calculates confidence scores for responses"""
    
    def calculate_response_confidence(self, query: str, response: str, 
                                    sources: List[SourceDocument],
                                    intent_confidence: float) -> float:
        """Calculate overall response confidence"""
        try:
            # Factor 1: Intent classification confidence (30%)
            intent_score = intent_confidence * 0.3
            
            # Factor 2: Source relevance scores (40%)
            if sources:
                avg_source_score = sum(doc.score for doc in sources) / len(sources)
                source_score = avg_source_score * 0.4
            else:
                source_score = 0.0
            
            # Factor 3: Response quality indicators (30%)
            response_score = self._calculate_response_quality(query, response) * 0.3
            
            # Combine scores
            total_confidence = intent_score + source_score + response_score
            
            return min(max(total_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _calculate_response_quality(self, query: str, response: str) -> float:
        """Calculate response quality based on various factors"""
        try:
            score = 0.0
            
            # Check response length (not too short, not too long)
            if 50 <= len(response) <= 1000:
                score += 0.3
            elif 20 <= len(response) <= 2000:
                score += 0.2
            
            # Check for fallback responses
            fallback_phrases = [
                "I don't have enough information",
                "I'm having trouble",
                "I apologize",
                "I cannot find"
            ]
            
            if not any(phrase in response for phrase in fallback_phrases):
                score += 0.4
            
            # Check for query terms in response
            query_terms = set(query.lower().split())
            response_terms = set(response.lower().split())
            
            if query_terms.intersection(response_terms):
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Response quality calculation error: {e}")
            return 0.5

class RAGEngine:
    """Main RAG engine that orchestrates the entire pipeline"""
    
    def __init__(self):
        self.retriever = RAGRetriever()
        self.generator = RAGGenerator()
        self.confidence_calculator = ConfidenceCalculator()
        self.intent_classifier = intent_classifier
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query through the complete RAG pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Intent classification
            if request.intent:
                # Use provided intent
                intent_classification = IntentClassification(
                    intent=request.intent,
                    confidence=1.0,
                    scores={request.intent.value: 1.0}
                )
            else:
                # Classify intent
                intent_classification = self.intent_classifier.classify_intent(request.query)
            
            # Step 2: Document retrieval
            source_docs = await self.retriever.retrieve_documents(
                query=request.query,
                intent=intent_classification.intent,
                max_results=request.max_sources
            )
            
            # Step 3: Prepare context
            context = self._prepare_context(source_docs)
            
            # Step 4: Generate response
            if context.strip():
                answer = await self.generator.generate_response(
                    query=request.query,
                    context=context,
                    intent=intent_classification.intent
                )
            else:
                answer = self._generate_fallback_response(intent_classification.intent)
            
            # Step 5: Calculate confidence
            confidence = self.confidence_calculator.calculate_response_confidence(
                query=request.query,
                response=answer,
                sources=source_docs,
                intent_confidence=intent_classification.confidence
            )
            
            # Step 6: Apply fallback if confidence is too low
            if confidence < settings.intent_confidence_threshold and settings.fallback_response_enabled:
                answer = self._generate_fallback_response(intent_classification.intent)
                confidence = 0.5
            
            processing_time = time.time() - start_time
            
            return QueryResponse(
                question=request.query,
                answer=answer,
                intent=intent_classification,
                sources=source_docs,
                confidence=confidence,
                response_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"RAG engine error: {e}")
            processing_time = time.time() - start_time
            
            return QueryResponse(
                question=request.query,
                answer="I apologize, but I'm experiencing technical difficulties. Please try again later.",
                intent=IntentClassification(
                    intent=IntentType.TECHNICAL,
                    confidence=0.1,
                    scores={"technical": 0.1, "billing": 0.1, "features": 0.1}
                ),
                sources=[],
                confidence=0.1,
                response_time=processing_time
            )
    
    async def process_streaming_query(self, request: QueryRequest) -> AsyncGenerator[StreamingResponse, None]:
        """Process a query with streaming response"""
        try:
            # Step 1: Intent classification
            if request.intent:
                intent_classification = IntentClassification(
                    intent=request.intent,
                    confidence=1.0,
                    scores={request.intent.value: 1.0}
                )
            else:
                intent_classification = self.intent_classifier.classify_intent(request.query)
            
            # Step 2: Document retrieval
            source_docs = await self.retriever.retrieve_documents(
                query=request.query,
                intent=intent_classification.intent,
                max_results=request.max_sources
            )
            
            # Step 3: Prepare context
            context = self._prepare_context(source_docs)
            
            # Step 4: Generate streaming response
            if context.strip():
                async for chunk in self.generator.generate_streaming_response(
                    query=request.query,
                    context=context,
                    intent=intent_classification.intent
                ):
                    yield StreamingResponse(
                        chunk=chunk,
                        is_final=False,
                        metadata={
                            "intent": intent_classification.intent.value,
                            "confidence": intent_classification.confidence
                        }
                    )
            else:
                fallback_response = self._generate_fallback_response(intent_classification.intent)
                yield StreamingResponse(
                    chunk=fallback_response,
                    is_final=True,
                    metadata={
                        "intent": intent_classification.intent.value,
                        "confidence": 0.5,
                        "fallback": True
                    }
                )
                return
            
            # Final chunk
            yield StreamingResponse(
                chunk="",
                is_final=True,
                metadata={
                    "intent": intent_classification.intent.value,
                    "confidence": intent_classification.confidence,
                    "sources": len(source_docs)
                }
            )
            
        except Exception as e:
            logger.error(f"Streaming RAG engine error: {e}")
            yield StreamingResponse(
                chunk="I apologize, but I'm experiencing technical difficulties.",
                is_final=True,
                metadata={"error": str(e)}
            )
    
    def _prepare_context(self, source_docs: List[SourceDocument]) -> str:
        """Prepare context from source documents"""
        if not source_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(source_docs):
            context_parts.append(f"Source {i+1}: {doc.content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_fallback_response(self, intent: IntentType) -> str:
        """Generate fallback response based on intent"""
        fallback_responses = {
            IntentType.TECHNICAL: "I don't have enough technical information to solve this issue. Please check our documentation or contact technical support.",
            IntentType.BILLING: "I don't have enough billing information to answer this question. Please contact our billing support team.",
            IntentType.FEATURES: "I don't have enough product information to answer this question. Please check our roadmap or contact our product team."
        }
        
        return fallback_responses.get(intent, "I don't have enough information to answer this question.")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all RAG components"""
        health_status = {}
        
        try:
            # Check LLM health
            llm_health = await self.generator.llm.health_check()
            health_status["llm"] = any(llm_health.values()) if isinstance(llm_health, dict) else llm_health
            
            # Check vector database health
            health_status["vector_db"] = self.retriever.chroma_manager.health_check()
            
            # Check intent classifier
            health_status["intent_classifier"] = self.intent_classifier.pipeline is not None
            
            # Overall health
            health_status["overall"] = all(health_status.values())
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_status["overall"] = False
        
        return health_status
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            stats = {
                "vector_db": self.retriever.chroma_manager.get_database_stats(),
                "llm": {
                    "current_provider": self.generator.llm.get_current_provider().value,
                    "current_model": self.generator.llm.get_current_model(),
                    "queue_status": self.generator.llm.get_queue_status()
                },
                "intent_classifier": self.intent_classifier.get_model_info()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"System stats error: {e}")
            return {}

# Global RAG engine instance
rag_engine = RAGEngine() 