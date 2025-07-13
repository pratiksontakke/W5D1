import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import pandas as pd
from datetime import datetime

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning("RAGAS not available, using fallback metrics")
    RAGAS_AVAILABLE = False

from pydantic_models import (
    TestCase, EvaluationResult, BatchEvaluationResult, EvaluationMetrics,
    IntentType, QueryRequest
)
from config import settings
from rag_engine import rag_engine
from data_ingestion import EmbeddingGenerator

class MetricsCalculator:
    """Calculates various evaluation metrics"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
    
    async def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            embeddings = await self.embedding_generator.generate_embeddings([text1, text2])
            
            if len(embeddings) != 2:
                return 0.0
            
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation error: {e}")
            return 0.0
    
    def calculate_token_usage(self, text: str) -> int:
        """Estimate token usage (approximate)"""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def calculate_context_utilization(self, query: str, context: str, answer: str) -> float:
        """Calculate how well the context was utilized in the answer"""
        try:
            if not context or not answer:
                return 0.0
            
            # Split into words
            context_words = set(context.lower().split())
            answer_words = set(answer.lower().split())
            query_words = set(query.lower().split())
            
            # Calculate overlap between context and answer
            context_answer_overlap = len(context_words.intersection(answer_words))
            
            # Calculate relevance to query
            query_context_overlap = len(query_words.intersection(context_words))
            
            # Combine metrics
            if len(context_words) > 0:
                utilization = (context_answer_overlap / len(context_words)) * 0.7
                if len(query_words) > 0:
                    utilization += (query_context_overlap / len(query_words)) * 0.3
            else:
                utilization = 0.0
            
            return min(utilization, 1.0)
            
        except Exception as e:
            logger.error(f"Context utilization calculation error: {e}")
            return 0.0
    
    async def calculate_answer_relevancy(self, query: str, answer: str) -> float:
        """Calculate answer relevancy using cosine similarity"""
        return await self.calculate_cosine_similarity(query, answer)
    
    def calculate_faithfulness(self, context: str, answer: str) -> float:
        """Calculate faithfulness (how well answer is supported by context)"""
        try:
            if not context or not answer:
                return 0.0
            
            # Simple keyword-based faithfulness
            context_words = set(context.lower().split())
            answer_words = set(answer.lower().split())
            
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            context_words -= common_words
            answer_words -= common_words
            
            if not answer_words:
                return 0.0
            
            # Calculate how many answer words are supported by context
            supported_words = len(answer_words.intersection(context_words))
            faithfulness = supported_words / len(answer_words)
            
            return min(faithfulness, 1.0)
            
        except Exception as e:
            logger.error(f"Faithfulness calculation error: {e}")
            return 0.0

class TestDataGenerator:
    """Generates test data for evaluation"""
    
    def __init__(self):
        self.test_cases = []
    
    def generate_test_cases(self) -> List[TestCase]:
        """Generate comprehensive test cases for all intents"""
        test_cases = []
        
        # Technical support test cases
        technical_cases = [
            {
                "question": "I'm getting a 500 error when calling the API",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "A 500 error indicates an internal server error. Check your API request format and authentication."
            },
            {
                "question": "How do I implement authentication in my application?",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "You can implement authentication using API keys or OAuth tokens in the request headers."
            },
            {
                "question": "The SDK is not working properly on my system",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Please check the SDK version compatibility and system requirements."
            },
            {
                "question": "I need help debugging this integration issue",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Please provide error logs and code snippets for debugging assistance."
            },
            {
                "question": "The API response format is different than documented",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Please check the API version and ensure you're using the correct endpoint."
            },
            {
                "question": "My webhook is not receiving events",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Check your webhook URL configuration and ensure it's accessible from our servers."
            },
            {
                "question": "How to handle rate limiting in my application?",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Implement exponential backoff and respect the rate limit headers in API responses."
            },
            {
                "question": "The SSL certificate validation is failing",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Ensure your SSL certificate is valid and properly configured."
            },
            {
                "question": "I'm getting timeout errors with large requests",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Consider breaking large requests into smaller batches or increasing timeout values."
            },
            {
                "question": "The API documentation is unclear about this endpoint",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Please refer to the specific endpoint documentation or contact support for clarification."
            },
            {
                "question": "How to properly handle API errors in production?",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Implement proper error handling with retry logic and user-friendly error messages."
            },
            {
                "question": "The database connection is failing intermittently",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Check your database connection pool settings and network stability."
            },
            {
                "question": "I need help with the JavaScript SDK integration",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Follow the JavaScript SDK documentation and ensure proper initialization."
            },
            {
                "question": "The API is returning malformed JSON responses",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Check the API endpoint and request parameters for proper formatting."
            },
            {
                "question": "How to implement proper error logging?",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Use structured logging with appropriate log levels and error context."
            },
            {
                "question": "The service is experiencing high latency",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Check network connectivity and consider implementing caching strategies."
            },
            {
                "question": "I need help with API versioning",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Use version headers or URL versioning to maintain API compatibility."
            },
            {
                "question": "The authentication token keeps expiring",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Implement token refresh logic and check token expiration times."
            },
            {
                "question": "How to optimize API performance?",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Use caching, pagination, and efficient query patterns to optimize performance."
            },
            {
                "question": "The mobile SDK is crashing on startup",
                "expected_intent": IntentType.TECHNICAL,
                "ground_truth": "Check device compatibility and ensure proper SDK initialization."
            }
        ]
        
        # Billing support test cases
        billing_cases = [
            {
                "question": "I was charged twice for this month",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Please contact billing support to review your charges and process any necessary refunds."
            },
            {
                "question": "How much does the premium plan cost?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "The premium plan costs $99/month and includes advanced features and priority support."
            },
            {
                "question": "I want to cancel my subscription",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "You can cancel your subscription from your account settings or contact billing support."
            },
            {
                "question": "When will my next invoice be generated?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Invoices are generated monthly on your billing date, typically 30 days after signup."
            },
            {
                "question": "I need to update my payment method",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "You can update your payment method in your account billing settings."
            },
            {
                "question": "What's included in the basic plan?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "The basic plan includes core features, standard support, and basic usage limits."
            },
            {
                "question": "I want to upgrade to the enterprise plan",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Contact our sales team to discuss enterprise plan options and pricing."
            },
            {
                "question": "My payment failed, what should I do?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Check your payment method details and ensure sufficient funds, or contact billing support."
            },
            {
                "question": "I need a refund for last month",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Contact billing support to request a refund and provide the reason for the request."
            },
            {
                "question": "How do I change my billing address?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "You can update your billing address in your account settings under billing information."
            },
            {
                "question": "What are the different pricing tiers?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "We offer Basic ($29/month), Pro ($79/month), and Enterprise (custom pricing) plans."
            },
            {
                "question": "I want to downgrade my plan",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "You can downgrade your plan from account settings, effective at the next billing cycle."
            },
            {
                "question": "My credit card expired, how do I update it?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Update your credit card information in your account billing settings before the next charge."
            },
            {
                "question": "I need to see my billing history",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Your billing history is available in your account dashboard under billing section."
            },
            {
                "question": "How do I add more users to my account?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Additional users can be added from your account settings, charges apply per user."
            },
            {
                "question": "What's the difference between Pro and Enterprise?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Enterprise includes advanced security, custom integrations, and dedicated support."
            },
            {
                "question": "I'm being charged for features I don't use",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Review your plan features and consider downgrading if you don't need all features."
            },
            {
                "question": "How can I get a discount on my subscription?",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Contact sales for annual billing discounts or check for available promotions."
            },
            {
                "question": "I need an invoice for my accounting department",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Download invoices from your billing dashboard or contact support for custom invoices."
            },
            {
                "question": "The payment processing is not working",
                "expected_intent": IntentType.BILLING,
                "ground_truth": "Check your payment method and contact billing support if issues persist."
            }
        ]
        
        # Feature request test cases
        feature_cases = [
            {
                "question": "Can you add dark mode to the dashboard?",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Dark mode is on our roadmap for Q2 2024. You can track progress in our product updates."
            },
            {
                "question": "I would like to see analytics features",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Advanced analytics features are planned for our next major release."
            },
            {
                "question": "When will the mobile app be available?",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Mobile app development is in progress with expected release in Q3 2024."
            },
            {
                "question": "Can you support multiple languages?",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Internationalization is planned for future releases, starting with Spanish and French."
            },
            {
                "question": "I need bulk import functionality",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Bulk import features are being developed and will be available in the next update."
            },
            {
                "question": "Please add export to PDF feature",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "PDF export functionality is on our development roadmap for this quarter."
            },
            {
                "question": "I want to customize the interface",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Interface customization options are being considered for future releases."
            },
            {
                "question": "Can you add real-time notifications?",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Real-time notifications are in development and will be available soon."
            },
            {
                "question": "I need better search functionality",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Enhanced search features with filters and advanced queries are being developed."
            },
            {
                "question": "Please add team collaboration features",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Team collaboration tools are planned for our enterprise features roadmap."
            },
            {
                "question": "I want to integrate with Slack",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Slack integration is on our roadmap along with other popular productivity tools."
            },
            {
                "question": "Can you add more chart types?",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Additional chart types and visualization options are being developed."
            },
            {
                "question": "I need advanced filtering options",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Advanced filtering capabilities are planned for the next major release."
            },
            {
                "question": "Please add automation features",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Workflow automation features are being designed for future implementation."
            },
            {
                "question": "I want to schedule reports",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Scheduled reporting functionality is on our development roadmap."
            },
            {
                "question": "Can you add two-factor authentication?",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Two-factor authentication is being implemented for enhanced security."
            },
            {
                "question": "I need better user management",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Enhanced user management features are planned for enterprise customers."
            },
            {
                "question": "Please add custom fields",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "Custom field functionality is being developed for better data organization."
            },
            {
                "question": "I want to white-label the product",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "White-label options are available for enterprise customers, contact sales for details."
            },
            {
                "question": "Can you add API rate limiting controls?",
                "expected_intent": IntentType.FEATURES,
                "ground_truth": "API rate limiting controls are being developed for better resource management."
            }
        ]
        
        # Convert to TestCase objects
        for case in technical_cases:
            test_cases.append(TestCase(
                question=case["question"],
                expected_intent=case["expected_intent"],
                ground_truth=case["ground_truth"]
            ))
        
        for case in billing_cases:
            test_cases.append(TestCase(
                question=case["question"],
                expected_intent=case["expected_intent"],
                ground_truth=case["ground_truth"]
            ))
        
        for case in feature_cases:
            test_cases.append(TestCase(
                question=case["question"],
                expected_intent=case["expected_intent"],
                ground_truth=case["ground_truth"]
            ))
        
        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases

class RAGEvaluator:
    """Main evaluation system for RAG pipeline"""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.test_generator = TestDataGenerator()
        self.rag_engine = rag_engine
    
    async def evaluate_single_query(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single query"""
        start_time = time.time()
        
        try:
            # Create query request
            request = QueryRequest(query=test_case.question)
            
            # Process query
            response = await self.rag_engine.process_query(request)
            
            # Calculate metrics
            metrics = await self._calculate_metrics(test_case, response)
            
            # Determine if test passed
            passed = self._evaluate_test_result(test_case, response, metrics)
            
            return EvaluationResult(
                test_case=test_case,
                response=response,
                metrics=metrics,
                passed=passed
            )
            
        except Exception as e:
            logger.error(f"Single query evaluation error: {e}")
            # Return failed result
            return EvaluationResult(
                test_case=test_case,
                response=QueryResponse(
                    question=test_case.question,
                    answer="Evaluation failed",
                    intent=test_case.expected_intent,
                    sources=[],
                    confidence=0.0,
                    response_time=time.time() - start_time
                ),
                metrics=EvaluationMetrics(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    response_time=time.time() - start_time,
                    token_usage=0
                ),
                passed=False
            )
    
    async def _calculate_metrics(self, test_case: TestCase, response: QueryResponse) -> EvaluationMetrics:
        """Calculate evaluation metrics"""
        try:
            # Prepare context from sources
            context = "\n".join([doc.content for doc in response.sources])
            
            # Calculate faithfulness
            faithfulness = self.metrics_calculator.calculate_faithfulness(context, response.answer)
            
            # Calculate answer relevancy
            answer_relevancy = await self.metrics_calculator.calculate_answer_relevancy(
                test_case.question, response.answer
            )
            
            # Calculate context precision and recall
            context_precision = self._calculate_context_precision(test_case, response.sources)
            context_recall = self._calculate_context_recall(test_case, response.sources)
            
            # Calculate token usage
            token_usage = self.metrics_calculator.calculate_token_usage(
                test_case.question + response.answer + context
            )
            
            return EvaluationMetrics(
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
                response_time=response.response_time,
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return EvaluationMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                response_time=response.response_time,
                token_usage=0
            )
    
    def _calculate_context_precision(self, test_case: TestCase, sources: List) -> float:
        """Calculate context precision"""
        if not sources:
            return 0.0
        
        # Simple implementation: check if sources are relevant to the query
        relevant_sources = 0
        for source in sources:
            if source.score and source.score > 0.7:  # Threshold for relevance
                relevant_sources += 1
        
        return relevant_sources / len(sources)
    
    def _calculate_context_recall(self, test_case: TestCase, sources: List) -> float:
        """Calculate context recall"""
        if not sources:
            return 0.0
        
        # Simple implementation: assume we have relevant information if we have sources
        return 1.0 if sources else 0.0
    
    def _evaluate_test_result(self, test_case: TestCase, response: QueryResponse, 
                            metrics: EvaluationMetrics) -> bool:
        """Evaluate if test passed based on metrics and intent classification"""
        try:
            # Check intent classification accuracy
            intent_correct = response.intent.intent == test_case.expected_intent
            
            # Check minimum thresholds
            faithfulness_ok = metrics.faithfulness >= 0.7
            relevancy_ok = metrics.answer_relevancy >= 0.7
            confidence_ok = response.confidence >= 0.6
            
            # Overall pass criteria
            return intent_correct and faithfulness_ok and relevancy_ok and confidence_ok
            
        except Exception as e:
            logger.error(f"Test evaluation error: {e}")
            return False
    
    async def run_batch_evaluation(self, test_cases: Optional[List[TestCase]] = None) -> BatchEvaluationResult:
        """Run batch evaluation on test cases"""
        if test_cases is None:
            test_cases = self.test_generator.generate_test_cases()
        
        logger.info(f"Starting batch evaluation with {len(test_cases)} test cases")
        
        results = []
        passed_count = 0
        intent_accuracy = {intent.value: {"correct": 0, "total": 0} for intent in IntentType}
        
        # Process test cases
        for test_case in test_cases:
            result = await self.evaluate_single_query(test_case)
            results.append(result)
            
            if result.passed:
                passed_count += 1
            
            # Track intent accuracy
            expected_intent = test_case.expected_intent.value
            actual_intent = result.response.intent.intent.value
            
            intent_accuracy[expected_intent]["total"] += 1
            if expected_intent == actual_intent:
                intent_accuracy[expected_intent]["correct"] += 1
        
        # Calculate overall metrics
        overall_accuracy = passed_count / len(test_cases)
        
        # Calculate intent-specific accuracy
        intent_acc_scores = {}
        for intent, stats in intent_accuracy.items():
            if stats["total"] > 0:
                intent_acc_scores[intent] = stats["correct"] / stats["total"]
            else:
                intent_acc_scores[intent] = 0.0
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(results)
        
        return BatchEvaluationResult(
            total_cases=len(test_cases),
            passed_cases=passed_count,
            failed_cases=len(test_cases) - passed_count,
            overall_accuracy=overall_accuracy,
            intent_accuracy=intent_acc_scores,
            average_metrics=avg_metrics,
            results=results
        )
    
    def _calculate_average_metrics(self, results: List[EvaluationResult]) -> EvaluationMetrics:
        """Calculate average metrics from results"""
        if not results:
            return EvaluationMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                response_time=0.0,
                token_usage=0
            )
        
        total_faithfulness = sum(r.metrics.faithfulness for r in results)
        total_relevancy = sum(r.metrics.answer_relevancy for r in results)
        total_precision = sum(r.metrics.context_precision for r in results)
        total_recall = sum(r.metrics.context_recall for r in results)
        total_time = sum(r.metrics.response_time for r in results)
        total_tokens = sum(r.metrics.token_usage for r in results)
        
        count = len(results)
        
        return EvaluationMetrics(
            faithfulness=total_faithfulness / count,
            answer_relevancy=total_relevancy / count,
            context_precision=total_precision / count,
            context_recall=total_recall / count,
            response_time=total_time / count,
            token_usage=total_tokens // count
        )
    
    def save_evaluation_results(self, results: BatchEvaluationResult, filepath: str):
        """Save evaluation results to file"""
        try:
            # Convert to serializable format
            results_dict = {
                "timestamp": datetime.now().isoformat(),
                "total_cases": results.total_cases,
                "passed_cases": results.passed_cases,
                "failed_cases": results.failed_cases,
                "overall_accuracy": results.overall_accuracy,
                "intent_accuracy": results.intent_accuracy,
                "average_metrics": {
                    "faithfulness": results.average_metrics.faithfulness,
                    "answer_relevancy": results.average_metrics.answer_relevancy,
                    "context_precision": results.average_metrics.context_precision,
                    "context_recall": results.average_metrics.context_recall,
                    "response_time": results.average_metrics.response_time,
                    "token_usage": results.average_metrics.token_usage
                },
                "individual_results": []
            }
            
            # Add individual results
            for result in results.results:
                results_dict["individual_results"].append({
                    "question": result.test_case.question,
                    "expected_intent": result.test_case.expected_intent.value,
                    "actual_intent": result.response.intent.intent.value,
                    "answer": result.response.answer,
                    "confidence": result.response.confidence,
                    "passed": result.passed,
                    "metrics": {
                        "faithfulness": result.metrics.faithfulness,
                        "answer_relevancy": result.metrics.answer_relevancy,
                        "context_precision": result.metrics.context_precision,
                        "context_recall": result.metrics.context_recall,
                        "response_time": result.metrics.response_time,
                        "token_usage": result.metrics.token_usage
                    }
                })
            
            # Save to file
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def generate_evaluation_report(self, results: BatchEvaluationResult) -> str:
        """Generate a human-readable evaluation report"""
        report = f"""
# RAG System Evaluation Report

## Overall Performance
- **Total Test Cases**: {results.total_cases}
- **Passed Cases**: {results.passed_cases}
- **Failed Cases**: {results.failed_cases}
- **Overall Accuracy**: {results.overall_accuracy:.2%}

## Intent Classification Accuracy
"""
        
        for intent, accuracy in results.intent_accuracy.items():
            report += f"- **{intent.title()}**: {accuracy:.2%}\n"
        
        report += f"""
## Average Metrics
- **Faithfulness**: {results.average_metrics.faithfulness:.3f}
- **Answer Relevancy**: {results.average_metrics.answer_relevancy:.3f}
- **Context Precision**: {results.average_metrics.context_precision:.3f}
- **Context Recall**: {results.average_metrics.context_recall:.3f}
- **Average Response Time**: {results.average_metrics.response_time:.3f}s
- **Average Token Usage**: {results.average_metrics.token_usage}

## Recommendations
"""
        
        # Add recommendations based on results
        if results.overall_accuracy < 0.8:
            report += "- Consider improving training data quality and quantity\n"
        
        if results.average_metrics.faithfulness < 0.7:
            report += "- Review context retrieval and improve document quality\n"
        
        if results.average_metrics.answer_relevancy < 0.7:
            report += "- Optimize prompt templates for better relevancy\n"
        
        if results.average_metrics.response_time > 2.0:
            report += "- Consider optimizing response generation for better performance\n"
        
        return report

# Global evaluator instance
evaluator = RAGEvaluator() 