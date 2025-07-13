#!/usr/bin/env python3
"""
Basic system tests for RAG Customer Support System
"""

import asyncio
import json
import requests
import time
from loguru import logger

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_QUERIES = [
    {
        "query": "I'm getting a 500 error when calling the API",
        "expected_intent": "technical",
        "description": "Technical support query"
    },
    {
        "query": "How much does the premium plan cost?",
        "expected_intent": "billing",
        "description": "Billing inquiry"
    },
    {
        "query": "Can you add dark mode to the dashboard?",
        "expected_intent": "features",
        "description": "Feature request"
    }
]

class SystemTester:
    """System tester for RAG API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def test_health_check(self):
        """Test health check endpoint"""
        logger.info("Testing health check...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health check passed: {data.get('status', 'unknown')}")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def test_query_endpoint(self, query_data: dict):
        """Test query endpoint"""
        logger.info(f"Testing query: {query_data['description']}")
        
        try:
            payload = {"query": query_data["query"]}
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ["question", "answer", "intent", "confidence", "response_time"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    logger.error(f"Missing fields in response: {missing_fields}")
                    return False
                
                # Check intent classification
                actual_intent = data["intent"]["intent"]
                expected_intent = query_data["expected_intent"]
                
                intent_correct = actual_intent == expected_intent
                confidence = data["confidence"]
                response_time = data["response_time"]
                
                logger.info(f"Query result:")
                logger.info(f"  - Intent: {actual_intent} (expected: {expected_intent}) - {'✓' if intent_correct else '✗'}")
                logger.info(f"  - Confidence: {confidence:.2%}")
                logger.info(f"  - Response time: {response_time:.2f}s")
                logger.info(f"  - Answer length: {len(data['answer'])} chars")
                
                self.results.append({
                    "query": query_data["query"],
                    "expected_intent": expected_intent,
                    "actual_intent": actual_intent,
                    "intent_correct": intent_correct,
                    "confidence": confidence,
                    "response_time": response_time,
                    "answer_length": len(data["answer"])
                })
                
                return True
                
            else:
                logger.error(f"Query failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            return False
    
    def test_streaming_endpoint(self, query_data: dict):
        """Test streaming endpoint"""
        logger.info(f"Testing streaming query: {query_data['description']}")
        
        try:
            payload = {"query": query_data["query"]}
            response = self.session.post(
                f"{self.base_url}/query/stream",
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code == 200:
                chunks = []
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            try:
                                chunk_data = json.loads(line_str[6:])
                                chunks.append(chunk_data)
                                if chunk_data.get('is_final', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                
                logger.info(f"Streaming result: {len(chunks)} chunks received")
                return True
                
            else:
                logger.error(f"Streaming failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            return False
    
    def test_intent_classification(self):
        """Test intent classification endpoint"""
        logger.info("Testing intent classification...")
        
        try:
            for query_data in TEST_QUERIES:
                response = self.session.post(
                    f"{self.base_url}/intent/classify",
                    params={"query": query_data["query"]}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    actual_intent = data["intent"]
                    expected_intent = query_data["expected_intent"]
                    confidence = data["confidence"]
                    
                    logger.info(f"Intent classification: {actual_intent} (confidence: {confidence:.2%})")
                else:
                    logger.error(f"Intent classification failed: {response.status_code}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return False
    
    def test_system_endpoints(self):
        """Test system management endpoints"""
        logger.info("Testing system endpoints...")
        
        endpoints = [
            "/metrics",
            "/system/stats",
            "/database/stats",
            "/llm/status"
        ]
        
        success_count = 0
        
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                
                if response.status_code == 200:
                    logger.info(f"✓ {endpoint}")
                    success_count += 1
                else:
                    logger.error(f"✗ {endpoint}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"✗ {endpoint}: {e}")
        
        return success_count == len(endpoints)
    
    def test_evaluation_endpoint(self):
        """Test evaluation endpoint"""
        logger.info("Testing evaluation endpoint...")
        
        try:
            # Test generating test cases
            response = self.session.get(f"{self.base_url}/evaluate/generate-tests")
            
            if response.status_code == 200:
                data = response.json()
                test_cases = data.get("test_cases", [])
                logger.info(f"Generated {len(test_cases)} test cases")
                
                # Test single evaluation with first test case
                if test_cases:
                    test_case = test_cases[0]
                    response = self.session.post(
                        f"{self.base_url}/evaluate/single",
                        json=test_case
                    )
                    
                    if response.status_code == 200:
                        logger.info("Single evaluation test passed")
                        return True
                    else:
                        logger.error(f"Single evaluation failed: {response.status_code}")
                        return False
                        
            else:
                logger.error(f"Test generation failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Evaluation test error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all system tests"""
        logger.info("Starting comprehensive system tests...")
        
        test_results = {
            "health_check": self.test_health_check(),
            "query_endpoint": all(self.test_query_endpoint(query) for query in TEST_QUERIES),
            "streaming_endpoint": all(self.test_streaming_endpoint(query) for query in TEST_QUERIES[:1]),  # Test one streaming
            "intent_classification": self.test_intent_classification(),
            "system_endpoints": self.test_system_endpoints(),
            "evaluation_endpoint": self.test_evaluation_endpoint()
        }
        
        # Summary
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Results Summary")
        logger.info(f"{'='*60}")
        
        for test_name, passed in test_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{test_name:<25} {status}")
        
        logger.info(f"{'='*60}")
        logger.info(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! System is working correctly.")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} test(s) failed. Check the logs above.")
        
        # Query performance summary
        if self.results:
            logger.info(f"\nQuery Performance Summary:")
            intent_accuracy = sum(1 for r in self.results if r["intent_correct"]) / len(self.results)
            avg_confidence = sum(r["confidence"] for r in self.results) / len(self.results)
            avg_response_time = sum(r["response_time"] for r in self.results) / len(self.results)
            
            logger.info(f"  - Intent Accuracy: {intent_accuracy:.2%}")
            logger.info(f"  - Average Confidence: {avg_confidence:.2%}")
            logger.info(f"  - Average Response Time: {avg_response_time:.2f}s")
        
        return passed_tests == total_tests

def main():
    """Main test function"""
    logger.info("RAG Customer Support System - Test Suite")
    logger.info(f"Testing API at: {BASE_URL}")
    
    # Wait for API to be ready
    logger.info("Waiting for API to be ready...")
    time.sleep(2)
    
    # Create tester and run tests
    tester = SystemTester(BASE_URL)
    success = tester.run_all_tests()
    
    if success:
        logger.info("\n✅ All tests passed! The RAG system is working correctly.")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Please check the system configuration.")
        return 1

if __name__ == "__main__":
    exit(main()) 