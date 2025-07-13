import re
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from loguru import logger
import joblib
import os

from pydantic_models import IntentType, IntentClassification
from config import INTENT_CATEGORIES, settings

class IntentClassifier:
    """Intent classification system for customer support queries"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.intent_keywords = self._extract_keywords()
        self.model_path = "./models/intent_classifier.joblib"
        
        # Initialize or load model
        self._initialize_model()
    
    def _extract_keywords(self) -> Dict[str, List[str]]:
        """Extract keywords from intent categories"""
        keywords = {}
        for intent, config in INTENT_CATEGORIES.items():
            keywords[intent] = config["keywords"]
        return keywords
    
    def _initialize_model(self):
        """Initialize or load the intent classification model"""
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default model with basic training data"""
        # Create basic training data
        training_data = self._generate_training_data()
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Train the model
        X = [item['text'] for item in training_data]
        y = [item['intent'] for item in training_data]
        
        self.pipeline.fit(X, y)
        
        # Save the model
        self._save_model()
        
        logger.info("Default intent classification model created and saved")
    
    def _generate_training_data(self) -> List[Dict]:
        """Generate basic training data for intent classification"""
        training_data = []
        
        # Technical support examples
        technical_examples = [
            "I'm getting a 500 error when calling the API",
            "How do I implement authentication in my app?",
            "The SDK is not working properly",
            "There's a bug in the payment processing",
            "I need help with the integration",
            "The API documentation is unclear",
            "My code is throwing an exception",
            "How to fix this technical issue?",
            "The service is returning invalid responses",
            "I'm having trouble with the webhook setup",
            "The SSL certificate is not working",
            "Database connection is failing",
            "Rate limiting is not working as expected",
            "How to debug this error?",
            "The API endpoint is not responding",
            "I need technical support for implementation",
            "The library is missing some functions",
            "How to handle API errors properly?",
            "The documentation needs updating",
            "I'm getting timeout errors"
        ]
        
        # Billing support examples
        billing_examples = [
            "I was charged twice this month",
            "How much does the premium plan cost?",
            "I want to cancel my subscription",
            "When will my invoice be generated?",
            "I need to update my payment method",
            "What's included in the basic plan?",
            "I want to upgrade my account",
            "My payment failed, what should I do?",
            "I need a refund for last month",
            "How to change my billing address?",
            "What are the pricing tiers?",
            "I want to downgrade my plan",
            "My credit card expired",
            "I need to see my billing history",
            "How to add more users to my account?",
            "What's the difference between plans?",
            "I'm being charged for features I don't use",
            "How to get a discount?",
            "I need an invoice for accounting",
            "Payment processing is not working"
        ]
        
        # Feature requests examples
        feature_examples = [
            "Can you add dark mode to the dashboard?",
            "I would like to see analytics features",
            "When will mobile app be available?",
            "Can you support multiple languages?",
            "I need bulk import functionality",
            "Please add export to PDF feature",
            "I want to customize the interface",
            "Can you add real-time notifications?",
            "I need better search functionality",
            "Please add team collaboration features",
            "I want to integrate with Slack",
            "Can you add more chart types?",
            "I need advanced filtering options",
            "Please add automation features",
            "I want to schedule reports",
            "Can you add two-factor authentication?",
            "I need better user management",
            "Please add custom fields",
            "I want to white-label the product",
            "Can you add API rate limiting controls?"
        ]
        
        # Add training examples
        for example in technical_examples:
            training_data.append({"text": example, "intent": "technical"})
        
        for example in billing_examples:
            training_data.append({"text": example, "intent": "billing"})
        
        for example in feature_examples:
            training_data.append({"text": example, "intent": "features"})
        
        return training_data
    
    def _save_model(self):
        """Save the trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info("Intent classification model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._create_default_model()
    
    def _keyword_based_classification(self, text: str) -> Dict[str, float]:
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            
            # Normalize score
            scores[intent] = score / len(keywords) if keywords else 0
        
        return scores
    
    def _rule_based_classification(self, text: str) -> Dict[str, float]:
        """Rule-based classification using patterns"""
        text_lower = text.lower()
        scores = {"technical": 0, "billing": 0, "features": 0}
        
        # Technical patterns
        technical_patterns = [
            r'\b(error|bug|issue|problem|fail|exception|timeout)\b',
            r'\b(api|sdk|code|implementation|integration)\b',
            r'\b(debug|fix|solve|troubleshoot)\b',
            r'\b(documentation|guide|tutorial)\b'
        ]
        
        # Billing patterns
        billing_patterns = [
            r'\b(price|cost|payment|billing|invoice|charge)\b',
            r'\b(plan|subscription|account|upgrade|downgrade)\b',
            r'\b(refund|cancel|discount|tier)\b',
            r'\b(credit card|payment method|billing address)\b'
        ]
        
        # Feature patterns
        feature_patterns = [
            r'\b(feature|add|new|enhance|improve|request)\b',
            r'\b(dashboard|interface|ui|ux|design)\b',
            r'\b(integration|export|import|automation)\b',
            r'\b(mobile|app|notification|search)\b'
        ]
        
        # Count pattern matches
        for pattern in technical_patterns:
            scores["technical"] += len(re.findall(pattern, text_lower))
        
        for pattern in billing_patterns:
            scores["billing"] += len(re.findall(pattern, text_lower))
        
        for pattern in feature_patterns:
            scores["features"] += len(re.findall(pattern, text_lower))
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        return scores
    
    def classify_intent(self, text: str) -> IntentClassification:
        """Classify the intent of a given text"""
        try:
            # Primary classification using trained model
            if self.pipeline:
                predictions = self.pipeline.predict_proba([text])[0]
                classes = self.pipeline.classes_
                
                # Create scores dictionary
                ml_scores = dict(zip(classes, predictions))
                
                # Ensure all intents are present
                for intent in ["technical", "billing", "features"]:
                    if intent not in ml_scores:
                        ml_scores[intent] = 0.0
            else:
                ml_scores = {"technical": 0.33, "billing": 0.33, "features": 0.33}
            
            # Fallback methods
            keyword_scores = self._keyword_based_classification(text)
            rule_scores = self._rule_based_classification(text)
            
            # Combine scores (weighted average)
            combined_scores = {}
            for intent in ["technical", "billing", "features"]:
                combined_scores[intent] = (
                    0.6 * ml_scores.get(intent, 0) +
                    0.2 * keyword_scores.get(intent, 0) +
                    0.2 * rule_scores.get(intent, 0)
                )
            
            # Find the intent with highest score
            best_intent = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[best_intent]
            
            # Apply confidence threshold
            if confidence < settings.intent_confidence_threshold:
                # If confidence is low, use keyword-based as fallback
                keyword_best = max(keyword_scores, key=keyword_scores.get)
                if keyword_scores[keyword_best] > 0:
                    best_intent = keyword_best
                    confidence = keyword_scores[keyword_best]
            
            return IntentClassification(
                intent=IntentType(best_intent),
                confidence=confidence,
                scores=combined_scores
            )
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            # Return default classification
            return IntentClassification(
                intent=IntentType.TECHNICAL,
                confidence=0.5,
                scores={"technical": 0.5, "billing": 0.25, "features": 0.25}
            )
    
    def train_model(self, training_data: List[Dict]) -> Dict:
        """Train the intent classification model with new data"""
        try:
            X = [item['text'] for item in training_data]
            y = [item['intent'] for item in training_data]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create and train pipeline
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            self.pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model
            self._save_model()
            
            logger.info(f"Model trained successfully with accuracy: {accuracy:.3f}")
            
            return {
                "accuracy": accuracy,
                "classification_report": report,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            raise
    
    def evaluate_model(self, test_data: List[Dict]) -> Dict:
        """Evaluate the current model performance"""
        try:
            if not self.pipeline:
                raise ValueError("No trained model available")
            
            X_test = [item['text'] for item in test_data]
            y_true = [item['intent'] for item in test_data]
            
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            
            return {
                "accuracy": accuracy,
                "classification_report": report,
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "model_type": "Naive Bayes with TF-IDF",
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path),
            "confidence_threshold": settings.intent_confidence_threshold,
            "supported_intents": list(INTENT_CATEGORIES.keys())
        }

# Global intent classifier instance
intent_classifier = IntentClassifier() 