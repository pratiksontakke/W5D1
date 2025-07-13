import asyncio
import time
from typing import Optional, Dict, Any, AsyncGenerator
from abc import ABC, abstractmethod
import openai
import requests
from loguru import logger
from pydantic_models import LLMProvider
from config import settings
import json

class LLMInterface(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text from prompt"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy"""
        pass

class OpenAIProvider(LLMInterface):
    """OpenAI LLM provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.name = "openai"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        try:
            temperature = kwargs.get('temperature', settings.temperature)
            max_tokens = kwargs.get('max_tokens', 1000)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text using OpenAI API"""
        try:
            temperature = kwargs.get('temperature', settings.temperature)
            max_tokens = kwargs.get('max_tokens', 1000)
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False

class OllamaProvider(LLMInterface):
    """Ollama LLM provider implementation"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.name = "ollama"
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API"""
        try:
            temperature = kwargs.get('temperature', settings.temperature)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text using Ollama API"""
        try:
            temperature = kwargs.get('temperature', settings.temperature)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Ollama API health"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

class RequestQueue:
    """Queue for managing concurrent requests"""
    
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue()
        self.active_requests = 0
    
    async def add_request(self, coro):
        """Add a request to the queue"""
        async with self.semaphore:
            self.active_requests += 1
            try:
                result = await coro
                return result
            finally:
                self.active_requests -= 1
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()
    
    def get_active_requests(self) -> int:
        """Get number of active requests"""
        return self.active_requests

class LLMWrapper:
    """Main LLM wrapper with fallback functionality"""
    
    def __init__(self):
        self.providers = {}
        self.current_provider = None
        self.fallback_provider = None
        self.request_queue = RequestQueue()
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        try:
            # Initialize OpenAI provider
            self.providers[LLMProvider.OPENAI] = OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model
            )
            logger.info("OpenAI provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        try:
            # Initialize Ollama provider
            self.providers[LLMProvider.OLLAMA] = OllamaProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model
            )
            logger.info("Ollama provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
        
        # Set default provider order
        self.current_provider = LLMProvider.OPENAI
        self.fallback_provider = LLMProvider.OLLAMA
    
    async def switch_provider(self, provider: LLMProvider, model: Optional[str] = None) -> bool:
        """Switch to a different LLM provider"""
        try:
            if provider not in self.providers:
                logger.error(f"Provider {provider} not available")
                return False
            
            # Check if provider is healthy
            if not await self.providers[provider].health_check():
                logger.error(f"Provider {provider} is not healthy")
                return False
            
            # Update model if provided
            if model:
                if provider == LLMProvider.OPENAI:
                    self.providers[provider].model = model
                elif provider == LLMProvider.OLLAMA:
                    self.providers[provider].model = model
            
            self.current_provider = provider
            logger.info(f"Switched to provider: {provider}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch provider: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with automatic fallback"""
        async def _generate():
            # Try current provider first
            try:
                provider = self.providers[self.current_provider]
                return await provider.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Current provider {self.current_provider} failed: {e}")
                
                # Try fallback provider
                if self.fallback_provider and self.fallback_provider in self.providers:
                    try:
                        logger.info(f"Falling back to {self.fallback_provider}")
                        provider = self.providers[self.fallback_provider]
                        return await provider.generate(prompt, **kwargs)
                    except Exception as fe:
                        logger.error(f"Fallback provider {self.fallback_provider} also failed: {fe}")
                
                # If all providers fail, raise the original exception
                raise e
        
        return await self.request_queue.add_request(_generate())
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text with automatic fallback"""
        try:
            provider = self.providers[self.current_provider]
            async for chunk in provider.generate_stream(prompt, **kwargs):
                yield chunk
        except Exception as e:
            logger.warning(f"Current provider {self.current_provider} streaming failed: {e}")
            
            # Try fallback provider for streaming
            if self.fallback_provider and self.fallback_provider in self.providers:
                try:
                    logger.info(f"Falling back to {self.fallback_provider} for streaming")
                    provider = self.providers[self.fallback_provider]
                    async for chunk in provider.generate_stream(prompt, **kwargs):
                        yield chunk
                except Exception as fe:
                    logger.error(f"Fallback provider {self.fallback_provider} streaming also failed: {fe}")
                    raise e
            else:
                raise e
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health_status = {}
        
        for provider_name, provider in self.providers.items():
            try:
                health_status[provider_name.value] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {provider_name}: {e}")
                health_status[provider_name.value] = False
        
        return health_status
    
    def get_current_provider(self) -> LLMProvider:
        """Get current active provider"""
        return self.current_provider
    
    def get_current_model(self) -> str:
        """Get current active model"""
        if self.current_provider in self.providers:
            return self.providers[self.current_provider].model
        return "unknown"
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get request queue status"""
        return {
            "queue_size": self.request_queue.get_queue_size(),
            "active_requests": self.request_queue.get_active_requests()
        }

# Global LLM wrapper instance
llm_wrapper = LLMWrapper() 