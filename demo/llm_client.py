"""
LLM Client for OpenRouter API
Handles API calls to different LLMs through OpenRouter
"""

import os
import time
import json
from typing import Dict, Optional, Tuple
import config

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenRouterClient:
    """Client for calling LLMs through OpenRouter API"""

    def __init__(self, api_key: str = None):
        """
        Initialize OpenRouter client

        Args:
            api_key: OpenRouter API key (uses config/env if None)
        """
        self.api_key = api_key or config.OPENROUTER_API_KEY

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY environment variable or pass to constructor"
            )

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/anthropics/router-research",
                "X-Title": "CoRE Router Demo"
            }
        )

        print("✅ OpenRouter client initialized")

    def call_llm(
        self,
        model_id: str,
        query: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        timeout: int = None
    ) -> Tuple[str, int]:
        """
        Call an LLM through OpenRouter

        Args:
            model_id: OpenRouter model ID (e.g., "meta-llama/llama-3.1-70b-instruct")
            query: User query to send to the model
            max_tokens: Maximum tokens to generate (None = unlimited)
            temperature: Sampling temperature
            timeout: Request timeout in seconds (uses config if None)

        Returns:
            Tuple of (response_text, actual_token_count)
        """
        timeout = timeout or config.LLM_TIMEOUT

        try:
            # Build request parameters
            request_params = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": query}
                ],
                "temperature": temperature,
            }

            # Add max_tokens if specified (including "unlimited" as very large number)
            if max_tokens is not None:
                if isinstance(max_tokens, str) and max_tokens == "unlimited":
                    request_params["max_tokens"] = 4096  # Reasonable unlimited
                else:
                    request_params["max_tokens"] = max_tokens

            # Make API call
            start_time = time.time()
            response = self.client.chat.completions.create(**request_params)
            elapsed_time = time.time() - start_time

            # Extract response text
            response_text = response.choices[0].message.content

            # Extract token usage
            if hasattr(response, 'usage') and response.usage:
                actual_tokens = response.usage.completion_tokens
            else:
                # Estimate if not provided (rough approximation)
                actual_tokens = len(response_text.split()) * 1.3

            print(f"  ✓ {model_id} responded ({actual_tokens} tokens, {elapsed_time:.1f}s)")

            return response_text, int(actual_tokens)

        except openai.APITimeoutError:
            raise TimeoutError(f"Request to {model_id} timed out after {timeout}s")
        except openai.RateLimitError:
            raise RuntimeError(f"Rate limit exceeded for {model_id}")
        except openai.APIError as e:
            raise RuntimeError(f"API error for {model_id}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling {model_id}: {e}")

    def call_llm_by_name(
        self,
        llm_name: str,
        query: str,
        token_limit: Optional[int] = None
    ) -> Tuple[str, int, str]:
        """
        Call an LLM by its name from the configured pool

        Args:
            llm_name: Name from config.LLM_POOL (e.g., "GLM-4.5-Air")
            query: User query
            token_limit: Token limit (None or "unlimited" for no limit)

        Returns:
            Tuple of (response_text, actual_token_count, actual_prompt_sent)
        """
        if llm_name not in config.LLM_POOL:
            raise ValueError(f"LLM '{llm_name}' not found in configured pool")

        llm_config = config.LLM_POOL[llm_name]
        model_id = llm_config["openrouter_id"]

        # Convert token_limit
        max_tokens = None if token_limit == "unlimited" else token_limit

        # Add instructional prompt ONLY for limited settings to enforce the budget
        # CARROT always uses unlimited, so it should never get the instructional prompt
        if token_limit is not None and token_limit != "unlimited" and str(token_limit).lower() != "unlimited":
            try:
                # Ensure token_limit is a valid integer
                limit_value = int(token_limit)
                # Add explicit instruction to respect token limit
                # Note: We use "words" as a more intuitive unit for the LLM,
                # though the actual limit is in tokens
                modified_query = (
                    f"{query}\n\n"
                    f"You have a strict budget of {limit_value} words. "
                    f"You must answer in at most {limit_value} words!\n"
                    f"Answer:"
                )
                # Don't enforce hard max_tokens cutoff since we have instructional prompt
                # Let the LLM naturally respect the limit instead of hard truncation
                api_max_tokens = None
            except (ValueError, TypeError):
                # If token_limit is not a valid integer, treat as unlimited
                modified_query = query
                api_max_tokens = max_tokens
        else:
            modified_query = query
            api_max_tokens = max_tokens

        response, token_count = self.call_llm(model_id, modified_query, max_tokens=api_max_tokens)
        return response, token_count, modified_query


class MockLLMClient:
    """Mock LLM client for testing without API calls"""

    def __init__(self):
        print("⚠️  Using MockLLMClient (no real API calls)")

    def call_llm(
        self,
        model_id: str,
        query: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        timeout: int = None
    ) -> Tuple[str, int]:
        """
        Generate mock LLM response

        Args:
            model_id: Model ID (ignored in mock)
            query: User query
            max_tokens: Maximum tokens (affects mock response length)
            temperature: Temperature (ignored in mock)
            timeout: Timeout (ignored in mock)

        Returns:
            Tuple of (mock_response, mock_token_count)
        """
        # Generate mock response based on query
        time.sleep(0.5)  # Simulate API latency

        # Simple mock responses
        query_lower = query.lower()

        if "capital" in query_lower and "france" in query_lower:
            response = "Paris"
            tokens = 1
        elif "solve" in query_lower or "equation" in query_lower or "2x" in query_lower:
            response = "To solve 2x + 5 = 13:\n1. Subtract 5 from both sides: 2x = 8\n2. Divide by 2: x = 4"
            tokens = 30
        elif "photosynthesis" in query_lower:
            response = "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."
            tokens = 25
        elif "factorial" in query_lower or "python" in query_lower:
            response = "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```"
            tokens = 35
        elif "climate change" in query_lower:
            response = "The main causes of climate change include:\n1. Greenhouse gas emissions from burning fossil fuels\n2. Deforestation\n3. Industrial processes\n4. Agriculture and livestock"
            tokens = 40
        else:
            response = f"This is a mock response to your query: '{query[:50]}...'"
            tokens = 15

        # Respect max_tokens if specified
        if max_tokens and max_tokens != "unlimited":
            # Roughly truncate response
            words = response.split()
            max_words = int(max_tokens * 0.75)  # Rough conversion
            if len(words) > max_words:
                response = " ".join(words[:max_words]) + "..."
                tokens = max_tokens

        print(f"  ✓ Mock {model_id} responded ({tokens} tokens)")

        return response, tokens

    def call_llm_by_name(
        self,
        llm_name: str,
        query: str,
        token_limit: Optional[int] = None
    ) -> Tuple[str, int, str]:
        """
        Mock call by name

        Args:
            llm_name: LLM name from pool
            query: User query
            token_limit: Token limit

        Returns:
            Tuple of (mock_response, mock_token_count, actual_prompt_sent)
        """
        if llm_name not in config.LLM_POOL:
            raise ValueError(f"LLM '{llm_name}' not found in configured pool")

        llm_config = config.LLM_POOL[llm_name]
        model_id = llm_config["openrouter_id"]

        max_tokens = None if token_limit == "unlimited" else token_limit

        # Add instructional prompt ONLY for limited settings (same as real client)
        # CARROT always uses unlimited, so it should never get the instructional prompt
        if token_limit is not None and token_limit != "unlimited" and str(token_limit).lower() != "unlimited":
            try:
                # Ensure token_limit is a valid integer
                limit_value = int(token_limit)
                modified_query = (
                    f"{query}\n\n"
                    f"You have a strict budget of {limit_value} words. "
                    f"You must answer in at most {limit_value} words!\n"
                    f"Answer:"
                )
                # Don't enforce hard max_tokens cutoff (same as real client)
                api_max_tokens = None
            except (ValueError, TypeError):
                # If token_limit is not a valid integer, treat as unlimited
                modified_query = query
                api_max_tokens = max_tokens
        else:
            modified_query = query
            api_max_tokens = max_tokens

        response, token_count = self.call_llm(model_id, modified_query, max_tokens=api_max_tokens)
        return response, token_count, modified_query


# ============================================================================
# Factory Function
# ============================================================================

def get_llm_client(mock: bool = None) -> OpenRouterClient:
    """
    Get an LLM client instance

    Args:
        mock: Use mock client (uses config.ENABLE_MOCK_MODE if None)

    Returns:
        OpenRouterClient or MockLLMClient
    """
    if mock is None:
        mock = config.ENABLE_MOCK_MODE

    if mock:
        return MockLLMClient()
    else:
        return OpenRouterClient()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing LLM Client...")
    print()

    # Test mock client
    print("=== Mock Client ===")
    mock_client = MockLLMClient()

    queries = [
        "What is the capital of France?",
        "Solve 2x + 5 = 13",
        "Explain photosynthesis"
    ]

    for query in queries:
        response, tokens = mock_client.call_llm("test-model", query, max_tokens=100)
        print(f"Query: {query}")
        print(f"Response: {response[:100]}...")
        print(f"Tokens: {tokens}")
        print()

    # Test real client (only if API key available)
    if config.OPENROUTER_API_KEY and not config.ENABLE_MOCK_MODE:
        print("=== Real Client ===")
        try:
            real_client = OpenRouterClient()

            # Test with a simple query
            query = "What is 2+2?"
            model_id = "qwen/qwen-2.5-7b-instruct"  # Cheap model for testing

            print(f"Testing with model: {model_id}")
            response, tokens = real_client.call_llm(model_id, query, max_tokens=20)
            print(f"Query: {query}")
            print(f"Response: {response}")
            print(f"Tokens: {tokens}")

        except Exception as e:
            print(f"⚠️  Real client test failed: {e}")
    else:
        print("⚠️  Skipping real client test (no API key or mock mode enabled)")
