"""
Judge Module for Response Evaluation
Uses a judge LLM to evaluate response quality and extract scores
"""

import json
import re
from typing import Dict, Tuple
import config
from llm_client import get_llm_client


class ResponseJudge:
    """Evaluates LLM responses using a judge model"""

    def __init__(self, judge_model: str = None, llm_client=None):
        """
        Initialize the judge

        Args:
            judge_model: OpenRouter model ID for judge (uses config if None)
            llm_client: LLM client instance (creates new if None)
        """
        self.judge_model = judge_model or config.JUDGE_MODEL
        self.llm_client = llm_client or get_llm_client()
        print(f"✅ Judge initialized (model: {self.judge_model})")

    def evaluate(self, query: str, response: str, golden_answer: str = None) -> Tuple[float, str]:
        """
        Evaluate a response and extract quality score

        Args:
            query: Original user query
            response: LLM response to evaluate
            golden_answer: Optional ground truth answer for comparison

        Returns:
            Tuple of (score, reasoning)
            - score: Float in [0, 1] indicating quality
            - reasoning: Explanation of the score
        """
        # Format judge prompt - use template with or without golden answer
        if golden_answer:
            prompt = config.JUDGE_PROMPT_TEMPLATE.format(
                query=query,
                golden_answer=golden_answer,
                response=response
            )
        else:
            # Use template without golden answer
            prompt = config.JUDGE_PROMPT_TEMPLATE_NO_GOLDEN.format(
                query=query,
                response=response
            )

        try:
            # Call judge LLM
            judge_response, _ = self.llm_client.call_llm(
                model_id=self.judge_model,
                query=prompt,
                max_tokens=200,
                temperature=0.0  # Deterministic evaluation
            )

            # Extract score and reasoning from JSON response
            score, reasoning = self._parse_judge_response(judge_response)

            return score, reasoning

        except Exception as e:
            print(f"⚠️  Judge evaluation failed: {e}")
            # Return neutral score on failure
            return 0.5, f"Evaluation failed: {str(e)}"

    def _parse_judge_response(self, response: str) -> Tuple[float, str]:
        """
        Parse judge response to extract score and reasoning

        Uses similar logic to ultimate_json_score_extractor from utils.py

        Args:
            response: Raw judge LLM response

        Returns:
            Tuple of (score, reasoning)
        """
        try:
            # Try direct JSON parsing
            data = json.loads(response)

            # Try new format first: "correctness_score" and "justification"
            score = data.get("correctness_score")
            reasoning = data.get("justification")

            # Fall back to old format: "score" and "reasoning"
            if score is None:
                score = data.get("score", 0.5)
            if reasoning is None:
                reasoning = data.get("reasoning", "No reasoning provided")

            score = float(score)

            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))

            return score, reasoning

        except json.JSONDecodeError:
            # JSON parsing failed, try to extract from text
            pass

        # Try to find JSON object in response (handles both field name formats)
        json_patterns = [
            r'\{[^{}]*"correctness_score"[^{}]*\}',  # New format
            r'\{[^{}]*"score"[^{}]*\}'  # Old format
        ]

        for json_pattern in json_patterns:
            json_match = re.search(json_pattern, response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())

                    # Try new format first
                    score = data.get("correctness_score")
                    reasoning = data.get("justification")

                    # Fall back to old format
                    if score is None:
                        score = data.get("score", 0.5)
                    if reasoning is None:
                        reasoning = data.get("reasoning", "No reasoning provided")

                    score = float(score)
                    score = max(0.0, min(1.0, score))
                    return score, reasoning
                except (json.JSONDecodeError, ValueError):
                    continue

        # Try to extract score from text patterns
        score_patterns = [
            r'"correctness_score"\s*:\s*([0-9.]+)',  # New format
            r'correctness_score\s*:\s*([0-9.]+)',
            r'"score"\s*:\s*([0-9.]+)',  # Old format
            r'score\s*:\s*([0-9.]+)',
            r'score\s*=\s*([0-9.]+)',
            r'([0-9.]+)\s*/\s*1\.0',
            r'([0-9.]+)\s*out of\s*1',
        ]

        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    score = max(0.0, min(1.0, score))
                    reasoning = response[:200]  # Use first 200 chars as reasoning
                    return score, reasoning
                except ValueError:
                    continue

        # If all parsing fails, return neutral score
        print(f"⚠️  Could not parse judge response: {response[:100]}...")
        return 0.5, "Could not parse judge response"


class MockJudge:
    """Mock judge for testing without API calls"""

    def __init__(self):
        print("⚠️  Using MockJudge (deterministic scores)")

    def evaluate(self, query: str, response: str) -> Tuple[float, str]:
        """
        Generate mock evaluation scores

        Args:
            query: Original user query
            response: LLM response to evaluate

        Returns:
            Tuple of (score, reasoning)
        """
        # Simple heuristic scoring
        score = 0.5  # Default neutral

        query_lower = query.lower()
        response_lower = response.lower()

        # Check for keywords that suggest correctness
        if "capital" in query_lower and "paris" in response_lower:
            score = 1.0
            reasoning = "Correct answer: Paris is the capital of France"

        elif ("solve" in query_lower or "equation" in query_lower) and "x = 4" in response_lower:
            score = 1.0
            reasoning = "Correct solution: x = 4"

        elif "photosynthesis" in query_lower and len(response) > 50:
            score = 0.9
            reasoning = "Good explanation of photosynthesis"

        elif "factorial" in query_lower and "def factorial" in response_lower:
            score = 0.95
            reasoning = "Correct Python factorial function"

        elif "climate change" in query_lower and "greenhouse" in response_lower:
            score = 0.85
            reasoning = "Mentions key causes of climate change"

        else:
            # Length-based heuristic
            if len(response) < 10:
                score = 0.6
                reasoning = "Response too short"
            elif len(response) > 20 and len(response) < 200:
                score = 0.75
                reasoning = "Reasonable response length"
            else:
                score = 0.7
                reasoning = "Generic response"

        return score, reasoning


# ============================================================================
# Factory Function
# ============================================================================

def get_judge(mock: bool = None) -> ResponseJudge:
    """
    Get a judge instance

    Args:
        mock: Use mock judge (uses config.ENABLE_MOCK_MODE if None)

    Returns:
        ResponseJudge or MockJudge
    """
    if mock is None:
        mock = config.ENABLE_MOCK_MODE

    if mock:
        return MockJudge()
    else:
        return ResponseJudge()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Response Judge...")
    print()

    # Test mock judge
    print("=== Mock Judge ===")
    mock_judge = MockJudge()

    test_cases = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of France?", "The capital of France is Paris, a beautiful city."),
        ("Solve 2x + 5 = 13", "x = 4"),
        ("Solve 2x + 5 = 13", "To solve: 2x + 5 = 13\nSubtract 5: 2x = 8\nDivide by 2: x = 4"),
        ("Explain photosynthesis", "Photosynthesis is how plants make food using sunlight."),
    ]

    for query, response in test_cases:
        score, reasoning = mock_judge.evaluate(query, response)
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Score: {score:.2f}")
        print(f"Reasoning: {reasoning}")
        print()

    # Test real judge (only if API available)
    if config.OPENROUTER_API_KEY and not config.ENABLE_MOCK_MODE:
        print("=== Real Judge ===")
        try:
            from llm_client import OpenRouterClient
            client = OpenRouterClient()
            real_judge = ResponseJudge(llm_client=client)

            query = "What is the capital of France?"
            response = "Paris"

            print(f"Evaluating response...")
            score, reasoning = real_judge.evaluate(query, response)
            print(f"Query: {query}")
            print(f"Response: {response}")
            print(f"Score: {score:.2f}")
            print(f"Reasoning: {reasoning}")

        except Exception as e:
            print(f"⚠️  Real judge test failed: {e}")
    else:
        print("⚠️  Skipping real judge test (no API key or mock mode enabled)")
