"""Tests for API integration framework."""

import asyncio
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
import torch

from src.integrations.base import APIIntegration, APIResponse
from src.integrations.knowledge_injection import (
    AdditiveInjection,
    AttentionInjection,
    KnowledgeInjectionManager,
    MultiplicativeInjection,
)
from src.integrations.validators import (
    ContentValidator,
    ResponseValidator,
    WolframResponseValidator,
)
from src.integrations.wolfram_alpha import (
    WolframAlphaIntegration,
    WolframKnowledgeInjector,
)


class MockAPIIntegration(APIIntegration):
    """Mock API integration for testing."""

    def __init__(self, api_key: str = "test_key", config: Dict[str, Any] = None):
        super().__init__(api_key, config or {})
        self.call_count = 0
        self.last_prompt = None

    def query(self, prompt: str, **kwargs) -> APIResponse:
        """Mock query implementation."""
        self.call_count += 1
        self.last_prompt = prompt

        # Return different responses based on prompt
        if "error" in prompt.lower():
            return APIResponse(success=False, data=None, error="Mock API error")
        elif "math" in prompt.lower():
            return APIResponse(
                success=True,
                data=[
                    {
                        "title": "Result",
                        "id": "result",
                        "subpods": [{"title": "", "plaintext": "42", "img": None}],
                    }
                ],
                metadata={"query": prompt},
            )
        else:
            return APIResponse(
                success=True,
                data=[
                    {
                        "title": "Response",
                        "id": "response",
                        "subpods": [
                            {"title": "", "plaintext": "Mock response", "img": None}
                        ],
                    }
                ],
                metadata={"query": prompt},
            )

    def validate_response(self, response: APIResponse) -> bool:
        """Mock validation."""
        return response.success and response.data is not None


class TestAPIIntegration:
    """Tests for base API integration functionality."""

    def test_api_integration_creation(self):
        """Test API integration can be created."""
        config = {"timeout": 30, "max_retries": 3}
        integration = MockAPIIntegration("test_key", config)

        assert integration.api_key == "test_key"
        assert integration.config == config
        assert integration.timeout == 30
        assert integration.max_retries == 3

    def test_api_integration_query(self):
        """Test basic query functionality."""
        integration = MockAPIIntegration()

        response = integration.query("test prompt")

        assert response.success is True
        assert response.data is not None
        assert integration.call_count == 1
        assert integration.last_prompt == "test prompt"

    def test_api_integration_error_handling(self):
        """Test error handling in queries."""
        integration = MockAPIIntegration()

        response = integration.query("error prompt")

        assert response.success is False
        assert response.error == "Mock API error"
        assert response.data is None

    def test_api_integration_validation(self):
        """Test response validation."""
        integration = MockAPIIntegration()

        valid_response = integration.query("test")
        invalid_response = APIResponse(success=False, data=None, error="test error")

        assert integration.validate_response(valid_response) is True
        assert integration.validate_response(invalid_response) is False


class TestWolframAlphaIntegration:
    """Tests for Wolfram Alpha integration."""

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_creation(self, mock_client_class):
        """Test Wolfram Alpha integration creation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = {
            "max_queries_per_day": 1000,
            "cache_dir": "./test_cache",
            "timeout": 30,
        }

        integration = WolframAlphaIntegration("test_key", config)

        assert integration.api_key == "test_key"
        assert integration.max_queries_per_day == 1000
        assert integration.daily_queries == 0
        mock_client_class.assert_called_once_with("test_key")

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_query_success(self, mock_client_class):
        """Test successful Wolfram Alpha query."""
        # Mock the Wolfram Alpha client and result
        mock_client = Mock()
        mock_result = Mock()
        mock_pod = Mock()
        mock_subpod = Mock()

        mock_pod.title = "Result"
        mock_pod.id = "result"
        mock_subpod.title = ""
        mock_subpod.plaintext = "42"
        mock_subpod.img.src = None

        mock_pod.subpods = [mock_subpod]
        mock_result.pods = [mock_pod]
        mock_result.success = True

        mock_client.query.return_value = mock_result
        mock_client_class.return_value = mock_client

        integration = WolframAlphaIntegration("test_key", {})
        response = integration.query("2 + 2")

        assert response.success is True
        assert len(response.data) == 1
        assert response.data[0]["title"] == "Result"
        assert response.data[0]["subpods"][0]["plaintext"] == "42"
        assert integration.daily_queries == 1

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_query_limit_exceeded(self, mock_client_class):
        """Test query limit enforcement."""
        mock_client_class.return_value = Mock()

        config = {"max_queries_per_day": 0}  # No queries allowed
        integration = WolframAlphaIntegration("test_key", config)

        response = integration.query("test")

        assert response.success is False
        assert "limit exceeded" in response.error.lower()

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_extract_result(self, mock_client_class):
        """Test mathematical result extraction."""
        mock_client_class.return_value = Mock()

        integration = WolframAlphaIntegration("test_key", {})

        # Test with result pod
        response = APIResponse(
            success=True,
            data=[
                {
                    "title": "Result",
                    "id": "result",
                    "subpods": [{"title": "", "plaintext": "3.14159", "img": None}],
                }
            ],
        )

        result = integration.extract_mathematical_result(response)
        assert result == "3.14159"

        # Test with no result
        empty_response = APIResponse(success=True, data=[])
        result = integration.extract_mathematical_result(empty_response)
        assert result is None


class TestValidators:
    """Tests for response validation utilities."""

    def test_response_validator_creation(self):
        """Test validator creation."""
        config = {"min_confidence": 0.8}
        validator = ResponseValidator(config)

        assert validator.config == config

    def test_wolfram_validator(self):
        """Test Wolfram response validation."""
        validator = WolframResponseValidator()

        # Valid response
        valid_response = APIResponse(
            success=True,
            data=[
                {
                    "title": "Result",
                    "id": "result",
                    "subpods": [{"title": "", "plaintext": "42", "img": None}],
                }
            ],
        )
        assert validator.validate(valid_response) is True

        # Invalid responses
        invalid_responses = [
            APIResponse(success=False, data=None),  # Not successful
            APIResponse(success=True, data="not_a_list"),  # Wrong data type
            APIResponse(success=True, data=[]),  # Empty data
            APIResponse(success=True, data=[{"missing": "fields"}]),  # Missing fields
        ]

        for invalid_response in invalid_responses:
            assert validator.validate(invalid_response) is False

    def test_content_validator(self):
        """Test content validation."""
        validator = ContentValidator({"min_confidence": 0.5})

        # Valid content
        assert validator.validate_content("This is a valid response.").valid is True

        # Invalid content
        invalid_contents = [
            "",  # Empty
            "x" * 10000,  # Too long
            None,  # Wrong type
        ]

        for invalid_content in invalid_contents:
            result = validator.validate_content(invalid_content)
            assert result.valid is False


class TestKnowledgeInjection:
    """Tests for knowledge injection system."""

    def test_additive_injection(self):
        """Test additive knowledge injection."""
        strategy = AdditiveInjection(weight=0.5)

        model_output = torch.randn(2, 10, 256)
        knowledge = torch.randn(2, 10, 256)

        result = strategy.inject(model_output, knowledge)

        # Should be weighted combination
        expected = model_output + 0.5 * knowledge
        assert torch.allclose(result, expected)

    def test_multiplicative_injection(self):
        """Test multiplicative knowledge injection."""
        strategy = MultiplicativeInjection(weight=0.1)

        model_output = torch.randn(2, 10, 256)

        result = strategy.inject(model_output, 2.0)

        # Should be scaled output
        expected = model_output * 1.1  # (1 + 0.1 * 2.0)
        assert torch.allclose(result, expected)

    def test_attention_injection(self):
        """Test attention-based knowledge injection."""
        strategy = AttentionInjection(hidden_dim=256, weight=0.5)

        model_output = torch.randn(2, 10, 256)
        knowledge = torch.randn(2, 10, 256)

        result = strategy.inject(model_output, knowledge)

        # Should have same shape as input
        assert result.shape == model_output.shape

    def test_injection_manager(self):
        """Test knowledge injection manager."""
        config = {"default_injection_weight": 0.1}
        manager = KnowledgeInjectionManager(config)

        # Register components
        mock_injector = Mock()
        mock_injector.inject_knowledge.return_value = {
            "injected": True,
            "validations": [{"expression": "2+2", "wolfram_result": "4"}],
        }

        manager.register_injector("mock", mock_injector)
        manager.register_strategy("additive", AdditiveInjection())

        # Test injection
        input_data = {"text": "What is 2+2?"}
        model_output = torch.randn(1, 256)

        result = manager.inject_knowledge(input_data, model_output, "mock", "additive")

        assert result["success"] is True
        assert "modified_output" in result
        mock_injector.inject_knowledge.assert_called_once()


class TestWolframKnowledgeInjector:
    """Tests for Wolfram knowledge injector."""

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_knowledge_injection(self, mock_client_class):
        """Test knowledge injection with Wolfram Alpha."""
        mock_client_class.return_value = Mock()

        wolfram = WolframAlphaIntegration("test_key", {})
        injector = WolframKnowledgeInjector(wolfram, {"injection_weight": 0.2})

        # Test with mathematical content
        input_data = "Calculate 3 * 4"
        model_output = torch.randn(1, 256)

        # Mock Wolfram response
        with patch.object(wolfram, "query") as mock_query:
            mock_query.return_value = APIResponse(
                success=True,
                data=[
                    {"title": "Result", "subpods": [{"plaintext": "12", "img": None}]}
                ],
            )

            result = injector.inject_knowledge(input_data, model_output)

            assert result["injected"] is True
            assert len(result["validations"]) > 0
            assert result["validations"][0]["expression"] == "3 * 4"

    def test_math_expression_extraction(self):
        """Test mathematical expression extraction."""
        injector = WolframKnowledgeInjector(Mock(), {})

        test_cases = [
            ("What is 2 + 2?", ["2 + 2"]),
            ("Calculate 3*4 and 5-2", ["3*4", "5-2"]),
            ("Solve x^2 = 4", ["x^2 = 4"]),
            ("No math here", []),
        ]

        for text, expected in test_cases:
            result = injector._extract_math_expressions(text)
            assert result == expected


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Tests for async API integration capabilities."""

    async def test_async_query_simulation(self):
        """Test async query simulation."""
        # This would be expanded for actual async API integrations
        integration = MockAPIIntegration()

        # Simulate async behavior

        async def async_query():
            await asyncio.sleep(0.01)  # Simulate network delay
            return integration.query("async test")

        result = await async_query()

        assert result.success is True
        assert integration.call_count == 1


# Performance and benchmarking tests
@pytest.mark.slow
@pytest.mark.benchmark
class TestPerformance:
    """Performance tests for API integrations."""

    def test_query_performance(self, benchmark):
        """Benchmark query performance."""
        integration = MockAPIIntegration()

        def run_query():
            return integration.query("performance test")

        result = benchmark(run_query)

        assert result.success is True

    def test_batch_query_performance(self, benchmark):
        """Benchmark batch query performance."""
        integration = MockAPIIntegration()

        def run_batch_queries():
            queries = [f"query {i}" for i in range(10)]
            return [integration.query(q) for q in queries]

        results = benchmark(run_batch_queries)

        assert len(results) == 10
        assert all(r.success for r in results)
