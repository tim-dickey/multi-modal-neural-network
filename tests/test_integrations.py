"""Tests for API integration framework."""

import asyncio
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
import torch

from src.integrations.base import APIIntegration, APIResponse, KnowledgeInjector
from src.integrations.knowledge_injection import (
    AdditiveInjection,
    AttentionInjection,
    KnowledgeInjectionManager,
    MultiplicativeInjection,
    create_injection_strategy,
)
from src.integrations.validators import (
    ContentValidator,
    KnowledgeInjectionValidator,
    ResponseValidator,
    WolframResponseValidator,
    create_validator,
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

    def test_api_integration_defaults(self):
        """Test API integration default config values."""
        integration = MockAPIIntegration("test_key", {})
        assert integration.timeout == 30
        assert integration.max_retries == 3
        assert integration.retry_delay == 1.0
        assert integration.cache_dir == "./cache"

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

    def test_api_response_timestamp(self):
        """Test APIResponse auto-timestamp."""
        import time
        before = time.time()
        response = APIResponse(success=True, data="test")
        after = time.time()
        assert before <= response.timestamp <= after

    def test_make_request_with_retry_success(self):
        """Test retry logic with successful request."""
        integration = MockAPIIntegration()
        
        def successful_request():
            return APIResponse(success=True, data="result")
        
        result = integration._make_request_with_retry(successful_request)
        assert result.success is True
        assert result.data == "result"

    def test_make_request_with_retry_failure(self):
        """Test retry logic with all failures."""
        integration = MockAPIIntegration("key", {"max_retries": 2, "retry_delay": 0.01})
        
        call_count = [0]
        def failing_request():
            call_count[0] += 1
            raise Exception("Network error")
        
        result = integration._make_request_with_retry(failing_request)
        assert result.success is False
        assert "failed after" in result.error.lower()
        assert call_count[0] == 2


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
    def test_wolfram_query_not_success(self, mock_client_class):
        """Test Wolfram Alpha query when result.success is False."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.success = False
        mock_result.pods = []

        mock_client.query.return_value = mock_result
        mock_client_class.return_value = mock_client

        integration = WolframAlphaIntegration("test_key", {})
        response = integration.query("invalid query")

        assert response.success is False
        assert response.error == "Wolfram Alpha query failed"
        assert response.metadata["query"] == "invalid query"

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

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_extract_result_fallback(self, mock_client_class):
        """Test extraction falls back to first pod with plaintext."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        # Response with no "Result" or "Solution" pod, but has plaintext
        response = APIResponse(
            success=True,
            data=[
                {
                    "title": "Input",
                    "id": "input",
                    "subpods": [{"title": "", "plaintext": "original input", "img": None}],
                }
            ],
        )

        result = integration.extract_mathematical_result(response)
        assert result == "original input"

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_extract_result_no_plaintext(self, mock_client_class):
        """Test extraction when no pod has plaintext."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        # Response where no subpod has plaintext
        response = APIResponse(
            success=True,
            data=[
                {
                    "title": "Input",
                    "id": "input",
                    "subpods": [{"title": "", "plaintext": None, "img": "image.png"}],
                }
            ],
        )

        result = integration.extract_mathematical_result(response)
        assert result is None

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_response_not_success(self, mock_client_class):
        """Test validate_response returns False when not successful."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        response = APIResponse(success=False, data=None)
        assert integration.validate_response(response) is False

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_response_no_data(self, mock_client_class):
        """Test validate_response returns False when data is None."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        response = APIResponse(success=True, data=None)
        assert integration.validate_response(response) is False

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_response_not_list(self, mock_client_class):
        """Test validate_response returns False when data is not a list."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        response = APIResponse(success=True, data="not a list")
        assert integration.validate_response(response) is False

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_response_empty_list(self, mock_client_class):
        """Test validate_response returns False for empty list."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        response = APIResponse(success=True, data=[])
        assert integration.validate_response(response) is False

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_response_pod_not_dict(self, mock_client_class):
        """Test validate_response returns False when pod is not a dict."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        response = APIResponse(success=True, data=["not a dict"])
        assert integration.validate_response(response) is False

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_response_missing_title(self, mock_client_class):
        """Test validate_response returns False when pod missing title."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        response = APIResponse(success=True, data=[{"subpods": []}])
        assert integration.validate_response(response) is False

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_response_missing_subpods(self, mock_client_class):
        """Test validate_response returns False when pod missing subpods."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        response = APIResponse(success=True, data=[{"title": "Test"}])
        assert integration.validate_response(response) is False

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_response_subpods_not_list(self, mock_client_class):
        """Test validate_response returns False when subpods is not a list."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        response = APIResponse(success=True, data=[{"title": "Test", "subpods": "not a list"}])
        assert integration.validate_response(response) is False

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_query_exception(self, mock_client_class):
        """Test query returns error response when exception is raised."""
        mock_client = Mock()
        mock_client.query.side_effect = Exception("Network error")
        mock_client_class.return_value = mock_client

        integration = WolframAlphaIntegration("test_key", {})
        response = integration.query("test query")

        assert response.success is False
        assert "Wolfram Alpha API error" in response.error
        assert "Network error" in response.error
        assert response.metadata["query"] == "test query"

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_wolfram_validate_multiple_pods_mixed_validity(self, mock_client_class):
        """Test validate_response when first pod is valid but second is invalid."""
        mock_client_class.return_value = Mock()
        integration = WolframAlphaIntegration("test_key", {})

        # First pod valid, second pod invalid (missing title)
        response = APIResponse(
            success=True,
            data=[
                {"title": "Valid", "subpods": []},
                {"subpods": []},  # Missing title
            ]
        )
        assert integration.validate_response(response) is False


class TestValidators:
    """Tests for response validation utilities."""

    def test_response_validator_creation(self):
        """Test validator creation."""
        config = {"min_confidence": 0.8}
        validator = ResponseValidator(config)

        assert validator.config == config

    def test_response_validator_not_implemented(self):
        """Test base ResponseValidator.validate raises NotImplementedError."""
        validator = ResponseValidator({})
        with pytest.raises(NotImplementedError):
            validator.validate(APIResponse(success=True, data=None))

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

    def test_wolfram_validator_invalid_subpods(self):
        """Test Wolfram validator with invalid subpods."""
        validator = WolframResponseValidator()
        
        # subpods not a list
        response = APIResponse(
            success=True,
            data=[{"title": "Result", "id": "result", "subpods": "not_a_list"}]
        )
        assert validator.validate(response) is False
        
        # subpod not a dict
        response = APIResponse(
            success=True,
            data=[{"title": "Result", "id": "result", "subpods": ["string"]}]
        )
        assert validator.validate(response) is False

    def test_wolfram_validator_pod_not_dict(self):
        """Test Wolfram validator with pod not being a dict."""
        validator = WolframResponseValidator()
        response = APIResponse(success=True, data=["not_a_dict"])
        assert validator.validate(response) is False

    def test_content_validator(self):
        """Test content validation."""
        validator = ContentValidator({"min_confidence": 0.5})

        # Valid content
        assert validator.validate_content("This is a valid response.").valid is True

        # Invalid content
        invalid_contents = [
            "",  # Empty
            "x" * 10001,  # Too long (default max is 10000)
            None,  # Wrong type
        ]

        for invalid_content in invalid_contents:
            result = validator.validate_content(invalid_content)
            assert result.valid is False

    def test_content_validator_quality_score(self):
        """Test content quality score calculation."""
        validator = ContentValidator({"min_confidence": 0.3})
        
        # Content with letters, numbers, special chars, and structure
        good_content = "Hello World! 123\nAnother line."
        result = validator.validate_content(good_content)
        assert result.valid is True
        assert result.confidence >= 0.5
        assert result["has_numbers"] is True
        assert result["has_text"] is True
        
        # Very short content
        short_content = "ab"
        result = validator.validate_content(short_content)
        assert result.confidence < 0.5

    def test_content_validator_validation_result_attribute_access(self):
        """Test ValidationResult attribute access."""
        validator = ContentValidator({})
        result = validator.validate_content("test content")
        
        # Test dict-like access
        assert result["valid"] is not None
        
        # Test attribute access
        assert result.valid is not None
        assert result.confidence is not None
        
        # Test attribute error for missing key
        with pytest.raises(AttributeError):
            _ = result.nonexistent_key

    def test_create_validator_wolfram(self):
        """Test create_validator factory for wolfram."""
        validator = create_validator("wolfram", {})
        assert isinstance(validator, WolframResponseValidator)

    def test_create_validator_unknown(self):
        """Test create_validator factory for unknown API."""
        validator = create_validator("unknown_api", {"custom": "config"})
        assert isinstance(validator, ResponseValidator)
        assert validator.config == {"custom": "config"}

    def test_knowledge_injection_validator(self):
        """Test KnowledgeInjectionValidator."""
        validator = KnowledgeInjectionValidator({})
        
        # Valid injection data
        valid_data = {"injected": True, "injection_type": "additive"}
        result = validator.validate_injection(valid_data)
        assert result["valid"] is True
        
        # Not a dict
        result = validator.validate_injection("not_a_dict")
        assert result["valid"] is False
        
        # Missing required fields
        result = validator.validate_injection({"injected": True})
        assert result["valid"] is False

    def test_knowledge_injection_validator_with_validations(self):
        """Test KnowledgeInjectionValidator with validations field."""
        validator = KnowledgeInjectionValidator({"min_confidence": 0.5})
        
        # Valid with validations
        data = {
            "injected": True,
            "injection_type": "additive",
            "weight": 0.5,
            "validations": [
                {"expression": "2+2", "wolfram_result": "4 is a valid result."}
            ]
        }
        result = validator.validate_injection(data)
        assert result["valid"] is True
        assert result["weight"] == 0.5

    def test_knowledge_injection_validator_invalid_content(self):
        """Test KnowledgeInjectionValidator with invalid validation content."""
        validator = KnowledgeInjectionValidator({"max_length": 5})
        
        data = {
            "injected": True,
            "injection_type": "additive",
            "validations": [
                {"expression": "x", "wolfram_result": "this is too long"}
            ]
        }
        result = validator.validate_injection(data)
        assert result["valid"] is False


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

    def test_additive_injection_non_tensor(self):
        """Test additive injection with non-tensor knowledge."""
        strategy = AdditiveInjection(weight=0.5)
        model_output = torch.randn(2, 10, 256)
        
        # Non-tensor knowledge should return model_output unchanged
        result = strategy.inject(model_output, "not_a_tensor")
        assert torch.allclose(result, model_output)

    def test_multiplicative_injection(self):
        """Test multiplicative knowledge injection."""
        strategy = MultiplicativeInjection(weight=0.1)

        model_output = torch.randn(2, 10, 256)

        result = strategy.inject(model_output, 2.0)

        # Should be scaled output
        expected = model_output * 1.1  # (1 + 0.1)
        assert torch.allclose(result, expected)

    def test_multiplicative_injection_tensor(self):
        """Test multiplicative injection with tensor knowledge."""
        strategy = MultiplicativeInjection(weight=0.5)
        model_output = torch.ones(2, 10)
        knowledge = torch.ones(2, 10) * 2.0
        
        result = strategy.inject(model_output, knowledge)
        expected = model_output * (1 + 0.5 * knowledge)
        assert torch.allclose(result, expected)

    def test_multiplicative_injection_non_numeric(self):
        """Test multiplicative injection with non-numeric knowledge."""
        strategy = MultiplicativeInjection(weight=0.1)
        model_output = torch.randn(2, 10)
        
        result = strategy.inject(model_output, "not_numeric")
        assert torch.allclose(result, model_output)

    def test_attention_injection(self):
        """Test attention-based knowledge injection."""
        strategy = AttentionInjection(hidden_dim=256, weight=0.5)

        model_output = torch.randn(2, 10, 256)
        knowledge = torch.randn(2, 10, 256)

        result = strategy.inject(model_output, knowledge)

        # Should have same shape as input
        assert result.shape == model_output.shape

    def test_attention_injection_non_tensor(self):
        """Test attention injection with non-tensor knowledge."""
        strategy = AttentionInjection(hidden_dim=256, weight=0.5)
        model_output = torch.randn(2, 10, 256)
        
        result = strategy.inject(model_output, "not_a_tensor")
        assert torch.allclose(result, model_output)

    def test_create_injection_strategy_additive(self):
        """Test factory for additive strategy."""
        strategy = create_injection_strategy("additive", weight=0.3)
        assert isinstance(strategy, AdditiveInjection)
        assert strategy.weight == 0.3

    def test_create_injection_strategy_multiplicative(self):
        """Test factory for multiplicative strategy."""
        strategy = create_injection_strategy("multiplicative", weight=0.2)
        assert isinstance(strategy, MultiplicativeInjection)

    def test_create_injection_strategy_attention(self):
        """Test factory for attention strategy."""
        strategy = create_injection_strategy("attention", hidden_dim=128, weight=0.1)
        assert isinstance(strategy, AttentionInjection)

    def test_create_injection_strategy_unknown(self):
        """Test factory for unknown strategy defaults to additive."""
        strategy = create_injection_strategy("unknown_strategy", weight=0.4)
        assert isinstance(strategy, AdditiveInjection)

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

    def test_injection_manager_no_injectors(self):
        """Test injection manager with no registered injectors."""
        manager = KnowledgeInjectionManager({})
        
        result = manager.inject_knowledge({}, torch.randn(1, 256))
        assert result["success"] is False
        assert "No knowledge injectors" in result["error"]

    def test_injection_manager_injection_not_triggered(self):
        """Test injection manager when injection is not triggered."""
        manager = KnowledgeInjectionManager({})
        
        mock_injector = Mock()
        mock_injector.inject_knowledge.return_value = {"injected": False, "reason": "low confidence"}
        manager.register_injector("mock", mock_injector)
        
        result = manager.inject_knowledge({}, torch.randn(1, 256), "mock")
        assert result["success"] is False
        assert "low confidence" in result["reason"]

    def test_injection_manager_strategy_exception(self):
        """Test injection manager handles strategy exceptions."""
        manager = KnowledgeInjectionManager({})
        
        mock_injector = Mock()
        mock_injector.inject_knowledge.return_value = {"injected": True}
        manager.register_injector("mock", mock_injector)
        
        failing_strategy = Mock()
        failing_strategy.inject.side_effect = RuntimeError("Strategy failed")
        manager.register_strategy("failing", failing_strategy)
        
        result = manager.inject_knowledge({}, torch.randn(1, 256), "mock", "failing")
        assert result["success"] is False
        assert "failed" in result["error"].lower()

    def test_injection_manager_get_available(self):
        """Test getting available injectors and strategies."""
        manager = KnowledgeInjectionManager({})
        manager.register_injector("inj1", Mock())
        manager.register_injector("inj2", Mock())
        manager.register_strategy("strat1", Mock())
        
        assert manager.get_available_injectors() == ["inj1", "inj2"]
        assert manager.get_available_strategies() == ["strat1"]

    def test_injection_manager_uses_first_injector(self):
        """Test manager uses first injector when none specified."""
        manager = KnowledgeInjectionManager({})
        
        mock_injector = Mock()
        mock_injector.inject_knowledge.return_value = {"injected": True}
        mock_injector.__class__.__name__ = "MockInjector"
        manager.register_injector("first", mock_injector)
        
        result = manager.inject_knowledge({}, torch.randn(1, 256))
        assert result["success"] is True
        mock_injector.inject_knowledge.assert_called_once()


class TestKnowledgeInjectorBase:
    """Tests for KnowledgeInjector base class from base.py."""

    def test_knowledge_injector_should_inject_low_confidence(self):
        """Test should_inject returns True for low confidence."""
        injector = KnowledgeInjector(Mock(), {"validation_threshold": 0.5})
        # Low confidence should trigger injection
        assert injector.should_inject(0.3) is True
        assert injector.should_inject(0.1) is True

    def test_knowledge_injector_should_inject_high_confidence(self):
        """Test should_inject returns False for high confidence."""
        injector = KnowledgeInjector(Mock(), {"validation_threshold": 0.5})
        # High confidence should not trigger injection
        assert injector.should_inject(0.7) is False
        assert injector.should_inject(0.9) is False

    def test_knowledge_injector_should_inject_at_threshold(self):
        """Test should_inject at threshold boundary."""
        injector = KnowledgeInjector(Mock(), {"validation_threshold": 0.5})
        # At threshold should not inject (uses <, not <=)
        assert injector.should_inject(0.5) is False

    def test_knowledge_injector_inject_knowledge_not_implemented(self):
        """Test inject_knowledge raises NotImplementedError."""
        injector = KnowledgeInjector(Mock(), {})
        
        with pytest.raises(NotImplementedError):
            injector.inject_knowledge("input", torch.randn(1, 256))

    def test_knowledge_injector_stores_api_and_config(self):
        """Test injector stores api and config."""
        mock_api = Mock()
        config = {"key": "value", "validation_threshold": 0.3}
        
        injector = KnowledgeInjector(mock_api, config)
        
        assert injector.api is mock_api
        assert injector.config == config
        assert injector.validation_threshold == 0.3


class TestAPIResponseDataclass:
    """Tests for APIResponse dataclass."""

    def test_api_response_creation(self):
        """Test creating APIResponse."""
        response = APIResponse(success=True, data={"result": 42})
        assert response.success is True
        assert response.data == {"result": 42}
        assert response.error is None
        assert response.metadata is None
        assert response.timestamp is not None  # Auto-filled

    def test_api_response_with_error(self):
        """Test APIResponse with error."""
        response = APIResponse(
            success=False,
            data=None,
            error="Connection failed",
            metadata={"status": 500}
        )
        assert response.success is False
        assert response.data is None
        assert response.error == "Connection failed"
        assert response.metadata == {"status": 500}

    def test_api_response_timestamp_auto_fill(self):
        """Test APIResponse timestamp is auto-filled."""
        import time
        before = time.time()
        response = APIResponse(success=True, data="test")
        after = time.time()
        
        assert before <= response.timestamp <= after


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

    def test_math_expression_extraction_non_string(self):
        """Test _extract_math_expressions with non-string input."""
        injector = WolframKnowledgeInjector(Mock(), {})

        # Non-string input should return empty list
        assert injector._extract_math_expressions(123) == []
        assert injector._extract_math_expressions(None) == []
        assert injector._extract_math_expressions(["list"]) == []

    @patch("src.integrations.wolfram_alpha.wolframalpha.Client")
    def test_inject_knowledge_no_math(self, mock_client_class):
        """Test inject_knowledge when no math expressions are found."""
        mock_client_class.return_value = Mock()

        wolfram = WolframAlphaIntegration("test_key", {})
        injector = WolframKnowledgeInjector(wolfram, {})

        # Input with no math
        input_data = "Hello world, no math here"
        model_output = torch.randn(1, 256)

        result = injector.inject_knowledge(input_data, model_output)

        assert result["injected"] is False
        assert "No mathematical expressions found" in result["reason"]


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
