"""Mocking utilities for testing external API integrations."""

import json
from typing import Any, Callable, Dict, Optional
from unittest.mock import Mock, patch

import pytest

# Optional imports for HTTP mocking
try:
    import responses  # pylint: disable=import-error

    HAS_RESPONSES = True
except ImportError:
    HAS_RESPONSES = False
    responses = None  # type: ignore

try:
    import httpx  # pylint: disable=import-error

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None  # type: ignore

try:
    from aiohttp import web  # pylint: disable=import-error

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    web = None  # type: ignore


class MockAPIClient:
    """Base class for mocking API clients."""

    def __init__(self, base_url: str = "https://api.mock.com"):
        self.base_url = base_url
        self.call_history = []

    def mock_success_response(self, data: Any = None, status_code: int = 200) -> Mock:
        """Create a mock successful response."""
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data or {"success": True}
        response.text = json.dumps(data or {"success": True})
        response.raise_for_status.return_value = None
        return response

    def mock_error_response(
        self, error_message: str = "API Error", status_code: int = 400
    ) -> Mock:
        """Create a mock error response."""
        response = Mock()
        response.status_code = status_code
        response.json.side_effect = ValueError("Invalid JSON")
        response.text = error_message
        response.raise_for_status.side_effect = Exception(
            f"HTTP {status_code}: {error_message}"
        )
        return response

    def record_call(self, method: str, endpoint: str, **kwargs):
        """Record an API call for verification."""
        self.call_history.append(
            {"method": method, "endpoint": endpoint, "kwargs": kwargs}
        )


class MockWolframClient(MockAPIClient):
    """Mock client for Wolfram Alpha API."""

    def __init__(self):
        super().__init__("https://api.wolframalpha.com")
        self.mock_responses = {
            "2+2": self._create_wolfram_result("4"),
            "pi": self._create_wolfram_result("3.14159"),
            "error": self._create_error_result(),
        }

    def _create_wolfram_result(self, result: str) -> Mock:
        """Create a mock Wolfram Alpha result."""
        mock_result = Mock()
        mock_result.success = True

        mock_pod = Mock()
        mock_pod.title = "Result"
        mock_pod.id = "result"

        mock_subpod = Mock()
        mock_subpod.title = ""
        mock_subpod.plaintext = result
        mock_subpod.img.src = None

        mock_pod.subpods = [mock_subpod]
        mock_result.pods = [mock_pod]

        return mock_result

    def _create_error_result(self) -> Mock:
        """Create a mock error result."""
        mock_result = Mock()
        mock_result.success = False
        mock_result.pods = []
        return mock_result

    def query(self, query: str) -> Mock:
        """Mock query method."""
        self.record_call("GET", "/v2/query", query=query)
        return self.mock_responses.get(
            query, self._create_wolfram_result("Mock result")
        )


class MockOpenAIClient(MockAPIClient):
    """Mock client for OpenAI API."""

    def __init__(self):
        super().__init__("https://api.openai.com")
        self.mock_responses = {
            "simple": "This is a mock response from GPT.",
            "math": "The answer to 2+2 is 4.",
            "error": None,  # Will raise exception
        }

    def create_completion(self, **kwargs) -> Mock:
        """Mock completion creation."""
        prompt = kwargs.get("prompt", "")
        self.record_call("POST", "/v1/completions", **kwargs)

        if "error" in prompt.lower():
            raise Exception("OpenAI API Error")

        response = Mock()
        response.choices = [Mock()]
        response.choices[0].text = self.mock_responses.get(
            "simple" if "simple" in prompt else "math", "Mock completion response"
        )
        return response

    def create_chat_completion(self, **kwargs) -> Mock:
        """Mock chat completion creation."""
        messages = kwargs.get("messages", [])
        self.record_call("POST", "/v1/chat/completions", **kwargs)

        # Extract user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if "error" in user_message.lower():
            raise Exception("OpenAI Chat API Error")

        response = Mock()
        response.choices = [Mock()]
        mock_message = Mock()
        mock_message.content = self.mock_responses.get(
            "simple" if "simple" in user_message else "math", "Mock chat response"
        )
        response.choices[0].message = mock_message
        return response


class MockGoogleAIClient(MockAPIClient):
    """Mock client for Google AI (PaLM) API."""

    def __init__(self):
        super().__init__("https://generativelanguage.googleapis.com")
        self.mock_responses = {
            "generate": "This is a mock response from PaLM.",
            "math": "The calculation result is 42.",
        }

    def generate_text(self, **kwargs) -> Mock:
        """Mock text generation."""
        prompt = kwargs.get("prompt", "")
        self.record_call("POST", "/v1beta/models/text-bison-001:generateText", **kwargs)

        if "error" in prompt.lower():
            raise Exception("Google AI API Error")

        response = Mock()
        response.candidates = [Mock()]
        response.candidates[0].output = self.mock_responses.get(
            "generate" if "generate" in prompt else "math", "Mock PaLM response"
        )
        return response


class MockAnthropicClient(MockAPIClient):
    """Mock client for Anthropic Claude API."""

    def __init__(self):
        super().__init__("https://api.anthropic.com")
        self.mock_responses = {
            "simple": "This is a mock response from Claude.",
            "reasoning": "After careful analysis, the answer is 42.",
        }

    def messages_create(self, **kwargs) -> Mock:
        """Mock message creation."""
        messages = kwargs.get("messages", [])
        self.record_call("POST", "/v1/messages", **kwargs)

        # Extract human message
        human_message = ""
        for msg in messages:
            if msg.get("role") == "human" or msg.get("role") == "user":
                human_message = msg.get("content", "")
                break

        if "error" in human_message.lower():
            raise Exception("Anthropic API Error")

        response = Mock()
        response.content = [Mock()]
        response.content[0].text = self.mock_responses.get(
            "simple" if "simple" in human_message else "reasoning",
            "Mock Claude response",
        )
        return response


# Context managers for mocking
class mock_wolfram_alpha:
    """Context manager for mocking Wolfram Alpha."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.mock_client = MockWolframClient()

    def __enter__(self):
        self.patcher = patch("wolframalpha.Client")
        mock_class = self.patcher.start()
        mock_class.return_value = self.mock_client
        return self.mock_client

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()


class mock_openai:
    """Context manager for mocking OpenAI."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.mock_client = MockOpenAIClient()

    def __enter__(self):
        self.patcher = patch("openai.OpenAI")
        mock_class = self.patcher.start()
        mock_class.return_value = self.mock_client
        return self.mock_client

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()


class mock_google_ai:
    """Context manager for mocking Google AI."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.mock_client = MockGoogleAIClient()

    def __enter__(self):
        self.patcher = patch("google.generativeai.GenerativeModel")
        mock_class = self.patcher.start()
        mock_class.return_value = self.mock_client
        return self.mock_client

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()


class mock_anthropic:
    """Context manager for mocking Anthropic."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.mock_client = MockAnthropicClient()

    def __enter__(self):
        self.patcher = patch("anthropic.Anthropic")
        mock_class = self.patcher.start()
        mock_class.return_value = self.mock_client
        return self.mock_client

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()


# HTTP mocking utilities
class MockHTTPServer:
    """Mock HTTP server for testing API integrations."""

    def __init__(self, port: int = 8080):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for MockHTTPServer")
        self.port = port
        self.app = web.Application()  # type: ignore
        self.runner = None
        self.site = None

    async def start_server(self):
        """Start the mock server."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for MockHTTPServer")
        self.runner = web.AppRunner(self.app)  # type: ignore
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "localhost", self.port)  # type: ignore
        await self.site.start()

    async def stop_server(self):
        """Stop the mock server."""
        if not HAS_AIOHTTP:
            return
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    def add_route(self, method: str, path: str, handler: Callable):
        """Add a route to the mock server."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for MockHTTPServer")
        self.app.router.add_route(method, path, handler)


# Factory functions for creating mock clients
def create_mock_wolfram_client(**kwargs) -> MockWolframClient:
    """Create a mock Wolfram Alpha client."""
    return MockWolframClient()


def create_mock_openai_client(**kwargs) -> MockOpenAIClient:
    """Create a mock OpenAI client."""
    return MockOpenAIClient()


def create_mock_google_client(**kwargs) -> MockGoogleAIClient:
    """Create a mock Google AI client."""
    return MockGoogleAIClient()


def create_mock_anthropic_client(**kwargs) -> MockAnthropicClient:
    """Create a mock Anthropic client."""
    return MockAnthropicClient()


# Pytest fixtures for mocking
@pytest.fixture
def mock_wolfram_client():
    """Pytest fixture for mock Wolfram client."""
    return create_mock_wolfram_client()


@pytest.fixture
def mock_openai_client():
    """Pytest fixture for mock OpenAI client."""
    return create_mock_openai_client()


@pytest.fixture
def mock_google_client():
    """Pytest fixture for mock Google AI client."""
    return create_mock_google_client()


@pytest.fixture
def mock_anthropic_client():
    """Pytest fixture for mock Anthropic client."""
    return create_mock_anthropic_client()


# Utility functions for test data
def create_mock_api_response(
    success: bool = True, data: Any = None, error: Optional[str] = None
):
    """Create a mock API response."""
    from src.integrations.base import APIResponse

    return APIResponse(success=success, data=data, error=error)


def create_mock_wolfram_pod(title: str, result: str) -> Mock:
    """Create a mock Wolfram Alpha pod."""
    pod = Mock()
    pod.title = title
    pod.id = title.lower().replace(" ", "_")

    subpod = Mock()
    subpod.title = ""
    subpod.plaintext = result
    subpod.img = None

    pod.subpods = [subpod]
    return pod
