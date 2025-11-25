"""Base classes and utilities for API integrations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Standardized API response structure."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class APIIntegration(ABC):
    """Base class for API integrations."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Common config defaults
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.cache_dir = config.get('cache_dir', './cache')

    @abstractmethod
    def query(self, prompt: str, **kwargs) -> APIResponse:
        """Execute a query against the API.

        Args:
            prompt: The query string
            **kwargs: Additional parameters

        Returns:
            APIResponse with the query result
        """
        raise NotImplementedError

    @abstractmethod
    def validate_response(self, response: APIResponse) -> bool:
        """Validate that the API response is well-formed.

        Args:
            response: The API response to validate

        Returns:
            True if response is valid, False otherwise
        """
        raise NotImplementedError

    def _make_request_with_retry(self, request_func, *args, **kwargs) -> APIResponse:
        """Make a request with automatic retry logic.

        Args:
            request_func: Function that makes the actual request
            *args: Positional arguments for request_func
            **kwargs: Keyword arguments for request_func

        Returns:
            APIResponse from the request
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return request_func(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                last_error = str(e)
                self.logger.warning("Request attempt %d failed: %s", attempt + 1, e)

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        return APIResponse(
            success=False,
            data=None,
            error=f"Request failed after {self.max_retries} attempts: {last_error}"
        )


class KnowledgeInjector:
    """Base class for injecting external knowledge into the model."""

    def __init__(self, api_integration: APIIntegration, config: Dict[str, Any]):
        self.api = api_integration
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.injection_weight = config.get('injection_weight', 0.1)
        self.validation_threshold = config.get('validation_threshold', 0.8)

    @abstractmethod
    def inject_knowledge(self, input_data: Any, model_output: Any) -> Dict[str, Any]:
        """Inject external knowledge based on input and model output.

        Args:
            input_data: The original input to the model
            model_output: The model's current output

        Returns:
            Dictionary containing injection data and metadata
        """
        raise NotImplementedError

    def should_inject(self, confidence_score: float) -> bool:
        """Determine if knowledge injection should occur based on confidence.

        Args:
            confidence_score: Model's confidence in its prediction

        Returns:
            True if injection should occur
        """
        return confidence_score < self.validation_threshold