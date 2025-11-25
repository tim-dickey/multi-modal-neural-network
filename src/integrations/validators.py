"""Validation utilities for API responses and knowledge injection."""

from typing import Any, Dict, Optional
from collections import UserDict
import re
from .base import APIResponse


class ResponseValidator:
    """Base class for validating API responses."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def validate(self, response: APIResponse) -> bool:
        """Validate an API response.

        Args:
            response: The response to validate

        Returns:
            True if response is valid
        """
        raise NotImplementedError


class WolframResponseValidator(ResponseValidator):
    """Validator for Wolfram Alpha responses."""

    def validate(self, response: APIResponse) -> bool:
        """Validate Wolfram Alpha response structure.

        Args:
            response: Wolfram Alpha response

        Returns:
            True if response is valid
        """
        if not response.success:
            return False

        if not isinstance(response.data, list):
            return False

        if len(response.data) == 0:
            return False

        # Check pod structure
        for pod in response.data:
            if not isinstance(pod, dict):
                return False

            required_keys = ["title", "id", "subpods"]
            if not all(key in pod for key in required_keys):
                return False

            if not isinstance(pod["subpods"], list):
                return False

            # Check subpod structure
            for subpod in pod["subpods"]:
                if not isinstance(subpod, dict):
                    return False

        return True


class ContentValidator:
    """Validator for response content quality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.max_length = self.config.get('max_length', 10000)

    class ValidationResult(UserDict):
        """Dict-like result that also allows attribute access (e.g., .valid)."""

        def __getattr__(self, name: str) -> Any:
            try:
                return self.data[name]
            except KeyError as e:
                raise AttributeError(name) from e

    def validate_content(self, content: str) -> "ContentValidator.ValidationResult":
        """Validate content quality and extract metadata.

        Args:
            content: Content to validate

        Returns:
            Dictionary with validation results and metadata
        """
        if not content or not isinstance(content, str):
            return self.ValidationResult({
                "valid": False,
                "reason": "Empty or invalid content",
                "confidence": 0.0
            })

        if len(content) > self.max_length:
            return self.ValidationResult({
                "valid": False,
                "reason": f"Content too long ({len(content)} > {self.max_length})",
                "confidence": 0.0
            })

        # Basic quality checks
        quality_score = self._calculate_quality_score(content)

        return self.ValidationResult({
            "valid": quality_score >= self.min_confidence,
            "confidence": quality_score,
            "length": len(content),
            "has_numbers": bool(re.search(r'\d', content)),
            "has_text": len(content.strip()) > 0
        })

    def _calculate_quality_score(self, content: str) -> float:
        """Calculate a quality score for the content.

        Args:
            content: Content to score

        Returns:
            Quality score between 0 and 1
        """
        score = 0.0

        # Length check (prefer substantial content)
        if 10 <= len(content) <= 5000:
            score += 0.3
        elif len(content) < 10:
            score += 0.1

        # Content diversity
        if re.search(r'[a-zA-Z]', content):  # Has letters
            score += 0.2
        if re.search(r'\d', content):  # Has numbers
            score += 0.2
        if re.search(r'[^\w\s]', content):  # Has special characters
            score += 0.1

        # Structure check
        if '\n' in content or '.' in content:
            score += 0.2

        return min(score, 1.0)


class KnowledgeInjectionValidator:
    """Validator for knowledge injection operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.content_validator = ContentValidator(self.config)

    def validate_injection(self, injection_data: Any) -> Dict[str, Any]:
        """Validate knowledge injection data.

        Args:
            injection_data: Injection data to validate

        Returns:
            Validation results
        """
        if not isinstance(injection_data, dict):
            return {
                "valid": False,
                "reason": "Injection data must be a dictionary"
            }

        # Check required fields
        required_fields = ["injected", "injection_type"]
        missing_fields = [field for field in required_fields if field not in injection_data]

        if missing_fields:
            return {
                "valid": False,
                "reason": f"Missing required fields: {missing_fields}"
            }

        # Validate injection content if present
        if "validations" in injection_data:
            validations = injection_data["validations"]
            if isinstance(validations, list):
                for i, validation in enumerate(validations):
                    if isinstance(validation, dict) and "wolfram_result" in validation:
                        result = validation["wolfram_result"]
                        if result:
                            content_validation = self.content_validator.validate_content(result)
                            if not content_validation["valid"]:
                                return {
                                    "valid": False,
                                    "reason": f"Invalid validation content at index {i}: {content_validation['reason']}"
                                }

        return {
            "valid": True,
            "injection_type": injection_data.get("injection_type"),
            "weight": injection_data.get("weight", 0.0)
        }


def create_validator(api_name: str, config: Optional[Dict[str, Any]] = None) -> ResponseValidator:
    """Factory function to create appropriate validator for an API.

    Args:
        api_name: Name of the API
        config: Configuration for the validator

    Returns:
        Appropriate validator instance
    """
    validators = {
        "wolfram": WolframResponseValidator,
    }

    validator_class = validators.get(api_name.lower())
    if validator_class:
        return validator_class(config)

    # Default validator
    return ResponseValidator(config)