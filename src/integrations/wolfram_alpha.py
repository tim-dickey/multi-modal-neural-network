"""Wolfram Alpha API integration for symbolic computation and knowledge injection."""

import wolframalpha
from typing import Any, Dict, Optional, cast

from .base import APIIntegration, APIResponse, KnowledgeInjector


class WolframAlphaIntegration(APIIntegration):
    """Integration with Wolfram Alpha for computational knowledge."""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        super().__init__(api_key, config)
        self.client = wolframalpha.Client(api_key)

        # Wolfram-specific config
        self.max_queries_per_day = config.get('max_queries_per_day', 2000)
        self.cache_dir = config.get('cache_dir', './cache/wolfram')

        # Query tracking (in production, this would be persistent)
        self.daily_queries = 0

    def query(self, prompt: str, **kwargs: Any) -> APIResponse:
        """Query Wolfram Alpha with the given prompt.

        Args:
            prompt: Mathematical or computational query
            **kwargs: Additional parameters (format, etc.)

        Returns:
            APIResponse with Wolfram Alpha results
        """
        try:
            # Check daily limit
            if self.daily_queries >= self.max_queries_per_day:
                return APIResponse(
                    success=False,
                    data=None,
                    error="Daily query limit exceeded"
                )

            # Make the query
            result = self.client.query(prompt)

            # Increment counter
            self.daily_queries += 1

            # Process results
            if not result.success:
                return APIResponse(
                    success=False,
                    data=None,
                    error="Wolfram Alpha query failed",
                    metadata={"query": prompt}
                )

            # Extract pod data
            pods_data = []
            for pod in result.pods:
                pod_info = {
                    "title": pod.title,
                    "id": pod.id,
                    "subpods": []
                }

                for subpod in pod.subpods:
                    subpod_info = {
                        "title": subpod.title,
                        "plaintext": subpod.plaintext,
                        "img": subpod.img.src if subpod.img else None
                    }
                    pod_info["subpods"].append(subpod_info)

                pods_data.append(pod_info)

            return APIResponse(
                success=True,
                data=pods_data,
                metadata={
                    "query": prompt,
                    "num_pods": len(pods_data),
                    "query_count": self.daily_queries
                }
            )

        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Wolfram Alpha query error: %s", e)
            return APIResponse(
                success=False,
                data=None,
                error=f"Wolfram Alpha API error: {str(e)}",
                metadata={"query": prompt}
            )

    def validate_response(self, response: APIResponse) -> bool:
        """Validate Wolfram Alpha response structure.

        Args:
            response: The API response to validate

        Returns:
            True if response contains valid Wolfram data
        """
        if not response.success or not response.data:
            return False

        # Check if data is a list of pods
        if not isinstance(response.data, list):
            return False

        # Check if at least one pod exists
        if len(response.data) == 0:
            return False

        # Validate pod structure
        for pod in response.data:
            if not isinstance(pod, dict):
                return False
            if "title" not in pod or "subpods" not in pod:
                return False
            if not isinstance(pod["subpods"], list):
                return False

        return True

    def extract_mathematical_result(self, response: APIResponse) -> Optional[str]:
        """Extract the primary mathematical result from a Wolfram response.

        Args:
            response: Wolfram Alpha response

        Returns:
            Primary result as string, or None if not found
        """
        if not self.validate_response(response):
            return None

        # Look for "Result" or "Solution" pods first
        for pod in response.data:
            if pod["title"].lower() in ["result", "solution", "answer"]:
                if pod["subpods"] and pod["subpods"][0]["plaintext"]:
                    return cast(str, pod["subpods"][0]["plaintext"])

        # Fallback to first pod with plaintext
        for pod in response.data:
            if pod["subpods"] and pod["subpods"][0]["plaintext"]:
                return cast(str, pod["subpods"][0]["plaintext"])

        return None


class WolframKnowledgeInjector(KnowledgeInjector):
    """Knowledge injector using Wolfram Alpha for mathematical validation."""

    def __init__(self, api_integration: WolframAlphaIntegration, config: Dict[str, Any]):
        super().__init__(api_integration, config)
        self.wolfram = api_integration

    def inject_knowledge(self, input_data: Any, model_output: Any) -> Dict[str, Any]:
        """Inject mathematical knowledge validation.

        Args:
            input_data: Original input (could be text with math)
            model_output: Model's current output

        Returns:
            Dictionary with injection data
        """
        # Extract mathematical expressions from input
        math_expressions = self._extract_math_expressions(input_data)

        if not math_expressions:
            return {
                "injected": False,
                "reason": "No mathematical expressions found"
            }

        # Query Wolfram for validation
        validations = []
        for expr in math_expressions:
            response = self.wolfram.query(f"validate {expr}")
            if response.success:
                result = self.wolfram.extract_mathematical_result(response)
                validations.append({
                    "expression": expr,
                    "wolfram_result": result,
                    "confidence": 0.9 if result else 0.5
                })

        return {
            "injected": len(validations) > 0,
            "validations": validations,
            "injection_type": "mathematical_validation",
            "weight": self.injection_weight
        }

    def _extract_math_expressions(self, input_data: Any) -> list[str]:
        """Extract mathematical expressions from input data.

        Args:
            input_data: Input data (text, etc.)

        Returns:
            List of mathematical expressions
        """
        # Simple regex-based extraction (could be enhanced)
        import re

        if isinstance(input_data, str):
            # Look for common math patterns
            patterns = [
                r'\d+\s*[\+\-\*\/]\s*\d+',  # Basic arithmetic
                r'\d+\^\d+',  # Exponents
                r'sqrt\(\d+\)',  # Square roots
                r'\d+\s*=\s*\d+',  # Equations
            ]

            expressions = []
            for pattern in patterns:
                matches = re.findall(pattern, input_data)
                expressions.extend(matches)

            return list(set(expressions))  # Remove duplicates

        return []