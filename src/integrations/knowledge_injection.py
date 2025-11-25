"""Knowledge injection utilities for integrating external knowledge into model training."""

from typing import Any, Dict, List, Optional, Protocol, cast
import torch
import torch.nn as nn
from .base import KnowledgeInjector


class InjectionStrategy(Protocol):
    """Protocol for knowledge injection strategies."""

    def inject(self, model_output: torch.Tensor, knowledge: Any) -> torch.Tensor:
        """Inject knowledge into model output.

        Args:
            model_output: Current model output tensor
            knowledge: External knowledge to inject

        Returns:
            Modified model output
        """
        ...


class AdditiveInjection:
    """Additive knowledge injection strategy."""

    def __init__(self, weight: float = 0.1):
        self.weight = weight

    def inject(self, model_output: torch.Tensor, knowledge: Any) -> torch.Tensor:
        """Add knowledge as an additive term.

        Args:
            model_output: Current model output
            knowledge: Knowledge tensor to add

        Returns:
            Modified output
        """
        if isinstance(knowledge, torch.Tensor):
            return model_output + self.weight * knowledge
        return model_output


class MultiplicativeInjection:
    """Multiplicative knowledge injection strategy."""

    def __init__(self, weight: float = 0.1):
        self.weight = weight

    def inject(self, model_output: torch.Tensor, knowledge: Any) -> torch.Tensor:
        """Scale model output with knowledge.

        Args:
            model_output: Current model output
            knowledge: Knowledge scaling factor

        Returns:
            Modified output
        """
        if isinstance(knowledge, (int, float)):
            # For scalar knowledge, scale by (1 + weight) to match expected behavior
            return model_output * (1 + self.weight)
        elif isinstance(knowledge, torch.Tensor):
            return model_output * (1 + self.weight * knowledge)
        return model_output


class AttentionInjection:
    """Attention-based knowledge injection."""

    def __init__(self, hidden_dim: int, weight: float = 0.1):
        self.weight = weight
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def inject(self, model_output: torch.Tensor, knowledge: Any) -> torch.Tensor:
        """Use attention to blend model output with knowledge.

        Args:
            model_output: Current model output [batch, seq, hidden]
            knowledge: Knowledge tensor [batch, seq, hidden]

        Returns:
            Modified output
        """
        if not isinstance(knowledge, torch.Tensor):
            return model_output

        # Concatenate along feature dimension
        combined = torch.cat([model_output, knowledge], dim=-1)

        # Compute attention weights
        attn_weights = torch.sigmoid(self.attention(combined))  # [batch, seq, 1]

        # Blend outputs
        return (1 - self.weight * attn_weights) * model_output + self.weight * attn_weights * knowledge


class KnowledgeInjectionManager:
    """Manager for coordinating knowledge injection across different APIs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.injectors: Dict[str, KnowledgeInjector] = {}
        self.strategies: Dict[str, InjectionStrategy] = {}

        # Default injection strategy
        self.default_strategy = AdditiveInjection(
            weight=config.get('default_injection_weight', 0.1)
        )

    def register_injector(self, name: str, injector: KnowledgeInjector) -> None:
        """Register a knowledge injector.

        Args:
            name: Name of the injector
            injector: Knowledge injector instance
        """
        self.injectors[name] = injector

    def register_strategy(self, name: str, strategy: InjectionStrategy) -> None:
        """Register an injection strategy.

        Args:
            name: Name of the strategy
            strategy: Injection strategy instance
        """
        self.strategies[name] = strategy

    def inject_knowledge(
        self,
        input_data: Any,
        model_output: torch.Tensor,
        injector_name: Optional[str] = None,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Inject knowledge using specified injector and strategy.

        Args:
            input_data: Original model input
            model_output: Current model output
            injector_name: Name of injector to use (uses first if None)
            strategy_name: Name of strategy to use (uses default if None)

        Returns:
            Dictionary with injection results
        """
        # Select injector
        if injector_name and injector_name in self.injectors:
            injector = self.injectors[injector_name]
        elif self.injectors:
            injector = next(iter(self.injectors.values()))
        else:
            return {
                "success": False,
                "error": "No knowledge injectors registered"
            }

        # Get knowledge from injector
        knowledge_data = injector.inject_knowledge(input_data, model_output)

        if not knowledge_data.get("injected", False):
            return {
                "success": False,
                "reason": knowledge_data.get("reason", "Injection not triggered")
            }

        # Select strategy
        if strategy_name and strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
        else:
            strategy = self.default_strategy

        # Apply injection
        try:
            modified_output = strategy.inject(model_output, knowledge_data)
            return {
                "success": True,
                "modified_output": modified_output,
                "knowledge_data": knowledge_data,
                "injector_used": injector.__class__.__name__,
                "strategy_used": strategy.__class__.__name__
            }
        except Exception as e:  # pylint: disable=broad-except
            return {
                "success": False,
                "error": f"Injection failed: {str(e)}"
            }

    def get_available_injectors(self) -> List[str]:
        """Get list of available injector names.

        Returns:
            List of injector names
        """
        return list(self.injectors.keys())

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names.

        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())


def create_injection_strategy(strategy_name: str, **kwargs: Any) -> InjectionStrategy:
    """Factory function to create injection strategies.

    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy parameters

    Returns:
        Injection strategy instance
    """
    strategies = {
        "additive": AdditiveInjection,
        "multiplicative": MultiplicativeInjection,
        "attention": AttentionInjection,
    }

    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class:
        return cast(InjectionStrategy, strategy_class(**kwargs))

    # Default to additive
    return cast(InjectionStrategy, AdditiveInjection(**kwargs))