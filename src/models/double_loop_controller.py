"""Double-loop learning controller for meta-learning and structural adaptation."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class LSTMMetaController(nn.Module):
    """LSTM-based meta-controller for double-loop learning.

    This controller maintains a meta-state that adapts the learning process
    by modulating model parameters or learning rates based on training progress.
    """

    def __init__(
        self,
        model_hidden_dim: int = 512,
        controller_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_hidden_dim = model_hidden_dim
        self.controller_hidden_dim = controller_hidden_dim

        # LSTM for maintaining meta-state
        self.lstm = nn.LSTM(
            input_size=model_hidden_dim
            + 3,  # features + loss + accuracy + gradient_norm
            hidden_size=controller_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Projection layers for generating adaptations
        self.lr_modulator = nn.Sequential(
            nn.Linear(controller_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output between 0 and 1 to scale learning rate
        )

        # Architecture adaptation predictor
        self.arch_predictor = nn.Sequential(
            nn.Linear(controller_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),  # Architectural adjustment signals
        )

        # Meta-loss predictor
        self.meta_loss_predictor = nn.Sequential(
            nn.Linear(controller_hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.hidden_state = None

    def reset_state(self):
        """Reset the LSTM hidden state."""
        self.hidden_state = None

    def forward(
        self,
        model_features: torch.Tensor,
        loss: torch.Tensor,
        accuracy: torch.Tensor,
        gradient_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            model_features: (batch_size, model_hidden_dim) - aggregate model features
            loss: (batch_size, 1) - current training loss
            accuracy: (batch_size, 1) - current accuracy
            gradient_norm: (batch_size, 1) - norm of gradients
        Returns:
            lr_scale: (batch_size, 1) - learning rate multiplier
            arch_adaptation: (batch_size, 64) - architectural adaptation signals
            meta_loss: (batch_size, 1) - predicted meta-loss
        """
        batch_size = model_features.shape[0]

        # Concatenate all inputs
        controller_input = torch.cat(
            [
                model_features,
                loss.view(batch_size, 1),
                accuracy.view(batch_size, 1),
                gradient_norm.view(batch_size, 1),
            ],
            dim=-1,
        ).unsqueeze(
            1
        )  # Add sequence dimension

        # Process through LSTM
        if self.hidden_state is None:
            lstm_out, self.hidden_state = self.lstm(controller_input)
        else:
            lstm_out, self.hidden_state = self.lstm(controller_input, self.hidden_state)

        controller_state = lstm_out.squeeze(1)

        # Generate outputs
        lr_scale = self.lr_modulator(controller_state)
        arch_adaptation = self.arch_predictor(controller_state)
        meta_loss = self.meta_loss_predictor(controller_state)

        return lr_scale, arch_adaptation, meta_loss


class AdaptiveLayerNorm(nn.Module):
    """Layer normalization with adaptive parameters controlled by meta-controller."""

    def __init__(self, normalized_shape: int, adaptation_dim: int = 64):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)

        # Learn to transform adaptation signals into layer norm parameters
        self.scale_transform = nn.Linear(adaptation_dim, normalized_shape)
        self.bias_transform = nn.Linear(adaptation_dim, normalized_shape)

    def forward(
        self, x: torch.Tensor, adaptation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, ..., normalized_shape)
            adaptation: (batch_size, adaptation_dim) - from meta-controller
        Returns:
            (batch_size, ..., normalized_shape)
        """
        x = self.ln(x)

        if adaptation is not None:
            # Apply adaptive transformations
            scale = 1.0 + 0.1 * self.scale_transform(adaptation)  # Small adjustments
            bias = 0.1 * self.bias_transform(adaptation)

            # Broadcast to match x shape
            while len(scale.shape) < len(x.shape):
                scale = scale.unsqueeze(1)
                bias = bias.unsqueeze(1)

            x = x * scale + bias

        return x


class DoubleLoopController(nn.Module):
    """
    Double-loop learning controller that implements:
    - Inner loop: Standard gradient descent on task loss
    - Outer loop: Meta-learning to adapt learning process
    """

    def __init__(
        self,
        model_hidden_dim: int = 512,
        controller_type: str = "lstm",
        controller_hidden_dim: int = 256,
        update_frequency: int = 100,
        meta_lr: float = 1e-5,
        adaptation_strength: float = 0.1,
    ):
        super().__init__()
        self.update_frequency = update_frequency
        self.meta_lr = meta_lr
        self.adaptation_strength = adaptation_strength
        self.step_count = 0

        # Create meta-controller
        if controller_type == "lstm":
            self.meta_controller = LSTMMetaController(
                model_hidden_dim=model_hidden_dim,
                controller_hidden_dim=controller_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        # Track statistics for meta-learning
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    def should_update_meta(self) -> bool:
        """Check if meta-controller should be updated."""
        return self.step_count % self.update_frequency == 0

    def compute_meta_gradient(
        self, model: nn.Module, loss: torch.Tensor, accuracy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute meta-gradients for controller update.

        This implements the outer loop of double-loop learning by:
        1. Taking a virtual step with current parameters
        2. Evaluating the effect on validation performance
        3. Computing gradients w.r.t. controller parameters
        """
        # Store statistics
        self.loss_history.append(loss.item())
        self.accuracy_history.append(accuracy.item())

        # Keep history limited
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
            self.accuracy_history = self.accuracy_history[-1000:]

        # Compute meta-metrics
        if len(self.loss_history) > 10:
            loss_trend = (self.loss_history[-1] - self.loss_history[-10]) / 10
            acc_trend = (self.accuracy_history[-1] - self.accuracy_history[-10]) / 10

            meta_metrics = {
                "loss_trend": loss_trend,
                "accuracy_trend": acc_trend,
                "loss_variance": torch.var(torch.tensor(self.loss_history[-100:])),
                "accuracy_variance": torch.var(
                    torch.tensor(self.accuracy_history[-100:])
                ),
            }
        else:
            meta_metrics = {
                "loss_trend": 0.0,
                "accuracy_trend": 0.0,
                "loss_variance": torch.tensor(0.0),
                "accuracy_variance": torch.tensor(0.0),
            }

        return meta_metrics

    def forward(
        self,
        model_features: torch.Tensor,
        loss: torch.Tensor,
        accuracy: torch.Tensor,
        gradient_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate adaptation signals from meta-controller.

        Args:
            model_features: Aggregate features from the model
            loss: Current training loss
            accuracy: Current accuracy
            gradient_norm: Norm of model gradients
        Returns:
            Dictionary with adaptation signals
        """
        self.step_count += 1

        # Get controller outputs
        lr_scale, arch_adaptation, meta_loss = self.meta_controller(
            model_features, loss, accuracy, gradient_norm
        )

        return {
            "lr_scale": lr_scale,
            "arch_adaptation": arch_adaptation,
            "meta_loss": meta_loss,
            "should_update_meta": self.should_update_meta(),
        }

    def reset(self):
        """Reset controller state."""
        self.meta_controller.reset_state()
        self.step_count = 0
        self.loss_history = []
        self.accuracy_history = []


def create_double_loop_controller(config: dict) -> DoubleLoopController:
    """Factory function to create double-loop controller from config."""
    return DoubleLoopController(
        model_hidden_dim=config.get("model_hidden_dim", 512),
        controller_type=config.get("controller_type", "lstm"),
        controller_hidden_dim=config.get("hidden_dim", 256),
        update_frequency=config.get("update_frequency", 100),
        meta_lr=config.get("meta_lr", 1e-5),
        adaptation_strength=config.get("adaptation_strength", 0.1),
    )
