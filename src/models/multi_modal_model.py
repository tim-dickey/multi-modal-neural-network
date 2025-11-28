"""Main multi-modal neural network model with double-loop learning."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .double_loop_controller import create_double_loop_controller
from .fusion_layer import create_fusion_layer
from .heads import MultiTaskHead, create_task_head
from .text_encoder import create_text_encoder
from .vision_encoder import create_vision_encoder


class MultiModalModel(nn.Module):
    """
    Multi-modal neural network combining vision and text encoders with fusion layer,
    double-loop learning controller, and task-specific heads.
    """

    def __init__(
        self,
        vision_config: Dict,
        text_config: Dict,
        fusion_config: Dict,
        double_loop_config: Dict,
        head_config: Dict,
        *,
        use_double_loop: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        # Encoders
        self.vision_encoder = create_vision_encoder(vision_config)
        self.text_encoder = create_text_encoder(text_config)

        # Fusion layer
        self.fusion_layer = create_fusion_layer(fusion_config)
        self.fusion_type = fusion_config.get("type", "early")

        # Double-loop controller
        self.use_double_loop = use_double_loop
        if use_double_loop:
            self.double_loop_controller = create_double_loop_controller(
                double_loop_config
            )

        # Task head
        self.task_head = create_task_head(head_config)

        # Gradient checkpointing for memory efficiency
        self.gradient_checkpointing = gradient_checkpointing

        # Global pooling for fusion output
        self.hidden_dim = fusion_config.get("hidden_dim", 512)

    def _aggregate_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate sequence features into a single vector.

        Args:
            features: (batch_size, seq_len, hidden_dim)
        Returns:
            (batch_size, hidden_dim)
        """
        # Mean pooling
        return features.mean(dim=1)

    def _encode_modalities(
        self,
        images: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
    ) -> tuple:
        """Encode vision and text modalities and return pooled components.

        Returns: vision_cls, vision_features, text_cls, text_features
        """
        if images is not None:
            vision_cls, vision_features = self.vision_encoder(images)
        else:
            raise ValueError("Image input is required")

        if input_ids is not None:
            text_cls, text_features = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            device = images.device if images is not None else None
            text_cls = torch.zeros(
                batch_size := vision_cls.shape[0], self.hidden_dim, device=device
            )
            text_features = torch.zeros(batch_size, 1, self.hidden_dim, device=device)

        return vision_cls, vision_features, text_cls, text_features

    def _apply_task_head(
        self,
        pooled_features: torch.Tensor,
        vision_cls: torch.Tensor,
        text_cls: torch.Tensor,
        task_name: Optional[str],
    ) -> Dict[str, Any]:
        """Apply the configured task head and return outputs dict."""
        outputs: Dict[str, Any] = {}

        _cls = getattr(self.task_head, "__class__", type(None))
        cls_name = _cls.__name__
        is_multi_task = isinstance(self.task_head, MultiTaskHead) or (
            cls_name == "MultiTaskHead"
        )

        if is_multi_task:
            task_kwargs = {"task_name": task_name}
            task_outputs = self.task_head(pooled_features, **task_kwargs)
            outputs.update(task_outputs)
            return outputs

        # Single-task head
        if hasattr(self.task_head, "forward"):
            cls_name2 = getattr(self.task_head, "__class__", type(None)).__name__
            is_contrastive = task_name == "contrastive" or (
                cls_name2 == "ContrastiveHead"
            )

            if is_contrastive:
                logits = self.task_head(vision_cls, text_cls)
            else:
                logits = self.task_head(pooled_features)
        else:
            logits = pooled_features

        outputs["logits"] = logits
        return outputs

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        *,
        return_features: bool = False,
        task_name: Optional[str] = None,
        current_loss: Optional[torch.Tensor] = None,
        current_accuracy: Optional[torch.Tensor] = None,
        gradient_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the multi-modal model.

        Args:
            images: (batch_size, channels, height, width) - image inputs
            input_ids: (batch_size, seq_length) - text token IDs
            attention_mask: (batch_size, seq_length) - text attention mask
            token_type_ids: (batch_size, seq_length) - text segment IDs
            return_features: whether to return intermediate features
            task_name: specific task head to use (if multitask head)
            current_loss: current training loss (for double-loop)
            current_accuracy: current accuracy (for double-loop)
            gradient_norm: gradient norm (for double-loop)

        Returns:
            Dictionary containing:
                - logits: task predictions
                - features: intermediate features (if return_features=True)
                - meta_info: double-loop controller info (if enabled)
        """
        outputs = {}

        # Encode modalities (vision + text) and fuse to pooled features
        vision_cls, vision_features, text_cls, text_features = self._encode_modalities(
            images, input_ids, attention_mask, token_type_ids
        )

        if self.fusion_type == "early":
            fused_features, vision_seq_len = self.fusion_layer(
                vision_features=vision_features,
                text_features=text_features,
                text_mask=attention_mask,
            )
            pooled_features = self._aggregate_features(fused_features)
        else:
            pooled_features, _ = self.fusion_layer(
                vision_features=vision_features,
                text_features=text_features,
                vision_pooled=vision_cls,
                text_pooled=text_cls,
            )

        # Apply double-loop controller if enabled
        if self.use_double_loop and self.training:
            if (
                current_loss is not None
                and current_accuracy is not None
                and gradient_norm is not None
            ):
                controller_output = self.double_loop_controller(
                    model_features=pooled_features.detach(),
                    loss=current_loss,
                    accuracy=current_accuracy,
                    gradient_norm=gradient_norm,
                )
                outputs["meta_info"] = controller_output
            else:
                outputs["meta_info"] = None

        # Task prediction
        # Short helper flags to keep lines under length limits
        _cls = getattr(self.task_head, "__class__", type(None))
        cls_name = _cls.__name__
        is_multi_task = isinstance(self.task_head, MultiTaskHead) or (
            cls_name == "MultiTaskHead"
        )

        if is_multi_task:
            # Return a dict of task outputs
            task_kwargs = {"task_name": task_name}
            task_outputs = self.task_head(pooled_features, **task_kwargs)
            outputs.update(task_outputs)
        else:
            if hasattr(self.task_head, "forward"):
                # Check if it's a contrastive head that needs both modalities
                cls_name = getattr(self.task_head, "__class__", type(None)).__name__
                is_contrastive = task_name == "contrastive" or (
                    cls_name == "ContrastiveHead"
                )

                if is_contrastive:
                    logits = self.task_head(vision_cls, text_cls)
                else:
                    logits = self.task_head(pooled_features)
            else:
                logits = pooled_features

            outputs["logits"] = logits

        # Return intermediate features if requested
        if return_features:
            outputs["features"] = {
                "vision_cls": vision_cls,
                "vision_features": vision_features,
                "text_cls": text_cls,
                "text_features": text_features,
                "pooled_features": pooled_features,
            }

        return outputs

    def get_num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def freeze_vision_encoder(self) -> None:
        """Freeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def freeze_text_encoder(self) -> None:
        """Freeze text encoder parameters."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        # Note: Actual implementation would need to use torch.utils.checkpoint
        # in the forward passes of encoders

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            "total_parameters": self.get_num_parameters(trainable_only=False),
            "trainable_parameters": self.get_num_parameters(trainable_only=True),
            "fusion_type": self.fusion_type,
            "use_double_loop": self.use_double_loop,
            "gradient_checkpointing": self.gradient_checkpointing,
            "vision_encoder": str(self.vision_encoder.__class__.__name__),
            "text_encoder": str(self.text_encoder.__class__.__name__),
            "task_head": str(self.task_head.__class__.__name__),
        }


def create_multi_modal_model(config: Dict) -> MultiModalModel:
    """
    Factory function to create multi-modal model from config.

    Args:
        config: Configuration dictionary with keys:
            - model.vision_encoder
            - model.text_encoder
            - model.fusion
            - model.double_loop
            - model.heads

    Returns:
        MultiModalModel instance
    """
    model_config = config.get("model", {})

    # Allow aliasing via `model.head_type` and `model.task_configs`
    head_config = dict(model_config.get("heads", {}))
    alias_head_type = model_config.get("head_type")
    if alias_head_type is not None:
        # Normalize alias values (tests may use 'multi_task')
        normalized = alias_head_type.replace("_", "")
        head_config["type"] = (
            "multitask" if normalized == "multitask" else alias_head_type
        )
    alias_tasks = model_config.get("task_configs")
    if alias_tasks is not None:
        head_config["tasks"] = alias_tasks

    return MultiModalModel(
        vision_config=model_config.get("vision_encoder", {}),
        text_config=model_config.get("text_encoder", {}),
        fusion_config=model_config.get("fusion", {}),
        double_loop_config=model_config.get("double_loop", {}),
        head_config=head_config,
        use_double_loop=model_config.get("use_double_loop", True),
        gradient_checkpointing=config.get("training", {}).get(
            "gradient_checkpointing", False
        ),
    )


def load_pretrained_weights(
    model: MultiModalModel,
    vision_checkpoint: Optional[str] = None,
    text_checkpoint: Optional[str] = None,
    *,
    strict: bool = False,
    allow_external: bool = False,
) -> MultiModalModel:
    """
    Load pretrained weights for encoders.

    Args:
        model: MultiModalModel instance
        vision_checkpoint: path to vision encoder checkpoint
        text_checkpoint: path to text encoder checkpoint
        strict: whether to strictly enforce key matching

    Returns:
        Model with loaded weights
    """
    if vision_checkpoint is not None:
        from ..utils.safe_load import safe_load_checkpoint

        vision_state = safe_load_checkpoint(
            vision_checkpoint, map_location="cpu", expected_keys=None, allow_external=allow_external
        )
        # If the checkpoint is a wrapper dict (contains model_state_dict),
        # accept that as well for backward compatibility.
        if "model_state_dict" in vision_state:
            vision_state = vision_state["model_state_dict"]
        model.vision_encoder.load_state_dict(vision_state, strict=strict)
        print(f"Loaded vision encoder from {vision_checkpoint}")

    if text_checkpoint is not None:
        from ..utils.safe_load import safe_load_checkpoint

        text_state = safe_load_checkpoint(
            text_checkpoint, map_location="cpu", expected_keys=None, allow_external=allow_external
        )
        if "model_state_dict" in text_state:
            text_state = text_state["model_state_dict"]
        model.text_encoder.load_state_dict(text_state, strict=strict)
        print(f"Loaded text encoder from {text_checkpoint}")

    return model
