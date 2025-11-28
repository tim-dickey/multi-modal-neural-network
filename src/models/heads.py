"""Task-specific prediction heads for multi-modal model."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head for image/text classification tasks."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_classes: int = 1000,
        dropout: float = 0.1,
        *,
        use_intermediate_layer: bool = True,
    ):
        super().__init__()
        self.use_intermediate_layer = use_intermediate_layer

        if use_intermediate_layer:
            self.head: Union[nn.Sequential, nn.Linear] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, hidden_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.head(x)


class RegressionHead(nn.Module):
    """Regression head for continuous value prediction."""

    def __init__(
        self, hidden_dim: int = 512, output_dim: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, hidden_dim)
        Returns:
            predictions: (batch_size, output_dim)
        """
        return self.head(x)


class MultiLabelHead(nn.Module):
    """Multi-label classification head."""

    def __init__(
        self, hidden_dim: int = 512, num_labels: int = 80, dropout: float = 0.1
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),  # Independent probabilities for each label
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, hidden_dim)
        Returns:
            probabilities: (batch_size, num_labels)
        """
        return self.head(x)


class ContrastiveHead(nn.Module):
    """Contrastive learning head for image-text matching."""

    def __init__(
        self,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / temperature))
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        *,
        return_similarity: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            image_features: (batch_size, hidden_dim)
            text_features: (batch_size, hidden_dim)
            return_similarity: whether to compute similarity matrix
        Returns:
            If return_similarity:
                similarity: (batch_size, batch_size) - cosine similarity matrix
            Else:
                (image_proj, text_proj): projected features
        """
        # Project features
        image_proj = self.projection(image_features)
        text_proj = self.projection(text_features)

        if not return_similarity:
            return image_proj, text_proj

        # Normalize features
        image_proj = image_proj / image_proj.norm(dim=-1, keepdim=True)
        text_proj = text_proj / text_proj.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        similarity = logit_scale * image_proj @ text_proj.t()

        return similarity


class SequenceGenerationHead(nn.Module):
    """Sequence generation head for tasks like captioning."""

    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = 30522,
        max_seq_length: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Token embeddings for decoder
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_length, hidden_dim)

        self.register_buffer(
            "position_ids", torch.arange(max_seq_length).expand((1, -1))
        )
        self.position_ids: torch.Tensor

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch_size, seq_len, hidden_dim) - from encoder
            target_ids: (batch_size, target_len) - target token IDs for training
            encoder_mask: (batch_size, seq_len) - encoder attention mask
        Returns:
            logits: (batch_size, target_len, vocab_size)
        """
        if target_ids is None:
            # Inference mode - generate one token at a time
            raise NotImplementedError("Auto-regressive generation not implemented yet")

        # Get target sequence length for position embeddings
        target_len = target_ids.shape[1]

        # Embed target tokens
        position_ids = self.position_ids[:, :target_len]
        target_embeds = self.token_embed(target_ids) + self.pos_embed(position_ids)

        # Create causal mask for decoder
        causal_mask = torch.triu(
            torch.ones(target_len, target_len, device=target_ids.device)
            * float("-inf"),
            diagonal=1,
        )

        # Decode
        decoder_output = self.decoder(
            target_embeds,
            encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=encoder_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(decoder_output)

        return logits


class MultiTaskHead(nn.Module):
    """Multi-task head that combines multiple task-specific heads."""

    def __init__(
        self,
        hidden_dim: int = 512,
        tasks: Optional[Dict[str, Dict]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if tasks is None:
            tasks = {"classification": {"num_classes": 1000}}

        self.task_names = list(tasks.keys())
        self.heads = nn.ModuleDict()

        for task_name, task_config in tasks.items():
            task_type = task_config.get("type", "classification")

            head: Union[
                ClassificationHead, RegressionHead, MultiLabelHead, ContrastiveHead
            ]

            if task_type == "classification":
                head = ClassificationHead(
                    hidden_dim=hidden_dim,
                    num_classes=task_config.get("num_classes", 1000),
                    dropout=dropout,
                )
            elif task_type == "regression":
                head = RegressionHead(
                    hidden_dim=hidden_dim,
                    output_dim=task_config.get("output_dim", 1),
                    dropout=dropout,
                )
            elif task_type == "multilabel":
                head = MultiLabelHead(
                    hidden_dim=hidden_dim,
                    num_labels=task_config.get("num_labels", 80),
                    dropout=dropout,
                )
            elif task_type == "contrastive":
                head = ContrastiveHead(
                    hidden_dim=hidden_dim,
                    projection_dim=task_config.get("projection_dim", 256),
                    temperature=task_config.get("temperature", 0.07),
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            self.heads[task_name] = head

    def forward(
        self, features: torch.Tensor, task_name: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch_size, hidden_dim)
            task_name: specific task to run (if None, run all)
            **kwargs: additional arguments for specific heads
        Returns:
            Dictionary mapping task names to outputs
        """
        if task_name is not None:
            return {task_name: self.heads[task_name](features, **kwargs)}
        else:
            outputs = {}
            for name, head in self.heads.items():
                outputs[name] = head(features)
            return outputs


def create_task_head(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create task head from config."""
    head_type = config.get("type", "classification")

    if head_type == "classification":
        return ClassificationHead(
            hidden_dim=config.get("hidden_dim", 512),
            num_classes=config.get("num_classes", 1000),
            dropout=config.get("dropout", 0.1),
            use_intermediate_layer=config.get("use_intermediate_layer", True),
        )
    elif head_type == "regression":
        return RegressionHead(
            hidden_dim=config.get("hidden_dim", 512),
            output_dim=config.get("output_dim", 1),
            dropout=config.get("dropout", 0.1),
        )
    elif head_type == "multilabel":
        return MultiLabelHead(
            hidden_dim=config.get("hidden_dim", 512),
            num_labels=config.get("num_labels", 80),
            dropout=config.get("dropout", 0.1),
        )
    elif head_type == "contrastive":
        return ContrastiveHead(
            hidden_dim=config.get("hidden_dim", 512),
            projection_dim=config.get("projection_dim", 256),
            temperature=config.get("temperature", 0.07),
        )
    elif head_type == "generation":
        return SequenceGenerationHead(
            hidden_dim=config.get("hidden_dim", 512),
            vocab_size=config.get("vocab_size", 30522),
            max_seq_length=config.get("max_seq_length", 128),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
        )
    elif head_type == "multitask":
        return MultiTaskHead(
            hidden_dim=config.get("hidden_dim", 512),
            tasks=config.get("tasks", None),
            dropout=config.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")
