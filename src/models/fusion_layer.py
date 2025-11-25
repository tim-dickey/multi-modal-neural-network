"""Multi-modal fusion layer for combining image and text features."""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Cross-attention between two modalities."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_features: (batch_size, query_len, hidden_dim)
            key_value_features: (batch_size, kv_len, hidden_dim)
            attention_mask: (batch_size, kv_len)
        Returns:
            (batch_size, query_len, hidden_dim)
        """
        B, N_q, C = query_features.shape
        N_kv = key_value_features.shape[1]

        # Compute Q from query features, K and V from key_value features
        q = (
            self.query(query_features)
            .reshape(B, N_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(key_value_features)
            .reshape(B, N_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(key_value_features)
            .reshape(B, N_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_kv)
            attn = attn.masked_fill(attention_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Combine
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x  # type: ignore[no-any-return]


class FusionTransformerBlock(nn.Module):
    """Transformer block with cross-modal attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = CrossModalAttention(hidden_dim, num_heads, dropout)

        # Cross-attention
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)

        # MLP
        self.norm3 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len_x, hidden_dim) - query modality
            context: (batch_size, seq_len_context, hidden_dim) - key/value modality
            context_mask: (batch_size, seq_len_context)
        Returns:
            (batch_size, seq_len_x, hidden_dim)
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x))

        # Cross-attention with the other modality
        x = x + self.cross_attn(self.norm2(x), context, context_mask)

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x


class EarlyFusionLayer(nn.Module):
    """Early fusion: concatenate features and process together."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Projection layers to ensure both modalities have the same dimension
        self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)

        # Modality type embeddings
        self.modality_embed = nn.Parameter(torch.zeros(2, hidden_dim))

        # Fusion transformer blocks
        self.blocks = nn.ModuleList(
            [
                FusionTransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            vision_features: (batch_size, num_patches, hidden_dim)
            text_features: (batch_size, seq_length, hidden_dim)
            text_mask: (batch_size, seq_length)
        Returns:
            fused_features: (batch_size, total_seq_len, hidden_dim)
            vision_seq_len: int - length of vision sequence for later separation
        """
        N_vis = vision_features.shape[1]

        # Project features
        vision_features = self.vision_proj(vision_features)
        text_features = self.text_proj(text_features)

        # Add modality embeddings
        vision_features = vision_features + self.modality_embed[0].unsqueeze(
            0
        ).unsqueeze(0)
        text_features = text_features + self.modality_embed[1].unsqueeze(0).unsqueeze(0)

        # Concatenate vision and text features
        fused_features = torch.cat([vision_features, text_features], dim=1)

        # Create combined mask (currently unused but may be needed for
        # attention masking)
        # if text_mask is not None:
        #     vision_mask = torch.ones(B, N_vis, device=vision_features.device,
        #                              dtype=text_mask.dtype)
        #     combined_mask = torch.cat([vision_mask, text_mask], dim=1)

        # Process through fusion blocks with alternating modality attention
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                # Vision attends to text
                vision_part = fused_features[:, :N_vis]
                text_part = fused_features[:, N_vis:]
                vision_part = block(vision_part, text_part, text_mask)
                fused_features = torch.cat([vision_part, text_part], dim=1)
            else:
                # Text attends to vision
                vision_part = fused_features[:, :N_vis]
                text_part = fused_features[:, N_vis:]
                text_part = block(text_part, vision_part)
                fused_features = torch.cat([vision_part, text_part], dim=1)

        fused_features = self.norm(fused_features)

        return fused_features, N_vis


class LateFusionLayer(nn.Module):
    """Late fusion: process modalities separately then combine."""

    def __init__(
        self, hidden_dim: int, fusion_method: str = "concat", dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method

        if fusion_method == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif fusion_method == "add":
            # Weighted addition
            self.vision_weight = nn.Parameter(torch.tensor(0.5))
            self.text_weight = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == "attention":
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=dropout, batch_first=True
            )

    def forward(
        self, vision_pooled: torch.Tensor, text_pooled: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vision_pooled: (batch_size, hidden_dim)
            text_pooled: (batch_size, hidden_dim)
        Returns:
            (batch_size, hidden_dim)
        """
        if self.fusion_method == "concat":
            fused = torch.cat([vision_pooled, text_pooled], dim=-1)
            fused = self.fusion_proj(fused)
        elif self.fusion_method == "add":
            fused = self.vision_weight * vision_pooled + self.text_weight * text_pooled
        elif self.fusion_method == "attention":
            # Use vision as query, text as key/value
            vision_expanded = vision_pooled.unsqueeze(1)
            text_expanded = text_pooled.unsqueeze(1)
            fused, _ = self.attention(vision_expanded, text_expanded, text_expanded)
            fused = fused.squeeze(1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused


class FusionLayer(nn.Module):
    """Main fusion layer that supports both early and late fusion."""

    def __init__(
        self,
        fusion_type: str = "early",
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        late_fusion_method: str = "concat",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.fusion: Union[EarlyFusionLayer, LateFusionLayer]

        if fusion_type == "early":
            self.fusion = EarlyFusionLayer(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        elif fusion_type == "late":
            self.fusion = LateFusionLayer(
                hidden_dim=hidden_dim, fusion_method=late_fusion_method, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_pooled: Optional[torch.Tensor] = None,
        text_pooled: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Args:
            vision_features: (batch_size, num_patches, hidden_dim)
            text_features: (batch_size, seq_length, hidden_dim)
            vision_pooled: (batch_size, hidden_dim) - for late fusion
            text_pooled: (batch_size, hidden_dim) - for late fusion
            text_mask: (batch_size, seq_length)
        Returns:
            fused_features: depends on fusion type
            vision_seq_len: only for early fusion
        """
        if self.fusion_type == "early":
            return self.fusion(vision_features, text_features, text_mask)  # type: ignore[no-any-return]
        else:  # late fusion
            assert vision_pooled is not None and text_pooled is not None
            fused = self.fusion(vision_pooled, text_pooled)
            return fused, None


def create_fusion_layer(config: Dict[str, Any]) -> FusionLayer:
    """Factory function to create fusion layer from config."""
    return FusionLayer(
        fusion_type=config.get("type", "early"),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 8),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.1),
        late_fusion_method=config.get("late_fusion_method", "concat"),
    )
