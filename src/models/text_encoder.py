"""BERT-based text encoder for text processing."""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEmbedding(nn.Module):
    """Text embedding layer with token, position, and segment embeddings."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 512,
        max_seq_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.position_embed = nn.Embedding(max_seq_length, hidden_dim)
        self.segment_embed = nn.Embedding(2, hidden_dim)  # For sentence A/B

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "position_ids", torch.arange(max_seq_length).expand((1, -1))
        )
        self.position_ids: torch.Tensor

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length) - optional segment IDs
        Returns:
            (batch_size, seq_length, hidden_dim)
        """
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]

        token_embeddings = self.token_embed(input_ids)
        position_embeddings = self.position_embed(position_ids)

        embeddings = token_embeddings + position_embeddings

        if token_type_ids is not None:
            segment_embeddings = self.segment_embed(token_type_ids)
            embeddings = embeddings + segment_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TextMultiHeadAttention(nn.Module):
    """Multi-head attention for text processing with optional masking."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_dim)
            attention_mask: (batch_size, seq_length) - 1 for tokens to attend,
            0 for padding
        Returns:
            (batch_size, seq_length, hidden_dim)
        """
        B, N, C = hidden_states.shape

        # Compute Q, K, V
        q = (
            self.query(hidden_states)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(hidden_states)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(hidden_states)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting: (B, 1, 1, N)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attention_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class TextTransformerBlock(nn.Module):
    """BERT-style transformer block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = TextMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_dim)
            attention_mask: (batch_size, seq_length)
        Returns:
            (batch_size, seq_length, hidden_dim)
        """
        # Self-attention with residual and pre-norm
        attn_output = self.attention(self.norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + attn_output

        # MLP with residual and pre-norm
        mlp_output = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + mlp_output

        return hidden_states


class TextEncoder(nn.Module):
    """BERT-based encoder for text processing."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_seq_length: int = 512,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        *,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token

        self.embeddings = TextEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                TextTransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Pooler for CLS token (optional)
        if use_cls_token:
            self.pooler = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length) - 1 for real tokens, 0 for padding
            token_type_ids: (batch_size, seq_length) - segment IDs
        Returns:
            cls_token: (batch_size, hidden_dim) - pooled representation
            (if use_cls_token)
            sequence_output: (batch_size, seq_length, hidden_dim) - all token
            representations
        """
        # Get embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)

        # Apply transformer blocks
        for block in self.encoder_blocks:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        # Extract CLS token representation
        if self.use_cls_token:
            cls_token = hidden_states[:, 0]  # First token is CLS
            cls_token = self.pooler(cls_token)
            return cls_token, hidden_states
        else:
            # Use mean pooling over sequence
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(
                    hidden_states.size()
                )
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)
            return pooled, hidden_states

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.embeddings.token_embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the token embedding layer."""
        self.embeddings.token_embed = value


class SimpleTokenizer:
    """Simple character-level tokenizer for demonstration purposes.

    In production, use a proper tokenizer like transformers.AutoTokenizer.
    """

    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode text to input IDs."""
        # Simple character-based encoding (for demo only)
        input_ids = [self.cls_token_id]
        input_ids.extend(
            [
                min(ord(c) % (self.vocab_size - 10), self.vocab_size - 1)
                for c in text[: max_length - 2]
            ]
        )
        input_ids.append(self.sep_token_id)

        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(self.pad_token_id)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }


def create_text_encoder(config: Dict[str, Any]) -> TextEncoder:
    """Factory function to create text encoder from config."""
    return TextEncoder(
        vocab_size=config.get("vocab_size", 30522),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 12),
        num_heads=config.get("num_heads", 8),
        max_seq_length=config.get("max_seq_length", 512),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        dropout=config.get("dropout", 0.1),
        use_cls_token=config.get("use_cls_token", True),
    )
