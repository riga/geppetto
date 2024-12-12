# coding: utf-8

from __future__ import annotations

import math
from argparse import Namespace

import torch
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding for transformer models.
    For a given normaliuation N, the encoding at position T is given by vector of size E as:
    [
        sin(T / N**(2 * 0 / E)),
        cos(T / N**(2 * 0 / E)),
        sin(T / N**(2 * 1 / E)),
        cos(T / N**(2 * 1 / E)),
        ...
        sin(T / N**(2 * (E/2) / E)),
        cos(T / N**(2 * (E/2) / E)),
    ]
    """

    def __init__(
        self,
        /,
        n_embed_max: int,
        n_context_max: int,
        *,
        norm: int = 10_000,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # precompute the positional encodings up to a maximum size and safe as buffer
        assert n_embed_max % 2 == 0, "n_embed_max must be even"
        index = torch.arange(n_embed_max)
        k = index[:n_embed_max // 2].repeat_interleave(2)
        # register frequencies
        self.register_buffer("freq", (norm**(-2 * k / n_embed_max)).to(dtype))
        # register offsets that shift sin to cos for every second position
        self.register_buffer("sin_offset", ((index % 2 == 0) * math.pi / 2).to(dtype))
        # register positions
        self.register_buffer("pos", torch.arange(n_context_max, dtype=dtype)[:, None])

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        # x: (B, C, E)
        _, C, E = x.shape
        freq = self.freq[:E][None, :]  # (1, E)
        return torch.sin(self.pos[:C] @ freq + self.sin_offset[:E])  # (C, 1) @ (1, E) = (C, E)


class AttentionHead(torch.nn.Module):

    def __init__(
        self,
        *,
        n_embed: int,
        n_head: int,
        bias: bool = False,
        look_ahead: bool = True,
        n_context_max: int | None = None,
        qkv_dtype: torch.dtype = torch.float32,
        softmax_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.q = torch.nn.Linear(n_embed, n_head, bias=bias, dtype=qkv_dtype)
        self.k = torch.nn.Linear(n_embed, n_head, bias=bias, dtype=qkv_dtype)
        self.v = torch.nn.Linear(n_embed, n_head, bias=bias, dtype=qkv_dtype)

        # register tril mask for preserving causality (decoder style)
        self.ninf_mask: torch.Tensor | None
        if look_ahead:
            self.ninf_mask = None
        else:
            assert n_context_max, "n_context_max must be provided when look_ahead is disabled"
            self.register_buffer("ninf_mask", torch.tril(torch.ones(n_context_max, n_context_max)) == 0)

        self.softmax_dtype = softmax_dtype

    def forward(self, x: torch.Tensor, /, *, zero_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, E)
        _, C, _ = x.shape

        # attention
        a = self.attention(
            self.q(x),  # (B, C, E) @ (E, H) = (B, C, H)
            self.k(x),  # (B, C, H)
            self.v(x),  # (B, C, H)
            ninf_mask=None if self.ninf_mask is None else self.ninf_mask[:C, :C],
            zero_mask=zero_mask,
            softmax_dtype=self.softmax_dtype,
        )  # (B, C, H)

        return a

    @classmethod
    def attention(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        /,
        *,
        ninf_mask: torch.Tensor | None = None,  # sets values to -inf before softmax
        zero_mask: torch.Tensor | None = None,  # sets values to 0 after softmax
        softmax_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        # H is the head size, identical to embedding size for single-head attention
        # q, k, v: (B, C, H)
        # ninf_mask: (C, C)
        # zero_mask: (B, C)
        _, C, _ = q.shape

        # scaled dot product
        a = (q @ k.transpose(-2, -1)) * (C**-0.5)  # (B, C, C)

        # set values to -inf before softmax to ignore them while preserving normalization
        if ninf_mask is not None:
            a = a.masked_fill(ninf_mask, float("-inf"))

        # normalize
        a = F.softmax(a, dim=-1, dtype=softmax_dtype)  # (B, C, C)

        # zero mask after softmax to counteract uneven sequence lengths
        if zero_mask is not None:
            a = a.masked_fill(zero_mask[..., None], 0)

        # dot product with values
        a = a @ v  # (B, C, C) @ (B, C, H) = (B, C, H)

        return a


class MultiHeadedAttention(torch.nn.Module):

    def __init__(
        self,
        *,
        n_embed: int,
        n_heads: int,
        bias: bool = False,
        look_ahead: bool = True,
        n_context_max: int | None = None,
        qkv_dtype: torch.dtype = torch.float32,
        softmax_dtype: torch.dtype = torch.float32,
        proj_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # input checks
        assert n_embed % n_heads == 0, "n_embed must be divisible by n_heads"
        n_head = n_embed // n_heads

        # components
        self.attention_heads = torch.nn.ModuleList([
            AttentionHead(
                n_embed=n_embed,
                n_head=n_head,
                bias=bias,
                look_ahead=look_ahead,
                n_context_max=n_context_max,
                qkv_dtype=qkv_dtype,
                softmax_dtype=softmax_dtype,
            )
            for _ in range(n_heads)
        ])
        self.proj = torch.nn.Linear(n_embed, n_embed, bias=bias, dtype=proj_dtype)

    def forward(self, x: torch.Tensor, /, *, zero_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, E)

        # concatenate output from n heads, each with a size of H = E / n
        x = torch.cat([head(x, zero_mask=zero_mask) for head in self.attention_heads], dim=-1)  # (B, C, E)

        # projection
        x = self.proj(x)  # (B, C, E)

        return x


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        *,
        n_embed: int,
        n_heads: int,
        attention_bias: bool = False,
        attention_dropout_prob: float = 0.0,
        mlp_factor: int = 4,
        mlp_activation: str | torch.nn.Module = "GELU",
        mlp_bias: bool = True,
        mlp_dropout_prob: float = 0.0,
        look_ahead: bool = True,
        n_context_max: int | None = None,
        qkv_dtype: torch.dtype = torch.float32,
        softmax_dtype: torch.dtype = torch.float32,
        proj_dtype: torch.dtype = torch.float32,
        mlp_dtype: torch.dtype = torch.float32,
        norm_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # components
        self.multi_attention = MultiHeadedAttention(
            n_embed=n_embed,
            n_heads=n_heads,
            look_ahead=look_ahead,
            bias=attention_bias,
            n_context_max=n_context_max,
            qkv_dtype=qkv_dtype,
            softmax_dtype=softmax_dtype,
            proj_dtype=proj_dtype,
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_embed, n_embed * mlp_factor, bias=mlp_bias, dtype=mlp_dtype),
            (getattr(torch.nn, mlp_activation)() if isinstance(mlp_activation, str) else mlp_activation),
            torch.nn.Linear(n_embed * mlp_factor, n_embed, bias=mlp_bias, dtype=mlp_dtype),
        )

        self.attention_norm = torch.nn.LayerNorm(n_embed, dtype=norm_dtype)
        self.mlp_norm = torch.nn.LayerNorm(n_embed, dtype=norm_dtype)
        self.norm_dtype = norm_dtype

        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob) if attention_dropout_prob else None
        self.mlp_dropout = torch.nn.Dropout(mlp_dropout_prob) if mlp_dropout_prob else None

    def forward(self, x: torch.Tensor, /, *, zero_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, E)

        # attention
        a = self.attention_norm(x.to(self.norm_dtype)).to(x.dtype)  # (B, C, E)
        a = self.multi_attention(a, zero_mask=zero_mask)  # (B, C, E)
        if self.attention_dropout:
            a = self.attention_dropout(a)
        x = x + a

        # mlp
        m = self.mlp_norm(x.to(self.norm_dtype)).to(x.dtype)  # (B, C, E)
        m = self.mlp(m)  # (B, C, E)
        if self.mlp_dropout:
            m = self.mlp_dropout(m)
        x = x + m

        return x


class Geppetto(torch.nn.Module):

    def __init__(self, params: Namespace, /) -> None:
        super().__init__()

        # embedding
        self.embedding = torch.nn.Embedding(params.n_vocab, params.n_embed, dtype=params.embed_dtype)

        # positional encoding
        self.pe = PositionalEncoding(
            n_embed_max=params.n_embed,
            n_context_max=params.n_context,
            dtype=params.embed_dtype,
        )

        # transformer blocks
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                n_embed=params.n_embed,
                n_heads=params.n_heads,
                attention_bias=params.attention_bias,
                attention_dropout_prob=params.attention_dropout_prob,
                mlp_factor=params.mlp_factor,
                mlp_activation=params.mlp_activation,
                mlp_bias=params.mlp_bias,
                mlp_dropout_prob=params.mlp_dropout_prob,
                look_ahead=False,
                n_context_max=params.n_context,
                qkv_dtype=params.qkv_dtype,
                softmax_dtype=params.qkv_dtype,
                proj_dtype=params.proj_dtype,
                mlp_dtype=params.mlp_dtype,
                norm_dtype=params.norm_dtype,
            )
            for _ in range(params.n_layers)
        ])

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        # x: (B, C)

        # infer attention mask from negative tokens
        zero_mask = x < 0  # (B, C)

        # embedding
        # consider masked values as 0, the actual masking takes place in attention heads
        x = x * (~zero_mask).to(x.dtype)
        x = self.embedding(x)  # (B, C, E)

        # positional encoding
        x = x + self.pe(x)  # (B, C, E)

        # transformer blocks
        for block in self.blocks:
            x = block(x, zero_mask=zero_mask)  # (B, C, E)

        return x
