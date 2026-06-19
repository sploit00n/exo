"""Type stubs for mlx_lm.models.deepseek_v32"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import CacheList

class Indexer(nn.Module):
    def __init__(self, args: BaseModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array] = ...,
        cache: Optional[Any] = ...,
    ) -> Optional[mx.array]: ...

class DeepseekV32Attention(nn.Module):
    hidden_size: int
    num_heads: int
    q_head_dim: int
    qk_nope_head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    scale: float
    q_a_proj: nn.Linear
    q_b_proj: nn.Linear
    kv_a_proj_with_mqa: nn.Linear
    o_proj: nn.Linear
    q_a_layernorm: nn.RMSNorm
    kv_a_layernorm: nn.RMSNorm
    embed_q: nn.Module
    unembed_out: nn.Module
    indexer: Optional[nn.Module]
    rope: nn.Module

    def __init__(self, config: BaseModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = ...,
        cache: Optional[CacheList] = ...,
        prev_topk_indices: Optional[mx.array] = ...,
    ) -> tuple[mx.array, Optional[mx.array]]: ...

class DeepseekV32MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    def __init__(
        self,
        config: BaseModelArgs,
        hidden_size: Optional[int] = ...,
        intermediate_size: Optional[int] = ...,
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class DeepseekV32MoE(nn.Module):
    num_experts_per_tok: int

    def __init__(self, config: BaseModelArgs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class DeepseekV32DecoderLayer(nn.Module):
    self_attn: DeepseekV32Attention
    mlp: nn.Module
    input_layernorm: nn.RMSNorm
    post_attention_layernorm: nn.RMSNorm

    def __init__(self, config: BaseModelArgs, layer_idx: int) -> None: ...
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = ...,
        cache: Optional[CacheList] = ...,
        prev_topk_indices: Optional[mx.array] = ...,
    ) -> tuple[mx.array, Optional[mx.array]]: ...

class DeepseekV32Model(nn.Module):
    vocab_size: int
    embed_tokens: nn.Embedding
    layers: list[DeepseekV32DecoderLayer]
    norm: nn.RMSNorm
    start_idx: int
    end_idx: int
    num_layers: int
    pipeline_rank: int
    pipeline_size: int

    def __init__(self, config: BaseModelArgs) -> None: ...
    def __call__(
        self,
        x: mx.array,
        cache: Optional[list[CacheList | None]] = ...,
    ) -> mx.array: ...

class Model(nn.Module):
    model_type: str
    model: DeepseekV32Model
    lm_head: nn.Linear
    args: BaseModelArgs

    def __init__(self, config: BaseModelArgs) -> None: ...
    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list[CacheList | None]] = ...,
    ) -> mx.array: ...
    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]: ...
    def make_cache(self) -> list[CacheList]: ...
    @property
    def layers(self) -> list[DeepseekV32DecoderLayer]: ...
