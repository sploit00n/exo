"""Patch mlx_lm's ``glm_moe_dsa`` implementation for GLM-5.2.

GLM-5.2 (architecture ``GlmMoeDsaForCausalLM``) uses DeepSeek Sparse Attention
where an *indexer* selects the top-k KV positions each layer attends to. Unlike
GLM-5/5.1, GLM-5.2 ships a *per-layer indexer schedule*: only "full" layers carry
indexer weights and compute their own top-k; "shared" layers carry no indexer
weights and must reuse the previous full layer's top-k selection.

The pinned mlx-lm fork ships a bare ``glm_moe_dsa.Model(DeepseekV32Model)`` that
makes every layer run its own indexer. Because exo loads weights with
``strict=False``, the missing indexer weights on shared layers are silently left
uninitialized, producing garbage top-k selections and corrupted output (symbol
noise during long "thinking" generations).

In addition to IndexShare, the pinned mlx-lm indexer differs from the GLM
reference in one important numeric detail: GLM uses LayerNorm epsilon ``1e-6``.
The sparse selector can also evict the first attention-sink tokens after
``index_topk``. That has been reproduced as digit/punctuation noise on real
GLM-5.2 checkpoints. Full layers therefore use a GLM-specific indexer which
matches the reference epsilon and preserves four sinks plus a small recent
window before top-k selection. Set ``EXO_GLM_DSA_PRESERVE_SINKS=false`` to
disable the latter mitigation for reference-parity testing.

The patch installs its classes onto ``mlx_lm.models.glm_moe_dsa`` so the model
loader picks them up. Remove the corresponding pieces once upstream mlx-lm has
merged equivalent IndexShare and sparse-selector fixes.
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import glm_moe_dsa
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import BatchKVCache, CacheList, KVCache
from mlx_lm.models.deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from mlx_lm.models.deepseek_v32 import (
    Indexer as DeepseekV32Indexer,
)
from mlx_lm.models.glm_moe_dsa import Model as _BaseModel
from mlx_lm.models.glm_moe_dsa import ModelArgs as _BaseModelArgs
from mlx_lm.models.rope_utils import initialize_rope


class _IndexerCache(Protocol):
    @property
    def offset(self) -> int | mx.array: ...

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]: ...


type IndexerCache = _IndexerCache | BatchKVCache


def _environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off; got {value!r}"
    )


@dataclass
class ModelArgs(_BaseModelArgs):
    """GLM-5.2 args extended with the per-layer DSA indexer schedule."""

    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[str | list[str]] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2
    indexer_rope_interleave: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.indexer_types is not None:
            return

        if self.index_topk_pattern is not None:
            pattern = self.index_topk_pattern
            if isinstance(pattern, str):
                self.indexer_types = [
                    {"F": "full", "S": "shared"}[character] for character in pattern
                ]
            else:
                self.indexer_types = list(pattern)
        else:
            freq = max(self.index_topk_freq, 1)
            offset = self.index_skip_topk_offset
            self.indexer_types = [
                "full" if (max(layer_idx - offset + 1, 0) % freq) == 0 else "shared"
                for layer_idx in range(self.num_hidden_layers)
            ]


class GlmMoeDsaIndexer(DeepseekV32Indexer):
    """GLM-specific DSA selector with stable sparse top-k behavior."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)

        # The GLM/Hugging Face reference uses 1e-6. mlx-lm's inherited
        # DeepSeek indexer currently relies on LayerNorm's 1e-5 default.
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        assert config.rope_theta is not None
        self.rope = initialize_rope(
            dims=config.qk_rope_head_dim,
            base=config.rope_theta,
            traditional=config.indexer_rope_interleave,
            max_position_embeddings=config.max_position_embeddings,
            scaling_config=config.rope_scaling,
        )
        self.preserve_attention_sinks = _environment_flag(
            "EXO_GLM_DSA_PRESERVE_SINKS", default=True
        )
        self.force_dense = _environment_flag("EXO_GLM_DSA_FORCE_DENSE", default=False)

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[IndexerCache] = None,
    ) -> Optional[mx.array]:
        """Return sparse key indices, or ``None`` while attention stays dense."""
        batch_size, sequence_length, _ = x.shape
        offset = cache.offset if cache is not None else 0

        # Update K even in the dense regime so the indexer cache is complete
        # when sparse selection engages after index_topk.
        k = self.k_norm(self.wk(x))
        k = mx.reshape(k, (batch_size, 1, sequence_length, self.head_dim))
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(
                k, mx.zeros([batch_size, 1, sequence_length, 0])
            )

        if self.force_dense or k.shape[2] <= self.index_topk:
            return None

        q = self.wq_b(qr)
        q = q.reshape(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).swapaxes(1, 2)
        q = self.rope(q, offset=offset)

        scores = mx.maximum(q @ k.swapaxes(-1, -2), 0)
        head_scale = 1.0 / math.sqrt(self.n_heads)
        weights = self.weights_proj(x) * (head_scale * self.softmax_scale)
        scores = scores * weights.swapaxes(-1, -2)[..., None]
        scores = scores.sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))

        if self.preserve_attention_sinks:
            scores = self._preserve_sinks_and_recent_tokens(scores, cache, offset)

        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
            ..., -self.index_topk :
        ]

    def _preserve_sinks_and_recent_tokens(
        self,
        scores: mx.array,
        cache: Optional[IndexerCache],
        offset: int | mx.array,
    ) -> mx.array:
        """Reserve part of the top-k budget for sinks and recent tokens."""
        sink_count = min(4, self.index_topk)
        recent_window = min(128, max(self.index_topk - sink_count, 0))
        key_count = scores.shape[-1]

        # BatchKVCache stores offset and left_padding per sequence. A sink is
        # the first real token, not necessarily column zero in the padded
        # buffer, so all position math keeps an explicit batch dimension.
        left_padding = (
            cache.left_padding if isinstance(cache, BatchKVCache) else mx.array(0)
        )
        query_position = mx.arange(scores.shape[2]).reshape(1, -1, 1) + (
            mx.array(offset) + left_padding
        ).reshape(-1, 1, 1)
        key_position = mx.arange(key_count).reshape(1, 1, key_count)
        sink_start = left_padding.reshape(-1, 1, 1)
        force_keep = (key_position >= sink_start) & (
            key_position < sink_start + sink_count
        )
        if recent_window > 0:
            force_keep = force_keep | (
                (key_position <= query_position)
                & (key_position > query_position - recent_window)
            )

        return mx.where(
            force_keep[:, None],
            mx.array(float("inf"), scores.dtype),
            scores,
        )


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__(config)
        assert config.indexer_types is not None
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None
        else:
            self.indexer = GlmMoeDsaIndexer(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[CacheList] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[mx.array]]:
        B, L, _ = x.shape  # noqa: N806

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)

        if self.indexer is not None:
            indexer_cache = cache[1] if cache is not None else None
            topk_indices = self.indexer(x, qr, mask, cache=indexer_cache)
        else:
            topk_indices = prev_topk_indices

        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        # Ensure the indexer cache is evaluated even if the topk_indices are unused
        # to keep the graph from getting too large
        if self.indexer is not None and cache is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), topk_indices


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[CacheList] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[mx.array]]:
        r, topk_indices = self.self_attn(
            self.input_layernorm(x), mask, cache, prev_topk_indices
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_indices


class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Optional[list[CacheList | None]] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None for _ in range(self.num_layers)]
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices
            )

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(_BaseModel):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)

    def make_cache(self) -> list[CacheList]:
        # Shared layers run no indexer, so they get no indexer KVCache.
        caches: list[CacheList] = []
        for layer in self.layers:
            if cast(bool, getattr(layer.self_attn, "skip_topk", False)):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches


def patch_glm_moe_dsa() -> None:
    """Install the GLM-5.2 indexer-schedule classes onto the mlx-lm module.

    The model loader resolves the architecture via ``arch.Model`` / ``arch.ModelArgs``
    on the imported ``mlx_lm.models.glm_moe_dsa`` module, so replacing those
    attributes is sufficient for subsequently loaded models to use the fix.
    """
    glm_moe_dsa.ModelArgs = ModelArgs
    glm_moe_dsa.Model = Model
    glm_moe_dsa.GlmMoeDsaIndexer = GlmMoeDsaIndexer
    glm_moe_dsa.GlmMoeDsaAttention = GlmMoeDsaAttention
    glm_moe_dsa.GlmMoeDsaDecoderLayer = GlmMoeDsaDecoderLayer
    glm_moe_dsa.GlmMoeDsaModel = GlmMoeDsaModel
