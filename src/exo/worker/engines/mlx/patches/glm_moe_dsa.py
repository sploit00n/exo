"""Patch mlx_lm's `glm_moe_dsa` model to support GLM-5.2's DSA indexer schedule.

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

This patch ports the upstream fix (ml-explore/mlx-lm#1410) — cross-layer indexer
sharing plus a ``make_cache`` that omits the indexer cache on shared layers — and
installs it onto the ``mlx_lm.models.glm_moe_dsa`` module so the model loader
picks it up. Remove this patch once the fork includes #1410.
"""

from dataclasses import dataclass
from typing import List, Optional, cast

import mlx.core as mx
from mlx_lm.models import glm_moe_dsa
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from mlx_lm.models.glm_moe_dsa import Model as _BaseModel
from mlx_lm.models.glm_moe_dsa import ModelArgs as _BaseModelArgs


@dataclass
class ModelArgs(_BaseModelArgs):
    """GLM-5.2 args extended with the per-layer DSA indexer schedule."""

    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[str | list[str]] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2

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


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__(config)
        assert config.indexer_types is not None
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None

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
    glm_moe_dsa.GlmMoeDsaAttention = GlmMoeDsaAttention
    glm_moe_dsa.GlmMoeDsaDecoderLayer = GlmMoeDsaDecoderLayer
    glm_moe_dsa.GlmMoeDsaModel = GlmMoeDsaModel
