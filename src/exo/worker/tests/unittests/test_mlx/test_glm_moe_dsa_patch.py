# type: ignore

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import BatchKVCache

from exo.worker.engines.mlx.patches.glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaIndexer,
    ModelArgs,
)


def _tiny_args() -> ModelArgs:
    return ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=64,
        hidden_size=32,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=16,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_nope_head_dim=8,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10_000.0},
        attention_bias=False,
        indexer_types=["full", "shared", "full", "shared"],
    )


def test_attention_uses_indexer_only_on_full_layers() -> None:
    args = _tiny_args()
    assert isinstance(GlmMoeDsaAttention(args, 0).indexer, GlmMoeDsaIndexer)
    assert GlmMoeDsaAttention(args, 1).indexer is None


def test_sparse_indexer_keeps_sinks_and_recent_window(monkeypatch) -> None:
    monkeypatch.delenv("EXO_GLM_DSA_PRESERVE_SINKS", raising=False)
    monkeypatch.delenv("EXO_GLM_DSA_FORCE_DENSE", raising=False)
    mx.random.seed(0)

    args = _tiny_args()
    indexer = GlmMoeDsaIndexer(args)
    sequence_length = 32
    hidden = mx.random.normal((1, sequence_length, args.hidden_size))
    query_residual = mx.random.normal((1, sequence_length, args.q_lora_rank))

    indices = indexer(hidden, query_residual, mask=None, cache=None)
    assert indices is not None
    last_query = set(indices[0, 0, -1].tolist())
    assert set(range(4)) <= last_query
    assert set(range(sequence_length - 12, sequence_length)) <= last_query


def test_sparse_indexer_keeps_real_sinks_in_left_padded_batches(monkeypatch) -> None:
    monkeypatch.delenv("EXO_GLM_DSA_PRESERVE_SINKS", raising=False)
    monkeypatch.delenv("EXO_GLM_DSA_FORCE_DENSE", raising=False)
    mx.random.seed(0)

    args = _tiny_args()
    indexer = GlmMoeDsaIndexer(args)
    sequence_length = 32
    left_padding = [0, 5]
    hidden = mx.random.normal((2, sequence_length, args.hidden_size))
    query_residual = mx.random.normal((2, sequence_length, args.q_lora_rank))
    mask = create_causal_mask(sequence_length, left_padding=mx.array(left_padding))

    indices = indexer(
        hidden,
        query_residual,
        mask=mask,
        cache=BatchKVCache(left_padding),
    )
    assert indices is not None
    for batch_index, padding in enumerate(left_padding):
        last_query = set(indices[batch_index, 0, -1].tolist())
        assert set(range(padding, padding + 4)) <= last_query


def test_force_dense_is_an_explicit_diagnostic_switch(monkeypatch) -> None:
    monkeypatch.setenv("EXO_GLM_DSA_FORCE_DENSE", "true")
    indexer = GlmMoeDsaIndexer(_tiny_args())
    hidden = mx.zeros((1, 32, 32))
    query_residual = mx.zeros((1, 32, 16))

    assert indexer(hidden, query_residual, mask=None, cache=None) is None
