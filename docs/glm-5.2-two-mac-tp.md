# GLM-5.2 tensor parallelism on two 512 GB Mac Studios

This is the supported path for the patched exo tree. It uses true two-rank MLX
tensor parallelism over JACCL/RDMA; each Mac runs all 78 layers with the large
linear and MoE tensors split across ranks.

## What causes the corrupted output

There is no native 120K context boundary in GLM-5.2. Both tested checkpoints
declare `max_position_embeddings = 1048576`, `index_topk = 2048`, and the same
21-full/57-shared IndexShare schedule.

The observed failure has three credible, independently testable paths:

1. Stock mlx-lm does not yet have merged GLM-5.2 IndexShare support. Exo's local
   patch supplies it. Without that patch, 57 shared layers have no checkpoint
   indexer weights and `strict=False` leaves randomly initialized selectors.
2. Sparse DSA engages after 2,048 cached tokens. The inherited selector can
   evict the first attention-sink tokens and real GLM-5.2 runs then collapse
   into repetition, zeros, or punctuation noise. The local GLM-only patch uses
   the reference LayerNorm epsilon (`1e-6`) and preserves four sinks plus the
   latest 128 positions. The sink mitigation follows an upstream bug fix that
   is still open, so it has a kill switch for controlled comparison.
3. A 4,096-token prefill chunk near a 120K cache creates very large temporary
   score tensors. On each of two TP ranks, the main positional scores and the
   replicated 32-head indexer scores can together approach 59 GiB at BF16,
   before masks and Metal workspaces. Reducing the chunk to 1,024 cuts that
   estimate to about 15 GiB.

The JACCL `all_reduce` corruption race was also real, but it is fixed in MLX
0.32.0. Both nodes must actually be running that wheel rather than a stale
environment. See [MLX PR #3451](https://github.com/ml-explore/mlx/pull/3451).

Relevant upstream status:

- [IndexShare draft PR](https://github.com/ml-explore/mlx-lm/pull/1410)
- [same-stack GLM sparse failure](https://github.com/ml-explore/mlx-lm/issues/1453)
- [attention-sink root-cause and ablation](https://github.com/ml-explore/mlx-lm/issues/1443)
- [pending sink-preservation patch](https://github.com/ml-explore/mlx-lm/pull/1552)
- [current Transformers GLM reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)

## 1. Prepare both Macs

Use the same Mac model, exact macOS build, repository commit, `uv.lock`, Python
version, and environment variables on both machines.

JACCL requires macOS 26.2 or later and RDMA enabled from Recovery:

```text
rdma_ctl enable
```

After reboot, verify on both nodes:

```bash
ibv_devices
```

Connect the Macs directly with a certified Thunderbolt 5 cable. On Mac Studio,
do not use the Thunderbolt port next to Ethernet. Keep both Macs on a normal
LAN as well so exo discovery and the JACCL coordination side channel can work.
For a two-node cluster, the single direct cable is a fully connected mesh.

From the same exo checkout on each Mac:

```bash
sudo ./tmp/set_rdma_network_config.sh
uv sync --extra mlx --frozen \
  --reinstall-package mlx --reinstall-package mlx-lm
```

Confirm the effective packages on both nodes:

```bash
git rev-parse HEAD
uv run python -c \
  'import mlx; from importlib.metadata import version; print(mlx.__version__, version("mlx-lm"))'
```

For this tree, expect MLX `0.32.0` and mlx-lm `0.31.3`; the identical lock file
also pins the exact mlx-lm Git commit. Do not mix the old experimental MLX fork
with the official 0.32.0 macOS wheel.

Apple's current JACCL setup and topology details are in the
[MLX distributed guide](https://ml-explore.github.io/mlx/build/html/usage/distributed.html#getting-started-with-jaccl).

## 2. Start a conservative validation cluster

Run this on both Macs from the same checkout:

```bash
EXO_MLX_PREFILL_STEP_SIZE=1024 \
uv run exo -vv --namespace glm52-tp --no-fast-synch --no-batch
```

`--no-fast-synch` is intentional: MLX documents fast synchronization as
unreliable and recommends leaving it off. `--no-batch` removes continuous
batching from the first correctness pass. After the full sweep succeeds,
restart without `--no-batch` and repeat it before serving concurrent traffic.

## 3. Place exactly two Tensor/JACCL ranks

Use the API of whichever node is currently master. The kernelpool conversion is
recommended because its card documents conversion provenance. It occupies
about 790 GB on disk, so plan for at least 850 GB free on each Mac; each rank
loads approximately half of the packed weights into memory.

Preview placements:

```bash
curl --get 'http://MAC1:52415/instance/previews' \
  --data-urlencode 'model_id=kernelpool/GLM-5.2-8bit' | jq
```

Place the model, explicitly overriding exo's Pipeline/Ring defaults:

```bash
curl -fsS -X POST 'http://MAC1:52415/place_instance' \
  -H 'content-type: application/json' \
  -d '{
    "model_id": "kernelpool/GLM-5.2-8bit",
    "sharding": "Tensor",
    "instance_meta": "MlxJaccl",
    "min_nodes": 2
  }' | jq
```

Wait for loading to finish:

```bash
curl -N --get 'http://MAC1:52415/instance/await' \
  --data-urlencode 'model_id=kernelpool/GLM-5.2-8bit' \
  --data-urlencode 'timeout_seconds=0'
```

The patch rejects Pipeline placement for GLM DSA. A normal 39/39 split starts
one shard on a shared-indexer layer without the previous full layer's top-k, so
Pipeline output is not equivalent to the checkpoint.

GLM-5.2 also loads with strict checkpoint validation in this tree. A missing or
misnamed full-layer indexer tensor now stops model loading instead of leaving a
random parameter that only becomes visible once sparse DSA engages.

## 4. Prove the sparse boundary before testing 120K

Run deterministic needle probes from either Mac:

```bash
uv run python scripts/probe_glm52_context.py \
  --base-url http://MAC1:52415 \
  --model kernelpool/GLM-5.2-8bit \
  --lengths 1024,2049,4096
```

Every JSON line should report `"needle_found": true`, no zero-noise pattern,
and a sensible printable/repetition ratio. Then run the expensive sweep:

```bash
uv run python scripts/probe_glm52_context.py \
  --base-url http://MAC1:52415 \
  --model kernelpool/GLM-5.2-8bit \
  --lengths 32768,65536,120000,131072,200000 \
  --timeout 7200
```

The model config advertises 1,048,576 positions, but this tree intentionally
keeps 202,752 as the operational model-card limit until longer runs are proven.
The replicated BF16 MLA plus indexer cache is about 93 KiB/token/rank: roughly
10.6 GiB at 120K, 18.0 GiB at 202,752, and 93 GiB at 1M. Prefix-cache copies,
Metal workspaces, the OS, and roughly 395 GB/rank of weights are additional.

## 5. A/B diagnosis if a probe fails

Change only one variable at a time and restart both nodes after changing an
environment variable.

| Observation | Most likely path | Next comparison |
|---|---|---|
| Collapse just after 2,048 tokens | DSA selector | Keep the prompt below 8K and try `EXO_GLM_DSA_FORCE_DENSE=true` |
| Dense succeeds, sparse fails | DSA/indexer confirmed | Compare sink guard on/off; capture the exact prompt |
| 1,024 chunks pass but 4,096 fails near 120K | Metal transient memory | Use 512 or 1,024 and inspect Metal peak memory |
| `--no-batch` passes, batching fails | Batch/prefix-cache path | Repeat through `/bench/chat/completions`; test unequal histories |
| Tensor/Ring passes but Tensor/JACCL fails | Collective/transport path | Verify both package hashes, disable fast sync, inspect JACCL logs |
| Both Tensor transports fail identically | Model/indexer or memory | Do the dense and chunk-size comparisons |

Available GLM-specific switches:

```bash
# Default, high-evidence mitigation from mlx-lm PR #1552
EXO_GLM_DSA_PRESERVE_SINKS=true

# Reference-selection ablation; not recommended for serving
EXO_GLM_DSA_PRESERVE_SINKS=false

# Diagnostic only: dense attention becomes prohibitively large at long context
EXO_GLM_DSA_FORCE_DENSE=true
```

To isolate JACCL, request the same `Tensor` placement with
`"instance_meta":"MlxRing"`. Ring over IP is much slower, but even a short
2,049/4,096 comparison is useful. If a smaller GLM-5.2 quant fits on one Mac,
repeat the same prompt single-node; a single-node failure rules out collectives.

## Checkpoint choice

Use `kernelpool/GLM-5.2-8bit` first. Despite its name,
`mlx-community/GLM-5.2-fp8` also declares MLX affine 8-bit quantization
(`bits=8`, `group_size=64`, `mode=affine`); it is not a native runtime FP8
comparison. The repositories have different weight hashes, but switching
between them does not isolate FP8 versus integer-8 arithmetic.

CUDA-native servers such as vLLM and SGLang do not provide cross-Mac Metal
tensor parallelism. Bare mlx-lm uses the same underlying MLX model path and its
GLM-5.2 IndexShare work is still unmerged, so this patched Exo/MLX route is the
practical Mac-native option today, with the regression sweep treated as part of
deployment rather than assuming the advertised 1M window is already validated.
