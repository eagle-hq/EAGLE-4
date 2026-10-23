---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-23
---

# TokenSpeed guide

Serve Qwen3.8-27B and its EAGLE-4 draft with TokenSpeed on NVIDIA GPUs. Its measured cost-model mode needs cost
tables that EAGLE-4 does not ship yet, so the first server below uses a static tree to check that generation works.

## Start a server

Install the engine with the [TokenSpeed install command](install.md#tokenspeed-cuda), then start a
server with a static tree:

```bash
tokenspeed serve Qwen/Qwen3.8-27B --disable-kvstore \
  --attention-backend trtllm --max-model-len 65536 --sampling-backend greedy \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config '{"mode": "static"}'
```

The command serves on one GPU. `--disable-kvstore` turns off TokenSpeed's host-memory KV cache tier, which by
default pins twice the GPU's KV cache in host memory; the measured recipe turns it off too. Add
`--tensor-parallel-size 4` to serve on four, as the measured recipe below does.

Static mode drafts the full tree every round, without a cost model. Its twenty-prompt greedy check has one
first divergence above the fp32 gate, so it checks setup rather than correctness
([known limits](known-limits.md#tokenspeed)). `--sampling-backend greedy` is required, because
draft trees are verified by the greedy and triton sampling backends.

## Send a request

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "Explain how a hash table works."}], "max_tokens": 256}'
```

The response holds the generated message. The first start compiles tree-round CUDA sources, which needs `nvcc`.
On Ubuntu 26.04 / glibc 2.43, set `CUDA_HOME=/usr/local/cuda-13.2` first ([troubleshooting](troubleshooting.md#install-problems)).

## The measured recipe

The TokenSpeed rows of the [support matrix](support-matrix.md#011-speed-by-engine-and-concurrency) use serve mode, where the
cost model sizes every round from cost tables measured for TokenSpeed on B300 at TP4. Set `COST_ROOT` to a
directory whose `cost/` folder holds such tables. vLLM and ROCm tables are not substitutes.

```bash
export COST_ROOT=/path/to/tokenspeed-tp4-costs
tokenspeed serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 \
  --attention-backend trtllm --max-model-len 65536 --max-num-seqs 32 \
  --disable-kvstore --seed 721 --gpu-memory-utilization 0.6 --disable-prefix-caching \
  --reasoning-parser qwen3 --sampling-backend greedy \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-num-steps 12 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config "{\"mode\": \"serve\", \"data_path\": \"$COST_ROOT\"}"
```

The single-client recipe uses `--max-num-seqs 4` and the concurrent recipe `--max-num-seqs 32`. Neither setting
promises a speedup for your workload. The serve-mode check passed twenty greedy prompts with 4 exact and 16
fp32-tolerance results. These are first-divergence checks, not bitwise output equality or sampled-distribution
qualification.

The same command serves one GPU with `--tensor-parallel-size 1` (it reuses the TP4 tables; no TP1 tables exist
yet) and 32K-token prompts within `--max-model-len 65536`. The
[support matrix](support-matrix.md#012-speed-by-engine-and-concurrency) has one row per setup.

## Sampled requests

For temperature above 0, sample with FlashInfer, one random stream per batch, and speculative sampling. A draft
tree runs at most 32 requests at a time:

```bash
tokenspeed serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 \
  --attention-backend trtllm --max-model-len 65536 --max-num-seqs 32 \
  --disable-kvstore --seed 721 --gpu-memory-utilization 0.6 --disable-prefix-caching \
  --reasoning-parser qwen3 --sampling-backend flashinfer --sampling-stream batch \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-num-steps 12 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config "{\"mode\": \"serve\", \"data_path\": \"$COST_ROOT\"}" \
  --enable-speculative-sampling
```

To run more than 32 requests at a time, serve chains, which need no cost tables:

```bash
tokenspeed serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 \
  --attention-backend trtllm --max-model-len 65536 --max-num-seqs 64 \
  --disable-kvstore --seed 721 --gpu-memory-utilization 0.6 --disable-prefix-caching \
  --reasoning-parser qwen3 --sampling-backend flashinfer --sampling-stream batch \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
  --speculative-algorithm-config '{"mode": "serve", "intensity": "chain_intensity", "chain_width_table": [[1, 8], [32, 4]]}' \
  --enable-speculative-sampling
```

At 64 concurrent requests the chain server measured 1.12× its no-spec
control at temperature 1, the tree server 0.87×. `tools/sampling_check.py run --api tokenspeed` compares a sampled
server with a no-spec server; top-k/top-p and min-p follow TokenSpeed's own filters
([known limits](known-limits.md#tokenspeed)).

Prompt logprobs, per-request sampling streams with draft trees, live cost measurement and native no-speculation
rounds are not supported. The [known limits](known-limits.md#tokenspeed) list the rest.
