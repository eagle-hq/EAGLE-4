---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-21
---

# TokenSpeed guide, experimental

Serve Qwen3.8-27B with its EAGLE-4 draft on four NVIDIA GPUs with TokenSpeed. TokenSpeed support is experimental
and runs on NVIDIA GPUs only. Its measured cost-model mode needs cost tables that EAGLE-4 does not ship yet, so the
first server below uses a static tree to check that generation works.

## Start a server

Install the engine with the [TokenSpeed install command](install.md#tokenspeed-cuda-experimental), then start a
server with a static tree:

```bash
tokenspeed serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 \
  --attention-backend trtllm --max-model-len 65536 --sampling-backend greedy \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config '{"mode": "static"}'
```

Static mode drafts the full tree every round, without a cost model. Its twenty-prompt greedy check has one
first divergence above the fp32 gate, so it checks setup rather than correctness
([known limits](known-limits.md#tokenspeed-experimental)). `--sampling-backend greedy` is required, because
draft trees are verified by the greedy and triton sampling backends.

## Send a request

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "Explain how a hash table works."}], "max_tokens": 256}'
```

The response holds the generated message. The first start compiles tree-round CUDA sources, which needs `nvcc`.
On Ubuntu 26.04 / glibc 2.43, set `CUDA_HOME=/usr/local/cuda-13.2` first ([troubleshooting](troubleshooting.md#install-problems)).

## The measured recipe

The TokenSpeed rows of the [support matrix](support-matrix.md#tokenspeed-experimental) use serve mode, where the
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

Prompt logprobs, per-request sampled trees, live cost measurement and native no-speculation rounds are not
supported. The [known limits](known-limits.md#tokenspeed-experimental) list the rest.
