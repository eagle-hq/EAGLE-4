---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-26
---

# TokenSpeed guide

Serve Qwen3.8-27B and its EAGLE-4 draft with TokenSpeed on NVIDIA GPUs. TokenSpeed serves EAGLE-4 in serve mode,
where the cost model sizes every round from cost tables measured on your serving setup. EAGLE-4 0.1.3 ships no
TokenSpeed cost tables, so calibrate once with the driver below, then start the server with that profile.

Install the engine with the [TokenSpeed install command](install.md#tokenspeed-cuda) first. Static mode
(`"mode": "static"`) stops at server start with the released EAGLE-4 draft in this release
([known limits](known-limits.md#tokenspeed)).

## Calibrate costs

Stop other servers on your allocated GPUs, choose a fresh output directory, and measure the same draft shape and
TP size you will serve:

```bash
export COST_ROOT=$PWD/tokenspeed-tp4-costs
python -m eagle_harness.tokenspeed.measure_run \
  --model Qwen/Qwen3.8-27B --draft-model EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --run-dir "$COST_ROOT" --tp-size 4 --compute-intensity high_intensity \
  --topk 4 --num-steps 12 --num-draft-tokens 16 --max-bs 32 \
  --mamba-ssm-dtype float32
```

The driver starts its own greedy engine in `calibrate` mode, measures draft and target stages with CUDA events,
then measures tree-cascade attention over the context grid. It uses the shared calibration records and robust fits
to write `test/bs_<N>/` raw timings and `cost/bs_<N>/model_{slm,llm}/` profiles for every batch size from 1 through
`--max-bs`. Unstable fits are retried up to three times and a failed batch stops the command. Keep partial output
for diagnosis, but use only a completed profile for serving.

Calibration requests commit fixed tokens to control context growth. They are measurement traffic, not useful
generation or a correctness test. Costs must match GPU, model pair, engine build, TP, draft shape, attention backend
(`trtllm`), weight dtype (`bfloat16`) and Mamba state dtype. The driver records its settings in
`config/tokenspeed.json`. Recalibrate when any of these change. The default context grid ends at 8192 tokens.
Use `--context-stop` and `--context-step` for a longer grid. This is offline calibration, not live cost adaptation.
The first engine start compiles kernels and tunes GEMMs, which took about 20 minutes on an empty cache.

## Start a server

Point `data_path` at the completed calibration and start serve mode on the same four GPUs:

```bash
tokenspeed serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 --dtype bfloat16 \
  --attention-backend trtllm --max-model-len 65536 --max-num-seqs 32 \
  --disable-kvstore --seed 721 --gpu-memory-utilization 0.6 --disable-prefix-caching \
  --reasoning-parser qwen3 --sampling-backend greedy --mamba-ssm-dtype float32 \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-num-steps 12 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config "{\"mode\": \"serve\", \"data_path\": \"$COST_ROOT\"}"
```

`--dtype bfloat16` matches the calibration engine, so the server reuses its GEMM tuning instead of tuning again.
`--disable-kvstore` turns off TokenSpeed's host-memory KV cache tier, which by default pins twice the GPU's KV
cache in host memory. `--sampling-backend greedy` is required, because draft trees are verified by the greedy and
triton sampling backends. vLLM and ROCm cost tables are not substitutes.

The single-client recipe uses `--max-num-seqs 4` and the concurrent recipe `--max-num-seqs 32`. Neither setting
promises a speedup for your workload. On four B300 GPUs this command, with a fresh TP4 calibration, served twenty
greedy prompts with 3 outputs identical to TokenSpeed without speculation and 17 first divergences within the fp32
tolerance, the same in two independent starts. These are first-divergence checks, not bitwise output equality or
sampled-distribution qualification.

For one GPU, calibrate with `--tp-size 1` and serve with `--tensor-parallel-size 1`, using that new profile rather
than TP4 costs. The server also supports 32K-token prompts within `--max-model-len 65536`. The
[support matrix](support-matrix.md#012-speed-by-engine-and-concurrency) has one row per setup measured on 0.1.2.

## Send a request

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "Explain how a hash table works."}], "max_tokens": 256}'
```

The response holds the generated message. The first start compiles tree-round CUDA sources, which needs `nvcc`.
Send a few requests before you measure speed ([benchmarking](faq.md#how-do-i-benchmark-eagle-4)).
On Ubuntu 26.04 / glibc 2.43, set `CUDA_HOME=/usr/local/cuda-13.2` first ([troubleshooting](troubleshooting.md#install-problems)).

## Sampled requests

For temperature above 0, sample with FlashInfer, one random stream per batch, and speculative sampling. A draft
tree runs at most 32 requests at a time. The harness supplies its rejection sampler through TokenSpeed's
`sampling.tree_sampler` hook. The sampler and CUDA sources ship under the Eagle license in `eagle-harness`,
not in the engine. Native EAGLE-3 and MTP keep TokenSpeed's greedy and position-keyed Triton tree verification.
Installing the harness does not bind a sampler for native drafters.

Start sampled serving with:

```bash
tokenspeed serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 --dtype bfloat16 \
  --attention-backend trtllm --max-model-len 65536 --max-num-seqs 32 \
  --disable-kvstore --seed 721 --gpu-memory-utilization 0.6 --disable-prefix-caching \
  --reasoning-parser qwen3 --sampling-backend flashinfer --sampling-stream batch --mamba-ssm-dtype float32 \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-num-steps 12 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config "{\"mode\": \"serve\", \"data_path\": \"$COST_ROOT\", \"sampled_proposals\": true}"
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
  --speculative-algorithm-config '{"mode": "serve", "intensity": "chain_intensity", "chain_width_table": [[1, 8], [32, 4]], "sampled_proposals": true}'
```

On 0.1.2, at 64 concurrent requests, the chain server measured 1.12× its no-spec
control at temperature 1 and the tree server 0.87×. `tools/sampling_check.py run --api tokenspeed` compares a sampled
server with a no-spec server; top-k/top-p and min-p follow TokenSpeed's own filters
([known limits](known-limits.md#tokenspeed)).

Prompt logprobs, per-request sampling streams with draft trees, live cost measurement and native no-speculation
rounds are not supported. The [known limits](known-limits.md#tokenspeed) list the rest.
