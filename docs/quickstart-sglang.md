---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-23
---

# SGLang guide

Serve Qwen3.8-27B and its EAGLE-4 draft with SGLang. SGLang prices the draft tree from costs it
measures on the served rounds, so nothing needs calibrating first. For MI355X, use the [AMD guide](quickstart-amd.md).

## Start a server

Install the engine with the [SGLang install command](install.md#sglang-cuda), then start the server. The model and
the draft download from Hugging Face on the first start:

```bash
sglang serve --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4
```

Or run the published image, which has the engine and EAGLE-4 installed:

```bash
docker run --gpus all --ipc host -p 127.0.0.1:30000:30000 \
  -e HF_HOME=/hf -v ~/.cache/huggingface:/hf \
  ghcr.io/eagle-hq/eagle-4-sglang:0.1.2 \
  sglang serve --model-path Qwen/Qwen3.8-27B --host 0.0.0.0 \
  --disable-overlap-schedule --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-num-steps 12 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config '{}'
```

Both commands serve on one GPU. Add `--tp-size 4` to serve on four, as the measured recipe below does.

By default EAGLE-4 drafts a tree of up to 12 steps with 4 candidates per step, verifies up to 16 tokens, and
prices every round from live costs. `--speculative-num-steps`, `--speculative-eagle-topk`,
`--speculative-num-draft-tokens` and `--speculative-algorithm-config` override those
([configuration](configuration.md)). These defaults need `eagle-harness` 0.1.1 or newer, which the image carries.
The image command spells the same settings out and adds `--disable-overlap-schedule`; EAGLE-4 also serves with
SGLang's default overlap scheduler.

## Send a request

When the server logs `The server is fired up and ready to roll!`, send a request from another terminal:

```bash
curl http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "Explain how a hash table works."}], "max_tokens": 256}'
```

The response holds the generated message. The first requests compile kernels and seed the live costs, so send a
few before you measure speed. Compare against the same engine without the `--speculative-*` flags.

## The measured recipe

The SGLang rows of the [support matrix](support-matrix.md#011-speed-by-engine-and-concurrency) were measured with this command.
It adds the attention backends, graph settings, memory sizes and recurrent-state dtype of the benchmark:

```bash
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_TIMEOUT_KEEP_ALIVE=3600
sglang serve --model-path Qwen/Qwen3.8-27B --tp-size 4 --dtype bfloat16 \
  --trust-remote-code --attention-backend flashinfer --page-size 1 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --disable-overlap-schedule --cuda-graph-backend-decode full \
  --mem-fraction-static 0.75 --context-length 196608 --max-total-tokens 196608 \
  --chunked-prefill-size 16384 --mamba-ssm-dtype bfloat16 --disable-radix-cache \
  --reasoning-parser qwen3 --random-seed 721 --skip-server-warmup \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer \
  --speculative-attention-mode prefill --speculative-num-steps 12 \
  --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config '{"mode":"serve"}'
```

The benchmark warmed the server from its client, hence `--skip-server-warmup`: the first request after this start
compiles kernels and seeds the costs. Keep CUDA graph padding enabled, and do not add
`--disable-cuda-graph-padding`. `--max-running-requests 32` is the tree default; set a larger cap, up to 512, for
larger batches, or serve with the [high-load recipe](#high-load).

## High load

For many concurrent requests on four GPUs, serve with the high-load recipe. It raises the running cap and the decode
graph limit to 512 and turns the radix cache off, so EAGLE-4 can decode a batch without speculation wherever
speculation would be slower:

```bash
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_TIMEOUT_KEEP_ALIVE=3600
sglang serve --model-path Qwen/Qwen3.8-27B --tp-size 4 --dtype bfloat16 \
  --trust-remote-code --attention-backend flashinfer --page-size 1 \
  --max-running-requests 512 --cuda-graph-max-bs-decode 512 \
  --mem-fraction-static 0.70 --context-length 32768 \
  --chunked-prefill-size 16384 --mamba-ssm-dtype bfloat16 --disable-radix-cache \
  --random-seed 721 --skip-server-warmup \
  --speculative-algorithm EAGLE_HARNESS \
  --speculative-draft-model-path EAGLE-Corp/Qwen3.8-27B-EAGLE4 \
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer \
  --speculative-attention-mode prefill --speculative-num-steps 12 \
  --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config '{"mode":"serve"}'
```

With the radix cache on, this hybrid model tracks prefix-cache state that a batch switching to plain decoding
cannot carry, so the server speculates at every load. On one GPU, keep the radix cache on: high load on one GPU is
not supported yet ([known limits](known-limits.md#high-load)).

Each speedup divides EAGLE-4 throughput by the same command without the `--speculative-*` flags, in the same
allocation, on 4x B300 with Qwen3.8-27B and its EAGLE-4 draft at temperature 0. Columns are concurrent requests:

| Server start | 1 | 8 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First start on a machine | 2.80× | 1.67× | 0.95× | 0.97× | 1.00× | 0.99× |
| Restart with warm caches | 2.76× | 1.72× | 1.01× | 1.00× | 1.00× | 0.99× |

The first start compiles kernels and measures its live costs during its first requests, which costs up to 5% at 64
and 128 concurrent requests. A restart reuses the kernels and the costs and choices cached under
`~/.cache/eagle-harness/costs`. This setup ships packaged choices; on another GPU, TP size or model pair the first
start also searches them ([known limits](known-limits.md#high-load)). The
[support matrix](support-matrix.md#012-high-load) has the tokens/s and the correctness checks.

Live costs are cached per GPU, TP size, model pair, engine and dtype under `~/.cache/eagle-harness/costs`.
The [calibration guide](calibration.md) explains live costs and offline chain calibration, and the
[known limits](known-limits.md) cover sampled requests, logprobs and multimodal inputs.
