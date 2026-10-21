---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-20
---

# SGLang CUDA quickstart

Serve Qwen3.8-27B BF16 with its EAGLE-4 draft on four B300 GPUs. SGLang measures the draft and
verification costs during served rounds, so this recipe needs no offline calibration directory.
For MI355X, use the separate [AMD quickstart](quickstart-amd.md).

## Before starting

Complete the [SGLang CUDA installation](install.md#sglang-cuda) and run `eagle-harness-check`.
Set `MODEL` and `DRAFT` to readable local target and compatible draft directories.
The release recipe below uses TP4 and a running-request cap of 32. See the
[support matrix](support-matrix.md) before changing the model, hardware or tensor-parallel size.
Only enable `--trust-remote-code` for model code you have reviewed.

## Start the server

Use the release tree shape and keep CUDA graph padding enabled:

```bash
: "${MODEL:?set the local Qwen3.8-27B BF16 target directory}"
: "${DRAFT:?set the local compatible EAGLE-4 draft directory}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_TIMEOUT_KEEP_ALIVE=3600
python -m sglang.launch_server --model-path "$MODEL" --tp-size 4 --dtype bfloat16 \
  --trust-remote-code --attention-backend flashinfer --page-size 1 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --disable-overlap-schedule --cuda-graph-backend-decode full \
  --mem-fraction-static 0.75 --context-length 196608 --max-total-tokens 196608 \
  --chunked-prefill-size 16384 --mamba-ssm-dtype bfloat16 --disable-radix-cache \
  --reasoning-parser qwen3 --random-seed 721 --skip-server-warmup \
  --host 127.0.0.1 --port 30000 \
  --speculative-algorithm EAGLE_HARNESS --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer \
  --speculative-attention-mode prefill --speculative-num-steps 12 \
  --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 \
  --speculative-algorithm-config '{"mode":"serve"}'
```

Do not add `--disable-cuda-graph-padding` to this CUDA recipe. The release benchmark uses a separate
client warmup, hence `--skip-server-warmup`. With this exact command, the first generation below performs
cold compilation and cost seeding. Remove that flag if you want the engine's startup warmup instead,
but do not call the modified startup sequence an identical benchmark reproduction.

The shape flags bound the tree. Initial served rounds price captured verification widths and later
rounds refine their costs. The cache is keyed by GPU, TP size, model configurations, engine and dtype
under `~/.cache/eagle-harness/costs`. No private cost-table path is required.

## Generate text

In another terminal, wait until the model listing succeeds, then send a greedy request:

```bash
curl --fail http://127.0.0.1:30000/v1/models
curl --fail http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain how a hash table works.","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

The response must contain generated text and token metadata. Repeat the generation request after
compilation before measuring latency. A successful model listing alone does not check decoding.
For Docker, bind the server to `0.0.0.0` inside the container and keep the published host port on loopback.

## Next step

Use representative prompts and compare against the same engine without speculation before deployment.
The [calibration guide](calibration.md) explains live costs and offline chain calibration.
The [known limits](known-limits.md) distinguish greedy checks, sampled requests, logprobs and multimodal support.
Stop the server with Ctrl-C before starting another server or calibration on the same GPUs.
