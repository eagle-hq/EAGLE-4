---
type: tutorial
owner: eagle-harness
last_verified: 2026-10-20
---

# SGLang on AMD MI355X

Serve Qwen3.8-27B BF16 with its EAGLE-4 draft on one MI355X using the ROCm SGLang engine.
The matching MI355X cost profile is packaged with EAGLE-4 and selected automatically.
No `data_path` or profile environment variable is needed.

## Before starting

The fresh-install path remains under qualification and is deferred to a follow-up guide.
Use this measured serving recipe only in an already-qualified MI355X environment. It is not a
fresh-install procedure. See the [ROCm installation status](install.md#sglang-rocm-on-amd).
Run `eagle-harness-check`, but do not treat its import checks as runtime qualification.
Use the ROCm 7.2 runtime and locally downloaded target and draft directories. Profile selection compares
both local `config.json` files with the measured pair, so an unresolved Hub identifier is not sufficient.
The profile applies to MI355X, including VF, BF16, TP1 and no quantization. It is not a generic AMD profile.
Only enable `--trust-remote-code` for model code you have reviewed.

## Set the ROCm environment

Use the environment from the packaged-profile serving recipe:

```bash
: "${MODEL:?set the local Qwen3.8-27B BF16 target directory}"
: "${DRAFT:?set the local compatible EAGLE-4 draft directory}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1 SGLANG_INT4_WEIGHT=0 SGLANG_USE_ROCM700A=1
export SGLANG_ROCM_DISABLE_LINEARQUANT=0 SGLANG_USE_AITER=1 SGLANG_MOE_PADDING=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_SPLITKV_VERIFY_FLYDSL=0 SGLANG_SPLITKV_VERIFY_FLYDSL_COMBINE=0
```

Do not copy source-overlay `PYTHONPATH` settings into the installed environment.
Keep HIP prelaunch disabled and rounded recurrence enabled, as in the packaged-profile measurements.
The environment switch named `ROCM700A` is retained by the ROCm engine despite the runtime's 7.2 version.

## Start the server

The measured eight-request recipe uses AITER target attention, Triton draft attention and explicit
HIP graph batch sizes. Its padding setting differs from the CUDA recipe:

```bash
python -m sglang.launch_server --model-path "$MODEL" --tp-size 1 --dtype bfloat16 \
  --host 127.0.0.1 --port 30000 --trust-remote-code --attention-backend aiter --page-size 1 \
  --disable-overlap-schedule --disable-cuda-graph-padding --disable-prefill-cuda-graph \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 \
  --max-running-requests 8 --max-mamba-cache-size 8 --mem-fraction-static 0.90 \
  --context-length 32768 --chunked-prefill-size 8192 --max-total-tokens 65536 \
  --mamba-ssm-dtype bfloat16 --disable-radix-cache --reasoning-parser qwen3 \
  --random-seed 721 --skip-server-warmup \
  --speculative-algorithm EAGLE_HARNESS --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend triton \
  --speculative-attention-mode prefill --speculative-num-steps 12 \
  --speculative-eagle-topk 4 --speculative-num-draft-tokens 32 \
  --speculative-algorithm-config '{"mode":"serve"}'
```

Look for `selected packaged MI355X cost profile` in the startup log. If it is absent, check the
GPU, dtype, TP size and local model configurations before assuming the measured profile is in use.
An explicit `data_path` overrides automatic selection. The packaged profile includes measured costs
for batch sizes 1, 8 and 32, but that does not qualify every batch size or context length.
See the [support matrix](support-matrix.md) for correctness coverage.

## Generate text

In another terminal, wait for the model listing, then generate:

```bash
curl --fail http://127.0.0.1:30000/v1/models
curl --fail http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain how a hash table works.","sampling_params":{"temperature":0,"max_new_tokens":128}}'
```

The response must contain generated text and token metadata. The first request includes cold work because
the measured launch skips server warmup. Repeat it before timing. In Docker, bind the server to `0.0.0.0`
inside the container while publishing only the loopback host port.
Stop the server with Ctrl-C before changing its graph sizes or starting another GPU process.
