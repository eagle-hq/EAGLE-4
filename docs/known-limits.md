---
type: reference
owner: eagle-harness
last_verified: 2026-10-21
---

# Known limits

Use the [support matrix](support-matrix.md) for measured model, hardware and workload combinations.
Correctness and speed claims come from those measured runs. A loaded model, an admitted request cap or a compiled
kernel is a startup check.

## Models and hardware

- EAGLE-4 drafts need the trained candidate head and a supported draft structure.
- Dense and latent-MoE EAGLE-4 drafts and Mamba2 targets have adapters. Nemotron's SGLang comparison has
  first divergences under the fp32 gate that are being resolved. Release-stack Nemotron performance and correctness
  results for vLLM and TokenSpeed are in progress. An architecture that loads is not yet qualified.
- Serving measurements cover NVIDIA B300 and SGLang on AMD MI355X. Compilation on another NVIDIA or AMD
  architecture is separate evidence. Multi-node deployment and pipeline parallelism are being qualified.
- Throughput depends on the model, draft, engine, precision, TP size, attention backend, context length and
  workload. A result at one concurrency does not establish a result at another.

## Performance and correctness in progress

- SGLang TP4 single-request and 32K-context optimization is in progress, as is TP8 concurrency-8 prefill
  overhead and intermittent stalls. Repeated passes in one process are not independent server starts.
- SGLang above concurrency 64: with a running cap of 512 and the default tree (TP4), EAGLE-4 measured 1.03x the
  same engine without speculation at concurrency 64, 0.74x at 256 and 0.61x at 512 (3 starts each). Serve
  concurrency above 64 without speculation for now. The automatic high-load policy and high-load fixed-chain
  improvements are being qualified and stay off by default. Greedy agreement with no-spec at concurrency 512 is
  not claimed: the engine without speculation also differs between two starts there.
- SGLang native-OFF, including OFF forced by a cost profile, uses the engine's per-variant forward-mode
  decode graphs (the `forward_mode` field of the decode graph variant). The SGLang CUDA engine
  `0.5.23.dev252+eagle.5` has them. On the 0.1.0 SGLang CUDA engine and on ROCm, EAGLE-4 keeps native-OFF off.
- The SGLang temperature-1 sampled tree (draft proposals drawn in one fused launch and verified against their
  recorded probabilities) uses the engine's `sampling_step` verify hook, which the SGLang CUDA engine
  `0.5.23.dev252+eagle.5` has. On the 0.1.0 SGLang engines and on ROCm, temperature-1 requests use a greedy draft
  verified by the target's sampler.
  On every engine, a batch uses that path when any request has another temperature, a grammar or a penalty.
- The shipped vLLM TP4 rounds schedule improves the measured concurrency-64 and concurrency-128 cases at a
  running cap of 64, at temperature 0 and 1; see the
  [support matrix](support-matrix.md#010-high-load). The schedule has no TP1/TP8 profile.
  Running caps of 128 and above are being qualified. Native-OFF is off by default while its TP1 throughput is
  optimized to match native decoding.
- vLLM TP4 at concurrency 64: in a 64-prompt greedy comparison against no-spec, one prompt can take another token
  at one position, up to 5.4 bf16 ULP from the fp32 choice, beyond the 2-ULP rule. It appeared in 3 of 6 EAGLE-4
  starts with 0.1.1 and in 2 of 6 with 0.1.0 (the vLLM engine is the same); the other prompts and every
  twenty-prompt check at concurrency 1 pass. The cause is being investigated.
- AMD measurements apply to the MI355X TP1 Qwen3.8 profile. On one MI355X the speedup covers concurrency up
  to 32; concurrency 64 is being optimized, so serve it without speculation for now. HIP side-stream prelaunch
  stays off: its replay check passes and it measured no throughput gain on MI355X. Further target, draft and
  verify phase optimization is in progress.

## Requests

- Client concurrency can exceed the server's running cap and include queued requests. Keep both values with
  benchmark results. A cap of 512 is an admission limit, not a claim of acceleration or correctness at that load.
- SGLang supports accepted-token and input logprobs with the required host hooks. Images are refused.
- vLLM tree verification needs an unquantized KV cache. The target's sampling restrictions still apply.
  Sampling smoke tests check the path, not distribution equivalence on every workload. The vLLM
  sampled-distribution check covers temperature 1 with top-p 1 and no top-k; other temperatures, top-k/top-p
  and penalties are being qualified. vLLM refuses `min_p` and `logit_bias` with speculative decoding.
- Structured output and tool calls require the model's engine parser. Raw tool syntax in text is not a structured
  response. In a vLLM tree batch with a grammar constraint, every request in the batch advances one token per
  verify round. Check finish reasons and validate the completed output.
  See [compatibility](compatibility.md#structured-output-on-vllm).

## TokenSpeed

TokenSpeed serves EAGLE-4 with static and calibrated trees, chains and sampled trees on NVIDIA GPUs. Install the
engine, its native packages and `eagle-harness` together with the [install command](install.md#tokenspeed-cuda),
because the adapter needs the matching engine hooks. The 0.1.1 measurements cover serve mode at TP4 and
temperature 0, at concurrency 1, 8 and 64 ([support matrix](support-matrix.md#011-speed-by-engine-and-concurrency)).

- TokenSpeed runs on CUDA only. On AMD GPUs, serve with SGLang ROCm ([AMD guide](quickstart-amd.md)).
- On one GPU, start the server with `--disable-kvstore`. By default TokenSpeed adds a host KV cache tier twice the
  size of the GPU's KV cache, and a one-GPU server cannot pin that much host memory. The guide's commands pass the
  flag.
- TokenSpeed refuses prompt logprobs. Per-request sampled-tree overrides, live cost measurement and native
  no-speculation rounds are not available on TokenSpeed yet. A zero-draft speculative round is not a native
  no-speculation round.
- Serve mode needs cost tables measured for TokenSpeed. Tree and chain cost data must match the served shape, TP
  size, target, draft and recurrent-state dtype.
- These shapes are not yet measured on 0.1.1: serve mode at TP1 and TP8 (the
  [candidate rows](support-matrix.md#candidate-speed-by-engine-and-concurrency) have earlier results), temperature 1,
  where the sampled-output distribution is being qualified, and 32K-token contexts. TP4 single-request
  optimization, four-GPU chains, retraction under serving load and original-row output-logprob checks are also in
  progress. The sampled cap-64 path is refused.
- Static mode is a setup check, because its CUDA install smoke generates text with one first divergence above the
  fp32 gate. The serve-mode install smoke passes its greedy prompts with the CUDA 13.2 JIT. On Ubuntu 26.04 / glibc
  2.43, set `CUDA_HOME` to CUDA 13.2 ([troubleshooting](troubleshooting.md#install-problems)). These checks cover
  the toolkits named here.

## Calibration

Cost tables are configuration-specific. Keep the recorded settings with them and measure again after changing
the model, draft, TP size, precision, attention backend or engine. Keyed tables are checked at loading, but the
key does not encode every setting. A cache or table is not evidence of an optimal policy for every workload.

The vLLM public sweep measures batch-one tree forwards, not complete high-concurrency serving rounds.
The SGLang served-chain driver measures a separate path from dynamic-tree calibration. Its CLI uses the
per-variant native-decode graphs of the SGLang CUDA engine `0.5.23.dev252+eagle.5`; live-cost tree serving runs on
both CUDA engines. Built-in smoke prompts check the command, not representative acceptance. An interrupted
run keeps its partial output for diagnosis and writes `serve.sh` only on success. See
[calibration](calibration.md).

## Distribution

The [install guide](install.md) describes the release path: `eagle-harness` packages on PyPI,
patched engine wheels on the GitHub Pages indexes, and images on GHCR. Model and draft weights come from
Hugging Face under their owners' licenses.

- The published SGLang and vLLM images passed serving and correctness checks, and the published TokenSpeed image
  passed a GPU check in serve mode ([support matrix](support-matrix.md#011-installation-checks)). They carry
  `eagle-harness` 0.1.1 and the 0.1.1 engines. Use the image commands in the engine guides with them.
- The files under `docker/` show how the release images are built. The checked CUDA images carry a prebuilt
  `eagle-harness` wheel, so the published images are the checked path. The TokenSpeed Dockerfile takes an
  explicit compatible `BASE_IMAGE`.
- On AMD, install with pip: the ROCm closure is clean-installed from the public indexes and inference-tested on
  an MI355X. The ROCm Dockerfile is being container-qualified, and no ROCm image is published yet.
- TokenSpeed serve mode uses cost tables measured for TokenSpeed on the served setup. EAGLE-4 does not ship them
  or a command that measures them yet, so the [TokenSpeed guide](quickstart-tokenspeed.md) starts in static
  mode as a setup check.
- Use the pinned artifacts: stock engines and unpinned packages do not carry the EAGLE-4 hooks.
