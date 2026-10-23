---
type: reference
owner: eagle-harness
last_verified: 2026-10-23
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

## High load

On SGLang with four GPUs, the [high-load recipe](quickstart-sglang.md#high-load) keeps pace with plain decoding up
to 512 concurrent requests. These limits apply around it:

- The SGLang command of the README quick start is slower than plain decoding above about 32 to 64 concurrent
  requests. On one B300 it measured 0.73× the same engine without speculation at concurrency 64 and 0.59× at 256.
  With `--tp-size 4` it measured 0.80× at 64, 0.48× at 128, 0.37× at 256 and 0.32× at 512 (2 starts each). Three
  settings cause it. The radix cache, on by default, makes this hybrid model track prefix-cache state that a batch
  switching to plain decoding cannot carry yet, so the server speculates at every load. The default tree caps a
  four-GPU server at 32 running requests, while plain decoding admits several hundred. On one GPU, the engine
  reserves verify state for every admitted request, so a speculative server admits about a quarter of the
  requests plain decoding does. For high load on four GPUs, use the high-load recipe. A fix for the default
  command is planned.
- High load on one GPU is not supported yet. On one GPU, a speculative server started with
  `--disable-radix-cache` stops during draft graph capture, so keep the radix cache on there. The high-load recipe
  is for four GPUs, and the server log's advice to add `--disable-radix-cache` applies to four GPUs only.
- The first start of the high-load recipe on a machine compiles kernels and measures its live costs during its
  first requests. It measured 0.95× at 64 and 0.97× at 128 concurrent requests, and restarts with warm caches
  1.01× and 1.00× (3 starts each).
- The packaged choices cover B300 TP4 with the Qwen3.8-27B checkpoint pair. On other setups the first start also
  searches its choices while serving: with the packaged choices withheld on that same setup, first starts measured
  0.80× at concurrency 64 (4 starts, from 0.71× to 0.96×) and restarts 0.94× (2 starts). From 128 concurrent
  requests up they measured 0.97× or more.
- Requests decoded without speculation stop drafting until they finish, and so do requests that join their
  batch. After a load spike, a batch keeps decoding without speculation until it empties, even when the load
  drops in the meantime.
- Greedy agreement at 64 and 512 concurrent requests is checked with the fp32 rule, not claimed. At 64, one prompt
  takes another token at one position in 3 of 12 EAGLE-4 starts, a prefix where the engine without speculation
  emits the same token under another batch shape. At 512, the engine without speculation also differs between its
  own starts and between batch sizes: 3 and 1 of 512 prompts fall outside the 2-ULP rule, up to 3.8 bf16 ULP. Its
  default kernels are not batch-invariant. EAGLE-4 decodes without speculation at this load, and its differences
  fall on the same prompts. With `--enable-deterministic-inference` on both servers, restarts and three EAGLE-4
  starts match token for token at 512 (512 of 512 prompts). In that mode EAGLE-4 decodes without speculation at 64
  and 512, so it gives no speedup there, and the mode lowers throughput. Every twenty-prompt check passes.
- vLLM and TokenSpeed do not switch to plain decoding at high load yet. Above concurrency 64 they can be slower
  with speculation than without it, so serve those loads without speculation for now. The switch is planned for
  both engines.

## Performance and correctness in progress

- SGLang TP4 single-request and 32K-context optimization is in progress, as is TP8 concurrency-8 prefill
  overhead and intermittent stalls. Repeated passes in one process are not independent server starts.
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
  starts with 0.1.1 and in 2 of 6 with 0.1.0, and 0.1.2 shows it too (the vLLM engine is the same); the other
  prompts and every twenty-prompt check at concurrency 1 pass. vLLM without speculation emits the same token at
  that prefix when 64 sequences share the batch (up to 19 of 64 identical copies in one batch), so the difference
  comes from the engine's batch-dependent numerics, not from tree verification.
- AMD measurements apply to the MI355X TP1 Qwen3.8 profile. On one MI355X the speedup covers concurrency up
  to 32; concurrency 64 is being optimized, so serve it without speculation for now. HIP side-stream prelaunch
  stays off: its replay check passes and it measured no throughput gain on MI355X. Further target, draft and
  verify phase optimization is in progress.

## Requests

- Client concurrency can exceed the server's running cap and include queued requests. Keep both values with
  benchmark results. A cap of 512 is an admission limit, not a claim of acceleration or correctness at that load.
- SGLang supports accepted-token and input logprobs with the required host hooks. Images are refused.
- vLLM tree verification needs an unquantized KV cache. The target's sampling restrictions still apply.
  Sampling smoke tests check the path, not distribution equivalence on every workload. On TP4 at concurrency 1,
  8 and 64, the vLLM tree's sampled tokens match the engine without speculation at temperature 1 and 0.7, with
  top-k 8 and top-p 0.7, with a repetition penalty of 1.5, and with frequency and presence penalties of 0.5 (two
  tree starts against three starts without speculation, chi-square tests at a 1 % family-wise level). vLLM
  refuses `min_p` and `logit_bias` with speculative decoding.
- Structured output and tool calls require the model's engine parser. Raw tool syntax in text is not a structured
  response. In a vLLM tree batch with a grammar constraint, every request in the batch advances one token per
  verify round. Check finish reasons and validate the completed output.
  See [compatibility](compatibility.md#structured-output-on-vllm).

## TokenSpeed

TokenSpeed serves EAGLE-4 with static and calibrated trees, chains and sampled trees on NVIDIA GPUs. Install the
engine, its native packages and `eagle-harness` together with the [install command](install.md#tokenspeed-cuda),
because the adapter needs the matching engine hooks. The 0.1.2 measurements cover serve mode at TP1 and TP4,
temperature 0 and 1, 32K-token contexts and four-GPU chains
([support matrix](support-matrix.md#012-speed-by-engine-and-concurrency)).

- TokenSpeed runs on CUDA only. On AMD GPUs, serve with SGLang ROCm ([AMD guide](quickstart-amd.md)).
- On one GPU, start the server with `--disable-kvstore`. By default TokenSpeed adds a host KV cache tier twice the
  size of the GPU's KV cache, and a one-GPU server cannot pin that much host memory. The guide's commands pass the
  flag.
- Some request features are limits of the TokenSpeed engine, with or without speculation:
  - Prompt logprobs: a request with `logprob_start_len` 0 returns output-token logprobs only. Top-k logprobs need
    `max_new_tokens` of 1. Output-token logprobs match the engine without speculation (see the support matrix).
  - Sampling filters: with FlashInfer sampling, top-k/top-p and min-p cut the distribution where the engine's own
    sampler cuts it, which is not where SGLang cuts it. The sampled-distribution check therefore covers
    temperature, unrestricted and penalty settings; for top-k/top-p and min-p only its per-token tests apply.
  - Sampling stream: with FlashInfer sampling and a draft tree, TokenSpeed refuses `--sampling-stream per-request`,
    because its tree verifier draws one random stream for the batch. Use `--sampling-stream batch`, or chains.
  - Draft trees run at most 32 requests at a time: the engine sizes its tree buffers for 32. Chains run up to 512.
  - Native no-speculation rounds: TokenSpeed has no hook that skips the draft for a batch yet, so EAGLE-4 cannot
    fall back to plain decoding at high load on TokenSpeed ([high load](#high-load)). A zero-draft speculative round
    is not a native no-speculation round.
  - Live cost measurement: serve mode reads cost tables measured for TokenSpeed.
- Serve mode needs cost tables measured for TokenSpeed. Tree and chain cost data should match the served shape, TP
  size, target, draft and recurrent-state dtype. Only TP4 tables exist; the TP1 measurements reuse them.
- Retraction is not qualified. TokenSpeed reserves pages when it admits a request and queues the rest, so the
  measured servers never retract. With retractions forced by a test-only setting, one prompt in three chain starts
  fails the fp32 rule ([support matrix](support-matrix.md#012-correctness-and-request-features)).
- EAGLE-3 drafts are not checked on TokenSpeed. TP8 is not measured on 0.1.2 (the
  [candidate rows](support-matrix.md#candidate-speed-by-engine-and-concurrency) have earlier results). TP4
  single-request optimization is in progress.
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
