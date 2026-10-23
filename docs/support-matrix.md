---
type: reference
owner: eagle-harness
last_verified: 2026-10-23
---

# Measured support matrix

Each cell divides EAGLE-4 throughput by the same engine's matched no-speculation control, measured in the same
allocation, on Qwen3.8-27B with its EAGLE-4 draft unless the table names another model. The page has four parts:
[0.1.0](#010-released), [0.1.1](#011-released), [0.1.2](#012-released-current) and
[unreleased candidates](#unreleased-measured-on-candidates-not-in-a-release). Shipped and candidate numbers never
share a row. Column headers are client concurrency, the number of requests the client keeps in flight.

| Mark | Meaning |
|---|---|
| ✅ | The greedy check passed and EAGLE-4 is at least 0.95× its no-spec control |
| ⚠️ | EAGLE-4 is below 0.95× its no-spec control, or the cell has a limit named in its Sources line |
| ❌ | The greedy check has a failure under the fp32 rule |
| — | Not reported |

[How these were measured](#how-these-were-measured) defines the greedy check, the default workloads and the terms.

## 0.1.0 (released)

`eagle-harness` 0.1.0 with `sglang==0.5.23.dev252+eagle.4`, `vllm==0.32.0rc1.dev283+eagle.4`,
`tokenspeed==0.1.1+eagle.4` and, on AMD, `sglang==0.5.23.dev252+eagle.4.rocm`. Every row runs on NVIDIA B300 except
the MI355X row.

### 0.1.0 speed by engine and concurrency

| Engine and setup | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| SGLang TP4, temperature 0 | ✅ 3.40× | ✅ 1.91× | — | ✅ 1.35× |
| SGLang TP4, temperature 1 | ✅ 2.32× | ✅ 1.75× | — | ✅ 1.19× |
| SGLang TP4, 32K-token context | ✅ 2.99× | — | — | — |
| SGLang TP1 | ✅ 3.33× | ✅ 2.11× | — | — |
| SGLang TP8 | ✅ 3.43× | — | — | — |
| vLLM TP4 | ✅ 2.99× | ✅ 1.70× | — | ✅ 1.16× |
| vLLM TP1 | ✅ 2.69× | ✅ 1.94× | — | — |
| vLLM TP8 | ✅ 2.35× | ✅ 1.40× | — | — |
| TokenSpeed TP4 | ✅ 2.96× | ✅ 2.02× | — | ✅ 1.42× |
| SGLang ROCm, AMD MI355X TP1 | — | ✅ 2.07× | ✅ 1.56× | — |

Sources:

1. SGLang TP4, temperature 0: EAGLE-4 735.7 tokens/s at 1 (6 starts, no-spec 216.3 over 3 starts), 2820.6 at 8
   and 5281.3 at 64 (2 starts each, no-spec 1479.4 and 3925.3). Running cap 32, live costs. Concurrency 64 queues
   into the 32 running slots and was measured with graph padding disabled. Every start passes its twenty-prompt
   check with 5 exact / 15 near tie. In the 0.1.1 qualification allocation the same 0.1.0 packages, served by the
   0.1.1 tuned command, measured 3.45×, 2.11× and 1.38× (2 starts).
2. SGLang TP4, temperature 1, from those two 0.1.0 starts in the 0.1.1 qualification allocation: EAGLE-4 501.3,
   2684.6 and 5467.5 tokens/s. The no-spec control is SGLang without speculation at temperature 1 in the same
   allocation, from the 0.1.1 packages (1 start): 216.3, 1531.5 and 4589.6. Both starts give 4 exact / 16 near tie
   on the twenty-prompt check.
3. SGLang TP4, 32K: EAGLE-4 717.7 tokens/s over 3 starts with a warm live-cost cache, no-spec 240.3 (1 start). All
   six EAGLE-4 starts give 5 exact / 15 near tie. A start with an empty cost cache measured 608.9 (2.53×). Packages
   installed into a clean environment and served from outside any checkout, measured after the release.
4. SGLang TP1: 343.1 against 103.1 tokens/s at 1 and 1503.7 against 711.7 at 8, 2 starts each, graph padding
   disabled at 8. Checks give 6 exact / 14 near tie at 1 and 5 exact / 14 near tie / 1 fp32 pass at 8, in both
   starts.
5. SGLang TP8: 828.6 against 241.7 tokens/s, 2 starts, 5 exact / 14 near tie / 1 fp32 pass. Concurrency 8 is not
   reported while its throughput variability is investigated.
6. vLLM TP4: static tree (top-k 4, depth 14, verify width 16, `CUSTOM` attention), running cap 64. 722.8 against
   241.6 tokens/s at 1 and 2883.9 against 1699.9 at 8 (2 starts each). Concurrency 64 is the
   [0.1.0 high-load row](#010-high-load).
7. vLLM TP1 and TP8: static tree, running cap 8, 2 starts each. TP1 279.9 against 104.0 and 1389.1 against 715.5,
   check 8 exact / 12 near tie. TP8 745.4 against 317.2 and 2890.0 against 2063.0, and its two concurrency-8 starts
   differ by 11%.
8. TokenSpeed TP4: serve mode with TokenSpeed cost tables, running cap 4 at 1 and 32 at 8 and 64. Concurrency 1 is
   22 conversations and 31 turns with up to 1024 new tokens: 819.1 against 276.5 tokens/s. Eleven first turns with
   up to 256 tokens give 747.0 against 273.7 (2.73×). Concurrency 8 and 64 use 22 and 88 first turns with up to 256
   new tokens: 3150.2 against 1555.7 and 6950.2 against 4894.4. Every cell has 2 starts per arm, and the
   twenty-prompt check gives 4 exact / 16 fp32 pass. Wheels `tokenspeed==0.1.1+eagle.4`,
   `tokenspeed-kernel==0.1.4+eagle.4`, `tokenspeed-scheduler==0.1.40+eagle.4`.
9. MI355X: measured MI355X costs, 44 conversations and 60 turns with up to 1024 new tokens, running cap equal to
   the concurrency. 1145.9 against 552.7 tokens/s at 8 and 1980.8 against 1269.2 at 32 (2 EAGLE-4 starts, 1 no-spec
   start). Checks give 5 exact / 15 near tie at 8, and at 32 give 3 exact / 14 near tie / 3 fp32 pass and
   5 exact / 13 near tie / 1 fp32 pass / 1 fp32-argmax pass. Concurrency 1 reaches 289.4 tokens/s without a matched
   no-spec control. Concurrency 64 is being optimized
   ([known limits](known-limits.md#performance-and-correctness-in-progress)).

### 0.1.0 high load

| Engine and setup | 64 | 128 |
|---|---|---|
| vLLM TP4, temperature 0 | ✅ 1.16× | ✅ 1.21× |
| vLLM TP4, temperature 1 | ✅ 1.03× | — |

Sources:

1. The vLLM guide's recipe: running cap 64, calibrated TP4 costs and the packaged rounds file, which narrows the
   verify width from 16 to 4 at 20 or more running requests. SPEED-Bench first turns with up to 512 new tokens,
   220 requests at 64 and 440 at 128 (queued into the 64 slots). 8155.1 against 7002.5 and 8552.5 against
   7063.0 tokens/s, 2 starts per arm, interleaved. Without the rounds file concurrency 64 measures 0.76×. Twenty
   sequential prompts give 5 exact / 14 near tie / 1 fp32 pass and 64 concurrent prompts at 64 give
   17 exact / 46 near tie / 1 fp32 pass.
2. Temperature 1 with top-p 1, no top-k and a fixed seed: 7267.9 against 7034.3 tokens/s, 2 starts. Without the
   rounds file 0.65×. The sampled-distribution check passes all 36 per-token and joint tests at 1, 8 and 64.

### 0.1.0 correctness and request features

| Configuration or feature | Evidence |
|---|---|
| SGLang Qwen TP4 concurrency 8/64 replay | Twenty-prompt comparisons give 5 exact / 15 near tie / 0 fp32 / 0 failed in both starts |
| SGLang Qwen TP1 live costs | Concurrency 1 gives 6 exact / 14 near tie; concurrency 8 gives 5 exact / 14 near tie / 1 fp32-tolerance pass, in both starts |
| vLLM Qwen TP4 tree and chain | Twenty-prompt checks pass with fp32 arbitration for one prompt in each mode |
| vLLM Qwen TP1 tree | Twenty-prompt check gives 8 exact / 12 near tie / 0 fp32 / 0 failed |
| vLLM emitted-token logprobs | All 768 checked tokens per configuration are own-row top-1, shared-prefix deltas are below 0.12 |
| vLLM stop strings, stop IDs, EOS and `ignore_eos` | All 32 requests pass per checked server under concurrency 16 |
| vLLM temperature-1 smoke | All 18 requests per checked server finish legally and samples differ across seeds |
| vLLM temperature-1 sampled distribution | All 36 per-token and joint tests pass at concurrency 1, 8 and 64 (top-p 1, no top-k) |
| vLLM prefix caching | All eight second-pass outputs match the first pass per checked configuration |
| SGLang logprobs and sampling | Accepted-token and input logprobs and target-side speculative sampling are implemented |
| Nemotron / Mamba2 | Adapters are implemented on SGLang and vLLM, but a general correctness-qualified speed claim is not established |

### 0.1.0 artifact installation checks

| Artifact and mode | Correctness result | Tokens/s | Qualification limit |
|---|---|---:|---|
| SGLang image `eagle-4-sglang:0.1.0` | 4 exact / 16 near ties | 537.40 | Hook, dependency, quickstart request and correctness checks pass |
| vLLM image `eagle-4-vllm:0.1.0` | 5 exact / 14 near ties / 1 fp32-tolerance pass | 717.36 | Hook, dependency, quickstart request and correctness checks pass |
| TokenSpeed image `eagle-4-tokenspeed:0.1.0`, serve | 4 exact / 16 fp32-tolerance passes | 743.17 | Serve mode only; the first uncached start took 1084 s for kernel downloads and compilation |
| TokenSpeed wheel, serve | 4 exact / 16 fp32-tolerance passes | 786.33 | Host-compatible CUDA 13.2 JIT; CUDA 13.0 JIT fails on Ubuntu 26.04 / glibc 2.43 |
| TokenSpeed wheel, static | 5 exact / 14 fp32 passes / 1 failure | 706.36 | Not qualified: failing gap 0.178637 exceeds 0.125 |

These checks use one server start on B300 TP4 and twenty prompts with up to 256 generated tokens. The rate is total
generated tokens divided by summed request latency, not a multi-start concurrent throughput benchmark. The image rows
test the published `0.1.0` images, each pullable without credentials, and the SGLang and vLLM images contain the
`eagle-harness` 0.1.0 wheel from PyPI. TokenSpeed has no full-decode native top-logprobs yet, so each divergent
prefix in its checks goes directly to fp32 arbitration.

## 0.1.1 (released)

`eagle-harness` 0.1.1 with `sglang==0.5.23.dev252+eagle.5`, the unchanged `vllm==0.32.0rc1.dev283+eagle.4`, and
`tokenspeed==0.1.1+eagle.5` with `tokenspeed-kernel==0.1.4+eagle.5` and `tokenspeed-scheduler==0.1.40+eagle.4`.
The packages are installed from the published wheels into clean environments and served from outside any checkout.
The SGLang ROCm install keeps the 0.1.0 engine and `eagle-harness` 0.1.0 through its hash lock, so the 0.1.0 MI355X
row applies. TP8 and vLLM TP1/TP8 keep their 0.1.0 measurements.

### 0.1.1 speed by engine and concurrency

| Engine and setup | 1 | 8 | 64 |
|---|---|---|---|
| SGLang TP4, temperature 0 | ✅ 3.42× | ✅ 2.09× | ✅ 1.39× |
| SGLang TP4, temperature 1 | ✅ 2.76× | ✅ 1.78× | ✅ 1.26× |
| SGLang TP4, 32K-token context | ✅ 2.98× | — | — |
| SGLang TP1 | ✅ 3.35× | ✅ 2.21× | — |
| vLLM TP4 | ✅ 2.83× | ✅ 1.72× | ⚠️ 1.14× |
| TokenSpeed TP4 | ✅ 3.09× | ✅ 2.02× | ✅ 1.41× |

Sources:

1. SGLang TP4, temperature 0, on the published wheels: EAGLE-4 746.1, 3210.3 and 6475.9 tokens/s (3 starts each),
   no-spec 218.0, 1536.3 and 4654.1 (1 start), with the tuned command of the [SGLang guide](quickstart-sglang.md)
   and a running cap of 32. Every start gives 4 exact / 16 near tie / 0 failed. The release notes quote the first
   qualification run, on the release candidate build: 3.44×, 2.08× and 1.39×.
2. SGLang TP4, temperature 1, same starts: EAGLE-4 597.7, 2729.6 and 5800.3 tokens/s, no-spec at temperature 1
   216.3, 1531.5 and 4589.6. Temperature-1 requests draw their draft tree through the engine's `sampling_step` hook.
   The first qualification run measured 2.76×, 1.75× and 1.22×.
3. SGLang TP4, 32K: EAGLE-4 715.3 tokens/s over 4 starts (2 with an empty live-cost cache, 2 warm), no-spec 239.7
   (2 starts). Every start gives 5 exact / 15 near tie, and two no-spec starts give 20/20 identical outputs.
   Measured on the release candidate build. The one change after it applies only to chains narrower than 16 draft
   tokens.
4. SGLang TP1: the draft-path-only launch of the [SGLang guide](quickstart-sglang.md) on the overlap scheduler.
   341.7 against 102.0 tokens/s at 1 and 1581.1 against 716.4 at 8, 2 starts each, 7 exact / 12 near tie /
   1 fp32 pass per start. Measured on the 0.1.1 SGLang code and engine before the release build.
5. vLLM TP4: tuned command of the [vLLM guide](quickstart-vllm.md) (running cap 64, packaged rounds file).
   Concurrency 1 is 22 first turns with up to 512 new tokens. Mean of 6 starts over three allocations, 2 of them on
   the published wheels: EAGLE-4 681.0, 2894.8 and 7935.3 tokens/s, no-spec 240.5, 1686.7 and 6950.5. The two
   published-wheel starts alone give 2.85×, 1.74× and 1.17×. Every twenty-prompt check at concurrency 1 passes. In a
   64-prompt check at concurrency 64, one prompt takes another token at one position, up to 5.4 bf16 ULP from the
   fp32 choice, in 3 of 6 starts. The 0.1.0 packages show the same divergence in 2 of 6 starts on the same engine,
   and its cause is being investigated.
6. TokenSpeed TP4: serve mode with TokenSpeed cost tables, running cap 4 at 1 and 32 at 8 and 64, release
   candidate build. Concurrency 1 is 22 conversations and 31 turns with up to 1024 new tokens: 880.9 against
   284.8 tokens/s. Eleven first turns with up to 256 tokens give 815.2 against 281.9 (2.89×). Concurrency 8 and 64
   use 22 and 88 first turns with up to 256 tokens: 3274.5 against 1622.1 and 7075.5 against 5009.6. Every cell has
   2 starts per arm, and every start gives 6 exact / 14 fp32 pass.

### 0.1.1 high load

SGLang TP4 with a running cap of 512, 880 SPEED-Bench first turns with up to 256 new tokens at each concurrency.

| Engine and setup | 64 | 256 | 512 |
|---|---|---|---|
| SGLang TP4, default tree | ✅ 1.03× | ⚠️ 0.74× | ⚠️ 0.61× |
| SGLang TP4, 8-token chains with live costs | ✅ 1.08× | ⚠️ 0.88× | ⚠️ 0.78× |

Sources:

1. Default tree with live costs, release candidate build: EAGLE-4 7630.8, 10467.7 and 10003.6 tokens/s, no-spec
   7402.2, 14104.8 and 16481.9, 3 starts per arm, interleaved. Every twenty-prompt check passes. At 512 the
   512-prompt check passes on all 3 starts, and the engine without speculation also differs between two of its own
   starts there, so greedy agreement at 512 is not claimed. Serve concurrency above 64 without speculation for now.
   The [unreleased part](#candidate-high-load) has the high-load candidates.
2. Chains (`chain_intensity`, top-k 1, 8 draft tokens) with live costs, published wheels: EAGLE-4 7851.6, 12183.5
   and 12809.1 tokens/s, no-spec 7283.3, 13913.5 and 16325.3, 3 starts per arm. Every twenty-prompt check passes. At
   512, one to three of 512 prompts per start fall outside the fp32 rule, as do two per start between no-spec starts.

### 0.1.1 installation checks

| Quick-start command | Requests served |
|---|---|
| SGLang, draft path only | ✅ 40/40 |
| vLLM | ✅ 40/40 |
| TokenSpeed | ✅ 40/40 |

Each row starts the guide's serve command from the published 0.1.1 wheels on B300 TP4, outside any checkout, and
sends 8 requests at concurrency 1 and 32 at concurrency 8. `eagle_harness` loads from site-packages in all three.
A new TokenSpeed server compiles its kernels before the first request, which took 1105 s on an empty cache.

The published 0.1.1 images (`eagle-4-sglang:0.1.1`, `eagle-4-vllm:0.1.1`, `eagle-4-tokenspeed:0.1.1`) carry the
`eagle-harness` 0.1.1 wheel from PyPI and passed the same hook, dependency, request and twenty-prompt correctness
checks as the 0.1.0 images: SGLang 5 exact / 15 near ties, vLLM 7 exact / 12 near ties / 1 fp32-tolerance pass, and
TokenSpeed in serve mode 6 exact / 8 fp32-tolerance passes / 6 fp32-argmax.

## 0.1.2 (released, current)

`eagle-harness` 0.1.2 with the 0.1.1 engines: `sglang==0.5.23.dev252+eagle.5`, `vllm==0.32.0rc1.dev283+eagle.4`,
and `tokenspeed==0.1.1+eagle.5` with `tokenspeed-kernel==0.1.4+eagle.5` and `tokenspeed-scheduler==0.1.40+eagle.4`.
Every row runs on NVIDIA B300, on the release candidate build of `eagle-harness` 0.1.2 installed into clean
environments with the published engine wheels and served from outside any checkout. In the same allocations, 0.1.2
measured between 0.98× and 1.07× the 0.1.1 packages in every cell where both ran. The SGLang ROCm install keeps its
0.1.0 packages, so the 0.1.0 MI355X row applies.

### 0.1.2 speed by engine and concurrency

| Engine and setup | 1 | 8 | 64 |
|---|---|---|---|
| SGLang TP4, temperature 0 | ✅ 3.35× | ✅ 2.11× | ✅ 1.38× |
| SGLang TP4, temperature 1 | ✅ 2.74× | ✅ 1.78× | ✅ 1.25× |
| SGLang TP1, quick-start command | ✅ 2.87× | ✅ 1.87× | ⚠️ 0.73× |
| vLLM TP4 | ✅ 2.83× | ✅ 1.73× | ⚠️ 1.19× |
| TokenSpeed TP4 | ✅ 3.13× | ✅ 2.00× | ✅ 1.40× |

Sources:

1. SGLang TP4, temperature 0: the measured recipe of the [SGLang guide](quickstart-sglang.md#the-measured-recipe)
   (running cap 32). EAGLE-4 733.8, 3257.9 and 6488.2 tokens/s, no-spec 219.0, 1545.2 and 4718.6, 2 starts per
   arm. The 0.1.1 packages in the same allocation measured 3.36×, 2.11× and 1.40×. Every start gives 5 exact /
   15 near tie.
2. SGLang TP4, temperature 1, same starts: EAGLE-4 595.6, 2739.8 and 5827.9 tokens/s, no-spec at temperature 1
   217.4, 1539.4 and 4675.2.
3. SGLang TP1: the quick-start command on one GPU, 20, 64 and 384 SPEED-Bench first turns with up to 256 new
   tokens. EAGLE-4 302.7, 1271.8 and 2096.3 tokens/s, no-spec 105.5, 679.9 and 2870.2, 2 starts per arm. The 0.1.1
   packages measured 2.90×, 1.91× and 0.75×. The twenty-prompt and concurrency-64 checks pass. Concurrency 64 is a
   [known limit](known-limits.md#high-load).
4. vLLM TP4: the measured command of the [vLLM guide](quickstart-vllm.md) (running cap 64, packaged rounds file),
   two allocations with 2 starts per arm each. Concurrency 1 is 22 first turns with up to 512 new tokens. EAGLE-4
   678.5, 2887.4 and 8013.4 tokens/s, no-spec 239.4, 1668.9 and 6725.4. Every twenty-prompt check passes. The
   64-prompt check at concurrency 64 shows the one-prompt divergence of 0.1.1, which the 0.1.1 packages also show in
   the same allocation ([known limits](known-limits.md#performance-and-correctness-in-progress)).
5. TokenSpeed TP4: serve mode with TokenSpeed cost tables, running cap 4 at 1 and 32 at 8 and 64, 2 starts per
   arm. Concurrency 1 is 22 conversations and 31 turns with up to 1024 new tokens: 886.5 against 283.5 tokens/s.
   Eleven first turns with up to 256 tokens give 819.7 against 280.7 (2.92×). Concurrency 8 and 64 use 22 and 88
   first turns with up to 256 tokens: 3256.4 against 1628.2 and 6974.1 against 4969.8. Every start gives 6 exact /
   14 fp32 pass.

### 0.1.2 high load

SGLang TP4 with a running cap of 512: the [high-load recipe](quickstart-sglang.md#high-load), and the quick-start
command with `--tp-size 4`. Concurrency 64 serves 384 SPEED-Bench first turns and higher concurrencies serve four
times the concurrency, each with up to 256 new tokens.

| Engine and setup | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| SGLang TP4, high-load recipe, first start | ⚠️ 0.95× | ✅ 0.97× | ✅ 1.00× | ⚠️ 0.99× |
| SGLang TP4, high-load recipe, restart with warm caches | ⚠️ 1.01× | ✅ 1.00× | ✅ 1.00× | ⚠️ 0.99× |
| SGLang TP4, quick-start command with `--tp-size 4` | ⚠️ 0.80× | ⚠️ 0.48× | ⚠️ 0.37× | ⚠️ 0.32× |
| SGLang TP1, quick-start command | ⚠️ 0.73× | — | ⚠️ 0.59× | — |

Sources:

1. High-load recipe, first start on a machine (empty kernel and cost caches), 3 starts over three allocations,
   each against the mean of 3 no-spec starts in its own allocation: EAGLE-4 6935.2, 10737.0, 14253.5 and
   16152.6 tokens/s, no-spec 7313.7, 11086.4, 14264.9 and 16340.7. At 1 and 8 concurrent requests it measured 2.80×
   and 1.67× (704.3 and 2732.3 against 251.7 and 1634.7). Concurrency 64 ranges from 0.935× to 0.969× over the
   three starts.
2. Restart with the first start's caches, 3 starts in the same allocations: 7393.3, 11114.3, 14332.3 and
   16197.5 tokens/s, and 2.76× and 1.72× at 1 and 8. Every twenty-prompt check passes for both rows. The 384-prompt
   check at 64 has one prompt outside the fp32 rule in 2 of 6 starts, at a prefix where the engine without
   speculation emits the same token under another batch shape. The 2048-prompt check at 512 has one to three such
   prompts in 3 of 6 starts, and one to three between two no-spec starts in every allocation.
3. Quick-start command with `--tp-size 4`: 4377.7, 4723.1, 4490.4 and 4503.0 tokens/s against 5486.2, 9782.2,
   12034.3 and 14159.3, 2 starts per arm. The server admits 32 running requests (the default tree), and the radix
   cache keeps the batch on speculation. The 0.1.1 packages measured 0.75×, 0.48×, 0.38× and 0.32×.
4. TP1 quick-start command, the starts of the speed table: 2183.0 against 3691.8 tokens/s at 256. The 0.1.1
   packages measured 0.75× and 0.60×. See [known limits](known-limits.md#high-load).

### 0.1.2 correctness and request features

| Change | Result |
|---|---|
| `eagle-harness calibrate --engine vllm` completes on the Qwen3.8 hybrid target, and its `serve.sh` loads the tables it wrote | ✅ 20/20 |

The calibration ran on B300 TP4 with the 0.1.1 vLLM engine. Both the printed `serve.sh` and the quickstart command
pass the twenty-prompt check (4 exact / 15 near tie / 1 fp32 pass), 1 start each. No speed is claimed for it.

## Unreleased (measured on candidates, not in a release)

These rows measure changes that are not in a published release. Each row names the change and its engine build.
A row joins a release only after it passes the release qualification.

### Candidate speed by engine and concurrency

| Change, engine and setup | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| vLLM TP1, served-tree rounds measured on the setup | — | ✅ 1.94× | — | ✅ 1.07× |
| TokenSpeed TP1, serve mode | ⚠️ 3.22× | ⚠️ 2.34× | — | — |
| TokenSpeed TP8, serve mode | ⚠️ 2.79× | ⚠️ 1.68× | — | — |
| SGLang ROCm MI355X TP1, draft attention splits sized per batch | — | ⚠️ 2.21× | ⚠️ 1.66× | — |

Sources:

1. vLLM TP1: the 0.1.1 served-tree acceptance and round-weighted narrowing, with a TP1 rounds schedule measured on
   the served setup (no TP1 schedule ships). Running cap 64, unchanged vLLM engine, 88 and 220 first turns with up
   to 512 new tokens. 1366.5 against 705.4 tokens/s at 8 and 3402.4 against 3192.6 at 64, 3 starts per arm
   interleaved in one allocation. The 64-prompt checks at 64 give 0 failures in 4 pairs (1 fp32 pass).
2. TokenSpeed TP1 and TP8: the 0.1.0 TokenSpeed wheels with the pre-0.1.1 adapter, serve mode with the TP4 cost
   tables (no TP-matched tables exist yet). Concurrency 1 is 22 conversations and 31 turns: TP1 346.5 against
   107.6 tokens/s, TP8 962.1 against 345.4. Concurrency 8 is 88 first turns: TP1 1685.3 against 721.3, TP8 3916.6
   against 2337.9. Each cell has 2 starts per arm. The first start's twenty-prompt check passes (TP1 3 exact /
   17 fp32 pass, TP8 5 exact / 14 fp32 pass / 1 fp32-argmax pass), and the second start's check was not collected.
   Static mode refuses this tree at TP1 and TP8.
3. MI355X: draft attention key splits sized from the batch's programs, on the 0.1.0 ROCm engine, 44 conversations
   and 60 turns. 1202.7 against 545.2 tokens/s at 8 and 2201.8 against 1330.3 at 32 (2 EAGLE-4 starts, 1 no-spec
   start), and every MI355X check passes. The same change on B300 TP4 has one twenty-prompt failure in its first
   2 starts (2.7 bf16 ULP at a prefix where the MI355X engine without speculation picks the same token), so it
   stays a candidate.

### Candidate high load

SGLang TP4 with a running cap of 512 on the 0.1.1 engine unless a row says otherwise.

| Change | 64 | 256 | 512 |
|---|---|---|---|
| Automatic high-load chain widths, served from measured choices on a first start | ⚠️ 1.04× | ⚠️ 0.99× | ⚠️ 1.00× |
| Fixed 4-token chains with the Triton draft attention backend | — | ✅ 1.01× | ⚠️ 0.97× |
| FlashInfer draft-prefill planning without host waits, on top of fixed chains | — | ✅ 1.04× | ❌ 0.98× |

Sources:

1. Automatic chain widths: first starts with an empty policy cache, 2 starts, against the mean of 3 no-spec starts
   (8150, 15409 and 17411 tokens/s), workload 2× the concurrency in first turns with up to 256 new tokens. Restarts
   with a saved policy cache give 1.07×, 1.00× and 0.99× (3 starts). The first server on a machine compiles kernels
   during its first requests and measured 0.78× at 64. At 64, one prompt differs from no-spec at one token by
   2.69 bf16 ULP in 5 of 6 starts of one run, and a control run shows the engine without speculation emitting the
   same token there, with every row passing. At 256 and 512 the engine without speculation also differs between its
   own starts.
2. Fixed chains: 880 first turns with up to 256 new tokens, 3 starts per arm. 14178.3 against 14075.3 tokens/s at 256
   and 15812.4 against 16364.9 at 512. The twenty-prompt check passes. In the 512-prompt check at 512, one of two
   checked starts has one prompt at 2.47 bf16 ULP, inside the range the engine without speculation shows between
   its own starts.
3. FlashInfer draft-prefill engine patch with fixed chains, engine source overlays on the 0.1.0 environment:
   13997.1 against 13501.1 tokens/s at 256 and 15615.7 against 15957.6 at 512, 3 starts per arm. Twenty-prompt
   checks pass. The 512-prompt check at 512 has 4 failures (up to 4.36 bf16 ULP) against 1 for the same chains
   without the patch, so the patch is not qualified.

### Nemotron 3.5 Super candidates

Nemotron 3.5 Super 120B-A12B (language only) with its EAGLE-4 draft on B300 TP4, with the 0.1.0 engines and the
pre-0.1.1 adapters.

| Engine and setup | 1 | 8 | 64 |
|---|---|---|---|
| SGLang chain, TRT-LLM MHA target attention, page size 64 | ❌ 2.51× | — | — |
| vLLM tree | ❌ 2.37× | ❌ 1.84× | ❌ 1.57× |
| vLLM 8-token chain | — | ❌ 1.83× | ❌ 2.01× |
| TokenSpeed tree | ❌ 1.91× | ❌ 1.82× | — |

Sources:

1. SGLang: 22 conversations and 31 turns with up to 1024 new tokens, running cap 32. 611.5 against 244.0 tokens/s
   (3 EAGLE-4 starts, 1 no-spec start). The twenty-prompt check has 3 failures under the fp32 rule, and the engine
   without speculation emits the same token at the same prefix under other batch shapes in all three.
2. vLLM: running cap 64, concurrency 1 as SGLang, 88 and 220 first turns with up to 512 tokens at 8 and 64, 2 starts
   per arm. Tree 580.4, 2172.0 and 6359.5 against 244.5, 1181.0 and 4058.6 tokens/s, chain 2164.7 and 8145.2.
   The tree check has 5 failures (4 reproduced by the engine without speculation) and the chain check has 4
   (3 reproduced).
3. TokenSpeed: running cap 8, 11 and 22 first turns with up to 256 tokens, 2 starts per arm. 597.3 against 312.2 and
   2122.4 against 1167.0 tokens/s. The check has 1 failure, reproduced by the engine without speculation.

### In progress

1. vLLM sampled-distribution checks for settings other than temperature 1 with top-p 1.
2. vLLM native no-speculation rounds at TP4 that match native decoding throughput.
3. vLLM TP4 above concurrency 64, where the tree server's target attention sets the limit (engine work).
4. vLLM TP4 at concurrency 64: the one-prompt divergence in the 64-prompt check.
5. SGLang TP8 concurrency 8: prefill admission and intermittent stalls.
6. SGLang 32K-context acceptance.
7. SGLang greedy agreement at concurrency 512.
8. TokenSpeed: the candidate draft head, temperature 1 and 32K contexts, the sampled-output distribution and
   prompt logprobs.
9. AMD MI355X at concurrency 64.
10. Nemotron 3.5 Super correctness qualification on all three engines.

## How these were measured

Unless a Sources line says otherwise, every speed cell uses Qwen3.8-27B BF16 with its EAGLE-4 draft, temperature 0
and the SPEED-Bench HTTP client. Concurrency 1 uses 22 conversations and 31 turns with at most 1024 new tokens per
turn, and concurrency 8 and 64 use 88 and 220 first turns with at most 512 new tokens. Every timed pass follows a
matching warm pass, and EAGLE-4 and its no-spec control run in the same allocation with fresh server starts.
Throughput is generated tokens divided by elapsed workload time, in tokens/s, and higher is better. Client concurrency
is distinct from the server's running-request cap, and a client above that cap queues.

The greedy check compares twenty prompts with up to 256 generated tokens against the same engine without
speculation, and reports exact, near-tie, fp32-pass, fp32-argmax and failed outputs separately. At the first
divergence, a logprob gap of at most 0.13 in the engine without speculation is a near tie. A larger gap gets an
fp32 forward at the exact common prefix: a pair within two BF16 steps at that logit magnitude is an fp32 pass, and
an EAGLE-4 token equal to the fp32 argmax is an fp32-argmax pass. This is the fp32 rule. Passing it is not bitwise
equality to the native BF16 token stream.

These are configuration-specific measurements, not a claim of a speedup at every batch size. Rows from short
replays, full multi-turn workloads and synthetic contexts are not interchangeable, and they do not replace a full
SPEED-Bench qualitative evaluation or HumanEval closed-loop results. See [known limits](known-limits.md) for
configuration restrictions and [compatibility](compatibility.md) for the required engine hooks. Kernel compilation
on another GPU does not establish serving correctness or speed there. Raw launch manifests and request traces are
retained by the maintainers, not shipped in this source package.
