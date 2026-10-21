---
type: reference
owner: eagle-harness
last_verified: 2026-10-20
---

# Measured support matrix

Every row is Qwen3.8-27B with its EAGLE-4 draft, and each speedup divides EAGLE-4 throughput by the same engine's
matched no-speculation control. [Reading the numbers](#reading-the-numbers) defines the terms.

## SGLang on NVIDIA B300

| GPUs / TP | Client concurrency | Running cap | Graph padding | No-spec tokens/s | EAGLE-4 tokens/s | Speedup |
|---|---:|---:|---|---:|---:|---:|
| 1 | 1 | 1 | Irrelevant at batch 1 | 103.1 | 343.1 | 3.33× |
| 1 | 8 | 8 | Disabled in measurement | 711.7 | 1503.7 | 2.11× |
| 4 | 1 | 32 | Irrelevant at batch 1 | 216.3 | 735.7 | 3.40× |
| 4 | 8 | 32 | Default, enabled | 1479.4 | 2820.6 | 1.91× |
| 4 | 64 | 32 | Disabled in measurement | 3925.3 | 5281.3 | 1.35× |
| 8 | 1 | 1 | Irrelevant at batch 1 | 241.7 | 828.6 | 3.43× |

### B300 SGLang measurement conditions

The target and EAGLE-4 v3 draft use BF16. The dynamic tree has maximum depth 12, top-k 4 and verify width 16.
Target and draft attention use FlashInfer with page size 1. The SPEED-Bench HTTP client uses temperature 0.
Concurrency 1 uses 22 conversations and 31 turns with at most 1024 new tokens per turn. Concurrency 8 uses
88 first turns with at most 512 new tokens. Concurrency 64 uses 220 first turns with the same 512-token cap.
Every timed pass follows a warm pass. EAGLE-4 measures live costs. Each EAGLE-4 configuration has two starts.

TP4 concurrency 1 measures the released stack: `sglang==0.5.23.dev252+eagle.4` from the CUDA wheel index and
`eagle-harness==0.1.0` from PyPI, installed into a clean environment and served by the SGLang quickstart command
from a directory outside any checkout. It has six EAGLE-4 starts (725.2 to 752.7 tokens/s) and three no-spec
starts, interleaved in one allocation. Every EAGLE-4 start gives 5 exact / 15 near tie / 0 fp32 / 0 failed in its
twenty-prompt check.

TP4 concurrency 8 measures 2749.5 and 2891.8 tokens/s across its two starts.
TP8 concurrency 1 and 8 each give 5 exact / 14 near tie / 1 fp32-tolerance pass, with zero failures
in the configuration-specific twenty-prompt checks. TP8 concurrency-8 throughput variability is being investigated.

Concurrency 64 queues into 32 running slots. The measured padding-disabled settings identify those results,
not a recommendation to disable padding. Concurrent serving commands keep SGLang's default padding enabled.

## SGLang on AMD MI355X

| Configuration | Client concurrency | No-spec tokens/s | EAGLE-4 tokens/s | Speedup | EAGLE-4 starts |
|---|---:|---:|---:|---:|---:|
| Earlier one-request profile | 1 | — | 289.4 | — | 2 |
| Measured profile, explicit path | 8 | 552.7 | 1145.9 | 2.07× | 2 |
| Measured profile, explicit path | 32 | 1269.2 | 1980.8 | 1.56× | 2 |
| Packaged profile, automatic selection | 8 | — | 1209.2 | — | 1 |

Concurrency 64 on one MI355X is a [known limit](known-limits.md#performance-and-correctness-in-progress).

### MI355X measurement conditions

Qwen3.8-27B and its EAGLE-4 draft run on one MI355X at TP1 with BF16 target, KV cache and recurrent state.
AMD ROCm is included at all supported batch sizes.
The measured-profile workload uses the original SPEED-Bench client, 44 conversations and 60 turns,
temperature 0, at most 1024 new tokens per turn and a full matching warm pass. The running cap matches client
concurrency. The tree has maximum depth 12, top-k 4 and verify width 32. It uses measured MI355X costs,
AITER target attention, rounded recurrent-state updates and disabled draft prelaunch. The measured launches
disable graph padding but capture every integer batch size up to the running cap. These are recorded
measurement conditions, not a recommendation to disable padding.

The earlier one-request profile uses 22 conversations and 31 turns with the same temperature and token cap,
a running cap of 1 and two starts at 291.8 and 287.0 tokens/s. A separate packaged-profile check at concurrency 1
reaches 290.9 tokens/s on the 60-turn workload in one start. No matched no-speculation speedup is reported for
these one-request results or the packaged-profile row. The explicit-profile no-spec controls use one start each.

The explicit-profile starts are 1144.9 and 1146.8 tokens/s at concurrency 8, and 1991.7 and 1969.9 tokens/s at
concurrency 32. The packaged profile contains the same measured tables, but its one-start result is not pooled
with the earlier starts. Its twenty-prompt checks give 7 exact / 13 near tie at concurrency 1 and
6 exact / 14 near tie at concurrency 8, with no failures.

The explicit-profile concurrency-8 checks each give 5 exact / 15 near tie. At concurrency 32, start 1 gives
3 exact / 14 near tie / 3 fp32-tolerance passes. Start 2 gives 5 exact / 13 near tie / 1 fp32-tolerance pass /
1 candidate-is-fp32-argmax pass. Both starts have zero failures under the stated rule. The argmax pass is
separate from a tolerance pass and does not imply equality to the native BF16 token stream.

## vLLM on NVIDIA B300

| GPUs / TP | Client concurrency | Running cap | Mode | No-spec tokens/s | EAGLE-4 tokens/s | Speedup | EAGLE-4 starts |
|---|---:|---:|---|---:|---:|---:|---:|
| 1 | 1 | 8 | Static tree | 104.0 | 279.9 | 2.69× | 2 |
| 1 | 8 | 8 | Static tree | 715.5 | 1389.1 | 1.94× | 2 |
| 4 | 1 | 64 | Static tree | 241.6 | 722.8 | 2.99× | 2 |
| 4 | 8 | 64 | Static tree | 1699.9 | 2883.9 | 1.70× | 2 |
| 8 | 1 | 8 | Static tree | 317.2 | 745.4 | 2.35× | 2 |
| 8 | 8 | 8 | Static tree | 2063.0 | 2890.0 | 1.40× | 2 |

### B300 vLLM measurement conditions

Qwen3.8-27B BF16 uses the EAGLE-4 v3 draft and the SPEED-Bench HTTP client that preserves reasoning text between
turns. Temperature is 0. Concurrency 1 uses 22 conversations and 31 turns with at most 1024 new tokens per turn.
Concurrency 8 uses 88 first turns with at most 512 new tokens. Each timed pass follows a full matching warm pass.

The tree uses top-k 4, depth 14 and verify width 16 with `CUSTOM` attention. TP4 uses the derived depth stop.
TP1 and TP8 use fixed depth. No-spec uses the engine's default FlashInfer attention with the same cap.
The two TP8 concurrency-8 starts differ by 11%, and both are included in the mean.

### Measured high-load tree schedule

| Client concurrency | Temperature | No-spec tokens/s | EAGLE-4 tokens/s | Speedup | Verify width |
|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 1695.2 | 2932.7 | 1.73× | 16 |
| 64 | 0 | 7002.5 | 8155.1 | 1.16× | 4 |
| 128 | 0 | 7063.0 | 8552.5 | 1.21× | 4 |
| 64 | 1 | 7034.3 | 7267.9 | 1.03× | 4 |

Without the rounds file the tree keeps width 16 at every batch size, and the same concurrency-64 cells give
5349.4 tokens/s (0.76×) at temperature 0 and 4537.2 tokens/s (0.65×) at temperature 1.

#### How the high-load rows were measured

These rows measure the release vLLM engine with the `eagle-harness` 0.1.0 vLLM code on B300 TP4, served by
the vLLM quickstart recipe: running cap 64, calibrated TP4 costs and the packaged rounds file. The workload is
SPEED-Bench first turns with at most 512 new tokens: 88 requests at concurrency 8, 220 at 64 and 440 at 128.
Each timed pass follows a matching warm pass. Every arm, including the matched no-spec control, has two
independent server starts, interleaved in one allocation. Temperature 1 uses top-p 1, no top-k and a fixed seed.
Concurrency 128 queues into the 64 running slots.

Greedy checks against no-speculation decoding pass on both verify widths. Twenty sequential prompts (width 16)
give 5 exact / 14 near tie / 1 fp32-tolerance pass, and 64 concurrent prompts at concurrency 64 (width 4) give
17 exact / 46 near tie / 1 fp32-tolerance pass, with no failures. These are first-divergence checks, not
bitwise equality. Missing registration-time INFO logs do not imply the schedule failed to load: runtime
counters show the reduced verification width.

At temperature 1 (top-p 1, no top-k), a sampled-distribution check on the same stack passes all 36 per-token
and joint tests at concurrency 1, 8 and 64. Other sampling settings are not qualified by it; see
[known limits](known-limits.md#requests).

Use the [packaged rounds file with matching calibrated costs](../eagle_harness/core/profiles/vllm-qwen38-tp4/README.md).
The schedule narrows the verify width from 16 to 4 at batch size 20 and above, without enabling native-OFF.
No rounds file means no width change.

## Short-workload C8 checks

| Engine | Tokens/s | Running cap | Policy |
|---|---:|---:|---|
| SGLang | 2586.7 | 32 | Live costs, graph padding enabled |
| vLLM | 2708.8 | 64 | Width 16 at this batch size |

### Short-workload measurement conditions

These B300 TP4 checks use two independent starts, 16 requests with a 256-token cap and a matching
warm pass. They report absolute rates for the final runtime source on its pinned release engine.
They do not replace the longer-workload tables or establish a speedup over no-spec decoding.
Each measured pass lasts roughly 1.5 seconds with substantial start-to-start spread. No matching
no-spec mean is reported for these short checks.

## TokenSpeed (experimental)

| GPUs / TP | Client concurrency | Running cap | No-spec tokens/s | EAGLE-4 tokens/s | Speedup |
|---|---:|---:|---:|---:|---:|
| 4 | 8 | 32 | 1555.7 | 3150.2 | 2.02× |
| 4 | 64 | 32 | 4894.4 | 6950.2 | 1.42× |

### TokenSpeed measurement conditions

These rows measure the released wheels (`tokenspeed==0.1.1+eagle.4` with its scheduler and kernel wheels from
the CUDA index, `eagle-harness==0.1.0` from PyPI), installed by the install guide's TokenSpeed block and served
by the quickstart's serve-mode recipe on B300 TP4 with a running cap of 32. The workload is SPEED-Bench first
turns at temperature 0 with at most 256 new tokens: 22 requests at concurrency 8 and 88 at concurrency 64. Each
timed pass follows a matching warm pass. EAGLE-4 and the matched no-spec control each have two starts,
interleaved in one allocation.

Each pass is one short wave of requests. The EAGLE-4 starts are 3062.5 and 3238.0 tokens/s at concurrency 8,
and 6876.4 and 7023.9 tokens/s at concurrency 64. A twenty-prompt greedy check of the same installation
gives 4 exact / 16 fp32-tolerance passes and no failures.

The adapter implements static and calibrated trees, chains and sampled trees. This is implementation coverage,
not a performance or distribution-quality claim for every configuration.
Prompt logprobs, live cost measurement and native no-speculation rounds are not available on TokenSpeed yet.
No TokenSpeed throughput number is claimed for an unmeasured or non-matching workload. The CUDA install
checks completed real static and cost-model requests, but a static-mode first divergence remains above the
fp32 gate. Successful installation or generation is not blanket correctness qualification. TP1/TP8, sampled
distributions, long contexts and release-stack Nemotron need separate qualification.

## Artifact installation checks

| Artifact and mode | Correctness result | Tokens/s | Qualification limit |
|---|---|---:|---|
| SGLang image `eagle-4-sglang:0.1.0` | 4 exact / 16 near ties | 537.40 | Hook, dependency, quickstart request and correctness checks pass |
| vLLM image `eagle-4-vllm:0.1.0` | 5 exact / 14 near ties / 1 fp32-tolerance pass | 717.36 | Hook, dependency, quickstart request and correctness checks pass |
| TokenSpeed image `eagle-4-tokenspeed:0.1.0`, serve | 4 exact / 16 fp32-tolerance passes | 743.17 | Serve mode only; the first uncached start took 1084 s for kernel downloads and compilation |
| TokenSpeed wheel, serve | 4 exact / 16 fp32-tolerance passes | 786.33 | Host-compatible CUDA 13.2 JIT; CUDA 13.0 JIT fails on Ubuntu 26.04 / glibc 2.43 |
| TokenSpeed wheel, static | 5 exact / 14 fp32 passes / 1 failure | 706.36 | Not qualified: failing gap 0.178637 exceeds 0.125 |

### Installation-check conditions

These checks use one server start on B300 TP4 and twenty prompts with up to 256 generated tokens. The rate is
total generated tokens divided by summed request latency, not a multi-start concurrent throughput benchmark.
The image rows test the published `0.1.0` images, each of which is pullable without credentials. The wheel
rows predate publication and used a private index. The SGLang and vLLM images contain the `eagle-harness` 0.1.0
wheel published on PyPI. A near-tie or fp32-tolerance pass is not an exact-token match. Full-decode native
top-logprobs are not available on TokenSpeed yet, so each divergent prefix in these install checks goes directly
to fp32 arbitration.

## Correctness and request features

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

### How these were checked

The greedy gate reports exact, near-tie, fp32-tolerance, candidate-is-fp32-argmax and failed outputs separately.
A first-divergence reference-logprob gap of at most 0.13 is a near tie. Larger gaps require an fp32 forward
at the exact common prefix. A pair within two BF16 steps at that logit magnitude passes the tolerance check.
EAGLE-4 token equal to the fp32 argmax also passes, reported in its own category. Passing either rule
is not bitwise equality to the native BF16 token stream.

See [known limits](known-limits.md) for configuration restrictions and [compatibility](compatibility.md) for
required engine hooks. Kernel compilation on another GPU does not establish serving correctness or speed there.

## Reading the numbers

Throughput is generated tokens divided by elapsed workload time, in tokens/s, and higher is better. Client
concurrency is the number of requests the client keeps in flight. It is distinct from the server's running-request
cap, and a client above that cap queues. TokenSpeed is experimental.

These are configuration-specific measurements, not a claim of a speedup at every batch size. Rows from short
replays, full multi-turn workloads and synthetic contexts are not interchangeable. In particular, these replays do
not replace EAGLE-4's full SPEED-Bench qualitative result or its HumanEval closed-loop results. Raw launch manifests
and request traces are retained by the maintainers, not shipped in this source package.
