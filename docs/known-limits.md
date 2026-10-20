---
type: reference
owner: eagle-harness
last_verified: 2026-10-20
---

# Known limits

Use the [support matrix](support-matrix.md) for measured model, hardware and workload combinations.
A loaded model, an admitted request cap or a compiled kernel does not establish serving correctness or speed.

## Models and hardware

- EAGLE-4 drafts need the trained candidate head and a supported draft structure.
- Dense and latent-MoE EAGLE-4 drafts and Mamba2 targets have adapters. Nemotron's SGLang comparison has
  unresolved first divergences under the fp32 gate. vLLM and experimental TokenSpeed do not have a complete
  release-stack Nemotron performance and correctness result. Do not treat architecture admission as qualification.
- Serving measurements cover NVIDIA B300 and SGLang on AMD MI355X. Compilation on another NVIDIA or AMD
  architecture is separate evidence. Multi-node deployment and pipeline parallelism are not qualified.
- Throughput depends on the model, draft, engine, precision, TP size, attention backend, context length and
  workload. A result at one concurrency does not establish a result at another.

## Performance and correctness gaps

- SGLang TP4 single-request and 32K-context optimization remain open. TP8 concurrency-8 prefill
  overhead and intermittent stalls remain unresolved. Repeated passes in one process are not
  independent server starts.
- SGLang automatic high-load policy, fixed-chain improvements and sampled-tree improvements are not
  qualified release defaults. High-load and Nemotron first divergences remain open, as does TP4
  native-OFF arbitration. A release-engine cap-512 startup and request smoke passed, but it did not
  establish high-load acceleration or concurrent correctness.
- The shipped vLLM TP4 rounds schedule improves the measured greedy concurrency-64 case. The
  [support matrix](support-matrix.md#measured-high-load-tree-schedule) retains its no-spec-control and
  correctness limits. The schedule has no TP1/TP8 profile, and higher concurrency and temperature-1
  throughput are not qualified by that greedy result. Native-OFF is disabled by default and does not
  reach native-decode throughput in the measured TP1 cases.
- vLLM can drop EAGLE-4 INFO messages during plugin registration. An absent schedule-load line is
  not proof that the schedule is inactive. See [troubleshooting](troubleshooting.md#slower-than-expected).
- AMD measurements apply to the MI355X TP1 Qwen3.8 profile. HIP side-stream prelaunch stays disabled
  because its replay check failed. Remaining target, draft and verify phase overheads are separate
  from the measured-profile throughput results.

## Requests

- Client concurrency can exceed the server's running cap and include queued requests. Keep both values with
  benchmark results. A cap of 512 is an admission limit, not a claim of acceleration or correctness at that load.
- SGLang supports accepted-token and input logprobs with the required host hooks. Images are refused.
- vLLM tree verification needs an unquantized KV cache. The target's sampling restrictions still apply.
  Sampling smoke tests do not prove distribution equivalence on every workload.
- Structured output and tool calls require the model's engine parser. Raw tool syntax in text is not a structured
  response. A vLLM tree batch containing a grammar constraint emits only the root token per verify round,
  including unconstrained requests in that batch. Check finish reasons and validate the completed output.
  See [compatibility](compatibility.md#structured-output-on-vllm).

## TokenSpeed (experimental)

TokenSpeed implements static and calibrated trees, chains and sampled trees. Its engine and adapter must be
installed together with the matching hooks. Performance and correctness claims apply only to reported
configurations, not every implemented path. The sampled-output distribution is unqualified.

Prompt logprobs, per-request sampled-tree overrides, live cost measurement and native no-speculation rounds
are not supported. A zero-draft speculative round is not a native no-speculation round.
Tree and chain cost data must match the served shape, TP size, target, draft and recurrent-state dtype.
TP4 single-request optimization remains open. Release-stack TP1/TP8, long contexts, four-GPU chains,
retraction under serving load and original-row output-logprob checks are incomplete. The reported sampled
cap-64 path is refused. A CUDA static-mode install smoke generated text but retained one fp32-gate failure.
The serve-mode install smoke passed its checked greedy prompts only with the host-compatible CUDA 13.2
JIT. The original CUDA 13.0 JIT fails on Ubuntu 26.04 / glibc 2.43. Static mode is not qualified.
These checks do not qualify arbitrary CUDA-toolkit and host-compiler combinations.

## Calibration

Cost tables are configuration-specific. Keep the recorded settings with them and measure again after changing
the model, draft, TP size, precision, attention backend or engine. Keyed tables are checked at loading, but the
key does not encode every setting. A cache or table is not evidence of an optimal policy for every workload.

The vLLM public sweep measures batch-one tree forwards, not complete high-concurrency serving rounds.
The SGLang served-chain driver measures a separate path from dynamic-tree calibration. The pinned SGLang
CUDA engine lacks its required per-variant native-decode graph contract, so the CLI path is unavailable
on that engine. Live-cost tree serving does not require it. Built-in smoke prompts
check the command, not representative acceptance. A failed run retains partial output for diagnosis and does
not write a successful `serve.sh`. See [calibration](calibration.md).

## Distribution

The [install guide](install.md) describes the release path: `eagle-harness` packages on PyPI,
patched engine wheels on the GitHub Pages indexes, and images on GHCR. These destinations do not establish
publication or image qualification. Private-index CUDA installation and request checks do not test the public
URLs. AMD has a qualified private offline pip closure, but its fresh-install commands remain deferred
to the AMD guide update, and its Docker image is unqualified. Model and draft weights require separate access.
