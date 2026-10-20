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

## Calibration

Cost tables are configuration-specific. Keep the recorded settings with them and measure again after changing
the model, draft, TP size, precision, attention backend or engine. Keyed tables are checked at loading, but the
key does not encode every setting. A cache or table is not evidence of an optimal policy for every workload.

The vLLM public sweep measures batch-one tree forwards, not complete high-concurrency serving rounds.
The SGLang served-chain driver measures a separate path from dynamic-tree calibration. Built-in smoke prompts
check the command, not representative acceptance. A failed run retains partial output for diagnosis and does
not write a successful `serve.sh`. See [calibration](calibration.md).

## Distribution

The [install guide](install.md) uses supplied wheel bundles and source builds. Its local index is not a claim
of a public package service or anonymously accessible model weights. GPU serving and clean-machine container
installation require their own checks.
