# FAQ

## Does speculative decoding change the output?

Greedy verification commits the target model's argmax at each accepted position. Different verification shapes can
change target arithmetic, so this does not establish exact equality to a no-spec server. Some measured model pairs
have unexplained divergences beyond the near-tie threshold. Check the [support matrix](support-matrix.md) rather
than treating loading or native-engine equivalence as a no-spec parity guarantee.

## How does EAGLE-4 choose a draft tree shape?

The engines' built-in EAGLE methods verify a draft tree of fixed shape every step. eagle-harness sizes the tree each
round from measured costs: on sglang the cost model picks the draft depth and the verify width per round, and on vLLM
the draft depth stops as soon as more steps cannot pay. It also brings the EAGLE-4 draft model (sliding-window draft
layer, low-rank candidate head) and tree versions of the GDN kernels for hybrid targets.

## Which engine should I use?

Choose an engine with measured coverage for your model pair and request features in the
[support matrix](support-matrix.md). SGLang picks the verify width per round. vLLM fixes that width per batch size,
so only the draft depth adapts per round.

## Do I need cost tables?

When no `data_path` is supplied, SGLang selects a matching packaged MI355X profile if available, otherwise
it measures costs on served rounds. Offline tables remain an option.
vLLM static trees need no tables, while automatic policies require matching measured costs. TokenSpeed
needs offline costs for calibrated decisions, or a complete explicit chain schedule.
See [calibration](calibration.md).

## Does it patch sglang or vLLM?

No. It registers through the engines' plugin entry points and uses hooks the engine exposes. An engine without those
hooks keeps working: the plugin logs why and registers nothing.

## Which models work?

Targets need the engine's EAGLE-style auxiliary hidden states. Single-layer EAGLE-4 drafts use the low-rank
candidate head (`efficient_lm_head_config`).
Qwen3.8-27B has EAGLE-4 coverage. On sglang the draft may also use the latent ReLU^2 MoE, `attn_norm` and no rotary
positions, as in Nemotron 3.5 Super drafts. vLLM also implements those structures. Model-specific correctness
coverage is separate from admission and is listed in the support matrix. Multi-layer drafts are not supported.

## Does it support batch sizes above 1?

Configuration limits are not serving qualifications. Trees and chains (`chain_intensity`) admit up to 512 running requests. The engine's KV and recurrent-state
capacity can impose a lower limit. The sglang tree default is 32 requests. Set `--max-running-requests` explicitly
to raise it. vLLM uses `--max-num-seqs`.

A larger request cap does not guarantee higher throughput. Wider trees reserve more recurrent-state snapshots
and verify more tokens per round. Compare your tree shape against no speculation at the intended concurrency,
and calibrate the full batch-size range instead of treating a small-batch cost table as a large-batch measurement.
The fused accept step and tree-tail mask attention cover batch size 1 only. Other batch sizes use the engine's
own code for those parts.

## Does it support sampling, logprobs or images?

SGLang and vLLM support sampled requests with the engine-specific restrictions below.
`tools/sampling_check.py` compares a speculative server with a no-spec server (`--api sglang`, `vllm` or
`tokenspeed`). Sampling smoke tests alone
do not establish distribution equivalence for every model and workload.

- vLLM: seeded sampling uses the target sampler, but an identical seed is not a blanket guarantee of the same
  no-spec text or distribution qualification. vLLM itself
  refuses min-p and logit bias with any speculative method. A batch with
  repetition, frequency or presence penalties, bad words, a thinking budget or a custom logits processor is
  verified as chains using the host rejection sampler, which can accept fewer tokens. This fallback is not a
  claim of token-identical output across different numerical execution shapes.
- sglang: sampled requests go through sglang's own tree speculative sampling. min-p, and top-k together with top-p,
  need an sglang build with the verify filter fix: older builds ignore min-p in verify and cut top-p after top-k.
  Penalties use sglang's relaxed form (counts of committed tokens only), as in sglang's own EAGLE, so they are not
  exact. Seeds need `--enable-deterministic-inference`.

Logprobs are supported on SGLang and vLLM. Each request gets the logprobs of its accepted tokens. SGLang
input logprobs come from its prefill. Use the matching engine with the FlashInfer verify-wrapper graph fix.
TokenSpeed returns no prompt logprobs ([known limits](known-limits.md#tokenspeed)).

Requests with images are refused with a reason on sglang.

## Which GPUs work?

NVIDIA from sm_80 (A100) and AMD CDNA2 to CDNA4 through the Triton kernels, with faster native and Gluon kernels on
newer NVIDIA GPUs. Serving measurements cover B300 and SGLang on MI355X. Other architectures have compilation
coverage, not serving qualification. See [kernels](kernels.md).

## How do I cite it?

Cite the EAGLE-4 paper. `CITATION.cff` at the repository root has the entry, and GitHub shows it under "Cite this
repository".
