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

## How do I benchmark EAGLE-4?

- Warm the server up first. The first requests after a start compile kernels and seed the live costs. On
  4× B300 at one request (SGLang, eagle-harness 0.1.2, started with `--skip-server-warmup`), the first request
  after a start decoded at 63 to 85 tokens/s in three starts. The same request after warm-up decoded at 910 to
  915 tokens/s. Before timing, run one untimed pass of your workload at each concurrency
  ([what to warm up](#what-should-i-warm-up-before-timing-a-benchmark)). Restarts reuse the caches under
  `~/.cache/eagle-harness/costs`.
- Compare against the same engine and flags. Start the server without speculation from the same command minus
  the `--speculative-*` flags. Keep `--mamba-ssm-dtype`, the attention backend, the page size and the CUDA graph
  settings identical, because they change the speed of plain decoding too.
- Report decoding and prefill separately. Speculative decoding speeds up decoding, not prefill. With long
  prompts and short replies, as in agentic workloads, time to first token is a large share of each request, so
  end-to-end speedups are lower than decoding speedups for every speculative method. Report decoding throughput
  (first to last output token) and time to first token.
- Measure at the temperature you serve. At temperature 1, SGLang verifies a sampled draft tree. Other
  temperatures use a greedy draft verified by the target's sampler
  ([known limits](known-limits.md#performance-and-correctness-in-progress)).

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
- sglang: sampled requests go through sglang's own tree speculative sampling. Two settings need an sglang build with
  the verify filter fix, which the pinned engine has: min-p, and top-k combined with top-p. Older builds ignore min-p
  in verify and cut top-p after top-k. The temperature-1 sampled tree draws its proposals with the request's top-k,
  top-p and min-p, applied jointly as sglang's sampler applies them.
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

## What should I warm up before timing a benchmark?

Run one untimed pass of your workload at each concurrency you plan to measure, then time a separate pass.
Use the same prompt and output lengths, sampling settings and server flags for both passes. The first requests
compile kernels and fill cost caches for the batch shapes the workload exercises. A single request does not
warm the shapes used by a concurrent workload.

Repeat this warm-up after each server start, including restarts with saved caches. Warm both the EAGLE-4 server
and its same-engine no-speculation baseline. To isolate the effect of speculation, keep non-speculation settings
matched, including the attention backend, recurrent-state dtype, page size and CUDA graph settings. Measure at the
temperature you serve, and report decoding throughput separately from prefill latency or time to first token.

When reproducing a published ratio, use its documented no-speculation configuration and report any differences
between the servers. For example, the [AMD measurement](quickstart-amd.md#installed-wheel-qualification) uses
the ordinary overlapping scheduler and twice the Mamba cache cap for its no-speculation control.

The [candidate high-load measurement](support-matrix.md#candidate-high-load), source 1, illustrates why you
should distinguish a first machine start from a warm restart. It uses SGLang TP4 with Qwen3.8-27B,
automatic chain widths and a running cap of 512. At concurrency 64, the workload has 128 first turns with up to
256 new tokens per turn. Each ratio divides EAGLE-4 throughput by the same engine's no-speculation throughput,
so a higher ratio is better.

| Cache condition at concurrency 64 | Throughput relative to no speculation |
|---|---|
| First server start on the machine, compiling kernels during requests | 0.78× |
| Warm restarts with a saved policy cache, three starts | 1.07× |

These are candidate measurements, not a speed guarantee for another setup. If cold-start performance matters
to your application, report it separately rather than including it in a warm-throughput average.

## How do I cite it?

Cite the EAGLE-4 paper. `CITATION.cff` at the repository root has the entry, and GitHub shows it under "Cite this
repository".
