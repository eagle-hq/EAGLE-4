# Kernels

Every kernel of the tree round has a Triton version, so the package runs on any GPU the Triton tier covers. Faster
versions (a native CUDA library and a Gluon kernel) run only where the hardware has the feature and fall back to the
Triton kernel otherwise, with the same results. `eagle_harness/kernels/arch.py` is the one place that reads a GPU's
architecture and answers which version may run.

To see what your GPUs get:

```bash
python -m eagle_harness.kernels.arch
```

## Hardware tiers

| tier | hardware | what runs |
|---|---|---|
| Triton | NVIDIA sm_80 and newer (A100, H100, H200, B200, B300, sm_120), AMD CDNA2, CDNA3 and CDNA4 (gfx90a, gfx942, gfx950) | every kernel |
| native CUDA library | NVIDIA only: the candidate tail from sm_80, the cluster shard top-k from sm_90 | the candidate head |
| programmatic dependent launch | NVIDIA sm_90 and newer | launch overlap between dependent kernels |
| Gluon GDN tree verify | NVIDIA sm_103 (B300) | the GDN tree recurrence at one request |

On a GPU outside these tiers, or without an importable Triton, the server refuses `EAGLE_HARNESS` at startup with the
reason, for example `... is older than NVIDIA sm_80 (A100)`.

Runtime serving measurements cover B300 and SGLang on MI355X. Other listed architectures have compilation
coverage, not serving qualification. Native CUDA builds apply only to NVIDIA targets. See the
[support matrix](support-matrix.md) for the measured engine and model combinations.

## The kernels

| kernel | module | when it runs |
|---|---|---|
| candidate head: shard top-k, exact re-scoring of candidates (fp32 products), candidate tail | `candidate_ops`, `gather_dot`, `candidate_triton`, `csrc/candidate.cu` | every draft step and draft extend |
| draft tree step: scores, next frontier, ancestors, the cost model's work deltas | `tree_step` | every draft step |
| draft row attention: each row's sliding window plus its own tree ancestors, split over the keys (the split count follows the SM count and the draft heads per rank, fixed per server) | `sglang/row_attention.py`, `vllm/draft_attention.py` | every draft step and draft extend |
| GDN tree verify: ancestor tables, tree causal conv1d, tree gated delta rule recurrence | `gdn_tree`, `gdn_tree_gluon` | every verify of a hybrid (GDN) target |
| Mamba2 tree verify: ancestor tables, tree causal conv1d | `gdn_tree`, `gdn_tree_conv` | every verify of a Mamba2 hybrid (NemotronH) target on sglang |
| tree verify attention, split over the committed context | `vllm/verify_attention.py` | every verify on vLLM |
| tree-tail mask attention variant of FlashInfer's prefill | `sglang/tail_mask.py`, `verify_prep` | batch-size-1 verify graphs on sglang |
| accept tail: tree walk, accept lengths, bonus token, KV slots, compaction | `accept_tail` | greedy batch-size-1 rounds on sglang |
| L2 prefetch of a weight range | `l2_prefetch` | inside the sglang verify graphs (`l2_warm` config) |

## Native CUDA library

The candidate head has a native CUDA version (`eagle_harness/kernels/csrc/candidate.cu`), bit-identical to its Triton
version. It has a plain C ABI over raw pointers and links the CUDA runtime statically, so one binary per CUDA major
version serves every Python and torch version. It loads from the first of:

1. the file named by `$EAGLE_HARNESS_NATIVE_LIB`,
2. the prebuilt binary in the wheel (`_native/cu12/` or `_native/cu13/`, matching torch's CUDA major version),
3. an nvcc build for the visible GPUs, cached in `$EAGLE_HARNESS_CACHE` (default `~/.cache/eagle_harness`),
4. none: the Triton kernels run and a warning says why.

The prebuilt binaries hold SASS for sm_80, sm_90, sm_100, sm_103 and sm_120 plus sm_90 PTX for newer GPUs, and need
glibc 2.34 or newer. `draft_model.cuda_candidate_kernels: false` turns the library off. `eagle-harness-check
--kernels` loads it on the visible GPU and says where it came from.

To build or check the library yourself:

```bash
python -m eagle_harness.kernels.build prebuild --cuda-home /usr/local/cuda-13.0   # into the package
python -m eagle_harness.kernels.build check                                       # one compile per architecture
```

## GDN tree verify

Hybrid targets such as Qwen3.8 interleave full-attention layers with gated delta net (GDN) layers. A tree verify of a
GDN layer runs the width-4 causal conv1d and the gated delta rule recurrence per tree node, each node continuing from
its parent's state. The kernels derive the tree once per verify forward (parent table, the conv window ancestors of
each node) instead of following tree links in every layer. Conv outputs, conv windows and the parent table are
bit-identical to the engine's own tree kernels. The recurrence computes the same math, and its state snapshots can
differ in the last bit.

The recurrence has three forms. The default walks the nodes in order over narrow tiles of the value dimension. With
an fp32 state and few (request, head) pairs, it replays each first-child chain from the request state instead. On
B300 at one request, the Gluon kernel replaces the serial walk at every per-rank head shape with 128-wide keys and
values and whole GQA groups (TP1 to TP8 of Qwen3.8-27B measured, 1.3x to 1.7x faster than the walk at width 16):

- the first launch of each new shape runs a self-check that compares the Gluon kernel bitwise with the Triton kernel
  on that shape,
- a shape not yet checked during CUDA graph capture runs the Triton kernel (serving warms every verify graph before
  capture, so serving shapes are checked),
- any self-check failure, compile error or launch error turns the Gluon kernel off for the process, and a Triton
  build without Gluon keeps the Triton kernel.

## Mamba2 tree verify

Mamba2 hybrids such as NemotronH have the same width-4 causal conv1d in front of their selective state update. On
sglang the Mamba2 layers run the GDN ancestor tables and tree conv1d (bit-identical conv outputs, conv windows and
parent table); the selective state update stays sglang's own tree kernel, reading the parent table built once per
verify forward. This needs an sglang whose Mamba2 backend asks the method for a verify extension
(`LinearAttnStateLayout` with `kind`); older sglang builds run their own kernels for both steps.
