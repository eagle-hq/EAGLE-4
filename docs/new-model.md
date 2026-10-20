---
type: how-to
owner: eagle-harness
last_verified: 2026-10-19
---

# Onboard a target model

Train a draft whose structure the serving adapter implements, export its trained candidate head, then check
correctness before calibrating. Changing an architecture name or adding a candidate-head config to untrained
weights does not make a compatible draft.

The export steps below follow TorchSpec's `tools/convert_to_hf.py` and a real EAGLE4 safetensors export.
A complete training-to-serving run for an arbitrary new target is not measured. Treat the admission and greedy
checks as required stop points, not as proof that every target is supported.

## Check the target and draft contract

First serve the target without speculation in your pinned engine. Confirm that its tokenizer, chat template,
precision and attention backend work. Keep those settings for the speculative comparison.

For the measured EAGLE4 ports, train a dense single-layer draft with rotary positions, full target vocabulary,
three auxiliary hidden-state inputs and a low-rank candidate head. Grouped-query draft attention is implemented, but the query and KV head counts must be compatible
with the selected adapter and tensor-parallel layout. The target itself can use a hybrid architecture if the engine and EAGLE-4 support
its verify and state-commit paths.

Use the target's hidden size, vocabulary and token IDs when building your draft configuration. Keep its trained
RoPE parameters, auxiliary layer IDs, sliding window and normalization settings in the export. Only choose
structures admitted by `eagle_harness/models/eagle4_config.py` and the selected engine's draft loader.
See the [support matrix](support-matrix.md) for measured pairs.

## Train with TorchSpec

The upstream training project is [TorchSpec](https://github.com/lightseekorg/TorchSpec).
Obtain a source bundle with the efficient-head training and export support below from the training maintainer.
A reproducible repository revision for the inspected bundle is not available with this release. The commands
below are an external recipe checked against its exporter source, not a verified end-to-end training path.
Do not substitute an arbitrary TorchSpec revision. Install its training backend in a separate environment
from serving, following that bundle's setup instructions. Its
`docs/draft_training_guide.md` and `configs/recipes/` contain the training recipes.

Prepare a target-specific YAML, `train.yaml`, with these fields set rather than copied from another target:

| Field | What to supply |
|---|---|
| `model.target_model_path` | Target weights and tokenizer |
| `model.draft_model_config` | Compatible draft architecture JSON |
| `model.embedding_key`, `model.lm_head_key`, `model.norm_key` | Exact target checkpoint tensor names |
| `inference.aux_hidden_states_layers` | Three training layer-output indices |
| `dataset.train_data_path`, `dataset.chat_template`, `dataset.prompt_key` | Training conversations in the target's template |
| `inference.inference_num_gpus`, `training.training_num_gpus_per_node` | Allocated inference and training resources |
| `training.num_train_steps`, `training.save_interval`, `output_dir` | Run length and saved checkpoint locations |

Train the dense drafter first and retain a saved checkpoint. From the TorchSpec checkout:

```bash
python -m torchspec.train_entry --config train.yaml --print-config-only
python -m torchspec.train_entry --config train.yaml
```

The resolved config must name the intended target, layers and output directory. Do not use a smoke recipe with
checkpoint saving disabled for this step.

Next train the low-rank candidate factors from the trained dense drafter. In a second YAML,
`train-candidates.yaml`, retain the same target and draft architecture and set:

```yaml
model:
  initial_draft_model_path: ./dense-export
training:
  efficient_lm: true
  efficient_lm_rank: 128
  efficient_lm_top_k: 128
  loss_type: forward_kl
```

`dense-export` is the dense checkpoint exported by the command in the next section. Merge these fields into the
complete target-specific training YAML, not a standalone four-field recipe. Rank 128 and 128 candidates are
an explicit example, not a tuned choice for your target. TorchSpec requires a pretrained drafter through
`model.initial_draft_model_path` or `training.load_path`, and checks that its output head is dense.
Choose a fresh `output_dir` and a nonzero checkpoint interval, then run:

```bash
python -m torchspec.train_entry --config train-candidates.yaml --print-config-only
python -m torchspec.train_entry --config train-candidates.yaml
```

Check held-out candidate recall and served acceptance. A low training loss alone does not establish a useful
speculative draft.

## Export the checkpoint

Set `CHECKPOINT` to the saved iteration directory and `EXPORT` to a fresh output directory. Run this from the
TorchSpec checkout, once for the dense initialization and again for the trained candidate-head checkpoint:

```bash
python tools/convert_to_hf.py --input-dir "$CHECKPOINT" --output-dir "$EXPORT"
```

The converter uses the checkpoint's `config.json` and discovers its `model/` distributed-checkpoint directory.
If the config is stored elsewhere, pass `--config` with the exact training config JSON. Avoid regenerating a
config from the target when doing so would lose trained draft settings. If embeddings must be supplied from
the target, use `--target-model-path` and the correct `--embedding-key` prefix.

Do not pass `--prune-vocab`. The efficient-head export requires the full target vocabulary. Do not pass
`--vllm` for this export either, because TorchSpec's efficient-head converter refuses that option. EAGLE-4
vLLM adapter consumes the EAGLE4 native export and translates its auxiliary layer indices itself.
Keep output-layer indices in `eagle_config.eagle_aux_hidden_state_layer_ids`, without manually adding one.

The serving directory contains:

```text
export/
  config.json
  model.safetensors
```

A real inspected export also contained `conversion_validation.json`. That report is useful provenance,
but it is not required by the loader. The weight header carries `torchspec_version`, and the config carries
`_torchspec_version` and the output dtype.

Check these candidate-head tensors before serving. `V` is the full target vocabulary, `H` the draft hidden
size and `R` the trained candidate rank:

| Tensor | Expected shape |
|---|---|
| `lm_head.weight` | `[V, H]` |
| `efficient_lm_head_down_proj.weight` | `[R, H]` |
| `efficient_lm_head.weight` | `[V, R]` |
| `embed_tokens.weight` | Target-compatible embedding shape |
| `fc.weight` | Projection from the three auxiliary hidden states |
| `fc_norm.0.weight`, `.1.weight`, `.2.weight` | Present when the trained draft uses feature normalization |

The config must contain `efficient_lm_head_config` with `type: low_rank`, the same rank and the candidate count.
TorchSpec refuses missing candidate factors or incompatible shapes. A safetensors header can be inspected
without loading the full weights onto a GPU:

```bash
export EXPORT
python - <<'PY'
import json
import os
from pathlib import Path
from safetensors import safe_open

root = Path(os.environ["EXPORT"])
config = json.loads((root / "config.json").read_text())
head = config["efficient_lm_head_config"]
assert head["type"] == "low_rank"
vocab, hidden, rank = config["vocab_size"], config["hidden_size"], head["rank"]
expected = {
    "lm_head.weight": [vocab, hidden],
    "efficient_lm_head_down_proj.weight": [rank, hidden],
    "efficient_lm_head.weight": [vocab, rank],
}
with safe_open(root / "model.safetensors", framework="pt", device="cpu") as weights:
    for name, shape in expected.items():
        actual = weights.get_slice(name).get_shape()
        assert actual == shape, (name, actual, shape)
        print(name, actual)
PY
```

The inspected Nemotron draft used this format, including a rank-128 candidate head, latent MoE,
attention normalization and no RoPE. SGLang and vLLM implement those structures. Preserve them in the
export rather than removing fields to pass validation. Structural admission does not establish Nemotron
release-stack performance or correctness qualification. See [known limits](known-limits.md#models-and-hardware).

## Check generation before calibration

In your serving environment, run `eagle-harness-check`, then use the matching
[SGLang](quickstart-sglang.md) or [vLLM](quickstart-vllm.md) launch recipe with `DRAFT` pointing to the export.
These are release-specific recipes, not generic fixed-tree examples. SGLang seeds live costs, while vLLM needs
matching calibrated costs and rounds. For vLLM, prepare those inputs with the [calibration guide](calibration.md)
before launching if they are absent. Do not reuse another model/draft pair's calibration. Start with one request.
Refused config fields require an adapter implementation or a compatible trained draft, not a renamed architecture.

Compare greedy token IDs on held-out prompts against the same engine without speculation. For each first
mismatch, inspect the no-spec token logprobs and separate a near-tie from a genuine error. Record acceptance
length as well. Acceptance near one can indicate a wrong export, wrong hidden layers or an incomplete head
mapping even when the target verifies every output token.

Then test your required stops, concurrency, logprobs and sampling behavior. Keep every unsupported request
feature out of the deployment contract.

## Calibrate and serve

Stop the test servers and measure the admitted pair:

```bash
eagle-harness calibrate --engine vllm --model "$MODEL" --draft "$EXPORT" --tp "$TP" --output calibration-new-model &&
  sh calibration-new-model/serve.sh
```

This offline command is for vLLM. The pinned SGLang engine lacks the native-decode graph contract needed
by the served-chain calibration CLI. Use its live-cost serving path instead. On a host exposing the required
contract, SGLang chain calibration should use a representative `--data` file.
The [calibration guide](calibration.md) explains the different per-engine measurements. Benchmark against a
matched no-spec server at your deployment concurrency. Save the model and draft revisions, engine build,
GPU type, TP, precision and calibration settings alongside the result before calling the new pair supported.
