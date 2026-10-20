# MI355X Qwen3.8-27B TP1 serving costs

SGLang selects this packaged profile when `data_path` is omitted, on an
AMD Instinct MI355X (including VF), BF16, TP1, with the measured target and
draft checkpoint configurations. Selection compares parsed local `config.json`
content with `target_config.json` and `draft_config.json`; directory names do not
matter. Other devices, unresolved Hub paths, different model configurations,
quantization and other TP sizes do not select these costs. An explicit
`data_path` always takes precedence. No environment variable selects the profile.

The batch-1 curves, GEMM solutions and candidate-top-k choices are unchanged
from the measured single-request configuration. Batch 8 and 32 add served-round
measurements at verify widths 1, 8, 16 and 32. Outer verify timing includes
host-induced gaps; draft timing includes the graph-replayed tree step. Those
batch-specific costs prevent high-batch rounds from being priced as batch one.

The high-batch tables contain one measured context interval. Longer contexts
use the cost loader's existing clamping behavior; this is not calibration of
every context or batch size. Unmeasured batch sizes use its existing nearest
batch selection. Recalibrate when the model, device or execution costs change.

Measurements used rounded BF16 recurrence and disabled HIP prelaunch on MI355X,
with SGLang's ROCm tree-verification backend. The runtime and cost model are not
replaced by this profile. Model configuration files describe the measured
checkpoint pair; this directory contains no model weights.
