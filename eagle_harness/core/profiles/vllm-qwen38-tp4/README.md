# Measured vLLM tree rounds for Qwen3.8-27B on B300 TP4

Use this profile only with Qwen3.8-27B, its EAGLE-4 v3 draft, four B300 GPUs, and the vLLM tree configuration `mode: static`, `node_top_k: 4`, `tree_depth: 14`, `max_verify_width: 16`.

The rounds file is included in the `eagle-harness` package. Copy it into the matching calibrated cost directory, preserving that directory's `cost/` tables for the derived depth stop:

```bash
: "${COST_ROOT:?Set COST_ROOT to the matching calibrated TP4 cost directory}"
TREE_ROUNDS=$(python -c 'from importlib.resources import files; print(files("eagle_harness.core").joinpath("profiles/vllm-qwen38-tp4/rounds/tree.json"))')
mkdir -p "$COST_ROOT/rounds"
cp "$TREE_ROUNDS" "$COST_ROOT/rounds/tree.json"
```

Set `method_config.data_path` in the vLLM speculative configuration to `$COST_ROOT`. The measured schedule keeps verify width 16 below batch size 20 and uses width 4 from batch size 20. It does not enable native-OFF. Without `rounds/tree.json`, the configured tree width is unchanged. The packaged rounds alone are not a substitute for the calibrated target and draft costs used by the measured depth-stop policy.

These measurements do not select a policy for TP1, TP8, another model pair, or another GPU. To measure a different configuration, set `record_rounds: true` with a separate writable `data_path`, not this packaged directory.
