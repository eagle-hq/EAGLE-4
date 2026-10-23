# B300 Qwen3.8-27B TP4 served chain widths and tree choices

An SGLang chain without `data_path` (`"intensity": "chain_intensity"`, `--speculative-eagle-topk 1`) starts from
these widths on an NVIDIA B300 SXM6 AC, BF16, TP4, with the measured target and draft checkpoint configurations.
Selection compares parsed local `config.json` content with `target_config.json` and `draft_config.json`; directory
names do not matter. Other devices, TP sizes, dtypes, quantization or model configurations search on their served
rounds from the first round.

`chain.json` maps a batch key (the nearest calibration batch size) to the verify width served there; 0 is native
decode, used only when the engine captures native decode graphs (otherwise the server searches). The batch keys
come from forced-width runs on SPEED-Bench prompts with 256-token outputs, matched against no-spec in the same job:
width 4 at batch 16 and 64, native decode at 256. Native decode at a batch key also serves every larger one. The
remaining batch keys are searched on the served rounds, never wider than a smaller key's width.

`tree.json` does the same for a serve-mode tree (the default shape): 1 speculates (the cost model's tree), 0 decodes
natively. Two cold starts with the default tree at a 512-request cap (SPEED-Bench prompts, 256-token outputs) chose
speculation up to batch 8 and native decode at 64; native decode at 64 also serves every larger batch key, and
speculation at 8 every smaller one. Batch keys 12 to 48 are searched on the served rounds.

Choices the server makes itself (`<cost_cache>/<key>-chain.json`, `<key>-tree.json`) take precedence. A workload with very different
acceptance or output lengths may favor other widths; calibrate it (`"mode": "calibrate"`) and serve the resulting
`data_path`.
This directory contains no model weights.
