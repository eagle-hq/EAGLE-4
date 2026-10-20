# Security policy

Report a vulnerability privately, never in a public issue, discussion or pull request. Use either:

- GitHub's private vulnerability reporting: the "Report a vulnerability" button on this repository's Security tab, or
- email to security@eagle.ai.

Include the affected version or commit, the steps to reproduce, the impact you expect, any proof-of-concept, and
whether you disclosed the issue anywhere else. Do not test beyond what you need to confirm the issue, and do not
access or keep data that belongs to others.

Disclose publicly only after the maintainers have had a reasonable chance to investigate and fix the issue.

## Scope

eagle-harness runs inside a serving engine (sglang or vLLM), in the engine's processes. Report a vulnerability of
the engine itself to that project. Report one in eagle-harness here, including one that the plugin makes reachable
in the engine. The native CUDA library loads from `$EAGLE_HARNESS_NATIVE_LIB` when that variable is set, so treat it
like `LD_PRELOAD`: anyone who can set it can run code in the server.
