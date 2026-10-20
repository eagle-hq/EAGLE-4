# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""vLLM plugin: the EAGLE-Harness speculative method on vLLM's Model Runner V2.

Registered through the ``vllm.general_plugins`` entry point (see pyproject.toml)
and the generic hooks of vLLM branch ``eagle-hooks``; nothing in vLLM is patched.

``docs/architecture.md`` describes the package layout.
"""
