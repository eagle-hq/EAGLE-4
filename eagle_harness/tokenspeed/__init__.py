# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""TokenSpeed plugin: EAGLE-Harness as a TokenSpeed drafter.

Registered through the ``tokenspeed.plugins`` entry point (see pyproject.toml) and TokenSpeed's plugin registries, with
the hooks :mod:`eagle_harness.tokenspeed.hooks` checks (the release engine's, or a capability engine's); nothing in
TokenSpeed is patched.

``docs/architecture.md`` describes the package layout.
"""
