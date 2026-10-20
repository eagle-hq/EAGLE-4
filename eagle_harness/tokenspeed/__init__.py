# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""TokenSpeed plugin: EAGLE-Harness as a TokenSpeed drafter.

Registered through the ``tokenspeed.plugins`` entry point (see pyproject.toml) and TokenSpeed's plugin registries, with
the generic hooks of TokenSpeed branch ``pr-spec-plugin-hooks``; nothing in TokenSpeed is patched.

Modules (``cpu`` = tested without a GPU, ``gpu`` = tested on a GPU):

- ``plugin``      cpu  entry point: registers the ``EAGLE_HARNESS`` drafter and the EAGLE-4 draft architecture
- ``hooks``       cpu  the TokenSpeed API the adapter uses, checked against the installed TokenSpeed
- ``config``      cpu  ``--speculative-algorithm-config`` parsing (static trees)
- ``tree_lanes``  cpu  the core draft tree in TokenSpeed's terms: lane masks, verify window and parents
- ``drafter``     gpu  ``HarnessDrafter``: TokenSpeed's EAGLE tree round with the core tree and the candidate head
- ``models``      gpu  the EAGLE-4 draft model for TokenSpeed (candidate head, no whole-vocab LM head per step)
"""
