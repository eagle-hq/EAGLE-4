# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Table of SPEED-Bench results (``client_speedbench.py`` JSONs) with mean and spread per server.

``python tools/bench_table.py <results dir> [...]``: one row per ``speedbench_<name>[_r<N>].json``
(wall tok/s, accept length, tokens, turns, validity), then per server name the mean wall tok/s and
the spread (max - min) over its repeats.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

_NAME = re.compile(r"speedbench_(?P<name>.+?)(?:_r(?P<rep>\d+))?\.json$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    by_name = collections.defaultdict(list)
    print(f"{'file':60s} {'tok/s':>8} {'accept':>7} {'tokens':>8} {'turns':>6} valid")
    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, "speedbench_*.json"))):
            match = _NAME.search(os.path.basename(path))
            with open(path) as f:
                summary = json.load(f)["summary"]
            overall = summary["all"]
            tput, accept = overall["wall_tput"], overall.get("accept_len")
            by_name[match["name"]].append(tput)
            print(
                f"{os.path.relpath(path, d) if len(args.dirs) == 1 else path:60s} {tput:8.1f} "
                f"{accept if accept is not None else float('nan'):7.3f} {overall['tokens']:8d} "
                f"{summary['turns']:6d} {summary['valid']}"
            )
    print()
    for name, values in sorted(by_name.items()):
        mean = sum(values) / len(values)
        print(f"{name:20s} n={len(values)} mean {mean:7.1f} tok/s  spread {max(values) - min(values):6.1f}")


if __name__ == "__main__":
    main()
