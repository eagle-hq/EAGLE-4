# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Wheel packaging: pure Python, plus the prebuilt native kernels when ``python -m eagle_harness.kernels.build prebuild``
has put them in the package. The libraries need neither torch nor a Python ABI (ctypes, static CUDA runtime), so such a
wheel is tagged ``py3-none-<platform>``; without them the wheel is the usual pure one and nvcc builds on first use."""

from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution

PREBUILT = Path(__file__).parent / "eagle_harness" / "kernels" / "_native"


class _Distribution(Distribution):
    def has_ext_modules(self):
        return any(PREBUILT.glob("*/*.so"))


class _Wheel(bdist_wheel):
    def get_tag(self):
        python, abi, platform = super().get_tag()
        return ("py3", "none", platform) if self.distribution.has_ext_modules() else (python, abi, platform)


setup(distclass=_Distribution, cmdclass={"bdist_wheel": _Wheel})
