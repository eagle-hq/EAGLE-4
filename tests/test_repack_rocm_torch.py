# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""The ROCm Torch repack changes dependency metadata, never runtime payload."""

import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "repack_rocm_torch", Path(__file__).parents[1] / "tools/repack_rocm_torch.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RepackTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / f"{MODULE.TORCH}-cp310-cp310-linux_x86_64.whl"
        self.info = f"{MODULE.TORCH}.dist-info"

    def fixture(self, requirement=MODULE.OLD):
        with zipfile.ZipFile(self.source, "w") as wheel:
            wheel.writestr("torch/version.py", '__version__ = "unchanged"\n')
            binary = zipfile.ZipInfo("torch/lib/example.so", date_time=(2020, 1, 2, 3, 4, 6))
            binary.external_attr = 0o100755 << 16
            wheel.writestr(binary, b"\x00binary fixture\xff")
            wheel.writestr(
                f"{self.info}/METADATA",
                "Metadata-Version: 2.1\nName: torch\nVersion: 2.9.1\n" + requirement + ' platform_system == "Linux"\n',
            )
            wheel.writestr(f"{self.info}/WHEEL", "Wheel-Version: 1.0\nTag: cp310-cp310-linux_x86_64\n")
            wheel.writestr(f"{self.info}/RECORD", "")

    def test_valid_requirement_and_unchanged_payload(self):
        self.fixture()
        receipt = MODULE.repack(self.source, self.root / "out")
        with zipfile.ZipFile(self.root / "out" / receipt["output"]) as wheel:
            metadata = wheel.read(f"{self.info}/METADATA").decode()
            self.assertIn(
                'Requires-Dist: triton==3.7.0+amd.rocm7.2.0.git89002410; platform_system == "Linux"', metadata
            )
            self.assertNotIn(MODULE.OLD, metadata)
            self.assertEqual(wheel.read("torch/version.py"), b'__version__ = "unchanged"\n')
            self.assertEqual(wheel.read("torch/lib/example.so"), b"\x00binary fixture\xff")
            self.assertEqual(wheel.getinfo("torch/lib/example.so").external_attr >> 16, 0o100755)
            self.assertEqual(wheel.getinfo("torch/lib/example.so").date_time, (2020, 1, 2, 3, 4, 6))
            self.assertIn(b"Build: 1eagle\n", wheel.read(f"{self.info}/WHEEL"))
            rows = list(csv.reader(io.StringIO(wheel.read(f"{self.info}/RECORD").decode())))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row[1].startswith("sha256=") and row[2] for row in rows[:-1]))
        self.assertTrue(receipt["record_verified"])
        self.assertEqual(receipt["unchanged_payload_members"], 2)

    def test_rejects_unexpected_dependency(self):
        self.fixture("Requires-Dist: triton==0.0;")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MODULE.repack(self.source, self.root / "out")

    def test_refuses_to_overwrite(self):
        self.fixture()
        MODULE.repack(self.source, self.root / "out")
        with self.assertRaises(FileExistsError):
            MODULE.repack(self.source, self.root / "out")


if __name__ == "__main__":
    unittest.main()
