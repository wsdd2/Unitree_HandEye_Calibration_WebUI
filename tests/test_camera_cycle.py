from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handeye_calib.camera import RealSenseD435i  # noqa: E402


class CameraCycleTests(unittest.TestCase):
    def test_cycle_wraps_to_first(self) -> None:
        devices = [
            {"serial": "AAA", "model": "Intel RealSense D435"},
            {"serial": "BBB", "model": "Intel RealSense D435i"},
        ]
        nxt = RealSenseD435i.cycle_listed_device(devices, "BBB")
        self.assertEqual(nxt["serial"], "AAA")

    def test_cycle_from_unknown_starts_at_first(self) -> None:
        devices = [
            {"serial": "AAA", "model": "D435"},
            {"serial": "BBB", "model": "D435i"},
        ]
        nxt = RealSenseD435i.cycle_listed_device(devices, "")
        self.assertEqual(nxt["serial"], "AAA")

    def test_cycle_empty_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            RealSenseD435i.cycle_listed_device([], "AAA")

    def test_format_label_uses_model_and_serial(self) -> None:
        label = RealSenseD435i.format_device_label(
            {"model": "Intel RealSense D435", "serial": "000000000001"}
        )
        self.assertIn("Intel RealSense D435", label)
        self.assertIn("000000000001", label)


if __name__ == "__main__":
    unittest.main()
