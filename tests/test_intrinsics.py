# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handeye_calib.intrinsics import (
    MANIFEST_FILENAME,
    discover_intrinsics_manifest,
    validate_intrinsics_manifest,
    write_intrinsics_manifest,
)


class IntrinsicsManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.npy_dir = self.root / "camera_intrinsics_20260811_100000_npy"
        self.npy_dir.mkdir()
        self.camera_matrix = np.array(
            [[600.0, 0.0, 320.0], [0.0, 601.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.array([[0.1, -0.2, 0.001, 0.002, 0.03]], dtype=np.float64)
        np.save(self.npy_dir / "camera_matrix.npy", self.camera_matrix)
        np.save(self.npy_dir / "dist_coeffs.npy", self.dist_coeffs)
        self.manifest_path = write_intrinsics_manifest(
            self.npy_dir,
            self.camera_matrix,
            self.dist_coeffs,
            (640, 480),
            camera_serial="123456",
            camera_model="D435i",
            stream_name="color",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_manifest_creation_discovery_and_validation(self) -> None:
        self.assertEqual(self.manifest_path.name, MANIFEST_FILENAME)
        self.assertEqual(discover_intrinsics_manifest(self.root), self.manifest_path)

        manifest = validate_intrinsics_manifest(
            self.root,
            expected_camera_serial="123456",
            expected_camera_model="D435i",
            expected_stream_name="color",
            expected_image_size=(640, 480),
        )

        self.assertEqual(manifest["image_width"], 640)
        self.assertEqual(manifest["image_height"], 480)
        self.assertEqual(np.asarray(manifest["K"]).shape, (3, 3))
        self.assertEqual(len(manifest["files"]["camera_matrix"]["sha256"]), 64)

    def test_expected_capture_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "camera serial mismatch"):
            validate_intrinsics_manifest(
                self.manifest_path,
                expected_camera_serial="different-camera",
            )
        with self.assertRaisesRegex(ValueError, "image size mismatch"):
            validate_intrinsics_manifest(
                self.manifest_path,
                expected_image_size=(1280, 720),
            )

    def test_modified_npy_hash_is_rejected(self) -> None:
        np.save(self.npy_dir / "camera_matrix.npy", np.eye(3, dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_intrinsics_manifest(self.manifest_path)

    def test_manifest_value_mismatch_is_rejected_even_with_matching_hash(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["K"][0][0] = 999.0
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Manifest values do not match"):
            validate_intrinsics_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()
