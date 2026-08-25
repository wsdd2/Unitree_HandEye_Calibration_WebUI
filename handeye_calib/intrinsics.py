# -*- coding: utf-8 -*-
"""Camera-intrinsics manifest creation, discovery, and validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MANIFEST_FILENAME = "intrinsics_manifest.json"
MANIFEST_SCHEMA = "handeye_calib.intrinsics_manifest/v1"


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_intrinsics_manifest(
    npy_dir: str | Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    *,
    camera_serial: str = "",
    camera_model: str = "",
    stream_name: str = "",
) -> dict[str, Any]:
    """Build a manifest for the standard intrinsics NPY files in ``npy_dir``."""
    directory = Path(npy_dir)
    matrix_path = directory / "camera_matrix.npy"
    distortion_path = directory / "dist_coeffs.npy"
    width, height = (int(image_size[0]), int(image_size[1]))

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "image_width": width,
        "image_height": height,
        "K": np.asarray(camera_matrix, dtype=float).tolist(),
        "D": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
        "files": {
            "camera_matrix": {
                "path": matrix_path.name,
                "sha256": file_sha256(matrix_path),
            },
            "dist_coeffs": {
                "path": distortion_path.name,
                "sha256": file_sha256(distortion_path),
            },
        },
    }
    camera = {
        key: value
        for key, value in (
            ("serial", camera_serial.strip()),
            ("model", camera_model.strip()),
        )
        if value
    }
    if camera:
        manifest["camera"] = camera
    if stream_name.strip():
        manifest["stream"] = {"name": stream_name.strip()}
    return manifest


def write_intrinsics_manifest(
    npy_dir: str | Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    *,
    camera_serial: str = "",
    camera_model: str = "",
    stream_name: str = "",
) -> Path:
    """Write and return ``intrinsics_manifest.json`` in an intrinsics NPY directory."""
    directory = Path(npy_dir)
    manifest = build_intrinsics_manifest(
        directory,
        camera_matrix,
        dist_coeffs,
        image_size,
        camera_serial=camera_serial,
        camera_model=camera_model,
        stream_name=stream_name,
    )
    path = directory / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def discover_intrinsics_manifest(path: str | Path) -> Path:
    """Find a manifest from a manifest/NPY path, an NPY directory, or an output tree."""
    candidate = Path(path)
    if candidate.is_file():
        if candidate.name == MANIFEST_FILENAME:
            return candidate
        candidate = candidate.parent

    direct = candidate / MANIFEST_FILENAME
    if direct.is_file():
        return direct

    matches = list(candidate.rglob(MANIFEST_FILENAME)) if candidate.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"No {MANIFEST_FILENAME} found under {candidate}")
    return max(matches, key=lambda item: (item.stat().st_mtime_ns, str(item)))


def load_intrinsics_manifest(path: str | Path) -> dict[str, Any]:
    """Discover and parse an intrinsics manifest."""
    manifest_path = discover_intrinsics_manifest(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid intrinsics manifest root in {manifest_path}: expected object")
    return data


def _metadata_value(manifest: Mapping[str, Any], section: str, key: str) -> str:
    value = manifest.get(section, {})
    return str(value.get(key, "")) if isinstance(value, Mapping) else ""


def validate_intrinsics_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
    expected_camera_serial: str | None = None,
    expected_camera_model: str | None = None,
    expected_stream_name: str | None = None,
    expected_image_size: tuple[int, int] | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate schema, metadata, NPY hashes/content, and optional capture expectations."""
    if isinstance(manifest_or_path, Mapping):
        manifest = dict(manifest_or_path)
        directory = Path(base_dir) if base_dir is not None else None
    else:
        manifest_path = discover_intrinsics_manifest(manifest_or_path)
        manifest = load_intrinsics_manifest(manifest_path)
        directory = Path(base_dir) if base_dir is not None else manifest_path.parent

    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported intrinsics manifest schema: {manifest.get('schema')!r}; "
            f"expected {MANIFEST_SCHEMA!r}"
        )

    width = manifest.get("image_width")
    height = manifest.get("image_height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("Manifest image_width/image_height must be positive integers")

    matrix = np.asarray(manifest.get("K"), dtype=np.float64)
    distortion = np.asarray(manifest.get("D"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Manifest K must be a finite 3x3 matrix")
    if distortion.size == 0 or not np.isfinite(distortion).all():
        raise ValueError("Manifest D must contain finite distortion coefficients")

    expectations = (
        ("camera serial", expected_camera_serial, _metadata_value(manifest, "camera", "serial")),
        ("camera model", expected_camera_model, _metadata_value(manifest, "camera", "model")),
        ("stream name", expected_stream_name, _metadata_value(manifest, "stream", "name")),
    )
    for label, expected, actual in expectations:
        if expected is not None and expected != actual:
            raise ValueError(f"Intrinsics {label} mismatch: expected {expected!r}, got {actual!r}")
    if expected_image_size is not None and tuple(expected_image_size) != (width, height):
        raise ValueError(
            f"Intrinsics image size mismatch: expected {tuple(expected_image_size)}, "
            f"got {(width, height)}"
        )

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Manifest files must be an object")
    arrays = {"camera_matrix": matrix, "dist_coeffs": distortion.reshape(-1)}
    if verify_hashes and directory is None:
        raise ValueError("base_dir is required to verify hashes for an in-memory manifest")
    for name, expected_array in arrays.items():
        entry = files.get(name)
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError(f"Manifest files.{name} must contain path and sha256")
        if not verify_hashes:
            continue
        file_path = directory / str(entry["path"])
        if not file_path.is_file():
            raise ValueError(f"Intrinsics file does not exist: {file_path}")
        actual_hash = file_sha256(file_path)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {file_path}")
        loaded = np.asarray(np.load(file_path, allow_pickle=False), dtype=np.float64)
        if name == "dist_coeffs":
            loaded = loaded.reshape(-1)
        if loaded.shape != expected_array.shape or not np.array_equal(loaded, expected_array):
            raise ValueError(f"Manifest values do not match {file_path}")
    return manifest
