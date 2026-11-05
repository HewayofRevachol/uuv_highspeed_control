"""Shared provenance and artifact checks for the H15 supplemental runners."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    path = path.resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes project root: {value}") from exc
    return path


def validate_manifest(path, root, expected_split=None, expected_seeds=None):
    """Load a geometry/seed manifest and verify every referenced file hash."""
    manifest_path = project_path(root, path)
    manifest = json.loads(manifest_path.read_text())
    if expected_split is not None and manifest.get("split") != expected_split:
        raise ValueError(f"expected split {expected_split!r}, got {manifest.get('split')!r}")
    cases = manifest.get("cases")
    seeds = manifest.get("seeds")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be a non-empty list")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("manifest seeds must be unique and non-empty")
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("manifest seeds must be integers")
    if expected_seeds is not None and list(seeds) != list(expected_seeds):
        raise ValueError(f"manifest seeds differ: expected {list(expected_seeds)!r}, got {seeds!r}")
    ids = [case.get("id") for case in cases]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest case IDs must be unique")
    for case in cases:
        relative = case.get("path")
        recorded = case.get("path_sha256")
        if not relative or not recorded or len(recorded) != 64:
            raise ValueError(f"case {case.get('id')!r} lacks path_sha256")
        data_path = project_path(root, relative)
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        actual = sha256(data_path)
        if actual != recorded:
            raise ValueError(f"manifest hash mismatch for {relative}: {actual} != {recorded}")
    return manifest, manifest_path


def load_matching_predictor(normalization_path, predictor_path, mode,
                            expected_label_version):
    """Load predictor metadata only after binding it to normalization bytes."""
    normalization_path = Path(normalization_path).resolve()
    predictor_path = Path(predictor_path).resolve()
    normalization = json.loads(normalization_path.read_text())
    predictor = json.loads(predictor_path.read_text())
    if normalization.get("split") != "train":
        raise ValueError("normalization must be train-only")
    if normalization.get("compensation_mode") != mode:
        raise ValueError("normalization compensation mode mismatch")
    manifest = predictor.get("manifest", {})
    if manifest.get("split") != "train" or manifest.get("compensation_mode") != mode:
        raise ValueError("predictor split or compensation mode mismatch")
    if manifest.get("label_version") != expected_label_version:
        raise ValueError("predictor label version mismatch")
    expected_hash = sha256(normalization_path)
    if manifest.get("normalization_sha256") != expected_hash:
        raise ValueError("predictor normalization hash does not match adapter normalization")
    return normalization, predictor
