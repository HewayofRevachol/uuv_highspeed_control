import hashlib
import json

import pytest

from experiments.h15_cat_mpc.contracts import load_matching_predictor, validate_manifest


def test_manifest_hash_is_verified(tmp_path):
    data = tmp_path / "path.csv"
    data.write_text("x,y\n1,2\n")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "T_fit", "seeds": [1],
        "cases": [{"id": "a", "path": "path.csv", "path_sha256": digest}]}))
    loaded, loaded_path = validate_manifest(manifest, tmp_path, expected_split="T_fit",
                                            expected_seeds=[1])
    assert loaded["split"] == "T_fit"
    assert loaded_path == manifest.resolve()
    data.write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(manifest, tmp_path)


def test_predictor_must_bind_normalization_bytes(tmp_path):
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"split": "train", "compensation_mode": "on"}))
    predictor = tmp_path / "predictor.json"
    predictor.write_text(json.dumps({"manifest": {"split": "train",
        "compensation_mode": "on", "label_version": "v3",
        "normalization_sha256": hashlib.sha256(normalization.read_bytes()).hexdigest()}}))
    load_matching_predictor(normalization, predictor, "on", "v3")
    predictor.write_text(json.dumps({"manifest": {"split": "train",
        "compensation_mode": "on", "label_version": "v3",
        "normalization_sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="normalization hash"):
        load_matching_predictor(normalization, predictor, "on", "v3")
