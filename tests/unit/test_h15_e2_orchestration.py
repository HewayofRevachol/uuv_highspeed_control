"""Exercise the real E2 CLI planner with synthetic episode output, no simulator."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.h15_cat_mpc import e2_matrix
from experiments.h15_cat_mpc.residual_predictor import ResidualAR1, RESIDUAL_LABEL_VERSION


def prepare_inputs(tmp_path, monkeypatch, ar_passes):
    monkeypatch.setattr(e2_matrix, "ROOT", tmp_path)
    paths = []
    for geometry in ("dev_reversal", "dev_coupled"):
        path = tmp_path / f"{geometry}.csv"
        path.write_text("synthetic path fixture\n")
        paths.append({"id": geometry, "path": path.name,
                      "path_sha256": e2_matrix.sha256(path)})
    manifest = tmp_path / "dev.json"
    manifest.write_text(json.dumps({"split": "dev", "cases": paths,
                                   "seeds": [32001, 32002, 32003]}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"dev_manifest": manifest.name}))
    # An ancestor called main reproduces the old directory-name grouping bug.
    out = tmp_path / "main" / "e2"
    argv = ["e2_matrix", "--config", str(config), "--out", str(out)]
    for mode in ("on", "off"):
        normalization = tmp_path / f"normalization_{mode}.json"
        normalization.write_text(json.dumps({"split": "train", "compensation_mode": mode}))
        predictor = tmp_path / f"predictor_{mode}.json"
        ResidualAR1(phi=np.zeros(5), mean=np.ones(5) if ar_passes else np.zeros(5),
                    version="train_ar1_test").save(predictor, {
            "split": "train", "compensation_mode": mode,
            "label_version": RESIDUAL_LABEL_VERSION,
            "normalization_sha256": e2_matrix.sha256(normalization)})
        argv.extend([f"--normalization-{mode}", str(normalization),
                     f"--predictor-{mode}", str(predictor)])
    monkeypatch.setattr(sys, "argv", argv)
    return out


def write_synthetic_episode(job):
    out = job["out"]
    out.mkdir(parents=True, exist_ok=False)
    row = {"schema": 2, "command_schema": "issued_v2",
           "residual_label_version": RESIDUAL_LABEL_VERSION,
           "method": job["method"], "seed": job["seed"],
           "path": str(job["path"]), "predictor_mode": job["predictor_mode"],
           "compensation_mode": job["compensation_mode"], "periods": 80,
           "success": True, "physical_strict": True,
           "no_fallback_physical": True, "deadline_clean": True,
           "strict_substeps": [{"lateral_error_m": 1., "depth_error_m": .2,
                                "speed_error_mps": .1, "pitch_rad": .01}],
           "completed_rows": [{"residual_valid": True, "residual_body_acc": [1.] * 5,
                               "residual_label_version": RESIDUAL_LABEL_VERSION}
                              for _ in range(80)]}
    (out / "episode.json").write_text(json.dumps(row))


@pytest.mark.parametrize("ar_passes", [True, False], ids=["ar", "fallback"])
def test_e2_runs_six_a0_before_selecting_remaining_sixty(tmp_path, monkeypatch, ar_passes):
    out = prepare_inputs(tmp_path, monkeypatch, ar_passes)
    phases = []

    def fake_dispatch(jobs, workers):
        if not phases:
            assert len(jobs) == 6
            assert all(job["method"] == "A0" and job["compensation_mode"] == "on"
                       for job in jobs)
            assert not (out / "predictor_selection.json").exists()
        else:
            assert len(jobs) == 60
            assert (out / "predictor_selection.json").is_file()
        phases.append(jobs.copy())
        for job in jobs:
            write_synthetic_episode(job)

    monkeypatch.setattr(e2_matrix, "run_jobs", fake_dispatch)
    e2_matrix.main()

    selected = "ar_or_persistence" if ar_passes else "persistence"
    jobs = [job for phase in phases for job in phase]
    assert [len(phase) for phase in phases] == [6, 60]
    assert len({job["out"] for job in jobs}) == 66
    assert len({(str(job["path"]), job["seed"], job["method"],
                 job["compensation_mode"], job["predictor_mode"]) for job in jobs}) == 66
    main = [job for job in jobs if job["group"] == "main"]
    assert len(main) == 30
    assert all(job["predictor_mode"] == selected for job in main if job["method"] != "A0")
    assert all((job["predictor_checkpoint"] is not None) == ar_passes
               for job in main if job["method"] != "A0")
    selected_a4 = [job for job in jobs if job["method"] == "A4"
                   and job["compensation_mode"] == "on" and job["predictor_mode"] == selected]
    assert len(selected_a4) == 6
    assert all(job["group"] == "main" for job in selected_a4)
    # AR stays a measured ablation even after it fails the main predictor gate.
    assert len([job for job in jobs if job["method"] == "A4"
                and job["compensation_mode"] == "on"
                and job["predictor_mode"] == "ar_or_persistence"]) == 6
    selection = json.loads((out / "predictor_selection.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert selection["selected"] == selected
    assert summary["episodes_completed"] == 66
    assert summary["main_physical_strict"] == {method: 6 for method in ("A0", "A1", "A2", "A3", "A4")}
    assert len(summary["main_pairs"]) == 24
    assert len(summary["ablation_pairs"]) == 36
    assert summary["protocol"]["predictor_selection_sha256"] == e2_matrix.sha256(
        out / "predictor_selection.json")


def test_e2_rejects_mismatched_normalization_before_dispatch(tmp_path, monkeypatch):
    prepare_inputs(tmp_path, monkeypatch, True)
    predictor_path = tmp_path / "predictor_on.json"
    predictor = json.loads(predictor_path.read_text())
    predictor["manifest"]["normalization_sha256"] = "0" * 64
    predictor_path.write_text(json.dumps(predictor))
    monkeypatch.setattr(e2_matrix, "run_jobs", lambda *args: pytest.fail("dispatched invalid artifacts"))
    with pytest.raises(ValueError, match="normalization"):
        e2_matrix.main()
