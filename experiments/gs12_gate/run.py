"""Prepare, smoke, calibrate and compare the GS-12 gate experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments.gs12_gate.data_io import (digest, file_hash, immutable_json,
                                            immutable_jsonl, read_cases, read_json)
from experiments.gs12_gate.episode import run_episode, bootstrap_estimator_normalization, estimator_parameters
from experiments.gs12_gate.features import GateActorCritic, schema
from experiments.gs12_gate.model_bridge import HybridDynamics, finite_difference_check, THETA_SAFE
from experiments.gs12_gate.plant_harness import (make_case, capacity_limited,
                                                  PlantHarness, plant_from_case)
from simulator.batch_mpc import CONTROL_SCALE
from experiments.h15_mainline.observation_features import FEATURE_NAMES

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
FAMILIES = ("nominal", "fin", "mass", "damping", "thrust", "delay", "tau", "second_all")
FIXED = ((0., 0., 0., 0.), (1., 1., 1., 1.), (.5, .5, .5, .5),
         (1., 0., 0., 0.), (0., 1., 1., 1.), (1., 1., 0., 1.),
         (1., 0., 1., 1.), (1., 1., 1., 0.), (0., .5, .5, .5))


def _test_case(case_id, speed, family, maneuver, seed, combo=None):
    case = make_case(case_id, "test", speed, family if family in FAMILIES else "nominal",
                     maneuver, seed)
    p = case["plant"]
    if family == "fin": p["fin"] = .94
    elif family == "mass": p["mass"] = 1.06
    elif family == "damping": p["damping"] = 1.06
    elif family == "thrust": p["thrust"] = .94
    elif family == "delay": p["delay"] = 4
    elif family == "tau": p["tau"] = 1.6
    elif family == "second_fins": p["order"] = "second_fins"
    if combo == 0: p.update(fin=.94, order="second_fins")
    elif combo == 1: p.update(fin=.92, tau=1.7, delay=4)
    elif combo == 2: p.update(mass=1.06, damping=1.04)
    elif combo == 3: p.update(thrust=.94, tau=1.6, delay=4)
    case["family"] = family
    case["combination_id"] = combo
    case["capacity_limited"] = capacity_limited(case)
    case["initial_offset"] = [float(x) for x in np.random.default_rng(seed+1).uniform(-.05, .05, 3)]
    return case


def build_cases(config, direction_source=None):
    rng = np.random.default_rng(config["seed"])
    cases = []
    smoke = ("nominal", "fin", "tau")
    for i, family in enumerate(smoke):
        case = make_case(f"smoke_{i:02d}", "smoke", 15, family, "coupled",
                         31000+i, duration_s=4.)
        if family == "fin": case["plant"]["fin"] = .9; case["plant"]["order"] = "second_all"
        if family == "tau": case["plant"]["tau"] = 2.
        cases.append(case)
    for i in range(config["calibration_cases"]):
        family = rng.choice(FAMILIES if i % 10 else ("combination",))
        maneuver = rng.choice(("straight", "speed", "turn", "depth", "coupled"))
        cases.append(make_case(f"cal_{i:03d}", "train_calibration", int(rng.choice((12, 15, 18))),
                               str(family), str(maneuver), int(rng.integers(1, 2**31)), training=True))
    for i in range(config["train_cases"]):
        kind = rng.choice(("single", "combination", "nominal"), p=(.5, .4, .1))
        family = str(rng.choice(FAMILIES[1:])) if kind == "single" else str(kind)
        directional = rng.random() < .25
        maneuver = "frequency_direction" if directional else str(rng.choice(("straight", "speed", "turn", "depth", "coupled")))
        case = make_case(f"train_{i:05d}", "train", int(rng.choice((12, 15, 18))),
                         family, maneuver, int(rng.integers(1, 2**31)),
                         duration_s=64. if directional else 24., training=True)
        if directional:
            bin_number = int(rng.choice((4, 6, 8, 12, 16, 24),
                                        p=(.2, .25, .2, .15, .1, .1)))
            case["frequency_bin_64s"] = bin_number
            case["omega"] = float(bin_number*2*np.pi/64.)
            case["amplitude"] = float(rng.uniform(.006, .03))
            direction_class = str(rng.choice(("svd", "axis", "random_complex")))
            case["direction_class"] = direction_class
            if direction_class == "svd":
                case["complex_direction"] = (direction_source["right_singular_vector"]
                                              if direction_source is not None else None)
            elif direction_class == "axis":
                axis = int(rng.integers(3))
                case["complex_direction"] = [[float(j == axis), 0.] for j in range(3)]
            else:
                vector = rng.normal(size=3)+1j*rng.normal(size=3)
                vector /= np.linalg.norm(vector)
                case["complex_direction"] = [[float(z.real), float(z.imag)] for z in vector]
        cases.append(case)
    for i, family in enumerate(FAMILIES):
        cases.append(make_case(f"screen_{i:02d}", "dev_screen", 15, family,
                               "coupled", 51000+i))
    for speed in (12, 15, 18):
        for i, family in enumerate(FAMILIES):
            cases.append(make_case(f"quick_{speed}_{i:02d}", "dev_quick", speed,
                                   family, "coupled", 52000+speed*10+i))
    for speed in (12, 15, 18):
        for i, family in enumerate(FAMILIES[:-1]+("second_fins",)):
            for j, maneuver in enumerate(("straight", "speed", "turn", "depth")):
                cases.append(_test_case(f"test_{speed}_{i:02d}_{j}", speed,
                                        family, maneuver, 60000+speed*100+i*4+j))
        for combo in range(4):
            for j, maneuver in enumerate(("depth", "coupled")):
                cases.append(_test_case(f"test_combo_{speed}_{combo}_{j}", speed,
                                        "combination", maneuver, 70000+speed*100+combo*2+j, combo))
    for original in [case for case in cases if case["split"] == "test"]:
        fresh = _test_case(f"ablation_{original['case_id']}", original["speed"],
                           original["family"], original["maneuver"],
                           original["seed"]+1_000_000, original["combination_id"])
        fresh["split"] = "ablation_holdout"
        cases.append(fresh)
    for case in cases:
        case["physical_plant"] = asdict(plant_from_case(case))
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids))
    assert sum(c["split"] == "dev_quick" for c in cases) == 24
    assert sum(c["split"] == "test" for c in cases) == 120
    assert sum(c["split"] == "ablation_holdout" for c in cases) == 120
    return cases


def prepare(config_path, out, direction_source_path=None):
    config = read_json(config_path)
    if config["horizon"] != 35 or config["dt_control_s"] != .1 or config["dt_plant_s"] != .02:
        raise ValueError("GS-12 model requires N=35, control .1 s, plant .02 s")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    source_files = [PACKAGE/name for name in ("run.py", "episode.py", "model_bridge.py",
                    "features.py", "plant_harness.py", "data_io.py", "train.py",
                    "frequency.py", "report.py", "config.json", "README.md")]
    source_files += [ROOT/path for path in ("aav_mpc/condensed_mpc.py",
                     "experiments/h15_mainline/response_estimator.py",
                     "simulator/model.py", "simulator/batch_mpc.py", "training/ppo.py",
                     "docs/GS-12_RL实验执行清单.md")]
    dependencies = {str(path.relative_to(ROOT)): file_hash(path) for path in source_files if path.exists()}
    direction_source = read_json(direction_source_path) if direction_source_path else None
    if direction_source is not None and (direction_source.get("split") != "dev" or
                                         direction_source.get("frequency_bin") != 6):
        raise ValueError("training direction source must be frozen development bin 6")
    if direction_source is not None:
        vector = np.asarray(direction_source["right_singular_vector"], float)
        if vector.shape != (3, 2) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1., atol=1e-6):
            raise ValueError("development direction must be a unit complex three-vector")
    cases = build_cases(config, direction_source)
    if direction_source_path:
        for case in cases:
            if case.get("direction_class") == "svd":
                case["direction_source_hash"] = file_hash(direction_source_path)
    cases_hash = immutable_jsonl(out/"cases.jsonl", cases)
    feature_schema = schema()
    feature_schema["version"] = "gs12_observation_200_v1"
    schema_hash = immutable_json(out/"obs_schema.json", feature_schema)
    manifest = {"protocol": config["protocol"], "config": config,
                "config_hash": file_hash(config_path), "source_hashes": dependencies,
                "direction_source_hash": (file_hash(direction_source_path)
                                          if direction_source_path else None),
                "cases_hash": cases_hash, "schema_hash": schema_hash,
                "seed_space": {"prepare": config["seed"], "train_first": 41001,
                               "confirm": [42001, 42002, 42003, 42004, 42005]},
                "versions": {"model": "hybrid25_second_thrust_first_fins_v1",
                             "estimator": "sliding_window_1s_v1",
                             "residual": "completed_transition_ema_rebase_v1",
                             "policy": "gaussian4_sigmoid_slew_v1"},
                "environment": {"python": platform.python_version(),
                                "numpy": np.__version__, "torch": torch.__version__},
                "processes": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if (out/"run_manifest.json").exists():
        existing = read_json(out/"run_manifest.json")
        comparison = dict(manifest); comparison["created_utc"] = existing.get("created_utc")
        if comparison != existing:
            raise FileExistsError("existing run manifest has different config, cases or source hashes")
        return existing
    immutable_json(out/"run_manifest.json", manifest)
    return manifest


def _load_run(run):
    run = Path(run)
    manifest = read_json(run/"run_manifest.json")
    if file_hash(run/"cases.jsonl") != manifest["cases_hash"]:
        raise ValueError("case manifest hash mismatch")
    if file_hash(run/"obs_schema.json") != manifest["schema_hash"]:
        raise ValueError("observation schema hash mismatch")
    root = PACKAGE.parents[1]
    for name, expected in manifest["source_hashes"].items():
        path = root/name
        if not path.exists() or file_hash(path) != expected:
            raise ValueError(f"source changed since prepare: {name}")
    return manifest, read_cases(run/"cases.jsonl")


def _artifacts(run, required=True):
    run = Path(run)
    if not (run/"normalization.json").exists():
        if required: raise FileNotFoundError("run collect before train/tune/evaluate")
        return {"bootstrap": True}
    norm = read_json(run/"normalization.json")
    residual = read_json(run/"residual_scale.json")
    return {"bootstrap": False, "observation_scale": norm["observation_scale"],
            "estimator_normalization": norm["estimator_normalization"],
            "residual_limits": residual["limits"]}


def smoke(run):
    _, cases = _load_run(run)
    compile_started = time.perf_counter()
    dynamics = HybridDynamics()
    compile_ms = 1000*(time.perf_counter()-compile_started)
    state = np.zeros(25); state[5] = 15.; state[10:13] = [.6, .2, -.1]
    state[13] = .03
    fd = [finite_difference_check(dynamics, state, np.array([.6, .3, -.2]), theta)
          for theta in (THETA_SAFE, np.array([1.1, 1.2]))]
    if any(not row["propagation_finite"] or row["A_max_abs"] > 2e-7 or row["B_max_abs"] > 2e-7 for row in fd):
        raise AssertionError(f"hybrid Jacobian failed: {fd}")
    parity_case = make_case("internal_parity", "smoke", 15, "nominal", "straight", 999, 4.)
    parity_case["plant"].update(fin=.85, thrust_tau=1., fin_tau=2., order="second_thrust")
    parity_plant = PlantHarness(parity_case)
    z = np.r_[parity_plant.state, parity_plant.actual/CONTROL_SCALE,
              np.zeros(3), parity_plant.queue[:3].reshape(-1)/np.tile(CONTROL_SCALE, 3)]
    command = np.array([.62, .05, -.03])
    predicted = dynamics.evaluate(z[None, None], command[None, None], THETA_SAFE)[0][0, 0]
    parity_plant.step(command*CONTROL_SCALE)
    propagated = np.r_[parity_plant.state, parity_plant.actual/CONTROL_SCALE,
                       parity_plant.rate/CONTROL_SCALE,
                       parity_plant.queue[:3].reshape(-1)/np.tile(CONTROL_SCALE, 3)]
    parity_max_abs = float(np.max(np.abs(predicted-propagated)))
    if parity_max_abs > 1e-9:
        raise AssertionError(f"matched hybrid/plant propagation mismatch: {parity_max_abs}")
    results = []
    checks = []
    for case in cases:
        if case["split"] != "smoke": continue
        for method in ("hybrid_zero", "deterministic_sequence"):
            target = (Path(run)/"episodes"/"smoke"/case["case_id"]/
                      method.replace(":", "_").replace(",", "_"))
            results.append(run_episode(case, method, target, artifacts={"bootstrap": True}))
            with np.load(target/"control.npz") as control, np.load(target/"decisions.npz") as decisions:
                finite = all(np.isfinite(control[name]).all() for name in
                             ("state", "model_state", "actual", "issued", "d_ref",
                              "d_used", "theta_used"))
                shape_ok = control["model_state"].shape == (40, 25)
                success = control["qp_ok"].astype(bool)
                issued_equal = bool(np.allclose(control["issued"][success],
                                                control["mpc_command"][success], atol=1e-6))
                alpha_bounded = bool(np.all((decisions["alpha"] >= 0) & (decisions["alpha"] <= 1)))
                alpha_slew = bool(np.all(np.abs(decisions["alpha"]-decisions["previous_alpha"]) <= .25+1e-12))
                causal_initial = bool(np.allclose(control["d_ref"][0], 0.))
                check = {"case_id": case["case_id"], "method": method,
                         "finite": finite, "issued_equal_mpc_when_qp_ok": issued_equal,
                         "model_state_shape_ok": shape_ok,
                         "alpha_bounded": alpha_bounded, "alpha_slew": alpha_slew,
                         "initial_residual_zero": causal_initial,
                         "control_shape": list(control["state"].shape),
                         "model_state_shape": list(control["model_state"].shape),
                         "substeps": results[-1]["substeps"]}
                if not all((finite, shape_ok, issued_equal, alpha_bounded,
                            alpha_slew, causal_initial)):
                    raise AssertionError(check)
                checks.append(check)
    output = {"finite_difference": fd, "matched_propagation_max_abs": parity_max_abs,
              "compile_warmup_ms": compile_ms,
              "episodes": results, "checks": checks,
              "control_steps": sum(r["control_steps"] for r in results),
              "substeps": sum(r["substeps"] for r in results)}
    immutable_json(Path(run)/"smoke.json", output)
    return output


def collect(run, episodes=32, split="train_calibration"):
    if split != "train_calibration": raise ValueError("calibration may only use train_calibration")
    _, cases = _load_run(run)
    selected = [c for c in cases if c["split"] == split]
    if episodes != len(selected): raise ValueError(f"prepared calibration has {len(selected)} cases")
    results, obs, d_raw, estimator_features = [], [], [], []
    jobs = [(case, "hybrid_zero" if i < episodes//2 else "random")
            for i, case in enumerate(selected)]
    complete = {"hybrid_zero": 0, "random": 0}
    # Calibration precedes SVD direction freezing. Replenish from ordinary
    # training maneuvers, never from unresolved directional references.
    fallback_cases = [case for case in cases if case["split"] == "train"
                      and case["maneuver"] != "frequency_direction"]
    fallback_index = 0
    i = 0
    while i < len(jobs) or any(value < episodes//2 for value in complete.values()):
        if i >= len(jobs):
            if fallback_index >= min(100, len(fallback_cases)):
                raise RuntimeError("insufficient finite train-only calibration episodes")
            method = next(name for name in complete if complete[name] < episodes//2)
            jobs.append((fallback_cases[fallback_index], method))
            fallback_index += 1
        case, method = jobs[i]
        i += 1
        path = Path(run)/"episodes"/split/case["case_id"]/method
        summary = run_episode(case, method, path, artifacts={"bootstrap": True})
        results.append(summary)
        complete[method] += int(summary["complete"])
        with np.load(path/"control.npz") as data:
            d_raw.extend(row for row in data["d_raw"] if np.isfinite(row).all())
            for j in range(len(data["t"])):
                errors = data["errors"][j]
                actual, issued, queue = data["actual"][j], data["issued"][j], data["queue"][j]
                feature = [errors[0], errors[1], errors[2],
                    np.rad2deg(data["state"][j, 3]), actual[0]/1_500_000,
                    actual[1], actual[2], issued[0]/1_500_000,
                    issued[1], issued[2], queue[0, 0]/1_500_000,
                    queue[0, 1], queue[0, 2]]
                if np.isfinite(feature).all():
                    estimator_features.append(feature)
        with np.load(path/"decisions.npz") as data:
            obs.extend(row for row in data["observation"] if np.isfinite(row).all())
    if not obs or not d_raw or not estimator_features:
        raise RuntimeError("no finite train-only samples for calibration")
    observations = np.asarray(obs)*np.asarray(schema()["scale"])
    base_scale = np.asarray(schema()["scale"])
    scale = np.maximum(base_scale, np.quantile(np.abs(observations), .99, axis=0))
    features = np.asarray(estimator_features)
    est_scale = np.maximum(np.std(features, axis=0), np.array(bootstrap_estimator_normalization()["scale"])*.01)
    estimator_normalization = {"split": "train", "feature_names": list(FEATURE_NAMES),
                               "mean": np.mean(features, axis=0).tolist(),
                               "scale": est_scale.tolist()}
    residual_floor = np.array([.01, .01, .01, .001, .001])
    limits = np.maximum(3*np.quantile(np.abs(d_raw), .99, axis=0), residual_floor)
    norm = {"split": split, "observation_scale": scale.tolist(),
            "estimator_normalization": estimator_normalization,
            "bootstrap": bootstrap_estimator_normalization(),
            "calibration_case_ids": [case["case_id"] for case, _ in jobs]}
    immutable_json(Path(run)/"normalization.json", norm)
    immutable_json(Path(run)/"estimator_config.json",
                   {"bootstrap": estimator_parameters(True), "formal": estimator_parameters(False)})
    immutable_json(Path(run)/"residual_scale.json",
                   {"limits": limits.tolist(), "floor": residual_floor.tolist(),
                    "quantile": .99, "multiplier": 3.})
    immutable_json(Path(run)/"calibration_summary.json",
                   {"episodes": results, "finite_episodes": sum(r["complete"] for r in results),
                    "group_complete": complete, "extra_episodes": fallback_index})
    return {"episodes": len(results), "normalization_hash": file_hash(Path(run)/"normalization.json"),
            "residual_hash": file_hash(Path(run)/"residual_scale.json")}


def _method_name(values):
    return "fixed:"+",".join(str(v) for v in values)


def tune(run, split="dev_screen"):
    if split != "dev_screen": raise ValueError("tune uses dev_screen only")
    _, cases = _load_run(run)
    artifacts = _artifacts(run)
    candidates = [_method_name(x) for x in FIXED] + ["rule10", "rule5"]
    results = []
    for case in cases:
        if case["split"] != split: continue
        for method in candidates:
            output = Path(run)/"episodes"/split/case["case_id"]/method.replace(":", "_").replace(",", "_")
            results.append(run_episode(case, method, output, artifacts=artifacts))
    score = {}
    for method in candidates:
        rows = [r for r in results if r["method"] == method]
        score[method] = [sum(not r["complete"] for r in rows),
                         float(np.mean([r["tracking_cost_integral"] for r in rows]))]
    selected = {"fixed_best": min(candidates[:9], key=lambda m: score[m]),
                "rule_best": min(candidates[9:], key=lambda m: score[m]),
                "scores": score, "split": split}
    immutable_json(Path(run)/"selected_baselines.json", selected)
    return selected


def evaluate(run, split="dev_quick", methods="hybrid_zero,fixed_best,rule_best,ppo",
             checkpoint=None, checkpoints=None, tag=None, gate_phase_steps=0):
    if split not in ("dev_quick", "test", "ablation_holdout"):
        raise ValueError("evaluation split must be dev_quick/test/ablation_holdout")
    _, cases = _load_run(run)
    artifacts = _artifacts(run)
    selected = read_json(Path(run)/"selected_baselines.json")
    paths = ([Path(p) for p in checkpoints] if checkpoints else
             ([Path(checkpoint)] if checkpoint else []))
    policies = {}
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        for field, name in (("normalization_hash", "normalization.json"),
                            ("residual_hash", "residual_scale.json"),
                            ("schema_hash", "obs_schema.json")):
            if saved[field] != file_hash(Path(run)/name):
                raise ValueError(f"checkpoint {field} mismatch")
        variant = saved.get("policy_variant", "standard")
        label = ("ppo" if not checkpoints and variant == "standard" else
                 (f"ppo_{saved['seed']}" if variant == "standard" else
                  f"{variant}_{saved['seed']}"))
        if label in policies: raise ValueError("duplicate policy seed")
        policy = GateActorCritic(variant); policy.load_state_dict(saved["policy"]); policy.eval()
        policies[label] = policy
    names = []
    for name in methods.split(","):
        if name == "ppo" and policies and "ppo" not in policies:
            names.extend(policies)
        else:
            names.append(name)
    results = []
    if gate_phase_steps and tag is None:
        tag = f"phase{gate_phase_steps}"
    output_split = split if tag is None else f"{split}_{tag}"
    if split in ("test", "ablation_holdout"):
        filename = "test_manifest.json" if split == "test" else "ablation_holdout_manifest.json"
        immutable_json(Path(run)/filename,
            {"split": split,
             "case_ids": [case["case_id"] for case in cases if case["split"] == split],
             "cases_hash": file_hash(Path(run)/"cases.jsonl"),
             "methods": names,
             "checkpoint_hashes": {name: file_hash(path) for name, path in zip(policies, paths)},
             "selected_baselines_hash": file_hash(Path(run)/"selected_baselines.json"),
             "normalization_hash": file_hash(Path(run)/"normalization.json")})
    for case in cases:
        if case["split"] != split: continue
        for name in names:
            method = selected.get(name, name)
            if name in policies: method = "ppo_mean"
            if method == "ppo" and name not in policies:
                raise ValueError("ppo evaluation requires --checkpoint or --checkpoints")
            output = Path(run)/"episodes"/output_split/case["case_id"]/name
            results.append(run_episode(case, method, output, artifacts=artifacts,
                                       policy=policies.get(name),
                                       gate_phase_steps=gate_phase_steps))
    result = {"split": split, "output_split": output_split, "methods": names,
              "gate_phase_steps": gate_phase_steps,
              "checkpoint_hashes": {name: file_hash(path) for name, path in zip(policies, paths)},
              "results": results}
    immutable_json(Path(run)/f"evaluation_{output_split}.json", result)
    return {"episodes": len(results), "failures": sum(not r["complete"] for r in results)}


def evaluate_ablation(run, checkpoints, ablations=("freeze4", "no_theta")):
    _, cases = _load_run(run)
    dev = [case for case in cases if case["split"] == "dev_quick"]
    artifacts = _artifacts(run)
    results = []
    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for field, name in (("normalization_hash", "normalization.json"),
                            ("residual_hash", "residual_scale.json"),
                            ("schema_hash", "obs_schema.json")):
            if saved[field] != file_hash(Path(run)/name):
                raise ValueError(f"checkpoint {field} mismatch")
        policy = GateActorCritic(saved.get("policy_variant", "standard"))
        policy.load_state_dict(saved["policy"]); policy.eval()
        for ablation in ablations:
            for case in dev:
                target = (Path(run)/"episodes"/"ablation"/str(saved["seed"])/
                          ablation/case["case_id"])
                results.append(run_episode(case, "ppo_mean", target,
                    artifacts=artifacts, policy=policy, ablation=ablation))
    immutable_json(Path(run)/"ablation_results.json",
                   {"split": "dev_quick", "ablations": list(ablations), "results": results})
    return {"episodes": len(results), "failures": sum(not r["complete"] for r in results)}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--config", type=Path, default=PACKAGE/"config.json"); p.add_argument("--out", type=Path, required=True); p.add_argument("--direction-source", type=Path)
    p = sub.add_parser("smoke"); p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("collect"); p.add_argument("--run", type=Path, required=True); p.add_argument("--split", default="train_calibration"); p.add_argument("--episodes", type=int, default=32)
    p = sub.add_parser("tune"); p.add_argument("--run", type=Path, required=True); p.add_argument("--split", default="dev_screen")
    p = sub.add_parser("evaluate"); p.add_argument("--run", type=Path, required=True); p.add_argument("--split", default="dev_quick"); p.add_argument("--methods", default="hybrid_zero,fixed_best,rule_best,ppo"); p.add_argument("--checkpoint", type=Path); p.add_argument("--ablation"); p.add_argument("--checkpoints"); p.add_argument("--tag"); p.add_argument("--gate-phase-steps", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.config, args.out, args.direction_source)
    elif args.command == "smoke": result = smoke(args.run)
    elif args.command == "collect": result = collect(args.run, args.episodes, args.split)
    elif args.command == "tune": result = tune(args.run, args.split)
    elif args.command == "evaluate" and args.ablation:
        checkpoints = [Path(p) for p in (args.checkpoints or "").split(",") if p]
        if not checkpoints: raise ValueError("ablation requires --checkpoints")
        result = evaluate_ablation(args.run, checkpoints, tuple(args.ablation.split(",")))
    else: result = evaluate(args.run, args.split, args.methods, args.checkpoint,
                            [Path(p) for p in args.checkpoints.split(",") if p] if args.checkpoints else None,
                            args.tag, args.gate_phase_steps)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
