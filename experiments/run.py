"""Launch one isolated training or validation run from a JSON configuration.

Run from any directory: python3 experiments/run.py experiments/configs/ddat_cpu_smoke.json
Start several commands in separate terminals to compare variants concurrently.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "experiments"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def source_hashes():
    """Record the code used for a run so later edits remain auditable."""
    files = [ROOT / "DDAT_train.py", ROOT / "validate_ddat.py"]
    for directory in ("simulator", "research", "trajectories", "high_speed_experiments"):
        files.extend(sorted((ROOT / directory).rglob("*.py")))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def command_args(values):
    if not isinstance(values, dict):
        raise ValueError("arguments must be a JSON object")
    result = []
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"invalid argument name: {key!r}")
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                result.append(flag)
            elif key in {"fsp_enabled", "feasible_only", "randomize_plant"}:
                result.append("--no-" + key.replace("_", "-"))
            continue
        if isinstance(value, (str, int, float)):
            result.extend((flag, str(value)))
            continue
        raise ValueError(f"unsupported value for {key}: {value!r}")
    return result


def prepare(config_path, name_override=None, dry_run=False):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    mode = config.get("mode")
    if mode not in {"train", "validate"}:
        raise ValueError("mode must be train or validate")
    name = name_override or config.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError("name must contain only letters, digits, _ or -")
    values = config.get("arguments", {})
    arguments = command_args(values)
    if mode == "train" and any(k in values for k in ("output", "eval_actor", "list_training_paths")):
        raise ValueError("train arguments cannot set output, eval_actor or list_training_paths")
    if mode == "validate":
        if "output_dir" in values:
            raise ValueError("validate arguments cannot set output_dir")
        if not values.get("checkpoint"):
            raise ValueError("validation requires an explicit checkpoint")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = OUTPUT_ROOT / name / f"{stamp}_{os.getpid()}"
    script = ROOT / ("DDAT_train.py" if mode == "train" else "validate_ddat.py")
    output_flag = "--output" if mode == "train" else "--output-dir"
    command = [sys.executable, str(script), *arguments, output_flag, str(run_dir)]
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=False)
    return config, run_dir, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", help="override the experiment label")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, run_dir, command = prepare(args.config, args.name, args.dry_run)
    print(f"run: {run_dir}", flush=True)
    print("command: " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    manifest = {
        "source_config": str(args.config.resolve()),
        "config": config,
        "command": command,
        "cwd": str(ROOT),
        "source_sha256": source_hashes(),
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "running",
    }
    manifest_path = run_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    process = None
    try:
        with (run_dir / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        manifest["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        raise
    manifest["status"] = "completed" if return_code == 0 else "failed"
    manifest["exit_code"] = return_code
    manifest["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
