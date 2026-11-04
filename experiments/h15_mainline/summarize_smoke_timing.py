"""Summarize instrumented smoke timings without making online method claims."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("plant_integration_wall_ms", "mpc_wall_ms", "step_wall_ms",
          "control_period_wall_ms")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.smoke / "summary.json"
    summaries = json.loads(summary_path.read_text())
    if len(summaries) != 6 or not all(result.get("profile_stages") for result in summaries):
        raise RuntimeError("requires all six profiled smoke traces")
    lines = ["# H15 参考链路 smoke 分段耗时", "",
             "这不是 B0_shared/B2_gov/B3_RL 的端到端在线计时；估计器、候选筛选与 RL 均未运行。",
             "`step_wall_ms` 包含植物积分及下一周期 MPC，`control_period_wall_ms` 还包含子步日志序列化。",
             "二者均未覆盖诊断函数在 `env.step` 之前的预测重放。", "",
             "| 场景 / 意图 | 分段 | p50 | p95 | p99 | max | >100 ms |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    trace_hashes = {}
    for result in summaries:
        trace = ROOT / result["trace"]
        trace_hashes[result["trace"]] = sha(trace)
        with trace.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != result["steps"]:
            raise RuntimeError(f"trace length mismatch: {trace}")
        for field in FIELDS:
            values = np.array([float(row[field]) for row in rows])
            if not np.isfinite(values).all() or (values < 0).any():
                raise RuntimeError(f"invalid timing: {trace} {field}")
            p50, p95, p99 = np.percentile(values, [50, 95, 99])
            label = f"{result['case_label']} / {result['intent_mps']:g}"
            lines.append(f"| {label} | {field} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {values.max():.2f} | {(values > 100).mean():.3f} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    run_record = {"schema": 1, "method": "h15_smoke_timing_summary",
                  "input_summary_sha256": sha(summary_path), "trace_sha256": trace_hashes,
                  "source_sha256": sha(Path(__file__)), "report_sha256": sha(args.out),
                  "python_executable": "/bin/python3"}
    (args.out.parent / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    (args.out.parent / "console.log").write_text(f"summarized {len(summaries)} profiled smoke traces\n")
    print(f"summarized {len(summaries)} profiled smoke traces")


if __name__ == "__main__":
    main()
