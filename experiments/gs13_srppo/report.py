"""Immutable paired reports with explicit failures, truncations and capacity strata."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

FIELDS = ("scene_id", "method", "split", "capacity_limited", "failed", "truncated",
          "complete", "cost", "tracking_integral", "qp_failures", "deadline_misses",
          "saturation_fraction")
FLAGS = ("capacity_limited", "failed", "truncated", "complete")
NUMBERS = ("cost", "tracking_integral", "qp_failures", "deadline_misses", "saturation_fraction")


def _validate(rows: Iterable[Mapping[str, Any]], baseline: str) -> list[dict[str, Any]]:
    records, seen, metadata = [], set(), {}
    for original in rows:
        missing = set(FIELDS) - original.keys()
        if missing:
            raise ValueError(f"missing report fields: {sorted(missing)}")
        row = {key: original[key] for key in FIELDS}
        for key in ("scene_id", "method", "split"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        pair = row["method"], row["scene_id"]
        if pair in seen:
            raise ValueError(f"duplicate method/scene pair: {pair}")
        seen.add(pair)
        for key in FLAGS:
            if not isinstance(row[key], bool):
                raise ValueError(f"{key} must be boolean")
        if sum(row[key] for key in ("failed", "truncated", "complete")) != 1:
            raise ValueError("episode must be exactly one of failed, truncated or complete")
        for key in NUMBERS:
            if isinstance(row[key], bool):
                raise ValueError(f"{key} must be numeric")
            try:
                row[key] = float(row[key])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid numeric {key}") from exc
            if not math.isfinite(row[key]) or row[key] < 0.:
                raise ValueError(f"{key} must be finite and nonnegative")
        if row["saturation_fraction"] > 1.:
            raise ValueError("saturation_fraction must lie in [0,1]")
        for key in ("qp_failures", "deadline_misses"):
            if not row[key].is_integer():
                raise ValueError(f"{key} must be an integer count")
            row[key] = int(row[key])
        scene_metadata = row["split"], row["capacity_limited"]
        previous = metadata.setdefault(row["scene_id"], scene_metadata)
        if previous != scene_metadata:
            raise ValueError(f"split/capacity metadata differs across paired scene {row['scene_id']}")
        records.append(row)
    if not records:
        raise ValueError("cannot report an empty evaluation")
    if len({row["split"] for row in records}) != 1:
        raise ValueError("one report must use exactly one frozen split")
    methods = {row["method"] for row in records}
    if baseline not in methods:
        raise ValueError(f"required baseline missing: {baseline}")
    expected_scenes = {row["scene_id"] for row in records if row["method"] == baseline}
    for method in methods:
        scenes = {row["scene_id"] for row in records if row["method"] == method}
        if scenes != expected_scenes:
            raise ValueError(f"missing paired scenes for {method}: "
                             f"missing={sorted(expected_scenes - scenes)}, extra={sorted(scenes - expected_scenes)}")
    return sorted(records, key=lambda row: (row["scene_id"], row["method"]))


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _method(values: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(values)
    failures = sum(row["failed"] for row in values)
    truncated = sum(row["truncated"] for row in values)
    return {"n": n, "mean_cost": _mean([row["cost"] for row in values]),
            "mean_tracking_integral": _mean([row["tracking_integral"] for row in values]),
            "failures": failures, "failure_rate": failures / n if n else None,
            "completed": sum(row["complete"] for row in values), "truncated": truncated,
            "qp_failures": sum(row["qp_failures"] for row in values),
            "deadline_misses": sum(row["deadline_misses"] for row in values),
            "mean_saturation_fraction": _mean([row["saturation_fraction"] for row in values]),
            "comparison_eligible": n > 0 and truncated == 0}


def _gain(candidate: list[float], reference: list[float]) -> float | None:
    base = _mean(reference)
    return (1. - _mean(candidate) / base) if base is not None and base > 0. else None


def _paired(candidate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    base = {row["scene_id"]: row for row in reference}
    pairs = [(row, base[row["scene_id"]]) for row in candidate]
    eligible = bool(pairs) and not any(a["truncated"] or b["truncated"] for a, b in pairs)
    successes = [(a, b) for a, b in pairs if a["complete"] and b["complete"]]
    # All failures remain in full-pair counts and absorbing-cost statistics.
    # Common-success metrics are labelled a conditional diagnostic, never used
    # to erase a failed pair from the full comparison.
    common = {"n": len(successes), "excluded_pairs": len(pairs) - len(successes),
              "mean_cost_delta": _mean([a["cost"] - b["cost"] for a, b in successes]),
              "gain_fraction": _gain([a["cost"] for a, _ in successes], [b["cost"] for _, b in successes]),
              "tracking_integral_gain_fraction": _gain([a["tracking_integral"] for a, _ in successes],
                                                        [b["tracking_integral"] for _, b in successes]),
              "scene_ids": [a["scene_id"] for a, _ in successes]}
    return {"n": len(pairs), "comparison_eligible": eligible,
            "truncated_pairs": sum(a["truncated"] or b["truncated"] for a, b in pairs),
            "failures": sum(a["failed"] for a, _ in pairs),
            "baseline_failures": sum(b["failed"] for _, b in pairs),
            "failure_delta": sum(int(a["failed"]) - int(b["failed"]) for a, b in pairs),
            "mean_cost_delta": _mean([a["cost"] - b["cost"] for a, b in pairs]) if eligible else None,
            "gain_fraction": _gain([a["cost"] for a, _ in pairs], [b["cost"] for _, b in pairs]) if eligible else None,
            "qp_failure_delta": sum(a["qp_failures"] - b["qp_failures"] for a, b in pairs),
            "deadline_miss_delta": sum(a["deadline_misses"] - b["deadline_misses"] for a, b in pairs),
            "saturation_fraction_delta": _mean([a["saturation_fraction"] - b["saturation_fraction"] for a, b in pairs]),
            "common_success": common}


def paired_summary(rows: Iterable[Mapping[str, Any]], baseline: str = "fixed_best") -> dict[str, Any]:
    records = _validate(rows, baseline)
    methods = sorted({row["method"] for row in records})
    baselines = [baseline] + [name for name in ("fixed_best", "rule_best") if name in methods and name != baseline]
    summary = {"schema_version": "gs13_paired_report_v1", "baseline": baseline,
               "baselines": baselines, "split": records[0]["split"], "row_count": len(records),
               "scene_count": len({row["scene_id"] for row in records}),
               "complete_pairing": True, "strata": {},
               "cost_definition": "duration-normalized tracking plus absorbing failure cost supplied by evaluation",
               "common_prefix_available": False,
               "limitations": ["Summary rows do not contain shared surviving-prefix trajectories.",
                               "Common-success gains condition on both methods succeeding; failures remain in full pairs.",
                               "Truncated evaluations retain partial costs but cannot qualify for checkpoint selection."]}
    for name, capacity in (("all", None), ("feasible", False), ("capacity_limited", True)):
        values = [row for row in records if capacity is None or row["capacity_limited"] == capacity]
        grouped = {method: [row for row in values if row["method"] == method] for method in methods}
        stats = {method: _method(grouped[method]) for method in methods}
        comparisons = {base: {method: _paired(grouped[method], grouped[base])
                              for method in methods if method != base} for base in baselines}
        summary["strata"][name] = {"scene_count": len(values) // len(methods), "methods": stats,
                                   "paired": comparisons,
                                   "comparison_eligible": bool(values) and all(v["comparison_eligible"] for v in stats.values())}
    # Compatibility alias is the unfiltered table; selection should use feasible.
    summary["methods"] = summary["strata"]["all"]["methods"]
    return summary


def _number(value: Any) -> str:
    return "—" if value is None else f"{value:.6g}"


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# GS-13 配对实验报告", "", f"集合：`{summary['split']}`；场景 {summary['scene_count']}；记录 {summary['row_count']}。",
             "", "失败回合保留吸收成本。截断回合保留并标注，禁止用于晋级；共同成功增益是条件诊断。", ""]
    for name, stratum in summary["strata"].items():
        lines += [f"## {name}", "", "| 方法 | 场景 | 完成 | 失败 | 截断 | 平均成本 | QP 失败 | deadline miss | 饱和比例 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method, row in stratum["methods"].items():
            lines.append(f"| {method} | {row['n']} | {row['completed']} | {row['failures']} | {row['truncated']} | "
                         f"{_number(row['mean_cost'])} | {row['qp_failures']} | {row['deadline_misses']} | {_number(row['mean_saturation_fraction'])} |")
        lines += ["", "| 基线 → 方法 | 配对数 | 失败差 | 全配对增益 | 共同成功数 | 共同成功增益 | 可比较 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for base, candidates in stratum["paired"].items():
            for method, row in candidates.items():
                common = row["common_success"]
                lines.append(f"| {base} → {method} | {row['n']} | {row['failure_delta']} | {_number(row['gain_fraction'])} | "
                             f"{common['n']} | {_number(common['gain_fraction'])} | {row['comparison_eligible']} |")
        lines += [""]
    lines += ["增益 = 1 − 方法平均成本 / 基线平均成本；正值表示改善。基线成本为零时增益未定义。", "",
              "当前输入只有回合摘要，未计算共同存活前缀 RMSE、p99 延迟或置信区间；这些指标需要原始轨迹。", ""]
    return "\n".join(lines)


def _immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"existing report artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".gs13_report_", delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"concurrent report artifact differs: {path}")
    finally:
        temporary.unlink()


def write_report(rows: Iterable[Mapping[str, Any]], output: str | Path,
                 baseline: str = "fixed_best") -> dict[str, Any]:
    records = _validate(rows, baseline)
    summary = paired_summary(records, baseline)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    csv_bytes = stream.getvalue().encode("utf-8")
    summary["paired_metrics_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    contents = {"paired_metrics.csv": csv_bytes,
                "training_summary.json": (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
                "report.md": _markdown(summary).encode("utf-8")}
    output = Path(output)
    # Preflight all files before writing any, so a conflicting re-run cannot
    # partially replace a prior report. Identical inputs are idempotent.
    for name, content in contents.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"existing report artifact differs: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        _immutable(output / name, content)
    return summary
