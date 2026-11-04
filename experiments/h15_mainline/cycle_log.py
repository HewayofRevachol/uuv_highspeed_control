"""Frozen per-control-period log contract for H15 mainline methods."""

import math


SCHEMA = 1
FIELDS = ("schema", "method", "episode_id", "period", "time_s",
          "state_before", "state_after", "actuator_before", "actuator_after",
          "command_queue_before", "issued_command", "raw_rl_intent_mps",
          "ramped_nu_ref_mps", "actual_tangent_water_speed_mps",
          "candidate_empty", "candidate_rejections", "selected_candidate_mps",
          "fallback", "first_violation", "qp_status", "timing_ms")
TIMING_FIELDS = ("estimator", "candidate_screen", "policy", "mpc",
                 "plant_integration", "full_period")
FORMAL_METHODS = ("B0_shared", "B2_gov", "B3_RL")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, size):
    return isinstance(value, list) and len(value) == size and all(_finite_number(x) for x in value)


def validate(record):
    if tuple(record) != FIELDS or record["schema"] != SCHEMA:
        raise ValueError("cycle log schema or order differs")
    if record["method"] not in FORMAL_METHODS + ("reference_chain_smoke",):
        raise ValueError("unknown H15 cycle method")
    if (not isinstance(record["episode_id"], str) or not record["episode_id"] or
            not isinstance(record["period"], int) or record["period"] < 0 or
            not _finite_number(record["time_s"])):
        raise ValueError("invalid cycle identity or timestamp")
    for field in ("state_before", "state_after"):
        if not _vector(record[field], 10):
            raise ValueError(f"invalid {field}")
    for field in ("actuator_before", "actuator_after", "issued_command"):
        if not _vector(record[field], 3):
            raise ValueError(f"invalid {field}")
    if (not isinstance(record["command_queue_before"], list) or
            len(record["command_queue_before"]) != 5 or
            any(not _vector(item, 3) for item in record["command_queue_before"])):
        raise ValueError("invalid command queue")
    for field in ("ramped_nu_ref_mps", "actual_tangent_water_speed_mps"):
        if not _finite_number(record[field]):
            raise ValueError(f"invalid {field}")
    if not 13 <= record["ramped_nu_ref_mps"] <= 17:
        raise ValueError("reference progress rate outside frozen range")
    if not isinstance(record["qp_status"], str) or not record["qp_status"]:
        raise ValueError("missing QP status")
    if record["first_violation"] is not None:
        event = record["first_violation"]
        if (not isinstance(event, dict) or tuple(event) != ("time_s", "kind") or
                not _finite_number(event["time_s"]) or
                not isinstance(event["kind"], str)):
            raise ValueError("invalid first violation event")
    times = record["timing_ms"]
    if tuple(times) != TIMING_FIELDS or any(
            value is not None and (not _finite_number(value) or value < 0)
            for value in times.values()):
        raise ValueError("invalid timing fields")
    if record["method"] == "reference_chain_smoke":
        if any(record[field] is not None for field in
               ("raw_rl_intent_mps", "candidate_empty", "candidate_rejections",
                "selected_candidate_mps")):
            raise ValueError("diagnostic smoke cannot claim candidate or RL events")
    else:
        if (not isinstance(record["candidate_empty"], bool) or
                not isinstance(record["candidate_rejections"], list) or
                any(value is None for value in times.values())):
            raise ValueError("formal cycle lacks candidate events or complete timings")
        for item in record["candidate_rejections"]:
            if (not isinstance(item, dict) or tuple(item) != ("candidate_mps", "reason") or
                    not _finite_number(item["candidate_mps"]) or
                    (item["reason"] is not None and not isinstance(item["reason"], str))):
                raise ValueError("invalid candidate rejection entry")
        if record["candidate_empty"]:
            if record["selected_candidate_mps"] is not None or record["fallback"] is None:
                raise ValueError("empty set requires logged fallback and no selection")
        elif record["selected_candidate_mps"] is None:
            if record["fallback"] is None:
                raise ValueError("unselected intent requires logged fallback")
        elif not _finite_number(record["selected_candidate_mps"]):
            raise ValueError("invalid selected candidate")
        if record["method"] == "B3_RL":
            if not _finite_number(record["raw_rl_intent_mps"]):
                raise ValueError("B3_RL requires raw policy intent")
        elif record["raw_rl_intent_mps"] is not None:
            raise ValueError("non-RL method cannot claim raw RL intent")
    return record
