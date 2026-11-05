"""Bind experiment labels to the controller that actually generates actions."""
import hashlib
from pathlib import Path


FIXED_GATES = {"A0": (0., 0.), "A1": (1., 0.), "A2": (0., 1.),
               "A3": (1., 1.), "A4": (.5, .5)}
METHODS = (*FIXED_GATES, "PPO")


def checkpoint_method(checkpoint):
    """Stable checkpoint labels used by paired PPO analysis."""
    suffix = Path(checkpoint).stem.removeprefix("ckpt_")
    if not suffix:
        raise ValueError("policy checkpoint must have a nonempty label")
    return "PPO_" + suffix


def validate_method_checkpoint(method, policy_checkpoint):
    """Reject mislabeled runs before allocating output or constructing an env."""
    if method in FIXED_GATES:
        if policy_checkpoint is not None:
            raise ValueError(f"fixed method {method} must not carry a policy checkpoint")
        return "fixed_gate"
    if method == "PPO" or (isinstance(method, str) and method.startswith("PPO_")):
        if policy_checkpoint is None:
            raise ValueError(f"policy method {method} requires a policy checkpoint")
        if method != "PPO" and method != checkpoint_method(policy_checkpoint):
            raise ValueError(f"method {method} does not match checkpoint label "
                             f"{checkpoint_method(policy_checkpoint)}")
        return "PPO"
    raise ValueError(f"unknown method {method!r}; expected A0-A4 or PPO")


def policy_execution_metadata(method, policy_checkpoint, deterministic=False):
    family = validate_method_checkpoint(method, policy_checkpoint)
    checkpoint = Path(policy_checkpoint).resolve() if policy_checkpoint is not None else None
    return {"method_family": family,
            "policy_checkpoint": str(checkpoint) if checkpoint is not None else None,
            "policy_checkpoint_sha256": (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                                         if checkpoint is not None else None),
            "initial_alpha": list(FIXED_GATES.get(method, (0., 0.))),
            "action_generation": ("fixed_gate" if family == "fixed_gate" else
                                  "policy_mean" if deterministic else "policy_gaussian_sample"),
            "policy_deterministic": bool(deterministic) if family == "PPO" else False}
