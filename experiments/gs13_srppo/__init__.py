"""GS-13 safety-residual PPO experiment.

The package is deliberately kept independent of the expensive GS-12 plant
runner.  It provides the policy, safety projection, constrained costs and
training contracts that an episode adapter can call without changing the
MPC solve budget.
"""

from .safety import ProjectionResult, SafetyConfig, project_gate, shield_action
from .costs import StepCosts, compute_cost, scalarize_cost
from .policy import SRPPOActorCritic

__all__ = [
    "ProjectionResult", "SafetyConfig", "project_gate", "shield_action",
    "StepCosts", "compute_cost", "scalarize_cost", "SRPPOActorCritic",
]
