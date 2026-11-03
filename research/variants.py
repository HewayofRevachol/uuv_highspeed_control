"""Build components from the same configuration in training and evaluation.

Modules are trusted Python code under ``research``. Existing checkpoints may
omit all component keys; their original classes and constructor defaults remain
the fallback. See docs/extending.md for the DDAT-specific component contracts.
"""

from collections.abc import Mapping
from importlib import import_module
from inspect import isclass

DEFAULT_CONTROLLER = "research.controllers.ddat"
DEFAULT_NETWORK = "research.networks.ddat"
DEFAULT_MODEL = "research.models.uuv"

# Historical evaluation defaults, before these switches were checkpointed.
ENVIRONMENT_DEFAULTS = {
    "horizon": 30, "backend": "admm", "seed": 1026,
    "randomize_plant": True, "admm_iterations": 400, "admm_rho": 2.,
    "adaptation_time": .5, "current_max_mps": 1., "current_filter_time": 1.,
    "attitude_pitch_weight": 0., "attitude_yaw_weight": 0.,
    "fsp_enabled": False, "fsp_filter_time": .4, "fsp_preview_time": None,
    "preview_mode": "dense", "yaw_reward": "heading",
}


def component(config, key, default, class_name, methods=()):
    if not isinstance(config, Mapping):
        raise ValueError("component configuration must be a mapping")
    module_name = config.get(key, default)
    if not isinstance(module_name, str) or not module_name.startswith("research."):
        raise ValueError(f"{key} must name a research.* Python module")
    module = import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(f"{module_name} must define {class_name}") from error
    if not isclass(cls):
        raise ValueError(f"{module_name}.{class_name} must be a class")
    missing = [name for name in methods if not callable(getattr(cls, name, None))]
    if missing:
        raise ValueError(f"{module_name}.{class_name} is missing methods: {', '.join(missing)}")
    return cls


def controller_class(config):
    return component(config, "controller_module", DEFAULT_CONTROLLER,
                     "ExperimentDDATPathEnv", ("reset", "observe", "step"))


def network_class(config):
    return component(config, "network_module", DEFAULT_NETWORK, "ActorCritic",
                     ("deterministic", "value", "sample", "log_prob", "distribution",
                      "to", "parameters", "state_dict", "load_state_dict", "eval"))


def model_class(config):
    return component(config, "model_module", DEFAULT_MODEL, "DynamicsModel",
                     ("tensor", "derivative", "rk4", "linearize"))


def component_kwargs(config, key, reserved=()):
    values = config.get(key, {})
    if not isinstance(values, dict) or any(not isinstance(k, str) for k in values):
        raise ValueError(f"{key} must be a JSON object with string keys")
    collisions = sorted(set(values).intersection(reserved))
    if collisions:
        raise ValueError(f"{key} cannot override managed arguments: {', '.join(collisions)}")
    return dict(values)


def create_environment(config, path_bank, batch_size, device, **overrides):
    """Construct a DDAT-compatible environment, with explicit run overrides last."""
    cls = controller_class(config)
    options = component_kwargs(config, "controller_kwargs", {
        *ENVIRONMENT_DEFAULTS, "path_bank", "batch_size", "device", "model_factory",
    })
    unknown = set(overrides).difference(ENVIRONMENT_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown environment overrides: {', '.join(sorted(unknown))}")
    arguments = {key: config.get(key, value) for key, value in ENVIRONMENT_DEFAULTS.items()}
    arguments.update(options)
    arguments.update(overrides)
    # Do not add a new keyword to legacy controllers unless a model was selected.
    if config.get("model_module", DEFAULT_MODEL) != DEFAULT_MODEL:
        arguments["model_factory"] = model_class(config)
    return cls(path_bank, batch_size=batch_size, device=device, **arguments)


def create_network(config, observation_dim, device):
    """Rebuild a policy from checkpointed architecture options."""
    cls = network_class(config)
    options = component_kwargs(config, "network_kwargs", {"observation_dim"})
    return cls(observation_dim, **options).to(device)
