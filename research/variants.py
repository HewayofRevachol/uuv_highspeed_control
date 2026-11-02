"""Resolve experiment components recorded in training configurations."""

from importlib import import_module

DEFAULT_CONTROLLER = "research.controllers.ddat"
DEFAULT_NETWORK = "research.networks.ddat"


def component(config, key, default, class_name):
    module_name = config.get(key, default)
    if not isinstance(module_name, str) or not module_name.startswith("research."):
        raise ValueError(f"{key} must name a research.* Python module")
    module = import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(f"{module_name} must define {class_name}") from error


def controller_class(config):
    return component(config, "controller_module", DEFAULT_CONTROLLER,
                     "ExperimentDDATPathEnv")


def network_class(config):
    return component(config, "network_module", DEFAULT_NETWORK, "ActorCritic")
