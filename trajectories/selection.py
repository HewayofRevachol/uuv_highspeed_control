"""Steady-feasibility screening shared by training and evaluation."""

from dataclasses import replace
from itertools import product

from simulator.model import MISMATCHED_PLANT


def select_paths(files, feasible_only, randomize_plant=False):
    """Filter sustained steady command-limit violations; keep audit metadata."""
    if not feasible_only:
        return files, []
    from high_speed_experiments.screen_dynamic_feasibility import screen_path

    plant = MISMATCHED_PLANT
    if randomize_plant:
        # BatchPathResidualEnv.reset multiplies the five baseline parameters
        # by [.97,1.03], [.93,1.07], [.95,1.05], [.90,1.10], [.90,1.10].
        # Mass does not alter a zero-acceleration trim. Delays/time constants
        # affect transients, which this steady screen cannot certify.
        # Test the expected worst corner first, then every other gain corner.
        plant = replace(plant, mass_scale=plant.mass_scale*1.03,
                        damping_scale=plant.damping_scale*1.07,
                        thrust_gain=plant.thrust_gain*.95,
                        elevator_gain=plant.elevator_gain*.90,
                        rudder_gain=plant.rudder_gain*.90)
    screening = []
    for file in files:
        result = screen_path(file, plant)
        result["parameter_corners_checked"] = 1
        if result["status"] == "not_screened_out" and result["unresolved_steady_samples"]:
            result["status"] = "screened_out"
            result["reason"] = "unresolved_steady_trim"
        if randomize_plant and result["status"] == "not_screened_out":
            for damping, thrust, elevator, rudder in product(
                    (.93, 1.07), (.95, 1.05), (.90, 1.10), (.90, 1.10)):
                if (damping, thrust, elevator, rudder) == (1.07, .95, .90, .90):
                    continue
                corner = replace(MISMATCHED_PLANT,
                                 damping_scale=MISMATCHED_PLANT.damping_scale*damping,
                                 thrust_gain=MISMATCHED_PLANT.thrust_gain*thrust,
                                 elevator_gain=MISMATCHED_PLANT.elevator_gain*elevator,
                                 rudder_gain=MISMATCHED_PLANT.rudder_gain*rudder)
                checked = screen_path(file, corner)
                result["parameter_corners_checked"] += 1
                if (checked["status"] == "screened_out" or
                        checked["unresolved_steady_samples"]):
                    result["status"] = "screened_out"
                    result["reason"] = "randomized_parameter_corner_limit_or_unresolved"
                    result["failed_corner"] = {
                        "damping_multiplier": damping,
                        "thrust_multiplier": thrust,
                        "elevator_multiplier": elevator,
                        "rudder_multiplier": rudder,
                        "screen_result": checked,
                    }
                    break
        screening.append(result)
    selected = [file for file, result in zip(files, screening)
                if result["status"] == "not_screened_out"]
    if not selected:
        raise ValueError("steady-feasibility screen retained no paths")
    return selected, screening
