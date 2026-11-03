"""Independent plant replay checks for the actuator-aware prediction model."""
import numpy as np

from simulator.batch_mpc import BatchedUUVModel
from simulator.model import DT, MISMATCHED_PLANT
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


def test_augmented_five_substeps_match_torch_plant_with_pending_queue():
    model = BatchedUUVModel("cpu")
    dynamics = ActuatorAwareCpuDynamics(.1)
    hull = np.array([10., -4., 280., .04, .1, 6., .02, -.03, .01, .02])
    actuator = np.array([.1, .02, -.03])
    queue = np.array([[.3, .1, -.2], [.2, -.1, .4], [.1, .2, -.3]])
    command = np.array([.7, -.25, .45])
    predicted = dynamics.evaluate(np.r_[hull, actuator, queue.ravel()][None, None],
                                  command[None, None])[0][0, 0]

    import torch
    state = model.tensor(hull)[None]
    actual = model.tensor(actuator)
    history = model.tensor(queue)
    parameters = model.tensor([MISMATCHED_PLANT.mass_scale,
        MISMATCHED_PLANT.damping_scale, MISMATCHED_PLANT.thrust_gain,
        MISMATCHED_PLANT.elevator_gain, MISMATCHED_PLANT.rudder_gain])[None]
    alpha = model.tensor([-np.expm1(-DT/MISMATCHED_PLANT.thrust_tau_s),
                          -np.expm1(-DT/MISMATCHED_PLANT.fin_tau_s),
                          -np.expm1(-DT/MISMATCHED_PLANT.fin_tau_s)])
    for _ in range(5):
        target = history[2].clone()
        history = torch.cat((model.tensor(command)[None], history[:-1]), 0)
        actual = actual + alpha * (target-actual)
        state = model.rk4(state, (actual*model.control_scale)[None], parameters)
    expected = np.r_[state.numpy().ravel(), actual.numpy(), history.numpy().ravel()]
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-10)
