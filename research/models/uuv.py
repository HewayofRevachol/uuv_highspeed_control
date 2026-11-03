"""Default batched UUV dynamics and the compatible model interface.

Model modules export ``DynamicsModel``, constructed with one ``device`` argument.
The model supplies ``device``, ``dtype``, three-element ``control_scale``,
``lower`` and ``upper`` tensors, and ``tensor(value)``. Compatible methods are
``derivative(state, control, parameters=None, disturbance=None, current=None)``,
``rk4(state, control, parameters=None, disturbance=None, current=None, dt=DT)``,
and ``linearize(state, normalized_control, dt)``. Linearization returns batched
discrete ``A``, ``B`` and affine-offset tensors with shapes [B,10,10], [B,10,3]
and [B,10]. State layout and physical control units match ``BatchedUUVModel``.

The environment, linear MPC and filtered predictor share this model instance.
Custom models therefore change both simulated plant and nominal prediction;
the environment's existing parameter mismatch, actuator and disturbance logic
still applies. Custom models support the admm and osqp backends. RTI prediction
has separate symbolic dynamics and requires the default model.
"""

from simulator.batch_mpc import BatchedUUVModel


DynamicsModel = BatchedUUVModel
