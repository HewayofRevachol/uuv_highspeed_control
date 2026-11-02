"""UUV 路径、五自由度动力学与执行器仿真模型。

状态输入/输出均按 [north, east, depth, pitch, yaw, u, v, w, q, r] 排列；
位置单位为 m，角度为 rad，线速度为 m/s，角速度为 rad/s。
控制输入/输出按 [推力 N, 升降舵角 rad, 方向舵角 rad] 排列。
"""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PlantConfig:
    """艇体及执行器参数：前五项为无量纲正比例，延迟和时间常数单位为 s。"""

    mass_scale: float = 1.0
    damping_scale: float = 1.0
    thrust_gain: float = 1.0
    elevator_gain: float = 1.0
    rudder_gain: float = 1.0
    thrust_delay_s: float = 0.0
    fin_delay_s: float = 0.0
    thrust_tau_s: float = 0.0
    fin_tau_s: float = 0.0

    def __post_init__(self):
        """输入：本实例各参数；输出：无效参数抛出 ValueError，否则无返回值。"""
        from math import isfinite

        for name in ("mass_scale", "damping_scale", "thrust_gain",
                     "elevator_gain", "rudder_gain"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("thrust_delay_s", "fin_delay_s", "thrust_tau_s", "fin_tau_s"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")


NOMINAL_PLANT = PlantConfig()
# An illustrative fixed mismatch, not coefficients identified from a vehicle.
MISMATCHED_PLANT = PlantConfig(
    mass_scale=1.04, damping_scale=1.05, thrust_gain=0.98,
    elevator_gain=0.95, rudder_gain=0.95,
    thrust_delay_s=0.05, fin_delay_s=0.05,
    thrust_tau_s=0.10, fin_tau_s=0.08,
)


def add_plant_arguments(parser):
    """输入：argparse.ArgumentParser；输出：无，原位添加艇体参数选项。"""
    parser.add_argument("--plant-profile", choices=("nominal", "mismatch"),
                        default="mismatch",
                        help="艇体参数和执行器模型；默认 mismatch，包含示例失配与延迟")
    for name in ("mass_scale", "damping_scale", "thrust_gain",
                 "elevator_gain", "rudder_gain", "thrust_delay_s",
                 "fin_delay_s", "thrust_tau_s", "fin_tau_s"):
        parser.add_argument("--" + name.replace("_", "-"), type=float,
                            help=f"覆盖所选 profile 的 {name}")


def plant_from_args(args):
    """输入：含艇体命令行选项的 argparse.Namespace；输出：PlantConfig。"""
    profile = MISMATCHED_PLANT if args.plant_profile == "mismatch" else NOMINAL_PLANT
    overrides = {
        name: getattr(args, name)
        for name in PlantConfig.__dataclass_fields__
        if getattr(args, name) is not None
    }
    return replace(profile, **overrides)


# 仿真参数。轨迹文件由 LoopPath 的调用者提供。
DT = 0.02
STEPS = 4000
RENDER_EVERY = 10  # 每 10 个积分步刷新一次画面
# 播放倍率：1.0 为原速，2.0 为两倍速，0.5 为半速；必须大于 0。
PLAYBACK_SPEED = 10.0
LOOKAHEAD = 24.0
MAX_RUDDER = np.deg2rad(45)
RANDOM_SEED = 20268
# 体坐标系 [u, v, w, q, r] 加速度扰动的稳态标准差。
DISTURBANCE_STD = np.array([0.012, 0.012, 0.012, 0.0008, 0.0008])
DISTURBANCE_CORRELATION_S = 2.0
TRAJECTORY_COLUMNS = (
    "t_s", "s_m", "north_m", "east_m", "depth_m",
    "roll_rad", "pitch_rad", "yaw_rad",
    "u_mps", "v_mps", "w_mps", "p_radps", "q_radps", "r_radps",
)

MASS = 214_225.0
M_EFF = np.array([
    [217_198.87, 0, 0, 0, 0],
    [0, 401_214.92, 0, 0, 412_653.81],
    [0, 0, 401_214.92, -264_217.19, 0],
    [0, 0, 83_124.51, 31_994_175.01, 0],
    [0, 135_077.33, 0, 0, 31_994_175.01],
])
D_COEFF = np.array([
    [4_104.41, 0, 0, 0, 0],
    [0, 9_929.52, 0, 0, -80_016.28],
    [0, 0, 7_462.56, -103_550.48, 0],
    [0, 0, -30_166.57, -1_610_537.36, 0],
    [0, 30_166.57, 0, 0, 1_016_790.87],
])
# 舵效向量按 [u, v, w, q, r] 排列，作用项为 u² * 舵效 * 舵角。
ELEVATOR_COEFF = np.array([0, 0, -4_989.43, -72_742.08, 0])
RUDDER_COEFF = np.array([0, -4_989.43, 0, 0, 72_742.08])


class LoopPath:
    """从 CSV 读取位置和目标切线速度，并按几何弧长跟踪路径。

    支持 north_m、east_m、depth_m、target_tangent_speed_mps 四列，
    可另给 target_pitch_rad、target_yaw_rad 作为明确的姿态参考；
    缺省时由路径切线计算。旧轨迹的 pitch_rad、yaw_rad 为历史艇体状态。
    旧 CSV 若没有显式速度列，则由位置和 t_s 计算目标切线速度。
    其余旧列不会用作跟踪参考。
    """

    def __init__(self, file):
        """输入：必填的轨迹 CSV 路径；输出：无，初始化路径及进度属性。"""
        data = np.genfromtxt(file, delimiter=",", names=True, dtype=float,
                             encoding="utf-8")
        required = ("north_m", "east_m", "depth_m")
        names = data.dtype.names or ()
        if not set(required).issubset(names):
            raise ValueError(f"{file}: 轨迹 CSV 必须包含 {required}")
        data = np.atleast_1d(data)
        self.points = np.column_stack([data[name] for name in required])
        if len(data) < 2 or not np.isfinite(self.points).all():
            raise ValueError(f"{file}: 轨迹至少需要两个有限的位置采样点")
        segment_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        if np.any(segment_lengths <= 0):
            raise ValueError(f"{file}: 相邻轨迹位置必须不同")
        self.distances = np.r_[0.0, np.cumsum(segment_lengths)]
        if "target_tangent_speed_mps" in names:
            self.speed = np.asarray(data["target_tangent_speed_mps"], dtype=float)
            if not np.isfinite(self.speed).all() or np.any(self.speed <= 0):
                raise ValueError(f"{file}: target_tangent_speed_mps 必须为正的有限数")
            mean_speed = (self.speed[:-1] + self.speed[1:]) / 2
            self.times = np.r_[0.0, np.cumsum(segment_lengths / mean_speed)]
        elif "t_s" in names:
            times = np.asarray(data["t_s"], dtype=float)
            if (not np.isfinite(times).all() or
                    np.any(np.diff(times) <= 0)):
                raise ValueError(f"{file}: t_s 必须为有限且严格递增的时间")
            self.times = times - times[0]
            self.speed = np.gradient(self.distances, self.times)
            if not np.isfinite(self.speed).all() or np.any(self.speed <= 0):
                raise ValueError(f"{file}: 无法从位置和 t_s 计算正的切线速度")
        else:
            raise ValueError(f"{file}: 轨迹 CSV 需要 target_tangent_speed_mps 或 t_s")

        tangent = np.gradient(self.points, self.distances, axis=0)
        for index in np.flatnonzero(np.linalg.norm(tangent, axis=1) < 1e-12):
            next_index = min(index + 1, len(self.points) - 1)
            previous_index = max(index - 1, 0)
            tangent[index] = (self.points[next_index] - self.points[index]
                              if next_index > index else
                              self.points[index] - self.points[previous_index])
        self.tangent = tangent / np.linalg.norm(tangent, axis=1)[:, None]
        geometric_pitch = -np.arctan2(self.tangent[:, 2],
                                      np.linalg.norm(self.tangent[:, :2], axis=1))
        geometric_yaw = np.unwrap(np.arctan2(self.tangent[:, 1], self.tangent[:, 0]))
        if ("target_pitch_rad" in names) != ("target_yaw_rad" in names):
            raise ValueError(f"{file}: target_pitch_rad 和 target_yaw_rad 必须同时提供")
        if "target_pitch_rad" in names:
            self.pitch = np.asarray(data["target_pitch_rad"], dtype=float)
            self.yaw = np.unwrap(np.asarray(data["target_yaw_rad"], dtype=float))
            if not np.isfinite(self.pitch).all() or not np.isfinite(self.yaw).all():
                raise ValueError(f"{file}: 姿态参考必须为有限数")
        else:
            self.pitch, self.yaw = geometric_pitch, geometric_yaw
        self.length = self.distances[-1]
        self.progress = 0.0

    def position_at(self, distance):
        """输入：路径弧长 distance (m)；输出：[north, east, depth] (m) 数组。"""
        return np.array([np.interp(distance, self.distances, axis)
                         for axis in self.points.T])

    def pitch_at(self, distance):
        """输入：路径弧长 distance (m)；输出：俯仰角 (rad)。"""
        return float(np.interp(distance, self.distances, self.pitch))

    def speed_at(self, distance):
        """输入：路径弧长 distance (m)；输出：目标切线速度大小 (m/s)。"""
        return float(np.interp(distance, self.distances, self.speed))

    def tangent_at(self, distance):
        """输入：路径弧长 distance (m)；输出：NED 单位切线向量。"""
        direction = np.array([np.interp(distance, self.distances, axis)
                              for axis in self.tangent.T])
        length = np.linalg.norm(direction)
        if length < 1e-12:
            segment = min(np.searchsorted(self.distances, distance),
                          len(self.points) - 1)
            direction = self.points[segment] - self.points[segment - 1]
            length = np.linalg.norm(direction)
        return direction / length

    def advance(self, position):
        """输入：当前位置 [north, east, depth] (m)；输出：无，更新 progress (m)。"""
        # 只在当前进度附近找最近点，交会点的另一支不会被误选。
        start = np.searchsorted(self.distances, max(0, self.progress - 3))
        end = np.searchsorted(self.distances,
                              min(self.length, self.progress + LOOKAHEAD),
                              side="right")
        local = self.points[start:end]
        nearest = start + np.argmin(np.linalg.norm(local - position, axis=1))
        self.progress = max(self.progress, self.distances[nearest])


def rotation_ned(theta, psi):
    """输入：俯仰角 theta、航向角 psi (rad)；输出：体坐标到 NED 的 3×3 旋转矩阵。"""
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    return np.array([
        [cp * ct, -sp, cp * st],
        [sp * ct, cp, sp * st],
        [-st, 0, ct],
    ])


def derivative(state, control, plant_params=None):
    """五自由度动力学与 NED 坐标运动学。

    输入：state 为长度 10 的状态向量，control 为长度 3 的物理控制量；
    plant_params 为可选 PlantConfig，省略时使用标称参数。
    输出：长度 10 的状态时间导数，顺序与 state 相同。
    """
    params = NOMINAL_PLANT if plant_params is None else plant_params
    theta, psi = state[3:5]
    u, v, w, q, r = state[5:]
    tx, delta_e, delta_r = control

    coriolis = params.mass_scale * MASS * np.array(
        [-r * v + q * w, r * u, -q * u, 0, 0])
    damping = params.damping_scale * u * (D_COEFF @ state[5:])
    force = np.array([params.thrust_gain * tx, 0, 0, 0, 0], dtype=float)
    force += u**2 * (params.elevator_gain * ELEVATOR_COEFF * delta_e
                     + params.rudder_gain * RUDDER_COEFF * delta_r)
    nu_dot = np.linalg.solve(params.mass_scale * M_EFF,
                             force - coriolis - damping)

    position_dot = rotation_ned(theta, psi) @ state[5:8]
    return np.r_[position_dot, q, r / np.cos(theta), nu_dot]


def rk4_step(state, control, disturbance=None, plant_params=None):
    """控制和扰动在一个时间步内不变，使用四阶 Runge–Kutta 积分。

    输入：长度 10 的 state、长度 3 的物理 control；disturbance 是可选的
    [u, v, w, q, r] 加速度扰动（长度 5），plant_params 是可选 PlantConfig。
    输出：经过 DT 秒后的长度 10 的状态向量。
    """
    if disturbance is None:
        disturbance = np.zeros(5)

    def dynamics(x):
        """输入：长度 10 的中间状态；输出：叠加扰动后的状态时间导数。"""
        change = derivative(x, control, plant_params)
        change[5:] += disturbance
        return change

    k1 = dynamics(state)
    k2 = dynamics(state + DT * k1 / 2)
    k3 = dynamics(state + DT * k2 / 2)
    k4 = dynamics(state + DT * k3)
    return state + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class ActuatorModel:
    """推力、升降舵和方向舵分别经过指令延迟与一阶执行器响应。"""

    def __init__(self, config=NOMINAL_PLANT, dt=DT):
        """输入：PlantConfig 和执行器积分步长 dt (s)；输出：无，初始化执行器。"""
        if dt <= 0:
            raise ValueError("actuator integration step must be positive")
        self.config = config
        self.dt = dt
        self.thrust_delay_steps = int(np.ceil(config.thrust_delay_s / dt - 1e-12))
        self.fin_delay_steps = int(np.ceil(config.fin_delay_s / dt - 1e-12))
        self.has_dynamics = bool(self.thrust_delay_steps or self.fin_delay_steps
                                 or config.thrust_tau_s or config.fin_tau_s)
        self.reset()

    def reset(self, initial_control=None):
        """输入：可选初始物理控制量（长度 3，默认全零）；输出：实际控制量副本。

        同时重置执行器状态及延迟队列。
        """
        self.actual_control = (np.zeros(3) if initial_control is None else
                               np.asarray(initial_control, dtype=float).copy())
        if self.actual_control.shape != (3,):
            raise ValueError("initial control must have three channels")
        self.thrust_queue = deque([self.actual_control[0]] * self.thrust_delay_steps)
        self.fin_queue = deque([self.actual_control[1:].copy()
                                for _ in range(self.fin_delay_steps)])
        return self.actual_control.copy()

    def step(self, command):
        """输入：长度 3 的有限物理控制指令；输出：延迟及响应后的实际控制量副本。"""
        target = np.asarray(command, dtype=float).copy()
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("command must contain three finite values")
        if self.thrust_delay_steps:
            self.thrust_queue.append(target[0])
            target[0] = self.thrust_queue.popleft()
        if self.fin_delay_steps:
            self.fin_queue.append(target[1:].copy())
            target[1:] = self.fin_queue.popleft()
        for index, tau in enumerate((self.config.thrust_tau_s,
                                     self.config.fin_tau_s,
                                     self.config.fin_tau_s)):
            alpha = 1.0 if tau == 0 else -np.expm1(-self.dt / tau)
            self.actual_control[index] += alpha * (target[index] - self.actual_control[index])
        return self.actual_control.copy()

