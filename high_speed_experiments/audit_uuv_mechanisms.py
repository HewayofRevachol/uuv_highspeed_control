"""Reproduce UUV model and path checks without writing to the source repo.

Examples:
  PYTHONDONTWRITEBYTECODE=1 python3 high_speed_experiments/audit_uuv_mechanisms.py physics
  PYTHONDONTWRITEBYTECODE=1 python3 high_speed_experiments/audit_uuv_mechanisms.py control --speed 6 --horizon 25
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SOURCE = Path('/home/prufrock/uuv_simulator')
ARTIFACTS = Path(__file__).resolve().parents[1] / 'outputs/reports/uuv_mechanism_audit_20261102'
sys.path.insert(0, str(SOURCE))


def physics():
    from simulator import model
    from simulator.batch_mpc import CONTROL_SCALE
    from high_speed_experiments.screen_dynamic_feasibility import steady_planar_trim

    mass = model.M_EFF.copy()
    results = {
        'nominal_mass_kg': model.MASS,
        'effective_mass_matrix': mass.tolist(),
        'mass_matrix_max_asymmetry': float(np.max(np.abs(mass - mass.T))),
        'mass_matrix_relative_asymmetry': float(np.linalg.norm(mass - mass.T) / np.linalg.norm(mass)),
        'max_thrust_n': CONTROL_SCALE[0],
        'max_fin_deg': math.degrees(CONTROL_SCALE[1]),
        'straight_trim': [],
        'turn_trim': [],
        'paths': [],
        'fin_step': [],
        'fin_step_symmetric_mass_sensitivity': [],
    }
    plant = model.MISMATCHED_PLANT
    for speed in (3., 6., 9., 12., 15., 18., 19.2):
        thrust = plant.damping_scale * model.D_COEFF[0, 0] * speed**2 / plant.thrust_gain
        results['straight_trim'].append({
            'speed_mps': speed,
            'thrust_n': float(thrust),
            'max_thrust_fraction': float(thrust / CONTROL_SCALE[0]),
            'mechanical_power_proxy_w': float(thrust * speed),
        })
    results['max_steady_straight_speed_mps'] = math.sqrt(
        CONTROL_SCALE[0] * plant.thrust_gain / (plant.damping_scale * model.D_COEFF[0, 0]))
    for speed, radius in ((6., 48.), (18., 48.), (18., 45.), (18., 52.), (18., 63.4)):
        trim = steady_planar_trim(speed, 1 / radius, plant)
        results['turn_trim'].append({
            'speed_mps': speed, 'radius_m': radius,
            'converged': trim['converged'],
            'thrust_n': trim['thrust_n'],
            'rudder_deg': float(np.rad2deg(trim['rudder_rad'])),
            'surge_mps': trim['surge_mps'], 'sway_mps': trim['sway_mps'],
            'sideslip_deg': float(np.rad2deg(np.arctan2(trim['sway_mps'], trim['surge_mps']))),
        })

    for file in sorted((SOURCE / 'DDAT_trajectories/val').glob('*.csv')):
        path = model.LoopPath(file)
        s = path.distances
        kappa = np.gradient(path.yaw, s)
        pitch = path.pitch
        row = {
            'path': file.name,
            'length_m': float(path.length),
            'depth_min_m': float(path.points[:, 2].min()),
            'depth_max_m': float(path.points[:, 2].max()),
            'max_abs_curvature_inv_m': float(np.abs(kappa).max()),
            'min_horizontal_radius_m': float(1 / np.abs(kappa).max()) if np.abs(kappa).max() else None,
            'max_abs_path_pitch_deg': float(np.rad2deg(np.abs(pitch).max())),
            'max_abs_yaw_change_deg': float(np.rad2deg(np.abs(path.yaw[-1] - path.yaw[0]))),
            'peak_lateral_accel_6_mps2': float(6**2 * np.abs(kappa).max()),
            'peak_lateral_accel_18_mps2': float(18**2 * np.abs(kappa).max()),
            'peak_vertical_speed_18_mps': float(18 * np.sin(np.abs(pitch).max())),
        }
        results['paths'].append(row)

    for speed in (3., 6.):
        trim = plant.damping_scale * model.D_COEFF[0, 0] * speed**2 / plant.thrust_gain
        state = np.array([0., 0., 280., 0., 0., speed, 0., 0., 0., 0.])
        for step in range(round(12.5 / model.DT)):
            state = model.rk4_step(state, np.array([trim, 0., np.deg2rad(1.)]), plant_params=plant)
            if step + 1 in (round(t / model.DT) for t in (2.5, 5., 10., 12.5)):
                results['fin_step'].append({
                    'speed_mps': speed,
                    'time_s': round((step + 1) * model.DT, 2),
                    'east_displacement_m': float(state[1]),
                    'yaw_deg': float(np.rad2deg(state[4])),
                    'sway_mps': float(state[6]),
                    'yaw_rate_degps': float(np.rad2deg(state[9])),
                })
    try:
        model.M_EFF[:] = (mass + mass.T) / 2
        for speed in (3., 6.):
            trim = plant.damping_scale * model.D_COEFF[0, 0] * speed**2 / plant.thrust_gain
            state = np.array([0., 0., 280., 0., 0., speed, 0., 0., 0., 0.])
            for step in range(round(12.5 / model.DT)):
                state = model.rk4_step(state, np.array([trim, 0., np.deg2rad(1.)]),
                                       plant_params=plant)
                if step + 1 in (125, 250, 500, 625):
                    results['fin_step_symmetric_mass_sensitivity'].append({
                        'speed_mps': speed,
                        'time_s': round((step + 1) * model.DT, 2),
                        'east_displacement_m': float(state[1]),
                    })
    finally:
        model.M_EFF[:] = mass
    target = ARTIFACTS / 'summary/physics_path_audit.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print('saved', target)


def control(speed, horizon, backend, randomize, current, noise, ideal_plant,
            path_name, yaw_weight):
    import torch
    from DDAT_train import ExperimentDDATPathEnv
    from simulator.batch_path_env import PathBank
    from validate_ddat import controller_config
    from validate_mpc_speed_sweep import set_constant_speed

    torch.set_num_threads(2)
    checkpoint = torch.load(SOURCE / 'outputs/DDAT/run_20261029_114239/final_actor.pt',
                            map_location='cpu', weights_only=True)
    config = controller_config(checkpoint['config'])
    config['horizon'] = horizon
    config['backend'] = backend
    config['current_max_mps'] = current
    if yaw_weight is not None:
        config['attitude_yaw_weight'] = yaw_weight
    file = SOURCE / 'DDAT_trajectories/val' / f'{path_name}.csv'
    if not file.is_file():
        raise FileNotFoundError(file)
    bank = PathBank([file], 'cpu')
    set_constant_speed(bank, speed)
    env = ExperimentDDATPathEnv(
        bank, batch_size=1, device='cpu', horizon=horizon, backend=backend,
        seed=101027, randomize_plant=randomize,
        admm_iterations=config['admm_iterations'], admm_rho=config['admm_rho'],
        adaptation_time=config['adaptation_time'], current_max_mps=current,
        current_filter_time=config['current_filter_time'],
        attitude_pitch_weight=config['attitude_pitch_weight'],
        attitude_yaw_weight=config['attitude_yaw_weight'],
        fsp_enabled=config['fsp_enabled'], fsp_filter_time=config['fsp_filter_time'],
        fsp_preview_time=config['fsp_preview_time'],
        preview_mode=config['preview_mode'], yaw_reward=config['yaw_reward'],
    )
    env.reset(torch.zeros(1, dtype=torch.long))
    if ideal_plant:
        # Remove *all* modeled plant mismatch and actuator lag.  Re-solve the
        # first command because reset solved with the original plant/FSP state.
        env.parameters[:] = 1.
        env.delay_steps.zero_()
        env.tau.zero_()
        env.fsp = None
        env.mpc.reset(env.state)
        env.base_command, env.last_mpc_ok = env._solve_mpc()
    if not noise:
        env.disturbance_std.zero_()
    data = []
    start = time.perf_counter()
    while True:
        command = env.base_command[0].tolist()
        _, _, done, info = env.step(torch.zeros(1, 3))
        _, lateral, depth, _, _, speed_error, _ = env._errors()
        state = env.state[0]
        data.append({
            't_s': float(env.steps[0] * env.mpc.dt),
            'progress_m': float(env.progress[0]),
            'position_error_m': float(info['position_error_m'][0]),
            'lateral_error_m': float(lateral[0]),
            'depth_error_m': float(depth[0]),
            'speed_error_mps': float(speed_error[0]),
            'body_speed_mps': float(torch.linalg.vector_norm(state[5:8])),
            'pitch_deg': float(torch.rad2deg(state[3])),
            'yaw_deg': float(torch.rad2deg(state[4])),
            'rudder_command_deg': float(np.rad2deg(command[2])),
            'rudder_actual_deg': float(torch.rad2deg(env.actuator[0, 2])),
            'thrust_command_n': float(command[0]),
            'mpc_ok': bool(info['mpc_ok'][0]),
        })
        if bool(done[0]) or len(data) >= 2500:
            break
    elapsed = time.perf_counter() - start
    summary = {
        'source_checkpoint': 'outputs/DDAT/run_20261029_114239/final_actor.pt',
        'path': f'DDAT_trajectories/val/{path_name}.csv',
        'seed': 101027, 'target_speed_mps': speed, 'horizon': horizon,
        'backend': backend, 'randomize_plant': randomize,
        'current_max_mps': current, 'process_noise_enabled': noise,
        'ideal_plant': ideal_plant,
        'yaw_weight': config['attitude_yaw_weight'],
        'steps': len(data), 'wall_s': elapsed,
        'success': bool(info['success'][0]),
        'failed': bool(info['failed'][0]),
        'timeout': bool(info['timeout'][0]),
        'truncated': not bool(done[0]),
        'max_abs_lateral_m': max(abs(r['lateral_error_m']) for r in data),
        'max_abs_depth_m': max(abs(r['depth_error_m']) for r in data),
        'max_abs_pitch_deg': max(abs(r['pitch_deg']) for r in data),
        'speed_rmse_mps': float(np.sqrt(np.mean([r['speed_error_mps']**2 for r in data]))),
        'speed_within_1mps_fraction': float(np.mean([abs(r['speed_error_mps']) <= 1 for r in data])),
        'position_rmse_m': float(np.sqrt(np.mean([r['position_error_m']**2 for r in data]))),
        'fallback_fraction': float(np.mean([not r['mpc_ok'] for r in data])),
    }
    run_dir = ARTIFACTS / 'runs'
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f'control_{path_name}_v{speed:g}_h{horizon}_{backend}_{"ideal" if ideal_plant else "random" if randomize else "fixed"}_{"current" if current else "still"}_{"noise" if noise else "quiet"}_yaw{config["attitude_yaw_weight"]:g}'
    (run_dir / f'{stem}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (run_dir / f'{stem}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('physics', 'control'))
    parser.add_argument('--speed', type=float, default=6.)
    parser.add_argument('--path', default='constant_turn_01')
    parser.add_argument('--horizon', type=int, default=25)
    parser.add_argument('--backend', choices=('osqp', 'admm'), default='osqp')
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--current', type=float, default=0.)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--ideal-plant', action='store_true')
    parser.add_argument('--yaw-weight', type=float)
    args = parser.parse_args()
    if args.mode == 'physics':
        physics()
    else:
        control(args.speed, args.horizon, args.backend, args.randomize,
                args.current, args.noise, args.ideal_plant, args.path,
                args.yaw_weight)
