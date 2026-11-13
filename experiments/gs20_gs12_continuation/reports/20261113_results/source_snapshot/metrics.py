"""Physical output RMSE and a dimensionless three-output combined RMSE."""
import numpy as np

OUTPUT_NAMES = ('N_m', 'E_m', 'D_m', 'pitch_rad', 'yaw_rad', 'VN_mps', 'VE_mps', 'VD_mps')
ERROR_SCALES = np.array([8., 8., 5., np.deg2rad(10), np.deg2rad(10), 2., 1., 1.])
TRACK_SCALES = np.array([8., 5., 2.])


def error(reference, output):
    result = reference-output
    result[..., 3:5] = np.arctan2(np.sin(result[..., 3:5]), np.cos(result[..., 3:5]))
    return result


def score(arrays, mask):
    e = arrays['error'][mask]
    if not len(e):
        return None
    rmse = np.sqrt(np.mean(e*e, axis=0))
    r2 = np.mean(np.sum((e[:, [1, 2, 5]]/TRACK_SCALES)**2, axis=1))
    return dict(rmse=dict(zip(OUTPUT_NAMES, rmse.tolist())),
                position_vector_rmse_m=float(np.linalg.norm(rmse[:3])),
                R2=float(r2), R=float(np.sqrt(r2)))


def windows(task, t):
    result = {'full': t > 0., 'steady': t > (80. if task['kind'] == 'bridge' else 16.)}
    if task['kind'] == 'switch':
        result.update(stage_1=(t >= 16)&(t < 46), stage_2=(t > 50)&(t < 94),
                      stage_3=t > 98, transition=((t >= 46)&(t <= 50))|((t >= 94)&(t <= 98)))
    return result
