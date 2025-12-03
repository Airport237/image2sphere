import torch
import numpy as np


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R: (B, 3, 3)
    returns q: (B, 4) as (w, x, y, z), normalized
    """
    B = R.shape[0]
    q = torch.empty((B, 4), device=R.device, dtype=R.dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # w branch
    mask_w = trace > 0
    t = torch.sqrt(1.0 + trace[mask_w]) * 2.0  # 4*w
    q[mask_w, 0] = 0.25 * t
    q[mask_w, 1] = (R[mask_w, 2, 1] - R[mask_w, 1, 2]) / t
    q[mask_w, 2] = (R[mask_w, 0, 2] - R[mask_w, 2, 0]) / t
    q[mask_w, 3] = (R[mask_w, 1, 0] - R[mask_w, 0, 1]) / t

    # x branch
    mask_x = (~mask_w) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    t = torch.sqrt(1.0 + R[mask_x, 0, 0] -
                   R[mask_x, 1, 1] - R[mask_x, 2, 2]) * 2.0
    q[mask_x, 0] = (R[mask_x, 2, 1] - R[mask_x, 1, 2]) / t
    q[mask_x, 1] = 0.25 * t
    q[mask_x, 2] = (R[mask_x, 0, 1] + R[mask_x, 1, 0]) / t
    q[mask_x, 3] = (R[mask_x, 0, 2] + R[mask_x, 2, 0]) / t

    # y branch
    mask_y = (~mask_w) & (~mask_x) & (R[:, 1, 1] > R[:, 2, 2])
    t = torch.sqrt(1.0 + R[mask_y, 1, 1] -
                   R[mask_y, 0, 0] - R[mask_y, 2, 2]) * 2.0
    q[mask_y, 0] = (R[mask_y, 0, 2] - R[mask_y, 2, 0]) / t
    q[mask_y, 1] = (R[mask_y, 0, 1] + R[mask_y, 1, 0]) / t
    q[mask_y, 2] = 0.25 * t
    q[mask_y, 3] = (R[mask_y, 1, 2] + R[mask_y, 2, 1]) / t

    # z branch
    mask_z = (~mask_w) & (~mask_x) & (~mask_y)
    t = torch.sqrt(1.0 + R[mask_z, 2, 2] -
                   R[mask_z, 0, 0] - R[mask_z, 1, 1]) * 2.0
    q[mask_z, 0] = (R[mask_z, 1, 0] - R[mask_z, 0, 1]) / t
    q[mask_z, 1] = (R[mask_z, 0, 2] + R[mask_z, 2, 0]) / t
    q[mask_z, 2] = (R[mask_z, 1, 2] + R[mask_z, 2, 1]) / t
    q[mask_z, 3] = 0.25 * t

    # normalize just in case
    q = q / q.norm(dim=1, keepdim=True)
    return q


def kelvins_orientation_score(R_est, R_gt, tol_deg=0.169):
    """
    R_est, R_gt: (B, 3, 3)
    Returns per-sample orientation score in degrees, with tolerance applied.
    """
    # q_est = rotmat_to_quat(R_est)
    # q_gt  = rotmat_to_quat(R_gt)

    # # inner product of unit quaternions
    # dot = torch.sum(q_est * q_gt, dim=1)
    # dot = torch.clamp(dot.abs(), -1.0, 1.0)

    # err_rad = 2.0 * torch.acos(dot)
    # err_deg = err_rad * (180.0 / np.pi)

    # # apply tolerance
    # score = err_deg.clone()
    # score[err_deg < tol_deg] = 0.0
    # return score  # (B,)
    R_est = R_est.cpu()
    R_gt = R_gt.cpu()

    # print(R_est.shape, R_gt.shape, R_est.ndim)
    assert R_est.shape == R_gt.shape and R_est.ndim == 2
    assert np.abs(np.linalg.det(R_est) - 1) < 1e-6, \
        f'Determinant of R_pr is {np.linalg.det(R_est)}'

    Rdot = np.dot(R_est, np.transpose(R_gt))
    trace = (np.trace(Rdot) - 1.0)/2.0
    trace = np.clip(trace, -1.0, 1.0)
    return np.rad2deg(np.arccos(trace))


def kelvins_position_score(t_est, t_gt, tol=0.002173):
    """
    t_est, t_gt: (B, 3) in meters
    Returns per-sample position score (dimensionless).
    """
    # diff = torch.norm(t_gt - t_est, dim=1)         # ||r_gt - r_est||
    # denom = torch.norm(t_gt, dim=1).clamp(min=1e-8)  # ||r_gt||
    # err = diff / denom

    # score = err.clone()
    # score[err < tol] = 0.0
    # return score  # (B,)
    t_gt = t_gt.detach().cpu().numpy().reshape(3,)
    t_est = t_est.detach().cpu().numpy().reshape(3,)

    # print(f"GT T: {t_gt} | P T: {t_est}")
    return np.sqrt(np.sum(np.square(t_gt - t_est)))


# def kelvins_pose_score(R_est, R_gt, t_est, t_gt):
#     s_orient = kelvins_orientation_score(R_est, R_gt)   # deg
#     s_pos    = kelvins_position_score(t_est, t_gt)      # normalized
#     return s_orient + s_pos, s_orient, s_pos

def kelvins_pose_score(t_pr, ori_pr, t_gt, ori_gt, representation='quaternion',
                       applyThreshold=True, theta_q=0.5, theta_t=0.005):
    # theta_q: rotation threshold [deg]
    # theta_t: normalized translation threshold [m/m]
    # print(f"GT T: {t_gt} | P T: {t_pr}")

    err_q = kelvins_orientation_score(ori_pr, ori_gt)
    err_t = kelvins_position_score(t_pr, t_gt)  # [deg]
    # print(f"Err q: {err_q}|err_t: {err_t}")
    t_gt = t_gt.detach().cpu().numpy().reshape(3,)
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    speed_q = np.deg2rad(err_q)

    # Check if within threshold
    if applyThreshold and err_q < theta_q:
        speed_q = 0.0

    if applyThreshold and speed_t < theta_t:
        speed_t = 0.0

    speed = speed_t + speed_q

    return speed_t, speed_q, speed


def compute_translation_stats(loader):
    """Compute per-dim mean/std of translation vectors from the training loader."""
    all_t = []

    for batch in loader:
        if 'trans' not in batch:
            continue
        t = batch['trans']  # (B, 3)
        all_t.append(t.view(-1, 3).cpu())

    if not all_t:
        mean = torch.zeros(3)
        std = torch.ones(3)
    else:
        all_t = torch.cat(all_t, dim=0)  # (N, 3)
        mean = all_t.mean(dim=0)
        std = all_t.std(dim=0)
        # avoid division by zero
        std[std == 0] = 1.0

    return mean, std
