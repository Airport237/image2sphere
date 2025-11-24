import torch
import numpy as np

# ---- 1. Matrix -> quaternion (w, x, y, z) ----
def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R: (B, 3, 3)
    returns q: (B, 4) as (w, x, y, z), normalized
    """
    B = R.shape[0]
    q = torch.empty((B, 4), device=R.device, dtype=R.dtype)

    # numerically stable variant
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
    t = torch.sqrt(1.0 + R[mask_x, 0, 0] - R[mask_x, 1, 1] - R[mask_x, 2, 2]) * 2.0
    q[mask_x, 0] = (R[mask_x, 2, 1] - R[mask_x, 1, 2]) / t
    q[mask_x, 1] = 0.25 * t
    q[mask_x, 2] = (R[mask_x, 0, 1] + R[mask_x, 1, 0]) / t
    q[mask_x, 3] = (R[mask_x, 0, 2] + R[mask_x, 2, 0]) / t

    # y branch
    mask_y = (~mask_w) & (~mask_x) & (R[:, 1, 1] > R[:, 2, 2])
    t = torch.sqrt(1.0 + R[mask_y, 1, 1] - R[mask_y, 0, 0] - R[mask_y, 2, 2]) * 2.0
    q[mask_y, 0] = (R[mask_y, 0, 2] - R[mask_y, 2, 0]) / t
    q[mask_y, 1] = (R[mask_y, 0, 1] + R[mask_y, 1, 0]) / t
    q[mask_y, 2] = 0.25 * t
    q[mask_y, 3] = (R[mask_y, 1, 2] + R[mask_y, 2, 1]) / t

    # z branch
    mask_z = (~mask_w) & (~mask_x) & (~mask_y)
    t = torch.sqrt(1.0 + R[mask_z, 2, 2] - R[mask_z, 0, 0] - R[mask_z, 1, 1]) * 2.0
    q[mask_z, 0] = (R[mask_z, 1, 0] - R[mask_z, 0, 1]) / t
    q[mask_z, 1] = (R[mask_z, 0, 2] + R[mask_z, 2, 0]) / t
    q[mask_z, 2] = (R[mask_z, 1, 2] + R[mask_z, 2, 1]) / t
    q[mask_z, 3] = 0.25 * t

    # normalize just in case
    q = q / q.norm(dim=1, keepdim=True)
    return q

# ---- 2. Kelvins orientation score (deg) ----
def kelvins_orientation_score(R_est, R_gt, tol_deg=0.169):
    """
    R_est, R_gt: (B, 3, 3)
    Returns per-sample orientation score in degrees, with tolerance applied.
    """
    q_est = rotmat_to_quat(R_est)
    q_gt  = rotmat_to_quat(R_gt)

    # inner product of unit quaternions
    dot = torch.sum(q_est * q_gt, dim=1)
    dot = torch.clamp(dot.abs(), -1.0, 1.0)

    err_rad = 2.0 * torch.acos(dot)
    err_deg = err_rad * (180.0 / np.pi)

    # apply tolerance
    score = err_deg.clone()
    score[err_deg < tol_deg] = 0.0
    return score  # (B,)

# ---- 3. Kelvins position score (normalized) ----
def kelvins_position_score(t_est, t_gt, tol=0.002173):
    """
    t_est, t_gt: (B, 3) in meters
    Returns per-sample position score (dimensionless).
    """
    diff = torch.norm(t_gt - t_est, dim=1)         # ||r_gt - r_est||
    denom = torch.norm(t_gt, dim=1).clamp(min=1e-8)  # ||r_gt||
    err = diff / denom

    score = err.clone()
    score[err < tol] = 0.0
    return score  # (B,)

# ---- 4. Combined pose score ----
def kelvins_pose_score(R_est, R_gt, t_est, t_gt):
    s_orient = kelvins_orientation_score(R_est, R_gt)   # deg
    s_pos    = kelvins_position_score(t_est, t_gt)      # normalized
    return s_orient + s_pos, s_orient, s_pos
