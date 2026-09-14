import torch


def normalize_poses(poses, eps=1e-8):
    poses = torch.as_tensor(poses, dtype=torch.float32)
    if poses.ndim >= 2 and poses.shape[-2:] == (33, 3):
        joints = poses
    elif poses.ndim >= 2 and poses.shape[-1] == 99:
        joints = poses.reshape(*poses.shape[:-1], 33, 3)
    else:
        raise ValueError("Expected pose sequences ending in [33, 3] or [99].")
    centered = joints - joints.mean(dim=-2, keepdim=True)
    scale = centered.norm(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    return (centered / scale.clamp_min(eps)).flatten(-2)
