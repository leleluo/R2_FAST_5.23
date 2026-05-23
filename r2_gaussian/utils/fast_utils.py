import random
import torch
from r2_gaussian.gaussian.render_query import render
from r2_gaussian.utils.loss_utils import l1_loss, ssim


# ----------------------------
# 1. 随机采样多视角
# ----------------------------
def sampling_cameras(viewpoint_stack, num_cams=10):
    """从训练相机中随机采样若干 view（不会改原列表）"""
    vs = list(viewpoint_stack)
    num_cams = min(num_cams, len(vs))
    return random.sample(vs, num_cams)


# ----------------------------
# 2. photometric loss for X-ray (L1 + optional SSIM)
# ----------------------------
def compute_photometric_loss(viewpoint_cam, rendered, lambda_dssim=0.0):
    """保持与主训练 loss 一致的 photometric loss"""
    gt_image = viewpoint_cam.original_image.to(rendered.device)

    loss_l1 = l1_loss(rendered, gt_image)

    if lambda_dssim > 0:
        loss_dssim = 1 - ssim(rendered, gt_image)
        return (1 - lambda_dssim) * loss_l1 + lambda_dssim * loss_dssim

    return loss_l1


# ----------------------------
# 3. 多视图给每个高斯打分（近似版）
# ----------------------------
def compute_gaussian_score_r2gs(
    camlist, gaussians, pipe, lambda_dssim=0.0,
    score_mode="screen_grad",
):
    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]

    imp = torch.zeros(N, device=device)
    counts = torch.zeros(N, device=device)

    eps = 1e-6

    with torch.set_grad_enabled(True):
        for cam in camlist:
            # if gaussians._density.grad is not None:
            #     gaussians._density.grad.zero_()
            # if hasattr(gaussians, "optimizer") and gaussians.optimizer is not None:
            #     gaussians.optimizer.zero_grad(set_to_none=True)
            pkg = render(cam, gaussians, pipe)
            pred = pkg["render"]
            gt = cam.original_image.to(device)

            diff = (pred - gt).abs()
            loss = diff.sum()
            if lambda_dssim > 0:
                loss = (1 - lambda_dssim) * loss + lambda_dssim * (1 - ssim(pred, gt))

            vis = pkg["visibility_filter"].detach()
            counts += vis.float()

            if score_mode == "density_grad":
                g = torch.autograd.grad(
                    loss, gaussians._density,
                    retain_graph=False, create_graph=False,
                    allow_unused=True
                )[0]
                if g is None:
                    continue
                score = g.detach().abs().squeeze(-1)
            else:
                g2d = torch.autograd.grad(
                    loss, pkg["viewspace_points"],
                    retain_graph=False, create_graph=False,
                    allow_unused=True
                )[0]
                if g2d is None:
                    continue
                score = torch.norm(g2d.detach()[:, :2], dim=-1)

            imp += score * vis.float()

    if imp.max() > 0:
        imp_norm = (imp - imp.min()) / (imp.max() - imp.min() + eps)
    else:
        imp_norm = imp
    if counts.max() < 1:
        return imp_norm, torch.zeros_like(imp_norm)
    support = counts / (counts.max() + eps)
    pruning_score = (1.0 - support) * (1.0 - imp_norm)

    return imp_norm, pruning_score
