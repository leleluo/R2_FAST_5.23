import random
import torch
from tqdm import tqdm
from r2_gaussian.gaussian.render_query import render
from r2_gaussian.utils.loss_utils import l1_loss, ssim


# ----------------------------
# 1. 随机采样多视角
# ----------------------------
def sampling_cameras(viewpoint_stack, num_cams=10):
    """从训练相机列表中随机采样若干视角（不修改原列表）"""
    vs = list(viewpoint_stack)
    num_cams = min(num_cams, len(vs))
    return random.sample(vs, num_cams)


# ----------------------------
# 2. 单视角 photometric loss（L1 + 可选 SSIM）
# ----------------------------
def compute_photometric_loss(viewpoint_cam, rendered, lambda_dssim=0.0):
    """与主训练 loss 保持一致的 photometric loss（标量）"""
    gt_image = viewpoint_cam.original_image.to(rendered.device)
    loss_l1 = l1_loss(rendered, gt_image)
    if lambda_dssim > 0:
        loss_dssim = 1.0 - ssim(rendered, gt_image)
        return (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * loss_dssim
    return loss_l1


# ----------------------------
# 3. 多视角一致性评分（VCD + VCP 核心）
#
# 流程：
#   ① 对每个视角普通渲染，得到预测图
#   ② 逐像素计算 L1 误差，用百分位阈值生成 pixel_error_map（0/1）
#   ③ 携带 pixel_error_map 再次渲染，CUDA kernel 内部统计：
#      - gaussian_cnt  : 每个高斯命中高误差像素的次数  → importance_score（VCD）
#   ④ 用整视角 photo_loss 加权累加，得到 pruning_score（VCP）
# ----------------------------
def compute_gaussian_score_r2gs(
    camlist,
    gaussians,
    pipe,
    lambda_dssim=0.0,
    DENSIFY=True,
    quantile=0.70,          # 取误差前 (1-quantile)*100% 的像素为高误差区域
):
    """
    参数：
        camlist      : 采样好的相机列表
        gaussians    : 当前高斯模型
        pipe         : 渲染管线参数
        lambda_dssim : SSIM 权重（与主训练一致即可，默认 0）
        DENSIFY      : True → 同时计算 importance_score；False → 只算 pruning_score
        quantile     : 百分位阈值，0.70 表示误差前 30% 的像素被标记为高误差

    返回：
        importance_score : [N] float tensor，VCD 致密化判据（DENSIFY=False 时为 None）
        pruning_score    : [N] float tensor，VCP 剪枝判据，归一化到 [0,1]
    """
    device = gaussians.get_xyz.device
    n_pts  = gaussians.get_xyz.shape[0]
    eps    = 1e-6

    # 跨视角累积量
    full_metric_counts = torch.zeros(n_pts, device=device)   # 命中高误差像素次数之和
    full_metric_score  = torch.zeros(n_pts, device=device)   # photo_loss 加权累积

    for cam in camlist:
        # ---- ① 第一次渲染：得到预测图 ----
        with torch.no_grad():
            pkg1 = render(cam, gaussians, pipe)
        rendered_image = pkg1["render"].detach()              # [C, H, W]

        # ---- ② 构造逐像素误差图 & 百分位阈值 ----
        gt_image = cam.original_image.to(device)             # [C, H, W]
        l1_map   = torch.abs(rendered_image - gt_image).mean(dim=0)   # [H, W]

        threshold       = torch.quantile(l1_map, quantile)
        pixel_error_map = (l1_map > threshold).float()       # [H, W]，0/1

        # 整视角 photo_loss（标量），用于 pruning_score 加权
        photo_loss = compute_photometric_loss(
            cam, rendered_image, lambda_dssim
        ).detach().clamp_min(0.0)

        # ---- ③ 第二次渲染：携带 pixel_error_map，CUDA 统计每个高斯的命中 ----
        with torch.no_grad():
            pkg2 = render(cam, gaussians, pipe,
                          pixel_error_map=pixel_error_map)

        # gaussian_cnt：该视角下每个高斯命中高误差像素的次数，形状 [N]
        gaussian_cnt = pkg2["gaussian_cnt"].detach()

        # ---- ④ 跨视角累加 ----
        if DENSIFY:
            full_metric_counts += gaussian_cnt

        # pruning_score：误差越大的视角权重越高
        full_metric_score += photo_loss * gaussian_cnt

    # ---- ⑤ 换算为最终分数 ----

    # importance_score：平均每个视角命中高误差像素次数（向下取整，对齐 FastGS）
    if DENSIFY:
        importance_score = torch.div(
            full_metric_counts, len(camlist), rounding_mode='floor'
        )
    else:
        importance_score = None

    # pruning_score：归一化到 [0, 1]，越大表示该高斯在多视角下持续贡献高误差
    if full_metric_score.max() > eps:
        pruning_score = (
            (full_metric_score - full_metric_score.min()) /
            (full_metric_score.max() - full_metric_score.min() + eps)
        )
    else:
        pruning_score = torch.zeros_like(full_metric_score)

    return importance_score, pruning_score


# ----------------------------
# 4. 基于 3D 体素重建误差的多视角一致性评分（论文 3.2 节 Voxelizer-based VCD）
#
# 流程：
#   ① 用 query() 得到当前重建体 vol_pred [nx, ny, nz]
#   ② vol_err = |vol_pred - vol_gt|，得到 3D 误差体
#   ③ 对每个高斯位置 xyz 做三线性插值，取 vol_err 的局部值作为 importance_score
#
# 物理意义：
#   - vol_err 是局部归因（不像 2D 像素归因会被穿透积分均匀涂抹）
#   - 高斯是带宽 3σ 的局部 RBF，xyz 处的 vol_err 直接反映其负责的局部重建质量
#   - 因此 importance_score 分布应当呈现"双峰"（误差区高、正确区低），
#     而不是像 2D 像素版那样塌缩成"全员高分"
# ----------------------------
def compute_gaussian_score_voxel(
    gaussians,
    queryfunc,
    vol_gt,
    scanner_cfg,
    eps=1e-12,
):
    """
    体素版 VCD 打分。

    参数：
        gaussians   : 当前高斯模型
        queryfunc   : lambda x: query(x, offOrigin, nVoxel, sVoxel, pipe)
        vol_gt      : [nx, ny, nz] tensor，真值体
        scanner_cfg : dict，包含 'offOrigin', 'sVoxel', 'nVoxel'
        eps         : 数值保护

    返回：
        importance_score : [N] float tensor，体素误差驱动的致密化判据
        pruning_score    : None
    """
    device = gaussians.get_xyz.device

    # ---- ① 当前重建（注意：query 内部分配 ~1 GiB binning buffer，用 no_grad 释放） ----
    with torch.no_grad():
        vol_pred = queryfunc(gaussians)["vol"]           # [nx, ny, nz]
        if vol_gt.device != device:
            vol_gt_dev = vol_gt.to(device)
        else:
            vol_gt_dev = vol_gt
        vol_err = torch.abs(vol_pred - vol_gt_dev)        # [nx, ny, nz]
        del vol_pred  # 立即释放

    # ---- ② 世界坐标 → 连续体素索引 ----
    center = torch.as_tensor(scanner_cfg["offOrigin"], device=device, dtype=torch.float32)
    sVoxel = torch.as_tensor(scanner_cfg["sVoxel"],    device=device, dtype=torch.float32)
    nVoxel = torch.as_tensor(scanner_cfg["nVoxel"],    device=device, dtype=torch.float32)
    voxel_size = sVoxel / nVoxel                          # [3]
    origin_corner = center - sVoxel / 2.0                 # 体素体角点

    xyz = gaussians.get_xyz.detach()                      # [N, 3]
    cidx = (xyz - origin_corner) / voxel_size             # [N, 3] 连续索引

    nx, ny, nz = vol_err.shape
    max_idx = torch.tensor([nx - 1, ny - 1, nz - 1], device=device, dtype=torch.float32)

    # 体素体外的高斯：标记为 0（不参与致密化）
    out_of_volume = (
        (cidx[:, 0] < 0) | (cidx[:, 0] > max_idx[0]) |
        (cidx[:, 1] < 0) | (cidx[:, 1] > max_idx[1]) |
        (cidx[:, 2] < 0) | (cidx[:, 2] > max_idx[2])
    )
    cidx_clamped = torch.clamp(cidx, torch.zeros_like(max_idx), max_idx)

    idx_lo = cidx_clamped.floor().long()                  # [N, 3]
    idx_hi = (idx_lo + 1).clamp(max=max_idx.long())       # [N, 3]
    frac = cidx_clamped - idx_lo.float()                  # [N, 3]

    i0, j0, k0 = idx_lo[:, 0], idx_lo[:, 1], idx_lo[:, 2]
    i1, j1, k1 = idx_hi[:, 0], idx_hi[:, 1], idx_hi[:, 2]
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

    # ---- ③ 8 角三线性插值 ----
    v000 = vol_err[i0, j0, k0]
    v100 = vol_err[i1, j0, k0]
    v010 = vol_err[i0, j1, k0]
    v110 = vol_err[i1, j1, k0]
    v001 = vol_err[i0, j0, k1]
    v101 = vol_err[i1, j0, k1]
    v011 = vol_err[i0, j1, k1]
    v111 = vol_err[i1, j1, k1]

    importance_score = (
        (1 - fx) * (1 - fy) * (1 - fz) * v000 +
             fx  * (1 - fy) * (1 - fz) * v100 +
        (1 - fx) *      fy  * (1 - fz) * v010 +
             fx  *      fy  * (1 - fz) * v110 +
        (1 - fx) * (1 - fy) *      fz  * v001 +
             fx  * (1 - fy) *      fz  * v101 +
        (1 - fx) *      fy  *      fz  * v011 +
             fx  *      fy  *      fz  * v111
    )

    # 体素体外的高斯打 0
    importance_score = importance_score.masked_fill(out_of_volume, 0.0)

    return importance_score, None


# ----------------------------
# 5. 分布诊断：验证 score 是否真的有区分度
# ----------------------------
def log_score_distribution(iteration, score, thresh_val, name="VCD-VOXEL"):
    """
    打印 score 的分布统计，用于验证 VCD/VCP 评分是否有"分辨力"。

    判断标准：
        - 分布越接近"双峰 / 长尾"，分辨力越好（局部归因有效）
        - 分布越接近"全员高分 / 单峰"，分辨力越差（穿透涂抹失效）
        - 健康指标：median 远小于 max；>p70 通过率约 30%（说明阈值真的在筛）
    """
    s = score.detach().float()
    n = s.numel()
    if n == 0:
        return

    s_sorted = torch.sort(s).values
    p50 = s_sorted[int(0.50 * (n - 1))].item()
    p70 = s_sorted[int(0.70 * (n - 1))].item()
    p90 = s_sorted[int(0.90 * (n - 1))].item()
    p99 = s_sorted[int(0.99 * (n - 1))].item()
    s_min = s_sorted[0].item()
    s_max = s_sorted[-1].item()
    s_mean = s.mean().item()
    s_std = s.std().item()
    pct_zero = (s <= 1e-12).float().mean().item()
    pct_above_thresh = (s > thresh_val).float().mean().item()
    pct_above_mean = (s > s_mean).float().mean().item()

    tqdm.write(
        f"[{name}][ITER {iteration}] N={n}  "
        f"mean={s_mean:.4g}  median={p50:.4g}  std={s_std:.4g}\n"
        f"           p70={p70:.4g}  p90={p90:.4g}  p99={p99:.4g}  "
        f"min={s_min:.4g}  max={s_max:.4g}\n"
        f"           zero%={pct_zero:.1%}  >mean%={pct_above_mean:.1%}  "
        f"thresh={thresh_val:.4g}  >thresh%={pct_above_thresh:.1%}"
    )
