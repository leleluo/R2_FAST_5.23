#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
import numpy as np
import torch

sys.path.append("./")
from r2_gaussian.utils.loss_utils import ssim


def mse(img1, img2, mask=None):
    """MSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)

        mse = torch.stack(
            [
                (((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True)
                for i in range(img1.shape[0])
            ],
            dim=0,
        )

    else:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse


def rmse(img1, img2, mask=None):
    """RMSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    rmse = mse_out**0.5
    return rmse


@torch.no_grad()
def psnr(img1, img2, mask=None, pixel_max=1.0):
    """PSNR

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
    if mask is not None:
        if torch.isinf(psnr_out).any():
            print(mse_out.mean(), psnr_out.mean())
            psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
            psnr_out = psnr_out[~torch.isinf(psnr_out)]

    return psnr_out


@torch.no_grad()
def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT."""
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
        return psnr_out.item(), None
    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError
                if slice1.max() > 0:
                    result = ssim(slice1[None, None], slice2[None, None])
                    count += 1
                else:
                    result = 0
                results.append(result)
            results = torch.tensor(results)
            mean_results = torch.sum(results) / count
            ssims.append(mean_results.item())
        return float(np.mean(ssims)), ssims


# ---------------------------------------------------------------------------
# LPIPS (perceptual) metric – lazy-loaded, graceful degradation if not installed
# ---------------------------------------------------------------------------
_LPIPS_MODEL = None
_LPIPS_DEVICE = None
_LPIPS_AVAILABLE = None


def _get_lpips_model(device=None):
    """Return a cached LPIPS(net='vgg') model, or None if package missing."""
    global _LPIPS_MODEL, _LPIPS_DEVICE, _LPIPS_AVAILABLE
    if _LPIPS_AVAILABLE is False:
        return None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _LPIPS_MODEL is None or _LPIPS_DEVICE != device:
        try:
            import lpips as _lpips_pkg  # noqa: F401
            _LPIPS_MODEL = _lpips_pkg.LPIPS(net="vgg", verbose=False).to(device)
            _LPIPS_MODEL.eval()
            _LPIPS_DEVICE = device
            _LPIPS_AVAILABLE = True
        except Exception as e:
            print(f"[LPIPS] disabled: {e}. Install with `pip install lpips`.")
            _LPIPS_AVAILABLE = False
            return None
    return _LPIPS_MODEL


def _to_lpips_input(slice_2d, device):
    """[H, W] grayscale in [0, 1] -> [1, 3, H, W] in [-1, 1]."""
    if slice_2d.ndim == 2:
        slice_2d = slice_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    if slice_2d.shape[1] == 1:
        slice_2d = slice_2d.repeat(1, 3, 1, 1)         # [1, 3, H, W]
    return (slice_2d.to(device).float() * 2.0 - 1.0)


@torch.no_grad()
def lpips_metric_proj(img1, img2, axis=2):
    """LPIPS per-slice averaged. Same axis/normalization as metric_proj.

    Args:
        img1 (Tensor or ndarray): [x, y, z] (GT)
        img2 (Tensor or ndarray): [x, y, z] (pred)
        axis (int): slicing axis (0/1/2). Default 2 – matches metric_proj.

    Returns:
        (mean_lpips, per_slice_list)  or  (None, None) if LPIPS not installed.
    """
    model = _get_lpips_model()
    if model is None:
        return None, None

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    device = next(model.parameters()).device

    n_slice = img1.shape[axis]
    results = []
    count = 0
    for i in range(n_slice):
        if axis == 0:
            slice1, slice2 = img1[i, :, :], img2[i, :, :]
        elif axis == 1:
            slice1, slice2 = img1[:, i, :], img2[:, i, :]
        else:
            slice1, slice2 = img1[:, :, i], img2[:, :, i]

        if slice1.max() > 0:
            slice1 = slice1 / slice1.max()
            slice2 = slice2 / slice2.max()
            d = model(
                _to_lpips_input(slice1, device),
                _to_lpips_input(slice2, device),
            ).item()
            results.append(d)
            count += 1
        else:
            results.append(0.0)

    mean_val = sum(results) / max(count, 1)
    return mean_val, results


@torch.no_grad()
def lpips_metric_vol(img1, img2):
    """LPIPS over a 3D volume, averaged across all 3 axes (slice-wise mean)."""
    model = _get_lpips_model()
    if model is None:
        return None, None

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())
    device = next(model.parameters()).device

    axis_means = []
    for axis in [0, 1, 2]:
        n_slice = img1.shape[axis]
        results = []
        count = 0
        for i in range(n_slice):
            if axis == 0:
                slice1, slice2 = img1[i, :, :], img2[i, :, :]
            elif axis == 1:
                slice1, slice2 = img1[:, i, :], img2[:, i, :]
            else:
                slice1, slice2 = img1[:, :, i], img2[:, :, i]
            if slice1.max() > 0:
                slice1 = slice1 / slice1.max()
                slice2 = slice2 / slice2.max()
                d = model(
                    _to_lpips_input(slice1, device),
                    _to_lpips_input(slice2, device),
                ).item()
                results.append(d)
                count += 1
            else:
                results.append(0.0)
        axis_means.append(sum(results) / max(count, 1))

    return float(np.mean(axis_means)), axis_means


@torch.no_grad()
def metric_vol_per_slice(img1, img2, metric="psnr", axis=2, pixel_max=1.0):
    """Per-slice PSNR/SSIM along the given axis.

    Args:
        img1, img2: [x, y, z] volumes (GT, pred)
        metric: "psnr" or "ssim"
        axis: 0/1/2 — default 2 (z-axis) matches reconstruction/ slice order
    Returns:
        (mean_over_non_empty_slices, per_slice_list)
    """
    assert metric in ["psnr", "ssim"]
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())

    n_slice = img1.shape[axis]
    results = []
    valid = []
    for i in range(n_slice):
        if axis == 0:
            slice1, slice2 = img1[i, :, :], img2[i, :, :]
        elif axis == 1:
            slice1, slice2 = img1[:, i, :], img2[:, i, :]
        else:
            slice1, slice2 = img1[:, :, i], img2[:, :, i]

        if slice1.max() > 0:
            if metric == "psnr":
                mse_v = torch.mean((slice1 - slice2) ** 2)
                if mse_v.item() > 0:
                    val = float(
                        (10 * torch.log10(pixel_max**2 / mse_v.float())).item()
                    )
                else:
                    val = float("inf")
            else:  # ssim
                val = float(ssim(slice1[None, None], slice2[None, None]).item())
            results.append(val)
            if not np.isinf(val):
                valid.append(val)
        else:
            results.append(0.0)

    mean = sum(valid) / max(len(valid), 1) if valid else 0.0
    return mean, results


@torch.no_grad()
def lpips_metric_vol_per_slice(img1, img2, axis=2):
    """Per-slice LPIPS along the given axis.

    Args:
        img1, img2: [x, y, z] volumes (GT, pred)
        axis: 0/1/2 — default 2 (z-axis) matches reconstruction/ slice order
    Returns:
        (mean_over_non_empty_slices, per_slice_list) or (None, None) if LPIPS missing
    """
    model = _get_lpips_model()
    if model is None:
        return None, None

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())
    device = next(model.parameters()).device

    n_slice = img1.shape[axis]
    results = []
    valid = []
    for i in range(n_slice):
        if axis == 0:
            slice1, slice2 = img1[i, :, :], img2[i, :, :]
        elif axis == 1:
            slice1, slice2 = img1[:, i, :], img2[:, i, :]
        else:
            slice1, slice2 = img1[:, :, i], img2[:, :, i]

        if slice1.max() > 0:
            slice1 = slice1 / slice1.max()
            slice2 = slice2 / slice2.max()
            d = float(
                model(
                    _to_lpips_input(slice1, device),
                    _to_lpips_input(slice2, device),
                ).item()
            )
            results.append(d)
            valid.append(d)
        else:
            results.append(0.0)

    mean = sum(valid) / max(len(valid), 1) if valid else 0.0
    return mean, results


@torch.no_grad()
def metric_proj(img1, img2, metric="psnr", axis=2, pixel_max=1.0):
    """Metrics for projection

    Args:
        img1 (_type_): [x, y, z]
        img2 (_type_): [x, y, z]
        pixel_max (float, optional): _description_. Defaults to 1.0.
    """
    assert axis in [0, 1, 2, None]
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    n_slice = img1.shape[axis]

    results = []
    count = 0
    for i in range(n_slice):
        if axis == 0:
            slice1 = img1[i, :, :]
            slice2 = img2[i, :, :]
        elif axis == 1:
            slice1 = img1[:, i, :]
            slice2 = img2[:, i, :]
        elif axis == 2:
            slice1 = img1[:, :, i]
            slice2 = img2[:, :, i]
        else:
            raise NotImplementedError
        if slice1.max() > 0:
            slice1 = slice1 / slice1.max()
            slice2 = slice2 / slice2.max()
            if metric == "psnr":
                result = psnr(
                    slice1[None, None], slice2[None, None], pixel_max=pixel_max
                )
            elif metric == "ssim":
                result = ssim(slice1[None, None], slice2[None, None])
            else:
                raise NotImplementedError
            count += 1
        else:
            result = 0
        results.append(result)
    results = torch.tensor(results)
    mean_results = torch.sum(results) / count
    return mean_results.item(), results.tolist()
