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

import os
import os.path as osp
import sys
import csv
import torch
from tqdm import tqdm, trange
import torchvision
from time import time
import numpy as np
import concurrent.futures
import yaml
from argparse import ArgumentParser
from random import randint
import SimpleITK as sitk

sys.path.append("./")
from r2_gaussian.arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
)
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state, t2a
from r2_gaussian.utils.image_utils import (
    metric_vol,
    metric_proj,
    lpips_metric_proj,
    lpips_metric_vol,
    metric_vol_per_slice,
    lpips_metric_vol_per_slice,
)


def testing(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int,
    skip_render_train: bool,
    skip_render_test: bool,
    skip_recon: bool,
):
    # Set up dataset
    scene = Scene(
        dataset,
        shuffle=False,
    )

    # Set up Gaussians
    gaussians = GaussianModel(None)  # scale_bound will be loaded later
    loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
    scene.gaussians = gaussians

    save_path = osp.join(
        dataset.model_path,
        "test",
        "iter_{}".format(loaded_iter),
    )

    # Evaluate projection train
    if not skip_render_train:
        evaluate_render(
            save_path,
            "render_train",
            scene.getTrainCameras(),
            gaussians,
            pipeline,
        )
    # Evaluate projection test
    if not skip_render_test:
        evaluate_render(
            save_path,
            "render_test",
            scene.getTestCameras(),
            gaussians,
            pipeline,
        )
    # Evaluate volume reconstruction
    if not skip_recon:
        evaluate_volume(
            save_path,
            "reconstruction",
            scene.scanner_cfg,
            gaussians,
            pipeline,
            scene.vol_gt,
        )


def evaluate_volume(
    save_path,
    name,
    scanner_cfg,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    vol_gt,
):
    """Evaluate volume reconstruction."""
    slice_save_path = osp.join(save_path, name)
    os.makedirs(slice_save_path, exist_ok=True)

    query_pkg = query(
        gaussians,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipeline,
    )
    vol_pred = query_pkg["vol"]

    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
    lpips_3d, lpips_3d_axis = lpips_metric_vol(vol_gt, vol_pred)
    # Per-slice metrics along z-axis (axis=2) so slice_id matches reconstruction/*.png
    _, psnr_3d_per_slice = metric_vol_per_slice(vol_gt, vol_pred, "psnr", axis=2)
    _, ssim_3d_per_slice = metric_vol_per_slice(vol_gt, vol_pred, "ssim", axis=2)
    _, lpips_3d_per_slice = lpips_metric_vol_per_slice(vol_gt, vol_pred, axis=2)
    n_points = int(gaussians.get_xyz.shape[0])

    multithread_write(
        [vol_gt[..., i][None] for i in range(vol_gt.shape[2])],
        slice_save_path,
        "_gt",
    )
    multithread_write(
        [vol_pred[..., i][None] for i in range(vol_pred.shape[2])],
        slice_save_path,
        "_pred",
    )
    eval_dict = {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
        "ssim_3d_x": ssim_3d_axis[0],
        "ssim_3d_y": ssim_3d_axis[1],
        "ssim_3d_z": ssim_3d_axis[2],
        "lpips_3d": lpips_3d,
        "lpips_3d_x": lpips_3d_axis[0] if lpips_3d_axis is not None else None,
        "lpips_3d_y": lpips_3d_axis[1] if lpips_3d_axis is not None else None,
        "lpips_3d_z": lpips_3d_axis[2] if lpips_3d_axis is not None else None,
        "n_points": n_points,
    }

    with open(osp.join(save_path, "eval3d.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

    np.save(osp.join(save_path, "vol_gt.npy"), t2a(vol_gt))
    np.save(osp.join(save_path, "vol_pred.npy"), t2a(vol_pred))
    # For visualization with 3D slicer
    sitk.WriteImage(
        sitk.GetImageFromArray(t2a(vol_gt).transpose(2, 0, 1)),
        os.path.join(save_path, "vol_gt.nii.gz"),
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(t2a(vol_pred).transpose(2, 0, 1)),
        os.path.join(save_path, "vol_pred.nii.gz"),
    )

    # Per-slice CSV (one row per z-slice, slice_id matches reconstruction/{id:05d}_*.png)
    csv_path = osp.join(save_path, f"per_slice_{name}.csv")
    n_slices = len(psnr_3d_per_slice)
    lpips_list = (
        lpips_3d_per_slice if lpips_3d_per_slice is not None else [None] * n_slices
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice_id", "psnr_3d", "ssim_3d", "lpips_3d"])
        for i in range(n_slices):
            psnr_v = psnr_3d_per_slice[i]
            ssim_v = ssim_3d_per_slice[i]
            lpips_v = "" if lpips_list[i] is None else f"{float(lpips_list[i]):.6f}"
            writer.writerow([
                i,
                f"{float(psnr_v):.6f}" if not np.isinf(psnr_v) else "inf",
                f"{float(ssim_v):.6f}",
                lpips_v,
            ])

    lpips_3d_str = f"{lpips_3d:.4f}" if lpips_3d is not None else "n/a"
    print(
        f"{name} complete. "
        f"psnr_3d: {psnr_3d:.3f}, ssim_3d: {ssim_3d:.4f}, lpips_3d: {lpips_3d_str}, "
        f"n_points: {n_points}"
    )


def evaluate_render(save_path, name, views, gaussians, pipeline):
    """Evaluate projection rendering."""
    proj_save_path = osp.join(save_path, name)

    # If already rendered, skip.
    if osp.exists(osp.join(save_path, "eval.yml")):
        print("{} in {} already rendered. Skip.".format(name, save_path))
        return
    os.makedirs(proj_save_path, exist_ok=True)

    gt_list = []
    render_list = []
    for view in tqdm(views, desc="render {}".format(name), leave=False):
        rendering = render(view, gaussians, pipeline)["render"]
        gt = view.original_image[0:3, :, :]
        gt_list.append(gt)
        render_list.append(rendering)
    multithread_write(gt_list, proj_save_path, "_gt")
    multithread_write(render_list, proj_save_path, "_pred")

    images = torch.concat(render_list, 0).permute(1, 2, 0)
    gt_images = torch.concat(gt_list, 0).permute(1, 2, 0)
    psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
    ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
    lpips_2d, lpips_2d_projs = lpips_metric_proj(gt_images, images)
    n_points = int(gaussians.get_xyz.shape[0])
    eval_dict = {
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "lpips_2d": lpips_2d,
        "psnr_2d_projs": psnr_2d_projs,
        "ssim_2d_projs": ssim_2d_projs,
        "lpips_2d_projs": lpips_2d_projs,
        "n_points": n_points,
    }
    with open(osp.join(save_path, f"eval2d_{name}.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

    # Per-camera CSV (one row per camera, easy to open in Excel)
    csv_path = osp.join(save_path, f"per_camera_{name}.csv")
    n_cams = len(psnr_2d_projs)
    lpips_list = lpips_2d_projs if lpips_2d_projs is not None else [None] * n_cams
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "psnr_2d", "ssim_2d", "lpips_2d"])
        for i in range(n_cams):
            psnr_v = float(psnr_2d_projs[i]) if psnr_2d_projs[i] != 0 else 0.0
            ssim_v = float(ssim_2d_projs[i]) if ssim_2d_projs[i] != 0 else 0.0
            lpips_v = "" if lpips_list[i] is None else f"{float(lpips_list[i]):.6f}"
            writer.writerow([i, f"{psnr_v:.6f}", f"{ssim_v:.6f}", lpips_v])

    lpips_2d_str = f"{lpips_2d:.4f}" if lpips_2d is not None else "n/a"
    print(
        f"{name} complete. "
        f"psnr_2d: {psnr_2d:.3f}, ssim_2d: {ssim_2d:.4f}, lpips_2d: {lpips_2d_str}, "
        f"n_points: {n_points}"
    )


def multithread_write(image_list, path, suffix):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, osp.join(path, "{0:05d}".format(count) + "{}.png".format(suffix))
            )
            np.save(
                osp.join(path, "{0:05d}".format(count) + "{}.npy".format(suffix)),
                image.cpu().numpy()[0],
            )
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_render_train", action="store_true", default=False)
    parser.add_argument("--skip_render_test", action="store_true", default=False)
    parser.add_argument("--skip_recon", action="store_true", default=False)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    with torch.no_grad():
        testing(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.skip_render_train,
            args.skip_render_test,
            args.skip_recon,
        )
