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
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import csv

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import (
    metric_vol,
    metric_proj,
    lpips_metric_proj,
    lpips_metric_vol,
)
from r2_gaussian.utils.plot_utils import show_two_slice
from r2_gaussian.utils.fast_utils import (
    sampling_cameras,
    compute_gaussian_score_r2gs,
)


class GPUMemoryLogger:
    def __init__(self, enabled=False, interval=100, sync=False, csv_path=None):
        self.enabled = enabled and torch.cuda.is_available()
        self.interval = max(int(interval), 1)
        self.sync = sync
        self.csv_path = csv_path
        self.rows = []
        self.prev_row = None

    def should_profile(self, iteration):
        return self.enabled and (iteration == 1 or iteration % self.interval == 0)

    @staticmethod
    def _to_mb(value):
        return value / (1024.0**2)

    def mark(self, iteration, stage, n_points=None):
        if not self.enabled:
            return None
        if self.sync:
            torch.cuda.synchronize()

        free_b, total_b = torch.cuda.mem_get_info()
        alloc_mb = self._to_mb(torch.cuda.memory_allocated())
        reserved_mb = self._to_mb(torch.cuda.memory_reserved())
        max_alloc_mb = self._to_mb(torch.cuda.max_memory_allocated())
        max_reserved_mb = self._to_mb(torch.cuda.max_memory_reserved())
        free_mb = self._to_mb(free_b)
        total_mb = self._to_mb(total_b)

        delta_alloc_mb = 0.0
        delta_reserved_mb = 0.0
        transition = ""
        if self.prev_row is not None and self.prev_row["iteration"] == iteration:
            delta_alloc_mb = alloc_mb - self.prev_row["alloc_mb"]
            delta_reserved_mb = reserved_mb - self.prev_row["reserved_mb"]
            transition = f"{self.prev_row['stage']}->{stage}"

        row = {
            "iteration": iteration,
            "stage": stage,
            "transition": transition,
            "n_points": int(n_points) if n_points is not None else -1,
            "alloc_mb": alloc_mb,
            "reserved_mb": reserved_mb,
            "max_alloc_mb": max_alloc_mb,
            "max_reserved_mb": max_reserved_mb,
            "delta_alloc_mb": delta_alloc_mb,
            "delta_reserved_mb": delta_reserved_mb,
            "free_mb": free_mb,
            "total_mb": total_mb,
        }
        self.rows.append(row)
        self.prev_row = row
        return row

    def summarize_iteration(self, iteration):
        if not self.enabled:
            return None

        iter_rows = [r for r in self.rows if r["iteration"] == iteration]
        if len(iter_rows) == 0:
            return None

        inc_rows = [r for r in iter_rows if r["delta_alloc_mb"] > 0]
        if inc_rows:
            hot = max(inc_rows, key=lambda r: r["delta_alloc_mb"])
            hot_msg = f"+{hot['delta_alloc_mb']:.1f}MB at {hot['transition']}"
        else:
            hot_msg = "no positive alloc delta"

        peak = max(iter_rows, key=lambda r: r["reserved_mb"])
        n_points = peak.get("n_points", -1)
        pts_msg = f", pts {n_points}" if n_points >= 0 else ""
        return (
            f"[MEM][ITER {iteration}] max growth {hot_msg}; "
            f"peak reserved {peak['reserved_mb']:.1f}MB "
            f"(alloc {peak['alloc_mb']:.1f}MB, free {peak['free_mb']:.1f}MB{pts_msg})"
        )

    def summarize_transitions(self, topk=5):
        if not self.enabled:
            return []

        stats = {}
        for row in self.rows:
            transition = row["transition"]
            if not transition or row["delta_alloc_mb"] <= 0:
                continue
            if transition not in stats:
                stats[transition] = {
                    "total_delta_mb": 0.0,
                    "max_delta_mb": 0.0,
                    "count": 0,
                }
            stats[transition]["total_delta_mb"] += row["delta_alloc_mb"]
            stats[transition]["max_delta_mb"] = max(
                stats[transition]["max_delta_mb"], row["delta_alloc_mb"]
            )
            stats[transition]["count"] += 1

        ranked = sorted(
            stats.items(), key=lambda kv: kv[1]["total_delta_mb"], reverse=True
        )
        return ranked[:topk]

    def dump_csv(self):
        if not self.enabled or not self.csv_path or len(self.rows) == 0:
            return

        os.makedirs(osp.dirname(self.csv_path), exist_ok=True)
        fieldnames = [
            "iteration",
            "stage",
            "transition",
            "n_points",
            "alloc_mb",
            "reserved_mb",
            "max_alloc_mb",
            "max_reserved_mb",
            "delta_alloc_mb",
            "delta_reserved_mb",
            "free_mb",
            "total_mb",
        ]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    mem_profile=False,
    mem_profile_interval=100,
    mem_profile_sync=False,
    disable_vcd=False,
    disable_vcp=False,
    vcd_quantile=0.70,
    vcp_quantile=0.98,
    vcp_hard_quantile=0.985,
    mv_soft_start_iter=6000,
    mv_hard_start_iter=15000,
    mv_refine_start_iter=18000,
    mv_hard_interval=400,
):
    first_iter = 0
    mv_hard_interval = max(int(mv_hard_interval), 1)

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    mem_logger = GPUMemoryLogger(
        enabled=mem_profile,
        interval=mem_profile_interval,
        sync=mem_profile_sync,
        csv_path=osp.join(scene.model_path, "mem_profile.csv"),
    )
    if mem_logger.enabled:
        print(
            f"Enable memory profiling: every {mem_logger.interval} iter, "
            f"log file {mem_logger.csv_path}"
        )
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1

    def score_quantile_threshold(score, quantile, default_value):
        if score is None or score.numel() == 0:
            return default_value
        q = min(max(float(quantile), 0.0), 1.0)
        return torch.quantile(score.detach(), q).item()

    for iteration in range(first_iter, opt.iterations + 1):
        do_mem_profile = mem_logger.should_profile(iteration)
        iter_start.record()
        if do_mem_profile:
            mem_logger.mark(iteration, "iter_start", gaussians.get_density.shape[0])

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if do_mem_profile:
            mem_logger.mark(iteration, "render", gaussians.get_density.shape[0])

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv
        if do_mem_profile:
            mem_logger.mark(iteration, "loss", gaussians.get_density.shape[0])

        loss["total"].backward()
        if do_mem_profile:
            mem_logger.mark(iteration, "backward", gaussians.get_density.shape[0])

        iter_end.record()
        torch.cuda.synchronize()

        # Four-stage schedule:
        # 1) warmup: no VCP pruning
        # 2) soft: VCP + densify
        # 3) hard: VCP only (no densify)
        # 4) refine: no prune / no densify
        n_points = gaussians.get_xyz.shape[0]
        do_mv_base = (
            iteration > opt.densify_from_iter
            and iteration % opt.densification_interval == 0
            and n_points > 1000
        )
        do_mv_hard = (
            iteration >= mv_hard_start_iter
            and iteration < mv_refine_start_iter
            and iteration % mv_hard_interval == 0
            and n_points > 1000
        )

        mv_stage = "off"
        run_mv = False
        do_densify_mv = False
        run_vcp = False
        if not (disable_vcd and disable_vcp):
            if do_mv_base and iteration < mv_soft_start_iter:
                mv_stage = "warmup"
                run_mv = True
                do_densify_mv = iteration < opt.densify_until_iter
                run_vcp = False
            elif do_mv_base and iteration < mv_hard_start_iter:
                mv_stage = "soft"
                run_mv = True
                do_densify_mv = iteration < opt.densify_until_iter
                run_vcp = True
            elif do_mv_hard:
                mv_stage = "hard"
                run_mv = True
                do_densify_mv = False
                run_vcp = True
            elif iteration >= mv_refine_start_iter:
                mv_stage = "refine"

        importance_score, pruning_score = None, None
        if run_mv:
            camlist = sampling_cameras(
                scene.getTrainCameras(),
                num_cams=getattr(opt, "mv_num_cams", 6),
            )
            importance_score, pruning_score = compute_gaussian_score_r2gs(
                camlist,
                gaussians,
                pipe,
                lambda_dssim=opt.lambda_dssim,
                score_mode=getattr(opt, "mv_score_mode", "screen_grad"),
            )
            if disable_vcd:
                importance_score = None
            if disable_vcp or (not run_vcp):
                pruning_score = None

        prune_quantile = vcp_hard_quantile if mv_stage == "hard" else vcp_quantile
        importance_thresh = score_quantile_threshold(
            importance_score,
            vcd_quantile,
            getattr(opt, "mv_importance_thresh", 0.6),
        )
        prune_score_thresh = score_quantile_threshold(
            pruning_score,
            prune_quantile,
            getattr(opt, "mv_prune_score_thresh", 0.8),
        )

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if run_mv:
                before_pts = gaussians.get_xyz.shape[0]
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    opt.density_min_threshold,
                    opt.max_screen_size,
                    max_scale,
                    opt.max_num_gaussians,
                    densify_scale_threshold,
                    bbox,
                    importance_score=importance_score,
                    pruning_score=pruning_score,
                    importance_thresh=importance_thresh,
                    prune_score_thresh=prune_score_thresh,
                    do_densify=do_densify_mv,
                )
                after_pts = gaussians.get_xyz.shape[0]
                imp_mean = (
                    importance_score.mean().item()
                    if importance_score is not None
                    else float("nan")
                )
                prune_mean = (
                    pruning_score.mean().item()
                    if pruning_score is not None
                    else float("nan")
                )
                tqdm.write(
                    f"[MV][{mv_stage}] iter={iteration} pts {before_pts}->{after_pts} "
                    f"delta={after_pts - before_pts:+d} "
                    f"imp_mean={imp_mean:.3f} "
                    f"prune_mean={prune_mean:.3f} "
                    f"densify={int(do_densify_mv)} "
                    f"ith={importance_thresh:.3f} "
                    f"pth={prune_score_thresh:.3f}"
                )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )
            if do_mem_profile:
                mem_logger.mark(iteration, "adaptive_control", gaussians.get_density.shape[0])

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if do_mem_profile:
                    mem_logger.mark(iteration, "optimizer", gaussians.get_density.shape[0])

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            if do_mem_profile:
                mem_logger.mark(iteration, "pre_report", gaussians.get_density.shape[0])
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
            )
            if do_mem_profile:
                mem_logger.mark(iteration, "post_report", gaussians.get_density.shape[0])
                msg = mem_logger.summarize_iteration(iteration)
                if msg is not None:
                    tqdm.write(msg)

    if mem_logger.enabled:
        mem_logger.dump_csv()
        transition_stats = mem_logger.summarize_transitions(topk=5)
        if len(transition_stats) > 0:
            tqdm.write("[MEM] Top growth transitions:")
            for transition, stat in transition_stats:
                avg_delta = stat["total_delta_mb"] / max(stat["count"], 1)
                tqdm.write(
                    f"[MEM] {transition}: total +{stat['total_delta_mb']:.1f}MB, "
                    f"avg +{avg_delta:.1f}MB, max +{stat['max_delta_mb']:.1f}MB, "
                    f"count {stat['count']}"
                )
        tqdm.write(f"[MEM] Saved detailed log to {mem_logger.csv_path}")


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                lpips_2d, lpips_2d_projs = lpips_metric_proj(gt_images, images)
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "lpips_2d": lpips_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                    "lpips_2d_projs": lpips_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )
                    if lpips_2d is not None:
                        tb_writer.add_scalar(
                            config["name"] + "/lpips_2d", lpips_2d, iteration
                        )

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        lpips_3d, lpips_3d_axis = lpips_metric_vol(vol_gt, vol_pred)
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
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
            if lpips_3d is not None:
                tb_writer.add_scalar("reconstruction/lpips_3d", lpips_3d, iteration)
        lpips_3d_str = f"{lpips_3d:.3f}" if lpips_3d is not None else "n/a"
        lpips_2d_str = f"{lpips_2d:.3f}" if lpips_2d is not None else "n/a"
        tqdm.write(
            f"[ITER {iteration}] Evaluating: "
            f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, lpips3d {lpips_3d_str}, "
            f"psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}, lpips2d {lpips_2d_str}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mem_profile", action="store_true", default=False)
    parser.add_argument("--mem_profile_interval", type=int, default=100)
    parser.add_argument("--mem_profile_sync", action="store_true", default=False)
    parser.add_argument("--disable_vcd", action="store_true", default=False)
    parser.add_argument("--disable_vcp", action="store_true", default=False)
    parser.add_argument("--vcd_quantile", type=float, default=0.70)
    parser.add_argument("--vcp_quantile", type=float, default=0.98)
    parser.add_argument("--vcp_hard_quantile", type=float, default=0.985)
    parser.add_argument("--mv_soft_start_iter", type=int, default=6000)
    parser.add_argument("--mv_hard_start_iter", type=int, default=15000)
    parser.add_argument("--mv_refine_start_iter", type=int, default=18000)
    parser.add_argument("--mv_hard_interval", type=int, default=400)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.mem_profile,
        args.mem_profile_interval,
        args.mem_profile_sync,
        args.disable_vcd,
        args.disable_vcp,
        args.vcd_quantile,
        args.vcp_quantile,
        args.vcp_hard_quantile,
        args.mv_soft_start_iter,
        args.mv_hard_start_iter,
        args.mv_refine_start_iter,
        args.mv_hard_interval,
    )

    # All done
    print("Training complete.")
