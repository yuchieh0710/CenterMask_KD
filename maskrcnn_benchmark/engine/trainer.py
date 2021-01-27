# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, is_pytorch_1_1_0_or_later
from maskrcnn_benchmark.engine.inference import inference

from .kd_loss import *

from collections import OrderedDict

import numpy as np


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        total_model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        args,
):
    if len(total_model) > 1:
        model = total_model[1]
        t_model = total_model[0]
    else:
        model = total_model[0]
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()

    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg[0].MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg[0].MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg[0].DATASETS.TEST

    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict, features_dict = model(images, targets)
        if len(total_model) > 1:
            with torch.no_grad():
                t_loss_dict, t_features_dict = t_model(images, targets)
            # with torch.no_grad():
            #     # teacher_model = t_model
            #     t_weight = torch.load('./weights/centermask-V-19-eSE-FPN-ms-3x.pth')
            #     t_weight = t_weight['model']
            #     new_tweight = OrderedDict()
            #     for k, v in t_weight.items():
            #         name = k[7:]  # remove `module.`
            #         new_tweight[name] = v
            #     t_model.load_state_dict(new_tweight)
            #     t_loss_dict, t_features_dict = t_model(images, targets)

        if args.loss_head:

            loss_regression = new_box_loss(t_loss_dict['loss_reg'], loss_dict['loss_reg'])
            loss_center = new_center_loss(t_loss_dict['loss_centerness'], loss_dict['loss_centerness'])
            mode = 'KL'  # mode = 'KL' or 'cross-entropy'
            loss_pixel_wise = pixel_wise_loss(features_dict['box_cls'], t_features_dict['box_cls'], mode)
            loss_head = (loss_regression + loss_center + loss_pixel_wise)
            loss_dict.setdefault('loss_head', loss_head)
            del loss_dict['loss_reg']
            del loss_dict['loss_centerness']

        if iteration > cfg[0].SOLVER.WARMUP_ITERS:
            if args.loss_correlation:
                correlation = True
                loss_corr = get_feature(t_model, model, images, targets, correlation)
                loss_dict.setdefault('loss_corr', loss_corr)
            if args.loss_featuremap:
                correlation = False
                loss_featuremap = get_feature(t_model, model, images, targets, correlation)
                loss_dict.setdefault('loss_featuremap', loss_featuremap)


        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0 and iteration != 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg[0], is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg[0].MODEL.MASK_ON else cfg[0].MODEL.RPN_ONLY,
                device=cfg[0].MODEL.DEVICE,
                expected_results=cfg[0].TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg[0].TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    if len(loss_dict) > 1:
                        loss_dict = loss_dict[0]
                    else:
                        loss_dict = loss_dict
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
