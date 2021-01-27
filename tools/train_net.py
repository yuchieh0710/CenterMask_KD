# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import copy


# from FLOPs_counter import print_model_parm_flops
# from torchprofile import profile_macs

def train(total_cfg, local_rank, distributed):
    total_model = []
    for i in reversed(range(len(total_cfg))):
        model = build_detection_model(total_cfg[i])
        device = torch.device(total_cfg[i].MODEL.DEVICE)
        model.to(device)
        if total_cfg[i].MODEL.USE_SYNCBN:
            assert is_pytorch_1_1_0_or_later(), \
                "SyncBatchNorm is only available in pytorch >= 1.1.0"
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer = make_optimizer(total_cfg[i], model)
        scheduler = make_lr_scheduler(total_cfg[i], optimizer)

        if distributed:                                                     
            model = torch.nn.parallel.DistributedDataParallel(              
                model, device_ids=[local_rank], output_device=local_rank,   
                # this should be removed if we update BatchNorm stats       
                broadcast_buffers=False, )

        arguments = {}
        arguments["iteration"] = 0

        output_dir = total_cfg[i].OUTPUT_DIR

        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            total_cfg[i], model, optimizer, scheduler, output_dir, save_to_disk
        )
        extra_checkpoint_data = checkpointer.load(total_cfg[i].MODEL.WEIGHT)
        if i == 0:
            arguments.update(extra_checkpoint_data)
        total_model.append(model)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = total_cfg[0].SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(total_cfg[0], is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = total_cfg[0].SOLVER.CHECKPOINT_PERIOD
    if len(total_model)>1:
        params = sum([np.prod(p.size()) for p in total_model[1].parameters()])
        print('Number of Parameters:{:5f}M'.format(params / 1e6))
        params = sum([np.prod(p.size()) for p in total_model[0].parameters()])
        print('teacher_model Number of Parameters:{:5f}M'.format(params / 1e6))
    else:
        params = sum([np.prod(p.size()) for p in total_model[0].parameters()])
        print('Number of Parameters:{:5f}M'.format(params / 1e6))

    do_train(
        total_cfg,
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
    )

    return total_model[1]


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    # setting default
    class_num = 21   # change numbers of class
    batch_size = 2   # change training batch size
    save_period = 50   # each 5000 iterations save and test once
    max_iteration = 400000  # train how much iterations
    lr_reduce_step = (300000, 340000)  # reduce learning rate at 300000 and 340000 iterations
    save_path = 'checkpoints/test'  # where to save the model (ex. modify checkpoint/XXXX)
    train_mode = 'kd'  # choose training mode (teacher/student/kd)

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--teacher-config-file",
        default="../configs/centermask/centermask_V_19_eSE_FPN_ms_3x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--student-config-file",
        default="../configs/centermask/centermask_V_19_eSE_FPN_lite_res600_ms_bs16_4x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=['MODEL.FCOS.NUM_CLASSES', class_num, 'SOLVER.CHECKPOINT_PERIOD', save_period, 'SOLVER.TEST_PERIOD',
                 save_period, 'SOLVER.IMS_PER_BATCH', batch_size, 'SOLVER.MAX_ITER', max_iteration, 'SOLVER.STEPS',
                 lr_reduce_step, 'OUTPUT_DIR', save_path],
        nargs=argparse.REMAINDER,
    )

    # setting kd loss
    if train_mode == 'kd':
        parser.add_argument('--loss_head', default=True)
        parser.add_argument('--loss_correlation', default=True)
        parser.add_argument('--loss_featuremap', default=False)
    else:  # always False
        parser.add_argument('--loss_head', default=False)
        parser.add_argument('--loss_correlation', default=False)
        parser.add_argument('--loss_featuremap', default=False)

    global args
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    t_cfg = copy.deepcopy(cfg)

    cfg.merge_from_file(args.student_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    t_cfg.merge_from_file(args.teacher_config_file)
    t_cfg.merge_from_list(args.opts)
    t_cfg.freeze()
    if train_mode == 'teacher':
        total_cfg = [t_cfg]
    elif train_mode == 'student':
        total_cfg = [cfg]
    else:
        total_cfg = [cfg, t_cfg]

    output_dir = total_cfg[0].OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if train_mode == 'teacher':
        logger.info("Loaded configuration file {}".format(args.teacher_config_file))
    else:
        logger.info("Loaded configuration file {}".format(args.student_config_file))
    with open(args.student_config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(total_cfg[0]))

    model = train(total_cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(total_cfg[0], model, args.distributed)


if __name__ == "__main__":
    main()
