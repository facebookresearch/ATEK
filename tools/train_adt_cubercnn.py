# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import logging
import os
import sys
import numpy as np
import copy
from collections import OrderedDict
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

import pdb

logger = logging.getLogger("cubercnn")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.solver import build_optimizer, freeze_bn, PeriodicCheckpointerOnlyOne
from cubercnn.config import get_cfg_defaults
from cubercnn.evaluation import (
    Omni3DEvaluator,
    Omni3Deval,
    Omni3DEvaluationHelper,
    inference_on_dataset,
)
from cubercnn import util, vis, data
import cubercnn.vis.logperf as utils_logperf

from atek.dataset.adt_wds import get_loader
from atek.model.cubercnn import build_model_with_priors


def add_configs(_C):
    _C.MAX_TRAINING_ATTEMPTS = 3

    _C.TRAIN_LIST = ""
    _C.TEST_LIST = ""
    _C.ID_MAP_JSON = ""
    _C.CATEGORY_JSON = ""


def get_tars(tar_yaml, use_relative_path=False):
    with open(tar_yaml, "r") as f:
        tar_files = yaml.safe_load(f)["tars"]
    if use_relative_path:
        data_dir = os.path.dirname(tar_yaml)
        tar_files = [os.path.join(data_dir, x) for x in tar_files]
    return tar_files


def build_test_loader(cfg):
    test_files = get_tars(cfg.TEST_LIST, use_relative_path=True)
    test_adt_loader = get_loader(test_files, cfg.ID_MAP_JSON)

    dataset_name = os.path.basename(cfg.TEST_LIST).split(".")[0]
    MetadataCatalog.get(dataset_name).set(json_file="", image_root="", evaluator_type="coco")

    return test_adt_loader


def build_train_loader(cfg):
    print("Getting tars from rank:", comm.get_local_rank())
    train_files = get_tars(cfg.TRAIN_LIST, use_relative_path=True)

    local_rank = comm.get_local_rank()
    world_size = comm.get_world_size()
    train_files_local = train_files[local_rank::world_size]

    local_batch_size = cfg.SOLVER.IMS_PER_BATCH // world_size

    train_adt_loader = get_loader(
        train_files_local,
        cfg.ID_MAP_JSON,
        batch_size=local_batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        repeat=True,
    )

    dataset_name = os.path.basename(cfg.TRAIN_LIST).split(".")[0]
    MetadataCatalog.get(dataset_name).set(json_file="", image_root="", evaluator_type="coco")

    return train_adt_loader


def do_test(cfg, model, iteration="final", storage=None):
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    filter_settings["visibility_thres"] = cfg.TEST.VISIBILITY_THRES
    filter_settings["truncation_thres"] = cfg.TEST.TRUNCATION_THRES
    filter_settings["min_height_thres"] = 0.0625
    filter_settings["max_depth"] = 1e8

    dataset_names_test = cfg.DATASETS.TEST
    only_2d = cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(
        cfg.OUTPUT_DIR, "inference", "iter_{}".format(iteration)
    )

    eval_helper = Omni3DEvaluationHelper(
        dataset_names_test,
        filter_settings,
        output_folder,
        iter_label=iteration,
        only_2d=only_2d,
    )

    for dataset_name in dataset_names_test:
        """
        Cycle through each dataset and test them individually.
        This loop keeps track of each per-image evaluation result,
        so that it doesn't need to be re-computed for the collective.
        """

        """
        Distributed Cube R-CNN inference
        """
        data_loader = build_test_loader(cfg)
        results_json = inference_on_dataset(model, data_loader)

        if comm.is_main_process():
            """
            Individual dataset evaluation
            """
            eval_helper.add_predictions(dataset_name, results_json)
            eval_helper.save_predictions(dataset_name)
            eval_helper.evaluate(dataset_name)

            """
            Optionally, visualize some instances
            """
            instances = torch.load(
                os.path.join(output_folder, dataset_name, "instances_predictions.pth")
            )
            log_str = vis.visualize_from_instances(
                instances,
                data_loader.dataset,
                dataset_name,
                cfg.INPUT.MIN_SIZE_TEST,
                os.path.join(output_folder, dataset_name),
                MetadataCatalog.get("omni3d_model").thing_classes,
                iteration,
            )
            logger.info(log_str)

    if comm.is_main_process():
        """
        Summarize each Omni3D Evaluation metric
        """
        eval_helper.summarize_all()


def do_train(cfg, model, resume=False):
    max_iter = cfg.SOLVER.MAX_ITER
    do_eval = cfg.TEST.EVAL_PERIOD > 0

    model.train()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # bookkeeping
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    periodic_checkpointer = PeriodicCheckpointerOnlyOne(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    )

    # pdb.set_trace()
    print("====" * 20)
    print("Local rank:", comm.get_local_rank())

    # create the dataloader
    data_loader = build_train_loader(cfg)
    data_iter = iter(data_loader)

    if cfg.MODEL.WEIGHTS_PRETRAIN != "":
        # load ONLY the model, no checkpointables.
        checkpointer.load(cfg.MODEL.WEIGHTS_PRETRAIN, checkpointables=[])

    # determine the starting iteration, if resuming
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))

    if not cfg.MODEL.USE_BN:
        freeze_bn(model)

    world_size = comm.get_world_size()

    # if the loss diverges for more than the below TOLERANCE
    # as a percent of the iterations, the training will stop.
    # This is only enabled if "STABILIZE" is on, which
    # prevents a single example from exploding the training.
    iterations_success = 0
    iterations_explode = 0

    # when loss > recent_loss * TOLERANCE, then it could be a
    # diverging/failing model, which we should skip all updates for.
    TOLERANCE = 4.0

    GAMMA = 0.02  # rolling average weight gain
    recent_loss = None  # stores the most recent loss magnitude

    # model.parameters() is surprisingly expensive at 150ms, so cache it
    named_params = list(model.named_parameters())

    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(
    #         wait=2,
    #         warmup=2,
    #         active=6,
    #         repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.OUTPUT_DIR),
    #     with_stack=False
    # ) as profiler:

    with EventStorage(start_iter) as storage:
        while True:
            data = next(data_iter)
            storage.iter = iteration

            # forward
            loss_dict = model(data)
            losses = sum(loss_dict.values())

            # reduce
            loss_dict_reduced = {
                k: v.item() for k, v in allreduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # sync up
            comm.synchronize()

            if recent_loss is None:
                # init recent loss fairly high
                recent_loss = losses_reduced * 2.0

            # Is stabilization enabled, and loss high or NaN?
            diverging_model = cfg.MODEL.STABILIZE > 0 and (
                losses_reduced > recent_loss * TOLERANCE
                or not (np.isfinite(losses_reduced))
                or np.isnan(losses_reduced)
            )

            if diverging_model:
                # clip and warn the user.
                losses = losses.clip(0, 1)
                logger.warning(
                    "Skipping gradient update due to higher than normal loss {:.2f} vs. rolling mean {:.2f}, Dict-> {}".format(
                        losses_reduced, recent_loss, loss_dict_reduced
                    )
                )
            else:
                # compute rolling average of loss
                recent_loss = recent_loss * (1 - GAMMA) + losses_reduced * GAMMA

            if comm.is_main_process():
                # send loss scalars to tensorboard.
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # backward and step
            optimizer.zero_grad()
            losses.backward()

            # if the loss is not too high,
            # we still want to check gradients.
            if not diverging_model:
                if cfg.MODEL.STABILIZE > 0:
                    for name, param in named_params:
                        if param.grad is not None:
                            diverging_model = (
                                torch.isnan(param.grad).any()
                                or torch.isinf(param.grad).any()
                            )

                        if diverging_model:
                            logger.warning(
                                "Skipping gradient update due to inf/nan detection, loss is {}".format(
                                    loss_dict_reduced
                                )
                            )
                            break

            # convert exploded to a float, then allreduce it,
            # if any process gradients have exploded then we skip together.
            diverging_model = torch.tensor(float(diverging_model)).cuda()

            if world_size > 1:
                dist.all_reduce(diverging_model)

            # sync up
            comm.synchronize()

            if diverging_model > 0:
                optimizer.zero_grad()
                iterations_explode += 1

            else:
                optimizer.step()
                storage.put_scalar(
                    "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
                )
                iterations_success += 1

            # profiler.step()

            total_iterations = iterations_success + iterations_explode

            # Only retry if we have trained sufficiently long relative
            # to the latest checkpoint, which we would otherwise revert back to.
            retry = (iterations_explode / total_iterations) >= cfg.MODEL.STABILIZE and (
                total_iterations > cfg.SOLVER.CHECKPOINT_PERIOD * 1 / 2
            )

            # Important for dist training. Convert to a float, then allreduce it,
            # if any process gradients have exploded then we must skip together.
            retry = torch.tensor(float(retry)).cuda()

            if world_size > 1:
                dist.all_reduce(retry)

            # sync up
            comm.synchronize()

            # any processes need to retry
            if retry > 0:
                # instead of failing, try to resume the iteration instead.
                logger.warning(
                    "!! Restarting training at {} iters. Exploding loss {:d}% of iters !!".format(
                        iteration,
                        int(
                            100
                            * (
                                iterations_explode
                                / (iterations_success + iterations_explode)
                            )
                        ),
                    )
                )

                # send these to garbage, for ideally a cleaner restart.
                del data_mapper
                del data_loader
                del optimizer
                del checkpointer
                del periodic_checkpointer
                return False

            scheduler.step()

            # Evaluate only when the loss is not diverging.
            if not (diverging_model > 0) and (
                do_eval
                and ((iteration + 1) % cfg.TEST.EVAL_PERIOD) == 0
                and iteration != (max_iter - 1)
            ):
                logger.info("Skipping test")
                # logger.info("Starting test for iteration {}".format(iteration + 1))
                # do_test(cfg, model, iteration=iteration + 1, storage=storage)
                comm.synchronize()

                if not cfg.MODEL.USE_BN:
                    freeze_bn(model)

            # Flush events
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()

            # Do not bother checkpointing if there is potential for a diverging model.
            if (
                not (diverging_model > 0)
                and (iterations_explode / total_iterations) < 0.5 * cfg.MODEL.STABILIZE
            ):
                periodic_checkpointer.step(iteration)

            iteration += 1

            if iteration >= max_iter:
                break

    # success
    return True


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # add extra configs for data
    add_configs(cfg)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)

    with open(cfg.CATEGORY_JSON, "r") as f:
        category_id_to_names = json.load(f)
        cfg.DATASETS.CATEGORY_NAMES = list(category_id_to_names.values())

        MetadataCatalog.get('omni3d_model').thing_classes = cfg.DATASETS.CATEGORY_NAMES
        cat_ids = [int(x) for x in category_id_to_names.keys()]
        id_map = {id: i for i, id in enumerate(cat_ids)}
        MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id = id_map

    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cubercnn"
    )

    return cfg


def main(args):
    cfg = setup(args)

    """
    The training loops can attempt to train for N times.
    This catches a divergence or other failure modes.
    """

    remaining_attempts = cfg.MAX_TRAINING_ATTEMPTS
    while remaining_attempts > 0:
        # build the training model.
        model = build_model_with_priors(cfg, priors=None)

        if remaining_attempts == cfg.MAX_TRAINING_ATTEMPTS:
            # log the first attempt's settings.
            logger.info("Model:\n{}".format(model))

        if args.eval_only:
            # skip straight to eval mode
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            # return do_test(cfg, model)
            return

        # setup distributed training.
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        # train full model, potentially with resume.
        if do_train(cfg, model, resume=args.resume):
            break
        else:
            # allow restart when a model fails to train.
            remaining_attempts -= 1
            del model

    if remaining_attempts == 0:
        # Exit if the model could not finish without diverging.
        raise ValueError("Training failed")

    # return do_test(cfg, model)
    return


def allreduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = comm.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
