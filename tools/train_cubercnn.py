# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import logging
import os
import sys
from datetime import timedelta

import detectron2.utils.comm as comm
import numpy as np
import torch
import torch.distributed as dist
import yaml

from atek.dataset.atek_webdataset import create_wds_dataloader
from atek.dataset.omni3d_adapter import create_omni3d_webdataset, ObjectDetectionMode

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model
from cubercnn.solver import build_optimizer, freeze_bn, PeriodicCheckpointerOnlyOne
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from torch.nn.parallel import DistributedDataParallel

DEFAULT_TIMEOUT = timedelta(minutes=10)


logger = logging.getLogger("cubercnn")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)


def add_configs(_C):
    # TODO: Fix this, too ugly
    _C.MAX_TRAINING_ATTEMPTS = 3

    _C.TRAIN_LIST = ""
    _C.TEST_LIST = ""
    _C.ID_MAP_JSON = ""
    _C.CATEGORY_JSON = ""
    _C.DATASETS.OBJECT_DETECTION_MODE = "PER_CATEGORY"
    _C.SOLVER.VAL_MAX_ITER = 0


def get_tars(tar_yaml, use_relative_path=False):
    with open(tar_yaml, "r") as f:
        tar_files = yaml.safe_load(f)["tars"]
    if use_relative_path:
        data_dir = os.path.dirname(tar_yaml)
        tar_files = [os.path.join(data_dir, x) for x in tar_files]
    return tar_files


def build_test_loader(cfg):
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    print("World size:", world_size)
    print("Getting tars from rank:", rank)
    test_tars = get_tars(cfg.TEST_LIST, use_relative_path=True)

    test_tars_local = test_tars[rank::world_size]
    local_batch_size = max(cfg.SOLVER.IMS_PER_BATCH // world_size, 1)
    print("local_batch_size:", local_batch_size)

    test_wds = create_omni3d_webdataset(
        test_tars_local,
        batch_size=local_batch_size,
        repeat=True,
        category_id_remapping_json=cfg.ID_MAP_JSON,
        object_detection_mode=ObjectDetectionMode[cfg.DATASETS.OBJECT_DETECTION_MODE],
    )
    test_dataloader = create_wds_dataloader(
        test_wds, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True
    )

    dataset_name = os.path.basename(cfg.TEST_LIST).split(".")[0]
    MetadataCatalog.get(dataset_name).set(
        json_file="", image_root="", evaluator_type="coco"
    )

    return test_dataloader


def build_train_loader(cfg):
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    print("World size:", world_size)
    print("Getting tars from rank:", rank)
    train_tars = get_tars(cfg.TRAIN_LIST, use_relative_path=True)

    train_tars_local = train_tars[rank::world_size]
    local_batch_size = max(cfg.SOLVER.IMS_PER_BATCH // world_size, 1)
    print("local_batch_size:", local_batch_size)

    train_wds = create_omni3d_webdataset(
        train_tars_local,
        batch_size=local_batch_size,
        repeat=True,
        category_id_remapping_json=cfg.ID_MAP_JSON,
        object_detection_mode=ObjectDetectionMode[cfg.DATASETS.OBJECT_DETECTION_MODE],
    )
    train_dataloader = create_wds_dataloader(
        train_wds, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True
    )

    dataset_name = os.path.basename(cfg.TRAIN_LIST).split(".")[0]
    MetadataCatalog.get(dataset_name).set(
        json_file="", image_root="", evaluator_type="coco"
    )

    return train_dataloader


def do_val(cfg, model, iteration, writers, max_iter=100):
    data_loader = build_test_loader(cfg)
    start_iter = iteration

    with torch.no_grad():
        for data in data_loader:
            loss_dict = model(data)

            # reduce
            loss_dict_reduced = {
                k: v.item() for k, v in allreduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # sync up
            comm.synchronize()

            if comm.is_main_process():
                if iteration - start_iter > 5 and iteration % 20 == 0:
                    print("val_iter: {}".format(iteration))

                # send loss scalars to tensorboard.
                loss_dict_reduced = {
                    "Val_" + k: v for k, v in loss_dict_reduced.items()
                }
                loss_dict_reduced.update({"Val_total_loss": losses_reduced})

                if iteration - start_iter > 5 and (iteration + 1) % 20 == 0:
                    for k, v in loss_dict_reduced.items():
                        # TODO: use last writer (TensorboardXWriter) to save validation losses
                        writers[-1]._writer.add_scalar(k, v, iteration)

            iteration += 1

            if iteration - start_iter > max_iter:
                break

    return iteration


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

    # create the dataloader
    data_loader = build_train_loader(cfg)
    data_iter = iter(data_loader)

    if cfg.MODEL.WEIGHTS_PRETRAIN != "":
        # load ONLY the model, no checkpointables.
        checkpointer.load(cfg.MODEL.WEIGHTS_PRETRAIN, checkpointables=[])

    # determine the starting iteration, if resuming
    ckpt = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    start_iter = ckpt.get("iteration", -1) + 1
    val_iter = ckpt.get("val_iter", -1) + 1
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
                    for _, param in named_params:
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
                del data_loader
                del optimizer
                del checkpointer
                del periodic_checkpointer
                return False

            scheduler.step()

            # Flush events
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()

            # Evaluate only when the loss is not diverging.
            if not (diverging_model > 0) and (
                do_eval
                and ((iteration + 1) % cfg.TEST.EVAL_PERIOD) == 0
                and iteration != (max_iter - 1)
            ):
                print("Starting validation for iteration {}".format(iteration + 1))
                val_iter = do_val(
                    cfg, model, val_iter + 1, writers, max_iter=cfg.SOLVER.VAL_MAX_ITER
                )
                comm.synchronize()

                if not cfg.MODEL.USE_BN:
                    freeze_bn(model)

            # Do not bother checkpointing if there is potential for a diverging model.
            if (
                not (diverging_model > 0)
                and (iterations_explode / total_iterations) < 0.5 * cfg.MODEL.STABILIZE
            ):
                additional_state = {"val_iter": val_iter}
                periodic_checkpointer.step(iteration, **additional_state)

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

        MetadataCatalog.get("omni3d_model").thing_classes = cfg.DATASETS.CATEGORY_NAMES
        cat_ids = [int(x) for x in category_id_to_names.keys()]
        id_map = {id: i for i, id in enumerate(cat_ids)}
        MetadataCatalog.get("omni3d_model").thing_dataset_id_to_contiguous_id = id_map

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
        model = build_model(cfg, priors=None)

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
        reduced_dict = dict(zip(names, values))
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
        timeout=DEFAULT_TIMEOUT,
    )
