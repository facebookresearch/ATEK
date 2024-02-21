import argparse
import os
import sys

from atek.model.model_factory import create_inference_model

from detectron2.engine import launch
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel


def run_inference(seq_path, model, model_config, args):
    # TODO: add actual inference logic here
    pass


def main(args):
    # setup config and model
    model_config, model = create_inference_model(args)

    # setup distributed inference
    world_size = comm.get_world_size()
    rank = comm.get_rank()
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # run inference
    with open(args.input_file, "r") as f:
        seq_paths_all = f.read().splitlines()

    # split input files across all GPUs
    seq_paths_local = seq_paths_all[rank::world_size]
    for seq_path in seq_paths_local:
        run_inference(seq_path, model, model_config, args)


# TODO: make this more generic and rename the file to infer.py
def get_args():
    parser = argparse.ArgumentParser(
        epilog=None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-file", default=None, help="Path to file with test sequences"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model architecture name, e.g., cubercnn",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        metavar="FILE",
        help="Path to config file of a trained model",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
