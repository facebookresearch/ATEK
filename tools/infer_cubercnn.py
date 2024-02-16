import argparse
import os
import sys

import tqdm
from detectron2.engine import launch
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel

from atek.callback.callback_factory import create_inference_callback
from atek.dataset.dataset_factory import create_inference_dataset
from atek.model.model_factory import create_inference_model


def run_inference(args, model_config, seq_path, model):
    # setup dataset
    dataset = create_inference_dataset(seq_path, args, model_config)

    # setup callbacks
    callbacks = create_inference_callback(args, model_config)

    # run inference, with optional callbacks
    prediction_list = []
    for idx in tqdm.tqdm(range(len(dataset) // 20)):
        data = dataset[idx]
        prediction = model(data)

        # run callbacks for current iteration
        for callback in callbacks["iter_callback"]:
            callback(data, prediction)

        prediction_list.append(prediction)

    # run callbacks for whole sequence
    for callback in callbacks["seq_callback"]:
        callback(prediction_list)


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
    seq_paths_local = seq_paths_all[rank::world_size]
    for seq_path in seq_paths_local:
        run_inference(args, model_config, seq_path, model)


def get_args():
    parser = argparse.ArgumentParser(
        epilog=None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-file", default=None, help="Path to file with test sequences"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory to save model predictions"
    )
    parser.add_argument(
        "--data-type",
        default="raw",
        help="Input data type. wds: webdataset tars, raw: raw ADT data",
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
        "--metadata-file",
        default=None,
        help="File with metadata for all instances",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="threshold on score for visualizing",
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help="whether to visualize inference results",
    )
    parser.add_argument(
        "--web-port",
        default=8888,
        help="The port to serve the web viewer on (defaults to 8888).",
    )
    parser.add_argument(
        "--ws-port",
        default=8877,
        help="The port to serve the WebSocket server on (defaults to 8877)",
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
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
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
