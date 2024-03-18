import argparse
import json
import os
import sys
import time

import tqdm

from atek.dataset.dataset_factory import create_inference_dataset
from atek.inference.callback_factory import create_inference_callback
from atek.model.model_factory import create_inference_model

from detectron2.engine import launch
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel


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

    # setup callbacks
    callbacks = create_inference_callback(args, model_config)

    # load sequence paths for inference
    with open(args.input_file, "r") as f:
        seq_paths_all = f.read().splitlines()

    # setup profiling
    num_iters = 0
    num_samples = 0
    start_time = time.time()

    # run inference
    seq_paths_local = seq_paths_all[rank::world_size]
    for seq_path in tqdm.tqdm(seq_paths_local):
        # setup dataset
        dataset = create_inference_dataset(seq_path, args, model_config)

        prediction_list = []
        for data in dataset:
            model_output = model(data)

            # post-process model output
            prediction = callbacks["post_processor"](data, model_output)

            # run callbacks for current iteration
            for callback in callbacks["per_batch_callbacks"]:
                callback(data, prediction)

            for eval_callback in callbacks["eval_callbacks"]:
                eval_callback.update(data, prediction)

            prediction_list.append(prediction)

            # increment iteration and sample counter
            num_iters += 1
            num_samples += len(data)

        # run callbacks for whole sequence
        for callback in callbacks["per_sequence_callbacks"]:
            callback(prediction_list)

    eval_results = []
    for eval_callback in callbacks["eval_callbacks"]:
        eval_results.append(eval_callback.evaluate())

    print(f"Evaluation results:\n{eval_results}")
    if args.eval_save_path:
        assert args.eval_save_path.endswith(".json")
        with open(args.eval_save_path, "w") as f:
            json.dump(eval_results, f, indent=2)

    elapsed_time = time.time() - start_time
    profile_message = (
        f"Inference time: {elapsed_time:.3f} secs for {num_iters} iters, "
        + f"{elapsed_time / num_iters:.2f} sec/iter, "
        + f"{num_samples} total samples "
        + f"{elapsed_time / num_samples:.2f} sec/sample"
    )
    print(profile_message)


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
        "--output-dir", default=None, help="Directory to save model predictions"
    )
    parser.add_argument(
        "--data-type",
        default="raw",
        help="Input data type. wds: webdataset tars, raw: raw ADT data",
    )
    parser.add_argument(
        "--rotate_image_cw90deg",
        type=bool,
        default=True,
        help="Rotate images by 90 degrees clockwise",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Target image height to process the raw data",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Target image width to process the raw data",
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
        "--evaluate",
        default=False,
        action="store_true",
        help="whether to evaluate the model predictions",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default="IoU",
        choices=["IoU", "GIoU", "ChamferDistance", "HungarianDistance"],
        help="Name of the metric to use for 3D bounding box evaluation",
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=0.25,
        help="Threshold of metric for matching predicted and GT bounding boxes",
    )
    parser.add_argument(
        "--eval-save-path",
        type=str,
        default=None,
        help="Path to save evaluation results",
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
