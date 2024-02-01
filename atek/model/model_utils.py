from atek.model.cubercnn import (
    CubeRCNNViewer,
    build_cubercnn_model,
    convert_cubercnn_prediction,
    save_cubercnn_prediction,
)


def build_model(args):
    if args.model_name == "cubercnn":
        cfg, model = build_cubercnn_model(args)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_name}")

    return cfg, model


def setup_callback(dataset, args, cfg):
    if args.model_name == "cubercnn":
        iter_callbacks = []
        if args.visualize:
            viewer = CubeRCNNViewer(dataset, args)
            iter_callbacks.append(viewer)
        callbacks = {
            "iter_postprocess": convert_cubercnn_prediction,
            "iter_callback": iter_callbacks,
            "seq_callback": [save_cubercnn_prediction],
        }
    else:
        raise ValueError(
            f"Unknown model architecture for setting callbacks: {args.model_name}"
        )
    return callbacks
