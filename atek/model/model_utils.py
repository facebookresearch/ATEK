from atek.model.cubercnn import (
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


def setup_callback(args, cfg):
    if args.model_name == "cubercnn":
        callbacks = {
            "iter_postprocess": convert_cubercnn_prediction,
            "iter_callback": [],
            "seq_callback": [save_cubercnn_prediction],
        }
    return callbacks
