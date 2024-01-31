from atek.dataset.cubercnn_data import build_cubercnn_dataset


def build_dataset(data_path, args, cfg):
    if args.model_name == "cubercnn":
        dataset = build_cubercnn_dataset(data_path)
    else:
        raise ValueError(f"Unknown model name for building dataset: {args.model_name}")
    return dataset
