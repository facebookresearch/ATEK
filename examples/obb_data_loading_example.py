import os

import numpy as np

import rerun as rr
import torch

from atek.data_loaders.cubercnn_model_adaptor import (
    load_atek_wds_dataset,
    load_atek_wds_dataset_as_cubercnn,
)

from atek.data_preprocess.atek_data_sample import (
    create_atek_data_sample_from_flatten_dict,
)
from atek.viz.atek_visualizer import NativeAtekSampleVisualizer

from tqdm import tqdm

wds_dir = "/home/louy/Calibration_data_link/Atek/2024_08_05_DryRun/wds_output/adt_test"

# Load Native ATEK WDS data
print("-------------------- loading ATEK data natively --------------- ")
tars = [os.path.join(wds_dir, f"shards-000{i}.tar") for i in range(5)]

dataset = load_atek_wds_dataset(
    tars, batch_size=None, repeat_flag=False, shuffle_flag=False
)
test_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,
    num_workers=1,
    pin_memory=True,
)

# Visualize the WDS dataset
atek_viz = NativeAtekSampleVisualizer()
first_sample = None
for unbatched_sample_dict in test_dataloader:
    # First convert it back to ATEK data sample
    atek_sample = create_atek_data_sample_from_flatten_dict(unbatched_sample_dict)
    atek_viz.plot_atek_sample(atek_sample)

    # log first sample to print information
    if first_sample is None:
        first_sample = unbatched_sample_dict

# Print content in a single atek data sample
print("Unbatched atek data sample is a Dict, containing the following: ")
for key, val in first_sample.items():
    print(f"{key}: is a {type(val)}")
    if isinstance(val, torch.Tensor):
        print(f"    with shape of : {val.shape}")
    elif isinstance(val, list):
        print(f"    with len of : {len(val)}")
    elif isinstance(val, str):
        print(f"    value is {val}")
    else:
        pass

# Load ATEK WDS data as CubeRCNN data
print(
    "-------------------- ATEK WDS data can also be loaded as CubeRCNN --------------- "
)
dataset_2 = load_atek_wds_dataset_as_cubercnn(tars, batch_size=8, repeat_flag=False)
test_dataloader_2 = torch.utils.data.DataLoader(
    dataset_2,
    batch_size=None,
    num_workers=1,
    pin_memory=True,
)

cubercnn_sample_list = next(iter(test_dataloader_2))

for key, val in cubercnn_sample_list[0].items():
    print(f"{key}: is a {type(val)}")
    if isinstance(val, torch.Tensor):
        print(f"    with shape of : {val.shape}")
    elif isinstance(val, list):
        print(f"    with len of : {len(val)}")
    elif isinstance(val, str):
        print(f"    value is {val}")
    else:
        pass
