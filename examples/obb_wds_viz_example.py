# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

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
from detectron2.data import detection_utils
from detectron2.structures import Boxes, BoxMode, Instances
from tqdm import tqdm


# Load Native ATEK WDS data
print("-------------------- loading ATEK data natively --------------- ")
wds_dir = "/home/louy/Calibration_data_link/Atek/2024_07_02_NewGtStructure/wds_output/adt_test_2"
tars = [os.path.join(wds_dir, f"shards-000{i}.tar") for i in range(5)]

dataset = load_atek_wds_dataset(tars, batch_size=None, repeat_flag=False)
test_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,
    num_workers=1,
    pin_memory=True,
)


# Create ATEK visualizer and visualize
from dataclasses import asdict

atek_viz = NativeAtekSampleVisualizer()
for unbatched_sample_dict in test_dataloader:
    # First convert it back to ATEK data sample
    atek_sample = create_atek_data_sample_from_flatten_dict(unbatched_sample_dict)
    atek_viz.plot_atek_sample(atek_sample)
