import faulthandler

import logging
import os
from logging import StreamHandler

from atek.data_preprocess.atek_wds_writer import AtekWdsWriter

from atek.data_preprocess.sample_builders.obb_sample_builder import ObbSampleBuilder
from atek.data_preprocess.subsampling_lib.temporal_subsampler import (
    CameraTemporalSubsampler,
)

from omegaconf import OmegaConf

faulthandler.enable()

handler = StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)

example_adt_data_dir = "/home/louy/Calibration_data_link/Atek/2024_05_07_EfmDataTest/adt_data_example/Apartment_release_clean_seq134/1WM103600M1292_optitrack_release_clean_seq134"
example_ase_data_dir = "/home/louy/Calibration_data_link/Atek/2024_05_07_EfmDataTest/ase_data_example/euston_simulation_100077_device0"
adt_config_path = os.path.join(
    "/home/louy/Calibration_data_link/Atek/2024_05_28_CubeRcnnTest/cubercnn_preprocess_adt_config.yaml"
)
adt_to_atek_category_mapping_file = (
    "/home/louy/atek_on_fbsource/data/adt_prototype_to_atek.csv"
)
output_wds_path = (
    "/home/louy/Calibration_data_link/Atek/2024_08_05_DryRun/wds_output/adt_test"
)

conf = OmegaConf.load(adt_config_path)
sequence_name = example_adt_data_dir.split("/")[-1]

sample_builder = ObbSampleBuilder(
    conf=conf.processors,
    vrs_file=os.path.join(example_adt_data_dir, "video.vrs"),
    sequence_name=sequence_name,
    mps_files={
        "mps_closedloop_traj_file": os.path.join(
            example_adt_data_dir, "aria_trajectory.csv"
        ),
    },
    gt_files={
        "obb3_file": os.path.join(example_adt_data_dir, "3d_bounding_box.csv"),
        "obb3_traj_file": os.path.join(example_adt_data_dir, "scene_objects.csv"),
        "obb2_file": os.path.join(example_adt_data_dir, "2d_bounding_box.csv"),
        "instance_json_file": os.path.join(example_adt_data_dir, "instances.json"),
        "category_mapping_file": adt_to_atek_category_mapping_file,
    },
)

subsampler = CameraTemporalSubsampler(
    vrs_file=os.path.join(example_adt_data_dir, "video.vrs"),
    conf=conf.camera_temporal_subsampler,
)

atek_wds_writer = AtekWdsWriter(
    output_path=output_wds_path,
    conf=conf.wds_writer,
)

for i in range(subsampler.get_total_num_samples()):
    timestamps_ns = subsampler.get_timestamps_by_sample_index(i)

    for t in timestamps_ns:
        sample = sample_builder.get_sample_by_timestamp_ns(t)
        if sample is not None:
            atek_wds_writer.add_sample(data_sample=sample)

atek_wds_writer.close()
