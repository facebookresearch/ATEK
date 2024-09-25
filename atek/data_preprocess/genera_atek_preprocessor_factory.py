# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# pyre-strict

import logging
from typing import Optional

from atek.data_preprocess.atek_wds_writer import AtekWdsWriter
from atek.data_preprocess.general_atek_preprocessor import GeneralAtekPreprocessor
from atek.data_preprocess.sample_builders.atek_data_paths_provider import (
    AtekDataPathsProvider,
)
from atek.data_preprocess.sample_builders.efm_sample_builder import EfmSampleBuilder
from atek.data_preprocess.sample_builders.obb_sample_builder import ObbSampleBuilder
from atek.data_preprocess.subsampling_lib.temporal_subsampler import (
    CameraTemporalSubsampler,
)
from atek.viz.atek_visualizer import NativeAtekSampleVisualizer
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _create_cubercnn_type_preprocessor(
    conf: DictConfig,
    raw_data_folder: str,
    sequence_name: str,
    output_wds_folder: Optional[str] = None,
    output_viz_file: Optional[str] = None,
    category_mapping_file: Optional[str] = None,
) -> GeneralAtekPreprocessor:
    # Get data paths
    data_path_provider = AtekDataPathsProvider(data_root_path=raw_data_folder)
    atek_data_paths = data_path_provider.get_data_paths()

    # Create Sample Builder
    # TODO: refactor to let the sample builder take the data paths object as an input
    sample_builder = ObbSampleBuilder(
        conf=conf.processors,
        vrs_file=atek_data_paths["video_vrs_file"],
        sequence_name=sequence_name,
        mps_files={
            "mps_closedloop_traj_file": atek_data_paths["mps_closedloop_traj_file"],
        },
        gt_files={
            "obb3_file": atek_data_paths["gt_obb3_file"],
            "obb3_traj_file": atek_data_paths["gt_obb3_traj_file"],
            "obb2_file": atek_data_paths["gt_obb2_file"],
            "instance_json_file": atek_data_paths["gt_instance_json_file"],
            "category_mapping_file": category_mapping_file,
        },
    )

    # Create temporal subsampler
    subsampler = CameraTemporalSubsampler(
        vrs_file=atek_data_paths["video_vrs_file"],
        conf=conf.camera_temporal_subsampler,
    )

    # Create WDS writer
    if "wds_writer" in conf and output_wds_folder is not None:
        atek_wds_writer = AtekWdsWriter(
            output_path=output_wds_folder,
            conf=conf.wds_writer,
        )
    else:
        atek_wds_writer = None

    # Create visualizer
    if "visualizer" in conf:
        atek_visualizer = NativeAtekSampleVisualizer(
            conf=conf.visualizer, output_viz_file=output_viz_file
        )
    else:
        atek_visualizer = None

    return GeneralAtekPreprocessor(
        sample_builder=sample_builder,
        subsampler=subsampler,
        atek_wds_writer=atek_wds_writer,
        atek_visualizer=atek_visualizer,
    )


def _create_efm_type_preprocessor(
    conf: DictConfig,
    raw_data_folder: str,
    sequence_name: str,
    output_wds_folder: Optional[str] = None,
    output_viz_file: Optional[str] = None,
    category_mapping_file: Optional[str] = None,
) -> GeneralAtekPreprocessor:
    # Get data paths
    data_path_provider = AtekDataPathsProvider(data_root_path=raw_data_folder)
    atek_data_paths = data_path_provider.get_data_paths()

    # Create Sample Builder
    # TODO: refactor to let the sample builder take the data paths object as an input
    depth_vrs_file = (
        atek_data_paths["depth_vrs_file"]
        if "depth_vrs_file" in atek_data_paths
        else None
    )
    sample_builder = EfmSampleBuilder(
        conf=conf.processors,
        sequence_name=sequence_name,
        vrs_file=atek_data_paths["video_vrs_file"],
        mps_files={
            "mps_closedloop_traj_file": atek_data_paths["mps_closedloop_traj_file"],
            "mps_semidense_points_file": atek_data_paths["mps_semidense_points_file"],
            "mps_semidense_observations_file": atek_data_paths[
                "mps_semidense_observations_file"
            ],
        },
        gt_files={
            "obb3_file": atek_data_paths["gt_obb3_file"],
            "obb3_traj_file": atek_data_paths["gt_obb3_traj_file"],
            "obb2_file": atek_data_paths["gt_obb2_file"],
            "instance_json_file": atek_data_paths["gt_instance_json_file"],
            "category_mapping_file": category_mapping_file,
        },
        depth_vrs_file=depth_vrs_file,
    )

    # Create temporal subsampler
    subsampler = CameraTemporalSubsampler(
        vrs_file=atek_data_paths["video_vrs_file"],
        conf=conf.camera_temporal_subsampler,
    )

    # Create WDS writer
    if "wds_writer" in conf and output_wds_folder is not None:
        atek_wds_writer = AtekWdsWriter(
            output_path=output_wds_folder,
            conf=conf.wds_writer,
        )
    else:
        atek_wds_writer = None

    # Create visualizer
    if "visualizer" in conf:
        atek_visualizer = NativeAtekSampleVisualizer(
            conf=conf.visualizer, output_viz_file=output_viz_file
        )
    else:
        atek_visualizer = None

    return GeneralAtekPreprocessor(
        sample_builder=sample_builder,
        subsampler=subsampler,
        atek_wds_writer=atek_wds_writer,
        atek_visualizer=atek_visualizer,
    )


def create_general_atek_preprocessor_from_conf(
    conf: DictConfig,
    raw_data_folder: str,
    sequence_name: str,
    output_wds_folder: Optional[str],
    output_viz_file=Optional[str],
    category_mapping_file: Optional[
        str
    ] = None,  # Optional object-detection category mapping file
) -> GeneralAtekPreprocessor:
    """
    A factory method to create a GeneralAtekPreprocessor from a Omega config object. The `atek_config_name` field in the config determines which ATEK config will be used in preprocessing
    """
    # CubeRCNN (or obb) flavor
    if conf.atek_config_name in ["cubercnn", "cubercnn_eval"]:
        return _create_cubercnn_type_preprocessor(
            conf=conf,
            raw_data_folder=raw_data_folder,
            sequence_name=sequence_name,
            output_wds_folder=output_wds_folder,
            output_viz_file=output_viz_file,
            category_mapping_file=category_mapping_file,
        )
    # EFM flavor
    if conf.atek_config_name in ["efm", "efm_eval"]:
        return _create_efm_type_preprocessor(
            conf=conf,
            raw_data_folder=raw_data_folder,
            sequence_name=sequence_name,
            output_wds_folder=output_wds_folder,
            output_viz_file=output_viz_file,
            category_mapping_file=category_mapping_file,
        )
