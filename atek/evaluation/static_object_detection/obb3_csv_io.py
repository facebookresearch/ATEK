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

# pyre-unsafe

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import fsspec
import numpy as np
import pandas as pd
import torch
from atek.data_loaders.cubercnn_model_adaptor import CubeRCNNModelAdaptor
from atek.util.tensor_utils import compute_bbox_corners_in_world
from projectaria_tools.core.sophus import SE3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ATEK_OBB3_CSV_HEADER_STR: str = (
    "time_ns,tx_world_object,ty_world_object,tz_world_object,qw_world_object,qx_world_object,qy_world_object,qz_world_object,scale_x,scale_y,scale_z,name,instance,sem_id,prob"
)


class AtekObb3CsvWriter:
    """
    A class to write Obb3 dict to a csv file.
    The csv file will have the following columns:
        "time_ns,tx_world_object,ty_world_object,tz_world_object,qw_world_object,qx_world_object,qy_world_object,qz_world_object,scale_x,scale_y,scale_z,name,instance,sem_id,prob"
    """

    def __init__(self, output_filename: str) -> None:
        if not output_filename:
            output_filename = "/tmp/obbs.csv"

        logger.info(f"starting writing obb3 to {output_filename}")
        self.output_filename = output_filename
        self.file_writer = fsspec.open(self.output_filename, "w").open()

        csv_headers = ATEK_OBB3_CSV_HEADER_STR.split(",")
        csv_header_row = ",".join(csv_headers)
        self.file_writer.write(csv_header_row + "\n")

    def write_from_atek_dict(
        self,
        atek_dict: Dict,
        confidence_score: Optional[torch.Tensor] = None,
        timestamp_ns: int = -1,
        flush_at_end: bool = True,
    ) -> None:
        """
        write a single row to the csv file, from an ATEK obb3 gt-format dict. See obb3_gt_processor for the format of the dict.
        """
        num_obbs = len(atek_dict["category_ids"])
        if num_obbs == 0:
            logger.warn(f"no obbs for timestamp {timestamp_ns}")
            return

        for i_obb in range(num_obbs):
            obb_category_name = atek_dict["category_names"][i_obb]
            obb_category_id = atek_dict["category_ids"][i_obb]
            obb_dims = atek_dict["object_dimensions"][i_obb]
            obb_dims_str = ",".join(obb_dims.numpy().astype(str))

            # instance id is optional
            if atek_dict["instance_ids"] is not None:
                obb_instance_id = atek_dict["instance_ids"][i_obb]
            else:
                obb_instance_id = -1

            # confidence score is optional, gt data will have a default value of 1.0
            if confidence_score is None:
                obb_confidence = 1.0
            else:
                obb_confidence = confidence_score[i_obb]

            T_world_obj = SE3.from_matrix3x4(atek_dict["ts_world_object"][i_obb])
            quat_and_translation = (
                T_world_obj.to_quat_and_translation().squeeze()
            )  # qw,qx,qy,qz,tx,ty,tz
            translation_xyz_str = ",".join(quat_and_translation[4:].astype(str))
            quat_wxyz_str = ",".join(quat_and_translation[:4].astype(str))

            self.file_writer.write(
                f"{timestamp_ns},{translation_xyz_str},{quat_wxyz_str},"
                f"{obb_dims_str},{obb_category_name},{obb_instance_id},{obb_category_id},{obb_confidence}\n"
            )

        if flush_at_end:
            self.file_writer.flush()

    def write_from_cubercnn_dict(
        self,
        cubercnn_dict: Dict,
        timestamp_ns: int = -1,
        flush_at_end: bool = True,
    ) -> None:
        """
        write a single row to the csv file, from a cubercnn-format dict.
        """
        # Convert to ATEK format
        atek_format_gt_dict = CubeRCNNModelAdaptor.cubercnn_gt_to_atek_gt(
            cubercnn_dict=cubercnn_dict,
            T_world_camera_np=cubercnn_dict["T_world_camera"],
            camera_label="camera-rgb",
        )

        # write to csv
        self.write_from_atek_dict(
            atek_dict=atek_format_gt_dict["obb3_gt"][
                "camera-rgb"
            ],  # cubercnn dict has only one camera
            confidence_score=atek_format_gt_dict["scores"],
            timestamp_ns=timestamp_ns,
            flush_at_end=flush_at_end,
        )

    # functions for cleaning up
    def flush(self) -> None:
        self.file_writer.flush()

    def __del__(self) -> None:
        if hasattr(self, "file_writer"):
            self.file_writer.close()


class AtekObb3CsvReader:
    """
    A class to read an eval Obb3 csv file back into a dict, in the format of {timestamp: Obb3GtDict}
    """

    def __init__(self, input_filename: str):
        logger.info(f"starting loading evaluation obb3s from {input_filename}")
        self.input_filename = input_filename

    def read_as_obb_dict(
        self,
    ) -> Dict:
        """
        Read the entire csv file into an ATEK obb3 gt-format dict. See obb3_gt_processor for the format of the dict.
        """
        # Load the CSV file as a DataFrame
        df = pd.read_csv(self.input_filename)

        # Initialize result dict
        obb3_dict_by_timestamp_ns = {}

        # Group by 'time_ns' which represents the timestamp
        for timestamp, obb_group in df.groupby("time_ns"):
            # Extract obb3 fields into tensors
            instance_ids = torch.tensor(obb_group["instance"].values, dtype=torch.int64)
            category_names = obb_group["name"].tolist()
            category_ids = torch.tensor(obb_group["sem_id"].values, dtype=torch.int64)
            object_dimensions = torch.tensor(
                obb_group[["scale_x", "scale_y", "scale_z"]].values, dtype=torch.float32
            )

            # Extract T_world_object poses from csv file
            Ts_world_object = self._extract_poses_from_data_frame(obb_group)

            # Compute a new field "bbox_corners in world", which is widely used in eval
            bbox_corners_in_world = compute_bbox_corners_in_world(
                object_dimensions, Ts_world_object
            )

            # Extract confidence score from csv file
            confidence_scores = torch.tensor(
                obb_group["prob"].values, dtype=torch.float32
            )

            # Store in dictionary
            single_timestamp_dict = {
                "instance_ids": instance_ids,
                "category_names": category_names,
                "category_ids": category_ids,
                "object_dimensions": object_dimensions,
                "ts_world_object": Ts_world_object,
                "bbox_corners_in_world": bbox_corners_in_world,  # (num_obbs, 8, 3)
                "confidence_scores": confidence_scores,
            }
            obb3_dict_by_timestamp_ns[timestamp] = single_timestamp_dict

        return obb3_dict_by_timestamp_ns

    def _extract_poses_from_data_frame(self, data_frame: pd.DataFrame) -> torch.Tensor:
        """
        Extract poses from a dataframe with
        'tx_world_object', 'ty_world_object', 'tz_world_object', 'qw_world_object', 'qx_world_object', 'qy_world_object', 'qz_world_object',
        into a tensor of shape (num_obbs, 3, 4)
        """
        translations = data_frame[
            ["tx_world_object", "ty_world_object", "tz_world_object"]
        ].values  # (num_obbs, 3)
        quats_wxyz = data_frame[
            ["qw_world_object", "qx_world_object", "qy_world_object", "qz_world_object"]
        ].values  # (num_obbs, 4)
        num_obbs = quats_wxyz.shape[0]
        T_world_object_list = []
        for i_obb in range(num_obbs):
            T_world_object = SE3.from_quat_and_translation(
                quats_wxyz[i_obb, 0].item(),
                quats_wxyz[i_obb, 1:],
                translations[i_obb],
            )
            T_world_object_list.append(
                torch.tensor(T_world_object.to_matrix3x4(), dtype=torch.float32)
            )

        Ts_world_object = torch.stack(T_world_object_list, dim=0)
        return Ts_world_object


class GroupAtekObb3CsvWriter:
    """
    A class to write obb3 dict to csv files into proper folder structure.
    It will write each sequence's data into their corresponding csv file
    """

    def __init__(self, output_folder: str, output_filename) -> None:
        logger.info(f"starting writing obb3 to {output_filename}")
        self.output_folder = output_folder
        self.output_filename = output_filename
        self.per_sequence_writers = {}

    def write_from_atek_dict(
        self,
        atek_dict: Dict,
        sequence_name: str,
        confidence_score: Optional[torch.Tensor] = None,
        timestamp_ns: int = -1,
        flush_at_end: bool = True,
    ) -> None:
        # Create a new writer for the current sequence if it doesn't exist
        if sequence_name not in self.per_sequence_writers:
            self.per_sequence_writers[sequence_name] = AtekObb3CsvWriter(
                output_filename=os.path.join(
                    self.output_folder, sequence_name, self.output_filename
                )
            )

        # Write the data to the current sequence's writer
        self.per_sequence_writers[sequence_name].write_from_atek_dict(
            atek_dict=atek_dict,
            confidence_score=confidence_score,
            timestamp_ns=timestamp_ns,
            flush_at_end=flush_at_end,
        )

    # functions for cleaning up
    def flush(self) -> None:
        for writer in self.per_sequence_writers.values():
            writer.flush()
