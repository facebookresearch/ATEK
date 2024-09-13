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


import os
import tempfile
import unittest

import torch
from atek.data_loaders.atek_wds_dataloader import load_atek_wds_dataset
from atek.data_preprocess.atek_wds_writer import AtekWdsWriter
from atek.data_preprocess.sample_builders.obb_sample_builder import ObbSampleBuilder
from atek.data_preprocess.subsampling_lib.temporal_subsampler import (
    CameraTemporalSubsampler,
)
from atek.util.tensor_utils import check_dicts_same_w_tensors

from omegaconf import OmegaConf


# test data paths
TEST_DIR_PATH = os.path.join(os.getenv("TEST_FOLDER"))
CONFIG_PATH = os.getenv("CONFIG_PATH")
CATEGORY_MAPPING_PATH = os.getenv("CATEGORY_MAPPING_PATH")


class LoadAtekWdsDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir_object = tempfile.TemporaryDirectory()
        self.output_wds_path = self.temp_dir_object.name

    def _preprocess_data(self) -> None:
        """
        A helper function to preprocess some ATEK data and write to WDS.
        """
        # From raw Aria dataset to WDS
        conf = OmegaConf.load(CONFIG_PATH)

        sample_builder = ObbSampleBuilder(
            conf=conf.processors,
            vrs_file=os.path.join(TEST_DIR_PATH, "test_ADT_unit_test_sequence.vrs"),
            sequence_name="test",
            mps_files={
                "mps_closedloop_traj_file": os.path.join(
                    TEST_DIR_PATH, "test_ADT_trajectory.csv"
                ),
            },
            gt_files={
                "obb3_file": os.path.join(TEST_DIR_PATH, "test_3d_bounding_box.csv"),
                "obb3_traj_file": os.path.join(
                    TEST_DIR_PATH, "test_3d_bounding_box_traj.csv"
                ),
                "obb2_file": os.path.join(TEST_DIR_PATH, "test_2d_bounding_box.csv"),
                "instance_json_file": os.path.join(
                    TEST_DIR_PATH, "test_instances.json"
                ),
                "category_mapping_file": CATEGORY_MAPPING_PATH,
            },
        )

        subsampler = CameraTemporalSubsampler(
            vrs_file=os.path.join(TEST_DIR_PATH, "test_ADT_unit_test_sequence.vrs"),
            conf=conf.camera_temporal_subsampler,
        )

        atek_wds_writer = AtekWdsWriter(
            output_path=self.output_wds_path,
            conf=conf.wds_writer,
        )

        # will be used for comparison after loading data
        self.expected_flattened_samples = []

        for i in range(subsampler.get_total_num_samples()):
            timestamps_ns = subsampler.get_timestamps_by_sample_index(i)

            for t in timestamps_ns:
                sample = sample_builder.get_sample_by_timestamp_ns(t)
                if sample is not None:
                    atek_wds_writer.add_sample(data_sample=sample)
                    self.expected_flattened_samples.append(sample.to_flatten_dict())
        atek_wds_writer.close()

    def test_atek_native_round_trip(self) -> None:
        # Preprocess data and create WDS files
        self._preprocess_data()

        # check number of tars created
        output_wds_names = os.listdir(self.output_wds_path)
        self.assertEqual(len(output_wds_names), 1)  # only 1 tar file created

        # Load tars back into ATEK
        tar_list = [os.path.join(self.output_wds_path, f) for f in output_wds_names]
        unbatched_dataset = load_atek_wds_dataset(
            tar_list, batch_size=None, repeat_flag=False
        )

        for expected_sample, sample in zip(
            self.expected_flattened_samples, unbatched_dataset
        ):
            # Check GT content
            self.assertTrue(
                check_dicts_same_w_tensors(
                    expected_sample["gt_data"], sample["gt_data"], atol=1e-3
                )
            )

            # Check sequence name
            self.assertEqual(expected_sample["sequence_name"], sample["sequence_name"])

            # Check other data content, note that expected_sample's keys have file extensions, like "gtdata.json",
            # while ATEK loaded samples won't have extensions.
            for key_wo_extension, expected_val in expected_sample.items():
                # skip certain fields
                if "image" in key_wo_extension or "gt_data" in key_wo_extension:
                    continue

                self.assertTrue(key_wo_extension in sample)
                sample_val = sample[key_wo_extension]

                if isinstance(expected_val, torch.Tensor):
                    self.assertTrue(
                        torch.allclose(
                            expected_val.to(torch.float32),
                            sample_val.to(torch.float32),
                            atol=1e-3,
                        )
                    )
                else:
                    self.assertEqual(expected_val, sample_val)

    def test_atek_default_collation(self) -> None:
        # Preprocess data and create WDS files
        self._preprocess_data()

        # check number of tars created
        output_wds_names = os.listdir(self.output_wds_path)
        self.assertEqual(len(output_wds_names), 1)  # only 1 tar file created

        # Load tars back into ATEK
        tar_list = [os.path.join(self.output_wds_path, f) for f in output_wds_names]
        batch_size = 2  # TODO: maybe support a unit test with a larger dataset
        batched_dataset = load_atek_wds_dataset(
            tar_list, batch_size=batch_size, repeat_flag=False
        )

        # Check batched data tensor shape
        batched_sample = next(iter(batched_dataset))
        for key, val in batched_sample.items():
            if key in ["__key__", "__url__"]:
                continue

            if isinstance(val, torch.Tensor):
                self.assertEqual(val.shape[0], batch_size)
            elif isinstance(val, list):
                self.assertEqual(len(val), batch_size)

    # This is like a DTOR
    def tearDown(self):
        # Explicitly cleanup the temporary directory
        self.temp_dir_object.cleanup()
        print("Temporary directory deleted")
