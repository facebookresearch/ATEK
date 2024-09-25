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

from atek.data_preprocess.atek_data_sample import AtekDataSample
from atek.data_preprocess.atek_wds_writer import AtekWdsWriter
from atek.viz.atek_visualizer import NativeAtekSampleVisualizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GeneralAtekPreprocessor:
    """
    A base class that defines a high-level interface for preprocessing ATEK data.
    It consists of a subsampler to control subsampling, a sample builder to build samples, a WDS writer to write samples to disk, and a visualizer to visualize the results.
    """

    def __init__(
        self,
        sample_builder,  # Should be Union[ObbSampleBuilder, EfmSampleBuidler], but intentionally not specifying the type here for extensibility
        subsampler,  # Intentionally not specifying the type here for extensibility
        atek_wds_writer: Optional[AtekWdsWriter],
        atek_visualizer: Optional[NativeAtekSampleVisualizer],
    ) -> None:
        """
        init function
        """
        self.subsampler = subsampler
        self.sample_builder = sample_builder
        self.atek_wds_writer = atek_wds_writer
        self.atek_visualizer = atek_visualizer

        # TODO: maybe perform an API check for subsampler and sample_builder

    def __getitem__(self, index) -> Optional[AtekDataSample]:
        """
        API to get an AtekDataSample by index
        """
        timestamps_ns = self.subsampler.get_timestamps_by_sample_index(index)
        return self.sample_builder.get_sample_by_timestamps_ns(timestamps_ns)

    def process_all_samples(
        self, write_to_wds_flag: bool = True, viz_flag: bool = False
    ) -> int:
        """
        API to process all samples, and (optionally) write them to WDS and visualize them.
        Return the total number of valid samples being processed.
        """
        # Check if the WDS writer and visualizer are initialized
        if write_to_wds_flag:
            assert (
                self.atek_wds_writer is not None
            ), "AtekWdsWriter is not initialized, cannot write to WDS"
        if viz_flag:
            assert (
                self.atek_visualizer is not None
            ), "AtekVisualizer is not initialized, cannot visualize samples"

        # Loop over all samples, check for validity, and write them to WDS and visualize them if specified
        num_samples = 0
        for i in range(self.subsampler.get_total_num_samples()):
            sample = self.__getitem__(i)
            if sample is not None:
                num_samples += 1
                if write_to_wds_flag:
                    self.atek_wds_writer.add_sample(sample)
                if viz_flag:
                    self.atek_visualizer.plot_atek_sample(sample)

        if write_to_wds_flag:
            self.atek_wds_writer.close()
        if viz_flag:
            self.atek_visualizer.save_viz()

        logger.info(f"ATEK has processed {num_samples} valid samples in total.")
        return num_samples
