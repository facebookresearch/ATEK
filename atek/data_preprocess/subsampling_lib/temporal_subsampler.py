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

from typing import List

from omegaconf.omegaconf import DictConfig
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain


class CameraTemporalSubsampler:
    """
    A subsampler class that subsamples the main camera stream to a target frequency.
    the subsampling is done by taking every Nth frame, where N is the subsampling factor.
    The target frequency must be dividable by the actual frequency of the main camera stream.
    Returns the timestamp of the i-th sample in the main camera stream.

    TODO: may expand this class to return multiple timestamps per sample.
    """

    def __init__(self, vrs_file, conf: DictConfig) -> None:
        """
        Args:
            vrs_file: the path to the vrs file
            conf: contains `main_camera_label`, `sample_target_freq_hz`, `time_domain`, `skip_begin_seconds`, and `skip_end_seconds`
        """

        self.conf = conf

        vrs_provider = data_provider.create_vrs_data_provider(vrs_file)
        assert vrs_provider is not None, f"Cannot open {vrs_file}"

        # get timestamps associated with main camera
        main_stream_id = vrs_provider.get_stream_id_from_label(conf.main_camera_label)
        assert (
            main_stream_id is not None
        ), f"Cannot find stream id for {conf.main_camera_label} in {vrs_file}"
        time_domain = getattr(TimeDomain, conf.time_domain)
        main_camera_timestamps = vrs_provider.get_timestamps_ns(
            main_stream_id, time_domain
        )

        # determine subfactor for the main camera, and uniformly subsample the main camera stream
        freq_in_vrs: float = vrs_provider.get_nominal_rate_hz(main_stream_id)
        subsampling_factor: int = self._compute_subsampling_factor(
            int(freq_in_vrs), int(conf.main_camera_target_freq_hz)
        )

        # If specified, skip first and last few seconds of recording
        begin_timestamp_id = 0
        if "skip_begin_seconds" in conf and conf.skip_begin_seconds > 0:
            begin_timestamp_id = int(conf.skip_begin_seconds * freq_in_vrs)
        end_timestamp_id = len(main_camera_timestamps)
        if "skip_end_seconds" in conf and conf.skip_end_seconds > 0:
            end_timestamp_id = len(main_camera_timestamps) - int(
                conf.skip_end_seconds * freq_in_vrs
            )
        self.subsampled_timestamps = main_camera_timestamps[
            begin_timestamp_id:end_timestamp_id:subsampling_factor
        ]

        # determine the total number of samples, where each sample may contain multiple sub-sampled frames.
        total_num_cam_frames = len(self.subsampled_timestamps)
        self.sample_length = conf.sample_length_in_num_frames
        self.stride = conf.stride_length_in_num_frames
        self.total_num_samples: int = (
            total_num_cam_frames - self.sample_length
        ) // self.stride + 1

    def _compute_subsampling_factor(self, freq_in_vrs: int, target_freq: int) -> int:
        if freq_in_vrs % target_freq != 0:
            raise ValueError(
                f"Cannot subsample {freq_in_vrs} to {target_freq} Hz, needs to be dividable."
            )
        return freq_in_vrs // target_freq

    def get_total_num_samples(self) -> int:
        """
        return the total number of samples in `target_freq_hz`.
        """
        return self.total_num_samples

    def get_timestamps_by_sample_index(self, sample_index: int) -> List[int]:
        """
        return the timestamps as a list of int corresponding to the sample, given the sample index (not sensor data index).
        """
        if sample_index >= self.total_num_samples or sample_index < 0:
            raise ValueError(
                f"sample_index {sample_index} is out of range, total number of samples under target freq is {self.total_num_samples}"
            )

        return self.subsampled_timestamps[
            sample_index * self.stride : sample_index * self.stride + self.sample_length
        ]
