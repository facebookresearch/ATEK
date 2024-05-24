# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from omegaconf.omegaconf import DictConfig, OmegaConf
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
            conf: contains `main_camera_label`, `sample_target_freq_hz`, and `time_domain`.
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
        self.main_camera_timestamps = vrs_provider.get_timestamps_ns(
            main_stream_id, time_domain
        )

        # determine subfactor for the main camera
        freq_in_vrs = vrs_provider.get_nominal_rate_hz(main_stream_id)
        self.subsampling_factor = self.compute_subsampling_factor(
            int(freq_in_vrs), int(conf.sample_target_freq_hz)
        )
        self.total_num_samples: int = (
            len(self.main_camera_timestamps) // self.subsampling_factor
        )

    def compute_subsampling_factor(self, freq_in_vrs: int, target_freq: int) -> int:
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

    def get_timestamp_by_sample_index(self, sample_index: int) -> int:
        """
        return the timestamp corresponding to the sample, given the sample index (not sensor data index)
        """
        if sample_index >= self.total_num_samples:
            raise ValueError(
                f"sample_index {sample_index} is out of range, total number of samples under target freq is {self.total_num_samples}"
            )

        return self.main_camera_timestamps[sample_index * self.subsampling_factor]
