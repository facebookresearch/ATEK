# Preprocessing configurations

In ATEK, many common customization in preprocessing can be done through changing the preprocessing configuration file, for example,

- Select a subset of data to include in preprocessing.
- Selectively perform certain image transformations to Aria Camera data, e.g. rotation, rescale, etc.
- Select how data subsampling is done.

Below we list tables of commonly used configurable params in ATEK preprocessing. User can also check out all configurations used in ATEK Data Store [here](../data/atek_data_store_confs/).

## Common configurable parameters

| Name           | Description                                                                                                                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `selected`     | Indicates if the processor data will be included in preprocessing                                                                                                                                                          |
| `tolerance_ns` | Tolerance window to synchronize data, in ns                                                                                                                                                                                |
| `time_domain`  | the time domain for synchronize data, for most of time this should be `DEVICE_TIME`. See [time domain](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/timestamps_in_aria_vrs) for details |

## Class specific configurable parameters

| Class                            | Knob                          | Description                                                                                                              |
| -------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `rgb`, `slam_left`, `slam_right` | `undistort_to_linear_cam`     | If set, undistort to a linear camera model                                                                               |
|                                  | `target_camera_resolution`    | rescale image resolution, e.g., [240, 240]                                                                               |
|                                  | `rescale_antialias`           | If set, perform anti-aliasing during rescaling                                                                           |
|                                  | `rotate_image_cw90deg`        | If set, rotate image by 90 degrees clockwise                                                                             |
| `rgb_depth`                      | `depth_stream_type_id`        | VRS file's type ID for the depth stream, set this to "214" for ASE data                                                  |
|                                  | `depth_stream_id`             | VRS file's stream ID for the depth stream, set this to "345-1" for ADT data                                              |
|                                  | `convert_zdepth_to_dist`      | If set, convert Z-depth to distance                                                                                      |
|                                  | `unit_scaling`                | Scaling unit, e.g., 0.001 to convert from mm to meters                                                                   |
| `obb_gt`                         | `bbox2d_num_samples_on_edge`  | The number of sampled points when applying image transformations to 2D bounding box annotations                          |
| `wds_writer`                     | `prefix_string`               | Prefix string for the writer                                                                                             |
|                                  | `max_samples_per_shard`       | Maximum number of samples per shard                                                                                      |
|                                  | `remove_last_tar_if_not_full` | If true, remove the last tar file if it is not full. This could be useful for load-balancing during multi-node training. |
| `camera_temporal_subsampler`     | `main_camera_target_freq_hz`  | Target frequency in Hz for the main camera used for subsampling data                                                     |
|                                  | `sample_length_in_num_frames` | Number of frames in a sample                                                                                             |
|                                  | `stride_length_in_num_frames` | Number of frames to stride over in a sample                                                                              |
