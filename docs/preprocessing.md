# Data preprocessing

## Introduction

The goal of data pre-processing is to convert "raw" Aria datasets, to PyTorch ready format data. The former consists Aria recording file in [VRS](https://www.projectaria.com/datasets/adt/) format, Aria [Machine Perception Service](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_summary) files, and dataset-specific annotation files, e.g. bounding box annotations. The later is often consists of PyTorch-compatible training samples, i.e. Dicts of Tensors, str, List, etc.

Performing such conversion often requires the following three steps:

1. Load VRS + MPS + annotation data.
2. Perform selected data processing operations, e.g. image rotation, rescaling, undistortion, etc.
3. Assemble data into individual training samples that can be loaded by PyTorch.

Before ATEK, users will need to hand-craft all these code on their own, which is non-trivial and prone to errors. With ATEK, users can first choose to download pre-processed data from ATEK Data Store thereby completely skip this step, or they can build their own processing workflow in two ways:

- Simple customization using a yaml config file.
- Advanced customization through customized code.

## Simple customization through preprocessing config

ATEK allows user to **customize the preprocessing workflow by simply modifying the preprocessing configuration yaml file** (see [preprocessing_configurations.md](./preprocessing_configurations.md) for details).

The following is the core code to load an open Aria data sequence, preprocess according to a given configuration file, and write the preprocessed results to disk as WebDataset ([full example](../examples/Demo_1_data_preprocessing.ipynb)). We also use a visualization library based on `ReRun` to visualize the preprocessed results. The results are stored as `Dict` in memory containing tensors, strings, and sub-dicts, and also saved to local disk in WebDataset (WDS) format for further use.

```python
from omegaconf import OmegaConf
config = OmegaConf.load('path/to/config.yaml')
preprocessor = create_general_atek_preprocessor_from_conf(
    conf=config,
    raw_data_folder="/path/to/raw/data",
    sequence_name="sequence_01",
)

num_samples = preprocessor.process_all_samples(write_to_wds_flag = True, viz_flag = True)
```

### `create_general_atek_preprocessor_from_conf`

This is a factory method that initializes a `GeneralAtekPreprocessor` based on a configuration object. It selects the appropriate preprocessor configuration for ATEK using the `atek_config_name` field in the provided Omega configuration. See [here](./preprocessing_configurations.md) for currently supported configs.

#### Parameters

- **conf** (`DictConfig`): Configuration object with preprocessing settings. The `atek_config_name` key specifies the preprocessor type,
- **raw_data_folder** (`str`): Path to the folder with raw data files.
- **sequence_name** (`str`): Name of the data sequence to process.
- **output_wds_folder** (`Optional[str]`): Path for saving preprocessed data in WebDataset (WDS) format. If `None`, data is not saved in WDS format.
- **output_viz_file** (`Optional[str]`): File path for saving visualization outputs. If `None`, no visualizations are generated.
- **category_mapping_file** (`Optional[str]`): Optional file path for object-detection category mappings.

#### Returns

- **GeneralAtekPreprocessor**: Configured instance of `GeneralAtekPreprocessor`.

### `GeneralAtekPreprocessor` Class

This is a high-level class that performs ATEK data preprocessing, including subsampling, sample building, WDS writing, and visualization. Please call `create_general_atek_preprocessor_from_conf` factory method to create this class instances.

#### Methods

- `__getitem__(self, index) -> Optional[AtekDataSample]` : Retrieves a `AtekDataSample` by index.
- `process_all_samples(self, write_to_wds_flag=True, viz_flag=False) -> int`: Processes all samples, with options to write to WDS and visualize. Returns the total number of valid samples processed.

## Advanced customization

For more customized preprocessing requirements, users are also free to fork any components in ATEK preprocessing library to implement their own features. Here we give an overview of the library structure.

### [`AtekDataSample`](../atek/data_preprocess/atek_data_sample.py)

This `dataclass` defines the data structure that stores a "training sample" in ATEK.

- It is flexible that all data are represented by PyTorch tensors, while each tensor's shape can be different for different applications.
- Annotation data is represented by `gt_data: Dict`, which provides full flexibility for the user to adapt to different ML tasks.
- This class comes with free converter function to convert to / from a flattened Python dictionary.

### [`processors`](../atek/data_preprocess/processors/)

In ATEK, each type of sensor, MPS, and annotation data are handled by a separate `processor` class. For example, `aria_camera_processor` handles camera image+calibration data, `mps_traj_processor` handles trajectory data from MPS, `obb2_gt_processor` handles 2D bounding box annotations data (in ADT format). Users can freely create their own processor class to handle new data that they want to add, and all these processors can be mixed-and-matched in the `sample_builder` class. Currently, we have the following processors in ATEK library:

Processor Name            | Description
:------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
`AriaCameraProcessor`     | Processes Aria SLAM + RGB camera data + calibration.
`MpsTrajProcessor`        | Processes the device trajectory data from Machine Perception Services.
`MpsSemidenseProcessor`   | Processes the semidense points data from Machine Perception Services.
`MpsOnlineCalibProcessor` | Processes camera online calibration data from Machine Perception Services.
`DepthImageProcessor`     | Processes depth data associated with Aria camera. This depth data is either generated from simulation (in ASE dataset), or computed from groundtruth annotations (in ADT dataset).
`Obb2GtProcessor`         | Processes object 2D bounding box annotation data.
`Obb3GtProcessor`         | Processes object 3D bounding box annotation data.
`EfmGtProcessor`          | Processes object 3D bounding box annotation data, specifically for the [EFM model](https://github.com/facebookresearch/efm3d).

### [`sample_builders`](../atek/data_preprocess/sample_builders/)

These classes defines how different processor's data are assembled into a `AtekDataSample`. The library contains 2 example sample builders:

SampleBuilder Name | Description
------------------ | -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
`ObbSampleBuilder` | This is a simple builder that aggregates data into single frames, and is used by the `cubercnn` config in ATEK Data Store, and all our `CubeRCNN` examples.
`EfmSampleBuilder` | This is a slightly more complicated sample builder that is used by the EFM paper, where multiple frames of data are aggregated into the same sample, and it also includes depth data and semidense point cloud data. It is used by the `efm` config in ATEK Data Store.

### [`subsampling_lib`](../atek/data_preprocess/subsampling_lib/)

This lib contains classes to subsample the sequence data. Currently we support [temporal subsampling according to a "main" camera](../atek/data_preprocess/subsampling_lib/temporal_subsampler.py).
