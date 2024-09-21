# Aria Training and Evaluation toolkit (ATEK)

Today, we’re introducing ATEK, an e2e framework for training and evaluating deep learning models on [Aria](https://www.projectaria.com/) data, for both 3D egocentric-specific and general machine perception tasks.


For full documentation, you can navitagete to:

- [Installation](docs/INSTALL.md)
- [Quick start](#quick-start)
- [Machine learning tasks supported by ATEK](docs/ml_tasks.md)
- [ATEK Data Store](docs/atek_data_store.md)
- [Core code snippets](docs/core_code_snippets.md)
- [Technical specifications](docs/technical_specifications.md)

<a id="quick-start"></a>
## Colab notebook
Data preprocessing, inference and evaluation example

[![Aria VRS Data Provider](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/ATEK/tree/main/examples/ATEK_CoLab_Notebook.ipynb)

## Quick start (TODO： add pypi installation)

1. Install miniconda3

Follow [instructions](https://docs.anaconda.com/free/miniconda/) to install miniconda3 and re-open a new terminal after install for this to take effect

2. Install mamba and initialize.

Mamba is a python venv tooling that is similar to Conda, but supposed to handle dependencies more elegantly.

```bash
export PATH=/home/$USER/miniconda3/bin:$PATH
conda init bash
conda install mamba -n base -c conda-forge
mamba init
```

3. clone ATEK code:

```
git clone https://github.com/facebookresearch/ATEK.git
cd ATEK
```

4. Create an ATEK environment and Install ATEK lib:

```bash
mamba env create -f env_atek.yml
mamba activate atek
python3 -m pip install -e ./
```

Verify the libs are installed correctly

```bash
mamba list atek
```

Details in [Complete installation guide](docs/INSTALL.md) includes：

1. Python installation from source code, on local Fedora + Macbook.
2. Python installation from source code on AWS.
3. We will also provide 2 types of installation flavor: `[default]` and `[demo]`, where the latter would require more dependencies e.g. SAM2, Omni3D, etc.

## Download example data

To download, you should use the dataverse_url_parser.py script in ATEK's lib, with --download-wds-to-local flag. You can select which preprocessing config to download, train/validation split, and number of sequences to download. Here we will use cubercnn configuration as example. Please download input json [here, TODO: get link](link) and specify its path in atek_json_path. Replace output_data_dir with the path you want to put your data.

```python
python3 tools/dataverse_url_parser.py \
--config-name cubercnn \
--input-json-path ${atek_json_path} \
--output-folder-path ${output_data_dir}/
--max-num-sequences 2
--download-wds-to-local
```
To directly stream from data store, you can also use dataverse_url_parser.py script without the --download-wds_to-local flag, which will just create 3 yaml files, streamable_all/train/validation_tars.yaml, each containing the urls of the WDS shard files. These yaml files can be consumed by ATEK lib.

```python
python3 tools/dataverse_url_parser.py \
--config-name cubercnn \
--input-json-path ${atek_json_path} \
--output-folder-path ${output_data_dir}/
--max-num-sequences 2
```
You can visualize the WDS content, using the streamable yaml files.
```python
# Loading local WDS files
tar_file_urls = load_yaml_and_extract_tar_list(yaml_path = os.path.join(data_dir, "streamable_yamls", "streamable_validation_tars.yaml"))

# Batch size is None so that no collation is invoked
atek_dataloader = create_native_atek_dataloader(urls = tar_file_urls, batch_size=None, repeat_flag=False)

# Loop over all samples in DataLoader and visualize
atek_visualizer = NativeAtekSampleVisualizer(viz_prefix = "dataloading_visualizer", conf = viz_conf)
for atek_sample_dict in atek_dataloader:
    # First convert it back to ATEK data sample and visualize
    atek_visualizer.plot_atek_sample_as_dict(atek_sample_dict)
```


## Benchmarking, inference, and training examples
The script expects provide 2 csv files as input, groundtruth and predictions, both in the same predefined ATEK format (see below). The benchmarking script will compute and report a number of detection metrics. See example training and evaluation [jupyter notebook](/examples/demo_3_training_and_eval.ipynb) for more details.

Currently we support 2 ML tasks in benchmakring.
1. Static 3D object detection.
2. 3D surface reconstruction.


Usage:
```bash
python tools/benchmarking_static_object_detection.py \
--pred-csv {workdir}/eval_results/prediction_obbs.csv \
--gt-csv {workdir}/eval_results/gt_obbs.csv \
--output-file {workdir}/eval_results/atek_metrics.json
```

## Data-preprocessing, and visualization examples
Handling raw sensor data from Project Aria can be challenging due to the need for detailed knowledge of various Aria specifications, such as camera calibration, sensor behavior, and data synchronization.

ATEK simplifies this process by offering robust processing functionalities for all types of Aria data. This approach replaces complex data processing pipelines with just a few API calls, using simple configuration JSON files, making it more accessible and efficient for developers to get started.

Refer to the data preprocesing [jupyter notebook](examples/demo_1_data_preprocessing.ipynb) to see the details.

## Machine learning tasks supported by ATEK

ATEK now support static 3D object detection and surface reconstruction tasks. Find out more on [Machine learning tasks supported by ATEK](docs/ml_tasks.md) section.


## License
<img alt="license" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg"/>


## Contributors
