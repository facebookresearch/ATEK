# Machine Learning tasks supported by ATEK: static 3D object detection

## Task introduction

Static 3D object detection is the task of perceiving objects in 3D from image / video inputs. The primary goal is to accurately estimate the 3D location, orientation, and extent of each object within the scene. This is achieved by placing a tight, oriented 3D bounding box around each detected object, effectively translating 2D data into a 3D spatial understanding.

**Input**: For 3D object detection task in Aria, the input would include the images captured by the RGB camera. In some variations, additional inputs may include Slam camera images, camera calibration, and trajecotry + point-cloud data from Aria's [Machine Perception Services](https://facebookresearch.github.io/projectaria_tools/docs/ARK/mps).

**Output**: the task expects to output detected 3D objects within the scene. Each detection consists of the pose of the 3D bounding box enclosing the object, and the detected category name of the object.

## ATEK standardized Evaluation library

In ATEK, we provide an standardized evaluation library for this task, and supports the following datasets and models:
- **supported Aria datasets**:
  - [Aria Digital Twin Dataset](https://www.projectaria.com/datasets/adt/)
  - [Aria Synthetic Environments Dataset](https://www.projectaria.com/datasets/ase/)
- **supported ML models**:
  - [Cube R-CNN](https://github.com/facebookresearch/omni3d/tree/main)
  - [Egocentric Voxel Lifting](https://arxiv.org/abs/2406.10224)

### Dependencies

User will need to install [`PyTorch3D`](https://github.com/facebookresearch/pytorch3d) following their official guidance in order to run ATEK evaluation for the object detection task. ATEK also provides a guide to install this using mamba / conda in [Install.md](./Install.md#full-dependencies-installation-using-mambaconda)

### ATEK Evaluation dataset

Follow the instructions in [ATEK Data Store](http://docs/atek_data_store.md) section to download **ASE** dataset with **cubercnn-eval** configuration.

### Prediction Files Format

For this task, ATEK defines both the prediction file and groundtruth files to be a `csv` file, with the following fields. For each Aria sequence, user just need to parse the GT data and their model output into this format (see [here](../atek/evaluation/static_object_detection/obb3_csv_io.py) for an example):

- **`​​time_ns`**: Timestamp of the detection in nanoseconds.
- **`t[x,y,z]_world_object`**: coordinate of the object's position in the world coordinate system, in meters.
- **`q[w,x,y,z]_world_object`**: orientation in quaternions of the object in the world coorindate system.
- **`scale_[x,y,z]`**: Size of the object.
- **`name`**: category name of the object.
- **`instance`**: Unique identifier for each instance of an object in a scene. If not available, this value is set to 1.
- **`sem_id`**: category id of the object.
- **`prob`**: Probability that the object is correctly identified, ranging from 0 to 1\. For groundtruth, this value is set to 1.

For multiple sequences, please follow the following folder structure to store the results:

```
results/
├── sequence_01/
│   ├── prediction.csv
│   └── gt.csv
├── sequence_02/
│   ├── prediction.csv
│   └── gt.csv
└── sequence_03/
    ├── prediction.csv
    └── gt.csv
```

### Run ATEK benchmarking

Once the `prediction.csv` and `gt.csv` files are prepared into the following folder structure, user can run ATEK-provided benchmarking script to generate metrics: [`tools/benchmarking_static_object_detection.py`](../tools/benchmarking_static_object_detection.py).

#### `benchmarking_static_object_detection.py` script

This Python script evaluates the performance of static 3D object detection models using oriented bounding boxes (OBBs). It supports both dataset-level and file-level evaluations.

##### Command Line Arguments

- `--input-folder`: If specificied, will perform Dataset-level evaluation. Path to the folder containing ground truth and prediction CSV files.
- `--pred-csv`: Filename of the prediction csv file.
- `--gt-csv`: Filename of the groundtruth csv file..
- `--output-file`: Path where the output metrics JSON file will be saved.
- `--iou-threshold`:[Optional] IOU threshold for determining a match between predicted and actual OBBs, used in computing average precision / recall metrics (default: 0.2).
- `--confidence-lower-threshold`:[Optional] Minimum confidence score for predictions to be considered, used in computing average precision / recall metrics (default: 0.3).
- `--max-num-sequences`:[Optional] Maximum number of sequences to evaluate (default: -1, which means no limit).

##### Reported metrics

- **`map_3D`** : the mean average precision computed across all the categories in the dataset, over a range of 3D IoU thresholds of [0.05, 0.10,..., 0.50].
- **`precision@IoU0.2,Confidence0.3`** : averaged precision value over all sequences, where the IoU threshold and confidence levels can be set through commandline arguments.
- **`recall@IoU0.2,Cofidence0.3`**: averaged recall value over all sequences, where the IoU threshold and confidence levels can be set through commandline arguments.
- **`map_per_class@...`**: the mean average precision computed

See the generated `--output-file` for full metrics.
