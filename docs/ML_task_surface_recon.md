# Machine Learning tasks supported by ATEK: 3D surface reconstruction

## Task introduction

Surface reconstruction refers to the process of estimating and recreating the 3D geometry of surfaces in a scene from egocentric video data. This task involves predicting the occupancy of a 3D voxel grid, which is aligned with the gravity direction of the scene. Local surface predictions are made for each voxel, which indicate whether the voxel is free space, part of a surface, or behind a surface.

**Input:** For 3D surface reconstruction task in Aria, the input data include camera images from RGB and 2 SLAM cameras, their calibration information, and trajectory + point cloud data from Aria's [Machine Perception Services](https://facebookresearch.github.io/projectaria_tools/docs/ARK/mps).

**Output:**: the task expects to output a mesh file that describes the 3D surfaces of the scene.

## ATEK standardized Evaluation library

In ATEK, we provide an standardized evaluation library for this task, and supports the following datasets and models:
- **supported Aria datasets**:
  - [Aria Digital Twin Dataset](https://www.projectaria.com/datasets/adt/)
  - [Aria Synthetic Environments Dataset](https://www.projectaria.com/datasets/ase/)
- **supported ML models**:
  - [Egocentric Voxel Lifting](https://arxiv.org/abs/2406.10224)

### Evaluation dataset

1. Follow the instructions in [ATEK Data Store](http://docs/atek_data_store.md) section to download **ASE** dataset with **cubercnn-eval** configurations. This will dowload the input data for surface recon eval.

2. Go to [Aria Synthetic Environments Dataset](https://www.projectaria.com/datasets/ase/) website, input your email at the bottom of the page, and click on **ACCESS THE DATASETS**. Then click on **EFM3D Eval Meshes** button, it will download a file named `ase_mesh_download_urls.json`. Then run this script [`tools/ase_mesh_downloader.py`](../tools/ase_mesh_downloader.py) to download the groundtruth mesh files for surface recon eval.

### Prediction file format

In ATEK's surface reconstruction benchmarking, we require both the groundtruth and prediction results to be exported into `.ply` format. And for multiple sequences, please follow the following format to save the results:

```
results/
├── sequence_01/
│   ├── pred_mesh.ply
│   └── gt_mesh.ply
├── sequence_02/
│   ├── pred_mesh.ply
│   └── gt_mesh.ply
└── sequence_03/
    ├── pred_mesh.ply
    └── gt_mesh.ply
```

### Run ATEK benchmarking

User can run ATEK-provided benchmarking script to generate metrics: [`tools/benchmarking_surface_reconstruction.py`](../tools/benchmarking_surface_reconstruction.py).

#### `benchmarking_surface_reconstruction.py` script

This Python script evaluates the performance of surface reconstruction models from mesh files. It supports both dataset-level and file-level evaluations.

##### Command Line Arguments

- `--input-folder`: If specificied, will perform Dataset-level evaluation. Path to the folder containing ground truth and prediction CSV files.
- `--pred-mesh`: Filename of the prediction mesh PLY file.
- `--gt-mesh`: Filename of the groundtruth mesh PLY file..
- `--output-file`: Path where the output metrics JSON file will be saved.

##### Reported metrics

- **`Accuracy`**: The average distance, from prediction mesh to groundtruth mesh, in the unit of meters. This reflects how accurate the reconstructed surfaces are.
- **`Completeness`**: The average distance from groundtruth mesh to prediction mesh, in the unit of meters. This reflects how complete the reconstructed surfaces are.
- **`Precision@5cm`**: The ratio of point-to-mesh distances that are within 5 cm, where points are sampled on prediction, and mesh takes the groundtruth mesh.
- **`Recall@5cm`**: The ratio of point-to-mesh distances that are within 5 cm, where points are sampled on groundtruth, and mesh takes the prediction mesh.

See the generated `--output-file` for full metrics.
