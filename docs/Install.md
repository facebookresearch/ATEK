# Installation

We provided 2 ways to install ATEK:

1. If you just need **the core functionalities of ATEK**, including data pre-processing, data loading, and visualization, you can simply [install ATEK's core lib](#core-lib-installation)
2. If you want to run the CubeRCNN demos and all task-specific evaluation benchmarking, you can follow this guide to [install **full dependencies**](#install-all-dependencies-using-mambaconda).

## Core lib installation

1. First install PyTorch following the [official guide](https://pytorch.org/).
2. Then install ATEK core lib:

```
pip install projectaria-atek
```

## Full dependencies installation using Mamba/Conda

The following steps will install ATEK with full dependencies, where you will be able to run all task-specific evaluations, along with our CubeRCNN examples. We have tested this under CUDA 12.4 + Python 3.9, on platforms including local Linux, AWS, and M1 MacBook. Note that all `mamba` commands below can be replaced by `conda`.

1. Install **`mamba` (recommended)** by following instructions [here](https://github.com/conda-forge/miniforge). `mamba` is a drop-in replacement for `conda` and is fully compatible with `conda` commands. For Mac / Linux, you can just do:

  ```bash
  cd ~/Documents/
  curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3-$(uname)-$(uname -m).sh
  mamba init  # may need to reopen terminal after this
  ```

2. Clone ATEK git repo.

  ```bash
  cd ~/Documents/
  git clone https://github.com/facebookresearch/ATEK.git
  cd ~/Documents/ATEK
  ```

3. Install dependencies in Mamba. You can choose from the following flavors: `linux`, `mac`, and `aws`:

  ```bash
  mamba env create -f envs/env_atek_linux[mac/aws].yml
  ```

4. Install ATEK library from source code:

  ```bash
  mamba activate atek_full
  cd ~/Documents/ATEK
  pip install -e .
  ```
