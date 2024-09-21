# Installation

## Install ATEK-core lib

1. First install PyTorch following the [official guide](https://pytorch.org/).
2. Install ATEK core lib. This should allow you to use all core functionalities
   of ATEK, including data preprocessing, data loading, and visualization

```
pip install projectaria-atek
```

## [Optional] Install dependencies for object detection evaluation lib

If you want to run ATEK's object detection evaluation lib and benchmarking, you
need to install Pytorch3D library by following their
[official guide](https://github.com/facebookresearch/pytorch3d). Or, ATEK
provides a [one-stop way](#bookmark_one_stop) to install full dependencies in
Mamba.

## [Optional] Install dependencies for running CubeRCNN examples

If you want to run ATEK's example inference and training scripts using CubeRCNN,
you need to install both
[Detectron2](https://github.com/facebookresearch/detectron2), and
[(a forked copy of) CubeRCNN](https://github.com/YLouWashU/omni3d). Or, ATEK
provides a [one-stop way](#bookmark_one_stop) to install full dependencies in
Mamba:

## One stop way to install all dependencies using Mamba {#bookmark_one_stop}

The following steps will install ATEK with full dependencies. We have tested
this under CUDA 12.4 + Python 3.9, AWS, and M1 macbook.

1. Install **`mamba` (recommended)** by following instructions
   [here](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).
   `mamba` is a drop-in replacement for `conda` and is fully compatible with
   `conda` commands. For Mac / Linux, you can just do:

```
cd ~/Documents/

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
mamba init; # may need to reopen terminal after this
```

2. Clone ATEK git repo.

```
cd ~/Documents/
git clone https://github.com/facebookresearch/ATEK.git
cd ~/Documents/ATEK
```

3. Install dependencies in Mamba. Depending on use cases, you can choose from
   the following flavors: `full`, `mac`, and `aws`:

   a. `full`: if you are on linux, run:

   ```
     mamba env create -f envs/env_atek_full.yml
   ```

   c. `aws`: if you are running on AWS, you can run the following command, it
   will install the `full` version:

   ```
     mamba env create -f envs/env_atek_aws.yml
   ```

   d. `mac`: if you are using mac, you can run the following command:

   ```
     mamba env create -f envs/env_atek_mac.yml
   ```

4. Install ATEK library from source code:

```
mamba activate atek_full
cd ~/Document/ATEK
pip install -e .
```
