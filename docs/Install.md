# Installation

## Prerequisites

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
   the following flavors: `core`, `full`, and `aws`: a. `core`: these are
   required by the ATEK core lib:
   ```
     mamba env create -f envs/env_atek_core.yml
     mamba activate atek_core
   ```
   b. `full`: these are required to run the CubeRCNN demo and examples,
   specifically you need to install both
   [`Detectron2`](https://github.com/facebookresearch/detectron2) and a forked
   copy of [`omni3d`](https://github.com/YLouWashU/omni3d). You can follow their
   official installation, or use ATEK-provided mamba env file:
   ```
     mamba env create -f envs/env_atek_full.yml
     mamba activate atek_full
   ```
   c. `aws`: if you are running on AWS, you can run the following command, it
   will install the `full` version:
   ```
     mamba env create -f envs/env_atek_aws.yml
     mamba activate atek_aws
   ```

## Install ATEK library:

First, make sure you have ran the `mamba activate ...` command in the above
section to activate the correct environment.

Then run the following:

```
cd ~/Document/ATEK
pip install -e ./
```
