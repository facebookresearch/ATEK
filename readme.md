# ATEK
Aria train and evaluation kits.

## How to set up ATEK on AWS
### 1. Make sure you have the right AWS access and can ssh to the frl surreal reserach AWS cluster.
### 2. In the submit node, go to the /work_1a/$USER folder to clone the ATEK repo. Make sure the github ssh access set up properly.
```
git clone git@github.com:fairinternal/ATEK.git
```

### 3. Create ATEK conda env (MUST BE in node shell)
```
srun -N 1 --gres=gpu:1 --pty bash
```

Init conda
```
export PATH=/data/home/$USER/miniconda3/bin:$PATH
conda init bash
```

Then create a conda environment with the ATEK environment.yaml

```
conda create -f environment_aws.yml
```

### 4. Setup the pybind library
In conda atek env
```
python setup.py build
```                                                                                                            