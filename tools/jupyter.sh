#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/$USER/jupyter_log/jupyter-%j.log
source activate base
conda activate atek
## jupyter-lab --ip=0.0.0.0 --port=${1:-8889} # use your desired port
jupyter-lab --ip=0.0.0.0 --port=${1:-9999}
