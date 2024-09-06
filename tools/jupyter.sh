#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --job-name=jupyter
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
## change $USER to actual user name
#SBATCH --output=/data/home/$USER/jupyter_log/jupyter-%j.log
source activate base
conda activate atek
## jupyter-lab --ip=0.0.0.0 --port=${1:-8889} # use your desired port
jupyter-lab --ip=0.0.0.0 --port=${1:-9999}
