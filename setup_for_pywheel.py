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

from setuptools import find_packages, setup


def main():
    setup(
        name="projectaria-atek",
        version="0.0.6",
        description="Aria trainining and evaluation kits",
        author="Meta Reality Labs Research",
        packages=find_packages(),  # automatically discover all packages and subpackages
        install_requires=[
            "torch==2.4.0",
            "torchvision",
            "pillow==9.5.0",
            "torchmetrics==0.10.1",
            "fvcore",
            "fsspec",
            "iopath",
            "pandas",
            "tqdm",
            "scipy",
            "webdataset",
            "trimesh",
            "pybind11",
            "toolz",
            "jupyter",
            "notebook",
            "opencv-python",
            "projectaria-tools",
            "omegaconf",
            # Add other dependencies that can be resolved by pip here
        ],
    )


if __name__ == "__main__":
    main()
