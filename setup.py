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
        name="atek",
        version="0.2",
        description="Aria train and evaluation kits",
        author="Meta Reality Labs Research",
        packages=find_packages(),  # automatically discover all packages and subpackages
    )


if __name__ == "__main__":
    main()
