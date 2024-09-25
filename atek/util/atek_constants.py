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

from typing import Dict, List, Optional

import numpy as np


# Category ID <-> name mappings for ATEK taxonomy
ATEK_CATEGORY_NAME_TO_ID = {
    "other": 0,
    "chair": 1,
    "lamp": 2,
    "picture": 3,
    "sofa": 4,
    "table": 5,
    "bed": 6,
    "pillow": 7,
    "window": 8,
    "cabinet": 9,
    "mirror": 10,
    "floor mat": 11,
    "shelves": 12,
    "plant": 13,
    "dresser": 14,
    "vase": 15,
    "container": 16,
    "cart": 17,
    "clothes_rack": 18,
    "ladder": 19,
    "exercise_weight": 20,
    "fan": 21,
    "air_conditioner": 22,
    "mount": 23,
    "jar": 24,
    "clock": 25,
    "tent": 26,
    "electronic_device": 27,
    "battery_charger": 28,
    "cutlery": 29,
    "books": 30,
    "box": 31,
    "door": 32,
    "bottle": 33,
    "candle": 34,
    "cup": 35,
    "bin": 36,
    "display": 37,
    "tray": 38,
    "night stand": 39,
    "bookcase": 40,
    "refrigerator": 41,
    "camera": 42,
    "laptop": 43,
    "floor": 44,
    "walls": 45,
    "island": 46,
    "whiteboard": 47,
    "ceiling": 48,
}

ATEK_CATEGORY_ID_TO_NAME = {v: k for k, v in ATEK_CATEGORY_NAME_TO_ID.items()}
