# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import torch
from atek.data_preprocess.util.file_io_utils import (
    merge_tensors_into_dict,
    separate_tensors_from_dict,
)
from atek.data_preprocess.util.tensor_utils import check_dicts_same_w_tensors


class FileIoUtilsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_separate_tensors_from_dict(self) -> None:
        input_dict = {
            "key_1": {
                "subkey_1": torch.tensor([1, 2, 3]),
                "subkey_2": torch.tensor([4, 5, 6]),
                "subkey_3": "value",
            },
            "key_2": [2, 3, 4],  # List should stay
            "key_3": {
                "key_4": {
                    "key_5": torch.tensor([7, 8]),
                    "key_6": {
                        "key_8": 1000,
                        "key_9": torch.tensor([10, 11, 12]),
                    },
                }
            },
        }

        # Test the function
        dict_wo_tensors, tensor_dict = separate_tensors_from_dict(input_dict)

        expected_dict_wo_tensors = {
            "key_1": {
                "subkey_3": "value",
            },
            "key_2": [2, 3, 4],  # List should stay
            "key_3": {
                "key_4": {
                    "key_6": {
                        "key_8": 1000,
                    }
                }
            },
        }
        expected_tensor_dict = {
            "key_1+subkey_1": torch.tensor([1, 2, 3]),
            "key_1+subkey_2": torch.tensor([4, 5, 6]),
            "key_3+key_4+key_5": torch.tensor([7, 8]),
            "key_3+key_4+key_6+key_9": torch.tensor([10, 11, 12]),
        }

        self.assertDictEqual(dict_wo_tensors, expected_dict_wo_tensors)
        self.assertEqual(set(tensor_dict.keys()), set(expected_tensor_dict.keys()))
        for key in tensor_dict:
            self.assertTrue(
                torch.allclose(tensor_dict[key], expected_tensor_dict[key], atol=0)
            )

    def test_separate_tensor_from_dict_round_trip(self) -> None:
        input_dict = {
            "key_1": {
                "subkey_1": torch.tensor([1, 2, 3]),
                "subkey_2": torch.tensor([4, 5, 6]),
                "subkey_3": "value",
            },
            "key_2": [2, 3, 4],  # List should stay
            "key_3": {
                "key_4": {
                    "key_5": torch.tensor([7, 8]),
                    "key_6": {
                        "key_8": 1000,
                        "key_9": torch.tensor([10, 11, 12]),
                    },
                }
            },
        }

        dict_wo_tensors, tensor_dict = separate_tensors_from_dict(input_dict)
        result_dict = merge_tensors_into_dict(dict_wo_tensors, tensor_dict)

        self.assertTrue(check_dicts_same_w_tensors(input_dict, result_dict))
