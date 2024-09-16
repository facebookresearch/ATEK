# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import List, Optional

import torch

from atek.data_loaders.atek_wds_dataloader import (
    load_atek_wds_dataset,
    simple_list_collation_fn,
)

from webdataset.filters import pipelinefilter


class Sam2ModelAdaptor:
    """
    A simple model adaptor class to convert ATEK WDS data to Sam2 format.
    Currently used for inference examples.
    """

    def __init__(
        self,
        num_boxes: int = 5,  # num of 2d bboxes to keep as prompts
    ):
        self.num_boxes = num_boxes

    @staticmethod
    def get_dict_key_mapping_all():
        dict_key_mapping = {
            "mfcd#camera-rgb+images": "image",
            # Needs GT data to get 2D bboxes, can be used as prompts.
            "gt_data": "gt_data",
        }
        return dict_key_mapping

    def atek_to_sam2(self, data):
        """
        Core data conversion function
        """
        for atek_wds_sample in data:
            sample = {}

            # Add images
            # from [1, C, H, W] to [H, W, C]
            image_torch = atek_wds_sample["image"].clone().detach()
            image_np = image_torch.squeeze(0).permute(1, 2, 0).numpy()
            sample["image"] = image_np

            # Select
            obb2_gt = atek_wds_sample["gt_data"]["obb2_gt"]["camera-rgb"]
            num_box = min(self.num_boxes, len(obb2_gt["category_names"]))
            bbox_ranges = obb2_gt["box_ranges"][
                0:num_box, [0, 2, 1, 3]
            ]  # First K bboxes, [K, 4], xxyy -> xyxy
            sample["boxes"] = bbox_ranges.numpy()  # xxyy -> xyxy

            yield sample


def create_atek_dataloader_as_sam2(
    urls: List[str],
    batch_size: Optional[int] = None,
    repeat_flag: bool = False,
    shuffle_flag: bool = False,
    num_workers: int = 0,
    num_prompt_boxes: int = 5,
) -> torch.utils.data.DataLoader:

    adaptor = Sam2ModelAdaptor(num_boxes=num_prompt_boxes)
    wds_dataset = load_atek_wds_dataset(
        urls,
        batch_size=batch_size,
        dict_key_mapping=Sam2ModelAdaptor.get_dict_key_mapping_all(),
        data_transform_fn=pipelinefilter(adaptor.atek_to_sam2)(),
        collation_fn=simple_list_collation_fn,
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )

    return torch.utils.data.DataLoader(
        wds_dataset, batch_size=None, num_workers=num_workers, pin_memory=True
    )
