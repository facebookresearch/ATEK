# Example: customization for SegmentAnything2 inference example

In this document, we will give an example on how to write customized code in ATEK for [SegmentAnything2 model](https://github.com/facebookresearch/segment-anything-2). User is also encouraged to check out [Demo_4](../examples/Demo_4_Sam2_example.ipynb) for end-to-end inference workflow.

## `SAM2` introduction

`sam2` is a image / video segmentation model designed to handle arbitrary objects in images.

To run inference on a `SAM2` model, the input should contain the following data:

1. An upright RGB image.
2. Prompt information to guide the segmentation process, e.g. 2D object bounding boxes to specify segmentation regions.

## Preprocessing Aria Data with `SAM2` requirements

In ATEK, users can customize preprocessing by simply adjusting the preprocessing configuration yaml file. For `SAM2`'s requirements listed above, we can adjust the following yaml fields accordingly (see [config.yaml](../atek/configs/obb_preprocess_base.yaml) for example config file):

- Setting the `selected` flags in `processors` will pick RGB camera data, and object annotation data to include in preprocessing.

  ```
  processors:
        rgb:
            selected: true
        obb_gt:
            selected: true
  ```

- Setting the following flags will transform the RGB image to upright position.

  ```
  rgb:
        rotate_image_cw90deg: true
  ```

- Setting the following value will automatically sync rgb image with groundtruth annotation data, with a tolerance window:

  ```
  tolerance_ns: 10_000_000
  ```

Then user can simply run the following code to generate preprocessed WDS data that is suitable for `sam2` model inference:

```python
preprocessor = create_general_atek_preprocessor_from_conf(
    conf=sam2_preprocessing_config,
    raw_data_folder="/path/to/raw/data",
    sequence_name="sequence_01",
    output_wds_folder = "./output",
)

num_samples = preprocessor.process_all_samples(write_to_wds_flag = True, viz_flag = True)
```

## Convert ATEK data format to `sam2` format via a `ModelAdaptor`

To feed ATEK-preprocessed data into a `sam2` model for inference, a `ModelAdaptor` class is needed. We will show the core code below, while users are encouraged to checkout the [source code](../atek/data_loaders/sam2_model_adaptor.py) for details.

```python
class Sam2ModelAdaptor:
    @staticmethod
    def get_dict_key_mapping_all():
        dict_key_mapping = {
            "mfcd#camera-rgb+images": "image",
            "gt_data": "gt_data",
        }
        return dict_key_mapping

    def atek_to_sam2(self, data):
        for atek_wds_sample in data:
            sample = {}

            # Add images
            # from [1, C, H, W] to [H, W, C]
            image_torch = atek_wds_sample["image"].clone().detach()
            image_np = image_torch.squeeze(0).permute(1, 2, 0).numpy()
            sample["image"] = image_np

            # Select boxes as prompts
            obb2_gt = atek_wds_sample["gt_data"]["obb2_gt"]["camera-rgb"]
            bbox_ranges = obb2_gt["box_ranges"][
                :, [0, 2, 1, 3]
            ]  # xxyy -> xyxy
            sample["boxes"] = bbox_ranges.numpy()

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
```

Within this class:

- `get_dict_key_mapping_all()` function returns a mapping from ATEK dictionary keys to `sam2` dictionary keys. Here, since we only need the RGB image and the 2D bounding box information, we only need to map 2 keys. ATEK will automatically discard other key-value content in ATEK dict.
- `atek_to_sam2` is the actual data transform function. The input `data` is a generator of dictionaries, whose keys are already remapped by `get_dict_key_mapping_all()`. We perform 2 operations in this data transform:

  - Reshape RGB image tensor from `[1, Channel, Height, Width]` to `[Height, Width, Channel]`, store it in `sample` dict.
  - From `gt_data` dictionary, take the first `num_box` 2D bounding boxes for the current RGB image, re-order the box corners from `[xmin, xmax, ymin, ymax]` to `[xmin, ymin, xmax, ymax]`, and store them to `sample` dict.

`create_atek_dataloader_as_sam2` is a thin wrapper on top of the `Sam2ModelAdaptor` class, which allows user to input the URLs of the WDS files, and return a PyTorch DataLoader object that produces data samples that can be directly used in SAM2 inference:

```
sam2_dataloader = create_atek_dataloader_as_sam2(tar_list)
first_sam2_sample = next(iter(sam2_dataloader))
print(f"Loading WDS into SAM2 format, each sample contains the following keys: {first_sam2_sample[0].keys()}")
```

## SAM2 model inference

With the created PyTorch DataLoader, user can run SAM2 inference easily with the following code:

```python
# create SAM2 predictor
predictor = SAM2ImagePredictor(build_sam2(sam2_model_cfg, sam2_model_checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for sam_dict in sam2_dataloader:
        # perform inference
        predictor.set_image(sam_dict["image"])

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=sam_dict["boxes"],
            multimask_output=False,
        )

        # Visualize results
        ...
```
