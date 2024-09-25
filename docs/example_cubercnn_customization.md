# Example: customization for CubeRCNN example

In this document, we will give an example of how to write customized code in ATEK for a 3D detection model called [`cubercnn`](https://github.com/facebookresearch/omni3d). User is also encouraged to check out [Demo_2](../examples/Demo_2_data_store_and_inference.ipynb) for end-to-end inference workflow, and [Demo_3](../examples/Demo_3_model_training.ipynb) and [Example: training](./example_training.md) for end-to-end training workflow.

## `cubercnn` introduction

`cubercnn` is a 3D detection model based on `maskrcnn`. It takes single frame RGB image as input, and detects 3D objects in the image, with category information.

To train a `cubercnn` model, it requires the model input to contain the following data:

1. An upright RGB image, with linear camera model.
2. Object annotations in both 2D and 3D.
3. camera calibration and pose data.

## Preprocessing Aria Data with `cubercnn` requirements

In ATEK, users can customize preprocessing by simply adjusting the preprocessing configuration yaml file. For `cubercnn`'s requirements listed above, we can adjust the following yaml fields accordingly (see [cubercnn config](./preprocessing_configurations.md#cubercnn-config) for full configuration):

- Setting the `selected` flags in `processors` will pick RGB camera data, trajectory (pose) data, and object annotation data to include in preprocessing.

  ```
  processors:
        rgb:
            selected: true
        mps_traj:
            selected: true
        obb_gt:
            selected: true
  ```

- Setting the following flags will transform the RGB image to upright position, and undistort the camera model from Fisheye to Linear. -

  ```
  rgb:
        undistort_to_linear_camera: true
        rotate_image_cw90deg: true
  ```

- Setting the following value will automatically sync data from different processors, with a tolerance window:

  ```
  tolerance_ns: 10_000_000
  ```

- Setting the following value will subsample the sequence by a factor of 2:

  ```
  stride_length_in_num_frames: 2
  ```

Then user can simply run the following code to generate preprocessed WDS data that is suitable for `cubercnn` model training:

```python
preprocessor = create_general_atek_preprocessor_from_conf(
    conf=cubercnn_preprocessing_config,
    raw_data_folder="/path/to/raw/data",
    sequence_name="sequence_01",
    output_wds_folder = "./output",
)

num_samples = preprocessor.process_all_samples(write_to_wds_flag = True, viz_flag = True)
```

## Convert ATEK data format to `cubercnn` format via a `ModelAdaptor`

To feed ATEK-preprocessed data into a `cubercnn` model for training, a `ModelAdaptor` class is needed. We will show the core code below, while users are encouraged to checkout the [source code](../atek/data_loaders/cubercnn_model_adaptor.py) for details.

```python
class CubeRCNNModelAdaptor:
    ...
    @staticmethod
    def get_dict_key_mapping_all():
        dict_key_mapping = {
            "mfcd#camera-rgb+images": "image",
            "mfcd#camera-rgb+projection_params": "camera_params",
            "mfcd#camera-rgb+camera_model_name": "camera_model",
            "mfcd#camera-rgb+t_device_camera": "t_device_rgbcam",
            "mfcd#camera-rgb+frame_ids": "frame_id",
            "mfcd#camera-rgb+capture_timestamps_ns": "timestamp_ns",
            "mtd#ts_world_device": "ts_world_device",
            "sequence_name": "sequence_name",
            "gt_data": "gt_data",
        }
        return dict_key_mapping

    def atek_to_cubercnn(self, data):
        for atek_wds_sample in data:
            sample = {}
            self._update_camera_data_in_sample(atek_wds_sample, sample)
            self._update_T_world_camera(atek_wds_sample, sample)

            # Skip if no gt data
            if "gt_data" in atek_wds_sample and len(atek_wds_sample["gt_data"]) > 0:
                self._update_gt_data_in_sample(atek_wds_sample, sample)

            yield sample
    ...

def create_atek_dataloader_as_cubercnn(
    urls: List[str],
    batch_size: Optional[int] = None,
    repeat_flag: bool = False,
    shuffle_flag: bool = False,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    cubercnn_model_adaptor = CubeRCNNModelAdaptor()

    wds_dataset = load_atek_wds_dataset(
        urls,
        batch_size=batch_size,
        dict_key_mapping=CubeRCNNModelAdaptor.get_dict_key_mapping_all(),
        data_transform_fn=pipelinefilter(cubercnn_model_adaptor.atek_to_cubercnn)(),
        collation_fn=cubercnn_collation_fn,
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )

    return torch.utils.data.DataLoader(
        wds_dataset, batch_size=None, num_workers=num_workers, pin_memory=True
    )
```

Within this class:

- `get_dict_key_mapping_all()` function returns a mapping from ATEK dictionary keys to `cubercnn` dictionary keys.
- `atek_to_cubercnn` is the actual data transform function. The input `data` is a generator of dictionaries, whose keys are already remapped by `get_dict_key_mapping_all()`. We perform 3 high-level data transform operations:

  - update the camera data (image + calibration).
  - compute the pose of the RGB camera in world.
  - update object annotation ground truth data.

`create_atek_dataloader_as_cubercnn` is a thin wrapper on top of the `CubeRCNNModelAdaptor` class, which allows user to input the URLs of the WDS files, and return a PyTorch DataLoader object that produces data samples that are converted to CubeRCNN format:

```
cubercnn_dataloader = create_atek_dataloader_as_cubercnn(urls = tar_file_urls, batch_size = 6, num_workers = 1)
first_cubercnn_sample = next(iter(cubercnn_dataloader))
print(f"Loading WDS into CubeRCNN format, each sample contains the following keys: {first_cubercnn_sample[0].keys()}")
```

## CuberCNN model trainng / inference

With the created Pytorch DataLoader, user will be able to easily run model training or inference for CubeRCNN model.

**Training script**
```python
# Load pre-trained model for training
model_config, model = create_training_model(model_config_file, model_ckpt_path)

# Training loop
for cubercnn_input_data in tqdm(
    cubercnn_dataloader,
    desc="Training progress: ",
):
    # Training step
    loss = model(cubercnn_input_data)
    losses = sum(loss.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
...
```


**Inference script**

```python
# Load pre-trained model for inference
model_config, model = create_inference_model(model_config_file, model_ckpt_path)

# Inference loop
for cubercnn_input_data in tqdm(
    cubercnn_dataloader,
    desc="Training progress: ",
):
    # Inference step
    cubercnn_model_output = model(cubercnn_input_data)
    ...
```
# Inference step
