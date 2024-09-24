# Model Adaptors

## Introduction

To adapt ATEK data samples from their native format for specific models, the data must undergo conversion. This conversion typically involves several lightweight operations, including:

1. **Dictionary key remapping**
2. **Tensor reshaping and remapping**
3. **Additional data transformations**

```python
class ModelAdaptor{
    # Define key remapping
    @staticmethod
    def get_dict_key_mapping_all():
        dict_key_mapping = {
            "mfcd#camera-rgb+images": "image",
            "gt_data": "gt_data",
            ...
        }
        return dict_key_mapping

    # Define data transform function
    def atek_to_model(self, data):
        # transform data
        ...
        yield sample
}

# A thin wrapper API to create a PyTorch DataLoader
def create_atek_dataloader_as_model(
    batch_size: Optional[int] = None,
    repeat_flag: bool = False,
    shuffle_flag: bool = False,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    adaptor = ModelAdaptor()

    wds_dataset = load_atek_wds_dataset(
        urls,
        batch_size=batch_size,
        dict_key_mapping=ModelAdaptor.get_dict_key_mapping_all(),
        data_transform_fn=pipelinefilter(adaptor.atek_to_model)(),
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )

    return torch.utils.data.DataLoader(
        wds_dataset, batch_size=None, num_workers=num_workers, pin_memory=True
    )
```

## Model Adaptor class specifications

In ATEK, `ModelAdaptor` classes are utilized to manage this data conversion process, and we provide streamlined APIs that generate a PyTorch DataLoader. This DataLoader outputs data samples ready to be directly fed into the model. Users are encouraged to develop their own `ModelAdaptor` classes tailored to their specific models. Here we show the basic concepts for building a `ModelAdaptor` class. User can refer to [`sam2_model_adaptor`](../atek/data_loaders/sam2_model_adaptor.py) and [`cubercnn_model_adaptor`](../atek/data_loaders/cubercnn_model_adaptor.py) for example implementations.

### `ModelAdaptor` Class API

#### Methods

- **`get_dict_key_mapping_all()`**: This method defines the key remappings from an ATEK data sample dictionary to a model-specific dictionary. Note that keys not included in the remapping will be omitted from the ATEK dictionary.

- **`atek_to_model`**: This function takes handles the actual data conversion, including tensor reshaping and re-ordering, some data transformations, etc. It takes an ATEK-format dictionary as input (with keys remapped), and returns a user-defined `sample` instance (e.g. `Dict`).

### `create_atek_dataloader_[as_model]` Function API

This is a thin wrapper API on top of `ModelAdaptor`, which loads ATEK WDS file URLs, and return a PyTorch DataLoader that yields converte data samples.

#### Parameters

- **urls** (`List[str]`): List of URLs or paths to WDS files.
- **batch_size** (`Optional[int]`, optional): Batch size for the DataLoader. If `None`, batch size is determined by the underlying dataset. Defaults to `None`.
- **repeat_flag** (`bool`, optional): Flag to repeat the dataset indefinitely. Defaults to `False`.
- **shuffle_flag** (`bool`, optional): Flag to shuffle the dataset. Defaults to `False`.
- **num_workers** (`int`, optional): Number of worker threads for loading data. Defaults to `0`.

#### Returns

- **torch.utils.data.DataLoader**: A DataLoader instance that yields converted data samples.
