# Model training and inference in ATEK

ATEK provides a robust framework for loading preprocessed data from preprocessed WebDataset (WDS) files into machine learning models. This document provides a guide on how to utilize ATEK's functionalities to efficiently load data into ML models.

```python
data_loader = create_native_atek_dataloader(urls, batch_size, num_workers)
# data_loader = create_atek_dataloader_for_some_model(urls, batch_size, num_workers) # with model-specific conversion

for data_sample in data_loader:
    # inference step
    output = model(data_sample)
    ...

    # or training step
    loss_dict = model(data)
    losses = sum(loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    ...
```

## Creating PyTorch Dataloader from ATEK preprocessed WDS files

From [ATEK preprocessed WDS files](./preprocessing.md), users can call the following API to create a **PyTorch DataLoader** object that yields data samples as ATEK-format python dictionaries. ATEK also supports adding a custom data transform function via [ModelAdaptors](./ModelAdaptors.md), to convert ATEK-format data samples to model-specific data formats. The returned DataLoader can be used in any training or inference workflows as any other iterable data loader.
See [example_training.md](./example_training.md) for how to run a full training example with CubeRCNN model in ATEK.

### `create_native_atek_dataloader` Function API

Creates a native ATEK DataLoader for loading and processing data from WebDataset (WDS) sources.

#### Parameters

- **urls** (`List[str]`): List of URLs or paths to WDS files.
- **nodesplitter** (`Callable`, optional): Node splitter function. Defaults to `wds.shardlists.single_node_only`.
- **dict_key_mapping** (`Optional[Dict[str, str]]`, optional): Dictionary key mapping for renaming keys in the dataset. Defaults to `None`.
- **data_transform_fn** (`Optional[Callable]`, optional): Data transformation function applied to each sample. Defaults to `None`.
- **collation_fn** (`Optional[Callable]`, optional): Collation function for aggregating samples into batches. Defaults to `atek_default_collation_fn`.
- **batch_size** (`Optional[int]`, optional): Batch size for the DataLoader. If `None`, batch size is determined by the underlying dataset. Defaults to `None`.
- **repeat_flag** (`bool`, optional): Flag to repeat the dataset indefinitely. Defaults to `False`.
- **shuffle_flag** (`bool`, optional): Flag to shuffle the dataset. Defaults to `False`.
- **num_workers** (`int`, optional): Number of worker threads for loading data. Defaults to `0`.

#### Returns

- **torch.utils.data.DataLoader**: A DataLoader instance configured for loading and processing ATEK data from WDS sources.
