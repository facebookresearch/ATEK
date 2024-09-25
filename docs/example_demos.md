# Example: Demo notebooks

We provided several demo notebooks to show how to run end-to-end ML workflows in ATEK.

## [Demo_1](../examples/Demo_1_data_preprocessing.ipynb)

This demo shows how to use ATEK to preprocess an AriaDigitalTwin data sequence, visualize the results, and write to WebDataset (WDS) files.

## [Demo_2](../examples/Demo_2_data_store_and_inference.ipynb)

This demo shows how to access preprocessed data from ATEK data store, how to load ATEK WDS data as a PyTorch DataLoader, how to run CubeRCNN model inference, and how to use ATEK's benchmarking script to evaluate the inference results. Note that you will need to [install full dependencies](./Install.md#full-dependencies-installation-using-mambaconda) in order to run this demo.


## [Demo_3](../examples/Demo_3_model_training.ipynb)
This demo shows how to run a mini-training of CubeRCNN model, on ATEK preprocessed data. Note that you will need to [install full dependencies](./Install.md#full-dependencies-installation-using-mambaconda) in order to run this demo.


## [Demo 4](../examples/Demo_4_Sam2_example.ipynb)
This demo shows how to run SAM2 infererence on ATEK preprocessed data. Note that you will need to install [SAM2](https://github.com/facebookresearch/segment-anything-2) model in order to run this demo.
